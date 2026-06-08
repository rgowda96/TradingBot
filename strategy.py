#!/usr/bin/env python3
"""
ETH 15m pyramiding-short — LIVE FORWARD paper trade (starts 'today').

On first run it anchors a start price = current ETH price and records the first
SHORT there. Every subsequent run it re-reads 15m candles since the start and
adds a 5-ETH SHORT for every +50 pt up-move, never booking, until 90% of the
$10k capital is committed as margin (then HOLD).

Two readings are tracked side by side:
  grid     -> short every +50 level ABOVE the first entry (pure "sell the rise")
  trailing -> re-arm the sell level off each new local low, so every +50 bounce
              is sold ("don't miss any up-move")

Usage:
  python3 strategy.py            # init if needed, then update
  python3 strategy.py --reset    # wipe state and start fresh from now
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone

# ---------------- parameters ----------------
SYMBOL      = "ETHUSDT"
INTERVAL    = "15m"
CAPITAL     = 10_000.0
LEVERAGE    = 20
MAX_MARGIN  = 0.90 * CAPITAL   # 9,000 -> stop adding past this, then HOLD
STEP        = 50.0             # add every +50 points
QTY         = 5.0             # ETH per add
MAINT_RATE  = 0.005           # 0.5% maintenance margin (liquidation estimate)
STATE_FILE  = "state.json"


def fetch_klines_since(start_ms):
    url = (f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}"
           f"&interval={INTERVAL}&startTime={int(start_ms)}&limit=1000")
    with urllib.request.urlopen(url, timeout=20) as r:
        raw = json.load(r)
    return [{"t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4])} for k in raw]


def fetch_price():
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return float(json.load(r)["price"])


def _rec(trades, ts, entry, cum_qty, cum_notional, margin_used):
    trades.append({
        "n": len(trades) + 1,
        "ts": ts,
        "time": datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "entry": round(entry, 2), "qty": QTY,
        "notional": round(entry * QTY, 2),
        "margin": round((QTY * entry) / LEVERAGE, 2),
        "cum_qty": round(cum_qty, 2),
        "avg_entry": round(cum_notional / cum_qty, 2),
        "margin_used": round(margin_used, 2),
    })


def run_grid(candles, start_price, start_ts):
    trades, margin_used, cum_qty, cum_notional, halted = [], 0.0, 0.0, 0.0, False
    # seed: first short at the anchor price, today
    margin_used = (QTY * start_price) / LEVERAGE
    cum_qty, cum_notional = QTY, start_price * QTY
    _rec(trades, start_ts, start_price, cum_qty, cum_notional, margin_used)
    next_level = start_price + STEP
    for c in candles:
        while not halted and c["high"] >= next_level:
            add = (QTY * next_level) / LEVERAGE
            if margin_used + add > MAX_MARGIN:
                halted = True; break
            margin_used += add; cum_qty += QTY; cum_notional += next_level * QTY
            _rec(trades, c["t"], next_level, cum_qty, cum_notional, margin_used)
            next_level += STEP
        if halted: break
    return trades, halted


def run_trailing(candles, start_price, start_ts):
    trades, margin_used, cum_qty, cum_notional, halted = [], 0.0, 0.0, 0.0, False
    margin_used = (QTY * start_price) / LEVERAGE
    cum_qty, cum_notional = QTY, start_price * QTY
    _rec(trades, start_ts, start_price, cum_qty, cum_notional, margin_used)
    next_sell = start_price + STEP
    base_low = start_price
    for c in candles:
        while not halted and c["high"] >= next_sell:
            add = (QTY * next_sell) / LEVERAGE
            if margin_used + add > MAX_MARGIN:
                halted = True; break
            margin_used += add; cum_qty += QTY; cum_notional += next_sell * QTY
            _rec(trades, c["t"], next_sell, cum_qty, cum_notional, margin_used)
            next_sell += STEP
        if halted: break
        if c["low"] < base_low:
            base_low = c["low"]
            next_sell = min(next_sell, base_low + STEP)
    return trades, halted


def package(trades, halted, start_price, live_price):
    cum_qty = sum(t["qty"] for t in trades)
    cum_notional = sum(t["entry"] * t["qty"] for t in trades)
    margin_used = sum(t["margin"] for t in trades)
    avg_entry = (cum_notional / cum_qty) if cum_qty else 0.0
    upnl = cum_qty * (avg_entry - live_price)
    equity = CAPITAL + upnl
    liq = (CAPITAL + cum_qty * avg_entry) / (cum_qty * (1 + MAINT_RATE)) if cum_qty else None
    return {
        "trades": trades, "halted": halted,
        "start_price": round(start_price, 2),
        "cum_qty": round(cum_qty, 2),
        "avg_entry": round(avg_entry, 2),
        "margin_used": round(margin_used, 2),
        "margin_free": round(CAPITAL - margin_used, 2),
        "margin_pct": round(100 * margin_used / CAPITAL, 1),
        "notional": round(cum_qty * avg_entry, 2),
        "live_price": round(live_price, 2),
        "upnl": round(upnl, 2),
        "equity": round(equity, 2),
        "roi": round(100 * upnl / CAPITAL, 2),
        "liq_price": round(liq, 2) if liq else None,
        "liq_dist_pct": round(100 * (liq - live_price) / live_price, 2) if liq else None,
    }


def load_or_init_state(reset=False):
    if not reset:
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except FileNotFoundError:
            pass
    price = fetch_price()
    now = datetime.now(tz=timezone.utc)
    state = {
        "start_ts": int(now.timestamp() * 1000),
        "started_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "start_price": round(price, 2),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return state


def main():
    reset = "--reset" in sys.argv
    st = load_or_init_state(reset)
    live = fetch_price()
    candles = fetch_klines_since(st["start_ts"])

    gt, gh = run_grid(candles, st["start_price"], st["start_ts"])
    tt, th = run_trailing(candles, st["start_price"], st["start_ts"])
    out = {
        "grid":     package(gt, gh, st["start_price"], live),
        "trailing": package(tt, th, st["start_price"], live),
        "candles":  candles,
        "live_price": round(live, 2),
        "start_ts": st["start_ts"],
        "started_at": st["started_at"],
        "generated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "params": {"symbol": SYMBOL, "interval": INTERVAL, "capital": CAPITAL,
                   "leverage": LEVERAGE, "step": STEP, "qty": QTY, "max_margin": MAX_MARGIN},
    }
    with open("data.json", "w") as f:
        json.dump(out, f)
    with open("data.js", "w") as f:
        f.write("window.DATA = " + json.dumps(out) + ";")

    print(f"started {st['started_at']} @ ${st['start_price']} | live ${live} | "
          f"candles since start: {len(candles)}")
    for m in ("grid", "trailing"):
        r = out[m]
        print(f"  [{m:8}] adds={len(r['trades'])} short={r['cum_qty']}ETH avg=${r['avg_entry']} "
              f"margin={r['margin_pct']}% uPnL=${r['upnl']} hold@90%={r['halted']}")


if __name__ == "__main__":
    main()
