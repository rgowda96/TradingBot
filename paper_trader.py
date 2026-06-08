"""
Paper trader — polls all TF/strategy combos, opens/closes virtual positions,
tracks P&L in real time, persists state to paper_trades.json.
"""
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timezone

from indicators import add_indicators
from strategies import STRATEGIES

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL        = "ETH/USDT"
CAPITAL       = 10_000.0   # virtual USD
RISK_PCT      = 0.01       # 1% per trade
ATR_SL_MULT   = 1.5
ATR_TP_MULT   = 3.0
FEE_PCT       = 0.001      # 0.1% per side
MAX_HOLD_BARS = 50
CANDLES       = 300
STATE_FILE    = "paper_trades.json"
LOG_FILE      = "paper_trade_log.txt"

WATCH = [
    # Best daily edges (positive Sharpe from backtest)
    ("1d", "obv_divergence"),
    ("1d", "engulf_reversal"),
    ("1d", "stoch_rsi_cross"),
    ("1d", "macd_momentum"),
    ("1d", "bb_breakout"),
    ("1d", "supertrend"),
    ("1d", "ema_trend"),
    ("1d", "volume_breakout"),
    # 4h
    ("4h", "macd_momentum"),
    ("4h", "volume_breakout"),
    ("4h", "stoch_rsi_cross"),
    ("4h", "bb_breakout"),
    ("4h", "supertrend"),
    ("4h", "ema_trend"),
    # 1h
    ("1h", "supertrend"),
    ("1h", "macd_momentum"),
    ("1h", "stoch_rsi_cross"),
    ("1h", "ema_trend"),
    # 15m (higher noise, smaller allocation)
    ("15m", "supertrend"),
    ("15m", "stoch_rsi_cross"),
]

REFRESH = {
    "1d":  3600,   # check every hour (daily candles)
    "4h":  900,    # every 15 min
    "1h":  300,    # every 5 min
    "15m": 120,    # every 2 min
    "5m":  60,
    "1m":  30,
}

# ── State ─────────────────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "equity":    CAPITAL,
        "positions": {},   # key: "tf|strategy"
        "closed":    [],
        "started_at": _now(),
    }

DASHBOARD_DIR = "/tmp/eth-dashboard"

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    _write_dashboard_data(state)

def _write_dashboard_data(state):
    """Write data.js for the live dashboard served from /tmp/eth-dashboard/."""
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    # Read log tail
    log_tail = ""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            lines = f.readlines()
            log_tail = "".join(lines[-80:])
    payload = json.dumps(state, default=str)
    log_payload = json.dumps(log_tail)
    js = f"window.__ETH_STATE__ = {payload};\nwindow.__ETH_LOG__ = {log_payload};\n"
    with open(os.path.join(DASHBOARD_DIR, "data.js"), "w") as f:
        f.write(js)

def _now():
    return datetime.now(timezone.utc).isoformat()

def _log(msg):
    line = f"[{_now()}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_df(exchange, tf):
    raw = exchange.fetch_ohlcv(SYMBOL, tf, limit=CANDLES)
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.dropna()
    df = add_indicators(df)
    return df.dropna()

# ── Core logic ────────────────────────────────────────────────────────────────

def process_pair(exchange, state, tf, strat_name, df):
    key = f"{tf}|{strat_name}"
    positions = state["positions"]

    strat_fn = STRATEGIES[strat_name]
    signal   = strat_fn(df)
    last_sig = int(signal.iloc[-1])
    px       = float(df["close"].iloc[-1])
    atr      = float(df["atr_14"].iloc[-1])
    equity   = state["equity"]

    # ── Check open position ───────────────────────────────────────────────────
    if key in positions:
        pos = positions[key]
        direction  = pos["direction"]
        entry_px   = pos["entry_px"]
        sl_px      = pos["sl_px"]
        tp_px      = pos["tp_px"]
        qty        = pos["qty"]
        bars_held  = pos["bars_held"] + 1
        pos["bars_held"] = bars_held

        hit_sl  = (direction ==  1 and px <= sl_px) or (direction == -1 and px >= sl_px)
        hit_tp  = (direction ==  1 and px >= tp_px) or (direction == -1 and px <= tp_px)
        hit_max = bars_held >= MAX_HOLD_BARS
        sig_flip = last_sig != 0 and last_sig != direction

        if hit_sl or hit_tp or hit_max or sig_flip:
            exit_px = sl_px if hit_sl else (tp_px if hit_tp else px)
            pnl_pct = direction * (exit_px / entry_px - 1) - 2 * FEE_PCT
            pnl     = pnl_pct * qty * entry_px
            state["equity"] = round(equity + pnl, 4)
            reason = "SL" if hit_sl else ("TP" if hit_tp else ("MAX_HOLD" if hit_max else "SIGNAL_FLIP"))

            closed = {
                "key":        key,
                "tf":         tf,
                "strategy":   strat_name,
                "direction":  direction,
                "entry_px":   entry_px,
                "exit_px":    round(exit_px, 4),
                "qty":        qty,
                "pnl_usd":    round(pnl, 4),
                "pnl_pct":    round(pnl_pct * 100, 3),
                "bars_held":  bars_held,
                "exit_reason":reason,
                "opened_at":  pos["opened_at"],
                "closed_at":  _now(),
            }
            state["closed"].append(closed)
            del positions[key]

            emoji = "✅" if pnl > 0 else "❌"
            _log(f"{emoji} CLOSED {key} {'+' if direction==1 else 'SHORT'}  "
                 f"entry=${entry_px:.2f} exit=${exit_px:.2f}  "
                 f"PnL={'+'if pnl>0 else ''}{pnl:.2f} USD ({pnl_pct*100:+.2f}%)  "
                 f"reason={reason}  equity=${state['equity']:,.2f}")

    # ── Open new position ─────────────────────────────────────────────────────
    if key not in positions and last_sig != 0:
        prev_sig = int(signal.iloc[-2]) if len(signal) > 1 else 0
        if last_sig != prev_sig:   # fresh signal only
            sl_dist   = ATR_SL_MULT * atr
            tp_dist   = ATR_TP_MULT * atr
            risk_amt  = equity * RISK_PCT
            qty       = risk_amt / sl_dist if sl_dist > 0 else 0

            if qty <= 0:
                return

            sl_px = round(px - last_sig * sl_dist, 4)
            tp_px = round(px + last_sig * tp_dist, 4)

            positions[key] = {
                "direction":  last_sig,
                "entry_px":   round(px, 4),
                "sl_px":      sl_px,
                "tp_px":      tp_px,
                "qty":        round(qty, 6),
                "bars_held":  0,
                "opened_at":  _now(),
            }

            d = "LONG" if last_sig == 1 else "SHORT"
            _log(f"📈 OPENED {key} {d}  entry=${px:.2f}  "
                 f"SL=${sl_px:.2f}  TP=${tp_px:.2f}  qty={qty:.4f} ETH  "
                 f"risk=${risk_amt:.2f}  equity=${equity:,.2f}")


def print_dashboard(state, exchange):
    positions = state["positions"]
    closed    = state["closed"]
    equity    = state["equity"]
    px        = float(exchange.fetch_ticker(SYMBOL)["last"])

    # Unrealised P&L
    unreal = 0.0
    for key, pos in positions.items():
        d  = pos["direction"]
        ep = pos["entry_px"]
        q  = pos["qty"]
        unreal += d * (px / ep - 1) * q * ep

    wins  = [t for t in closed if t["pnl_usd"] > 0]
    loses = [t for t in closed if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in closed)
    wr = len(wins) / len(closed) * 100 if closed else 0

    print(f"\n{'═'*68}")
    print(f"  📊  PAPER TRADING DASHBOARD  |  {_now()[:19]} UTC")
    print(f"{'═'*68}")
    print(f"  ETH Price   : ${px:>10,.2f}")
    print(f"  Equity      : ${equity:>10,.2f}  (started ${CAPITAL:,.2f})")
    print(f"  Unrealised  : ${unreal:>+10,.2f}")
    print(f"  Total PnL   : ${total_pnl:>+10,.2f}")
    print(f"  Return      : {(equity + unreal - CAPITAL) / CAPITAL * 100:>+9.2f}%")
    print(f"  Trades      : {len(closed)} closed  |  {len(positions)} open")
    print(f"  Win Rate    : {wr:.1f}%  ({len(wins)}W / {len(loses)}L)")
    print(f"{'─'*68}")

    if positions:
        print(f"  OPEN POSITIONS:")
        for key, pos in positions.items():
            d_str = "LONG " if pos["direction"] == 1 else "SHORT"
            unr = pos["direction"] * (px / pos["entry_px"] - 1) * pos["qty"] * pos["entry_px"]
            print(f"    {key:28s} {d_str}  entry=${pos['entry_px']:,.2f}  "
                  f"SL=${pos['sl_px']:,.2f}  TP=${pos['tp_px']:,.2f}  "
                  f"unreal=${unr:+.2f}  bars={pos['bars_held']}")
    else:
        print(f"  No open positions.")

    if closed:
        print(f"{'─'*68}")
        print(f"  LAST 10 CLOSED TRADES:")
        for t in closed[-10:]:
            e = "✅" if t["pnl_usd"] > 0 else "❌"
            d = "L" if t["direction"] == 1 else "S"
            print(f"  {e} {t['tf']:4s}/{t['strategy']:22s} {d}  "
                  f"${t['entry_px']:,.2f}→${t['exit_px']:,.2f}  "
                  f"{t['pnl_usd']:>+8.2f} USD ({t['pnl_pct']:>+6.2f}%)  "
                  f"{t['exit_reason']}")

    print(f"{'═'*68}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    exchange = ccxt.binance({"enableRateLimit": True})
    state    = load_state()
    _log(f"=== Paper trader started. Equity=${state['equity']:,.2f} ===")

    # Track last-fetched time per TF to avoid redundant API calls
    last_fetch = {}
    tf_cache   = {}   # latest df per tf

    try:
        while True:
            now_ts = time.time()

            # Refresh data per TF if interval elapsed
            for tf in set(t for t, _ in WATCH):
                interval = REFRESH.get(tf, 300)
                if now_ts - last_fetch.get(tf, 0) >= interval:
                    try:
                        tf_cache[tf] = fetch_df(exchange, tf)
                        last_fetch[tf] = now_ts
                    except Exception as e:
                        _log(f"  fetch error {tf}: {e}")

            # Process each pair
            for tf, strat in WATCH:
                if tf not in tf_cache:
                    continue
                try:
                    process_pair(exchange, state, tf, strat, tf_cache[tf])
                except Exception as e:
                    _log(f"  process error {tf}/{strat}: {e}")

            save_state(state)
            print_dashboard(state, exchange)

            print(f"\n  Sleeping 60s ... (Ctrl+C to stop)\n")
            time.sleep(60)

    except KeyboardInterrupt:
        save_state(state)
        _log("=== Paper trader stopped by user ===")
        print_dashboard(state, exchange)


if __name__ == "__main__":
    run()
