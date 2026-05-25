#!/usr/bin/env python3
"""
Backtest the Base-chain volume-spike strategy using GeckoTerminal (free API).

Looks back 30 days across:
  - All tokens currently in state.json
  - Recently trending / newly active Base pools from GeckoTerminal

A signal fires when:
  1. A token had >= MIN_SLEEP_DAYS of consecutive sleeping days (24h vol < LOW_VOL_MAX)
  2. The next day's 24h volume crossed SPIKE_MIN (5-6 figures)

Reports per-signal details and aggregate stats.
"""

import json
import time
import sys
from datetime import datetime, timezone

import requests

# ── Strategy parameters ───────────────────────────────────────────────────────
LOW_VOL_MAX    = 9_999    # "sleeping" if 24h vol ≤ this
SPIKE_MIN      = 10_000   # breakout if 24h vol ≥ this
MIN_SLEEP_DAYS = 2        # require at least N consecutive sleeping days

# ── GeckoTerminal ─────────────────────────────────────────────────────────────
GT      = "https://api.geckoterminal.com/api/v2"
HEADERS = {"Accept": "application/json"}


def gt_get(path, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(f"{GT}{path}", params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                wait = 12 * (attempt + 1)
                print(f"  [rate limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                return {}
            time.sleep(3 * (attempt + 1))
    return {}


def pools_for_token(token_addr):
    d = gt_get(f"/networks/base/tokens/{token_addr}/pools", {"page": 1})
    return [p["attributes"]["address"] for p in d.get("data", [])]


def daily_candles(pool_addr, limit=30):
    d = gt_get(
        f"/networks/base/pools/{pool_addr}/ohlcv/day",
        {"limit": limit, "currency": "usd"},
    )
    return d.get("data", {}).get("attributes", {}).get("ohlcv_list", []) or []


def hourly_candles(pool_addr, limit=168):
    d = gt_get(
        f"/networks/base/pools/{pool_addr}/ohlcv/hour",
        {"limit": limit, "currency": "usd"},
    )
    return d.get("data", {}).get("attributes", {}).get("ohlcv_list", []) or []


def trending_pools(pages=3):
    """Return pool addresses from GeckoTerminal's recently active Base pools."""
    addrs = []
    for page in range(1, pages + 1):
        d = gt_get("/networks/base/new_pools", {"page": page})
        for p in d.get("data", []):
            addr = p.get("attributes", {}).get("address")
            if addr:
                addrs.append(addr)
        time.sleep(0.3)
    return addrs


# ── Detection ─────────────────────────────────────────────────────────────────
def find_signals(candles):
    """
    candles: [[ts, open, high, low, close, volume], ...] in any order.
    Returns list of signal dicts.
    """
    if len(candles) < MIN_SLEEP_DAYS + 1:
        return []

    candles = sorted(candles, key=lambda c: c[0])  # oldest first
    signals = []

    for i in range(MIN_SLEEP_DAYS, len(candles)):
        ts, o, h, l, c, vol = candles[i]

        if vol < SPIKE_MIN:
            continue

        # Count consecutive sleeping days immediately before this candle
        sleeping = 0
        for j in range(i - 1, -1, -1):
            if candles[j][5] <= LOW_VOL_MAX:
                sleeping += 1
            else:
                break

        if sleeping < MIN_SLEEP_DAYS:
            continue

        baseline_vol = candles[i - 1][5]
        multiple     = vol / baseline_vol if baseline_vol > 0 else None
        day_change   = (c - o) / o * 100 if o > 0 else 0.0

        next_change = None
        if i + 1 < len(candles):
            nc = candles[i + 1]
            next_change = (nc[4] - o) / o * 100 if o > 0 else None

        signals.append({
            "date":           datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
            "sleeping_days":  sleeping,
            "baseline_vol":   baseline_vol,
            "spike_vol":      vol,
            "multiple":       multiple,
            "day_chg_pct":    day_change,
            "next_day_chg_pct": next_change,
            "open_price":     o,
            "close_price":    c,
        })

    return signals


def earliest_hour_spike(pool_addr, signal_date):
    """
    For a signal that fired on signal_date (YYYY-MM-DD), check hourly candles
    to find the earliest hour of that day where cumulative volume crossed SPIKE_MIN.
    Returns hour string like '14:00 UTC' or None.
    """
    candles = hourly_candles(pool_addr, limit=168)
    if not candles:
        return None

    candles = sorted(candles, key=lambda c: c[0])
    day_candles = [
        c for c in candles
        if datetime.fromtimestamp(c[0], tz=timezone.utc).strftime("%Y-%m-%d") == signal_date
    ]
    if not day_candles:
        return None

    cumulative = 0.0
    for ts, o, h, l, c, vol in day_candles:
        cumulative += vol
        if cumulative >= SPIKE_MIN:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.strftime("%H:%M UTC")
    return None


# ── Token universe ────────────────────────────────────────────────────────────
def build_pool_universe():
    pool_addrs = set()

    # 1. Pools from currently monitored tokens
    print("Loading monitored tokens from state.json ...")
    try:
        with open("state.json") as f:
            state = json.load(f)
        tokens = state.get("monitored_tokens", [])
        print(f"  {len(tokens)} tokens in state.json")
        for i, tok in enumerate(tokens):
            sys.stdout.write(f"\r  Mapping token {i+1}/{len(tokens)} ...")
            sys.stdout.flush()
            for addr in pools_for_token(tok):
                pool_addrs.add(addr)
            time.sleep(0.35)
        print()
    except FileNotFoundError:
        print("  state.json not found, skipping.")

    # 2. Recently created / newly active Base pools
    print("Fetching new/trending Base pools from GeckoTerminal ...")
    new_pools = trending_pools(pages=3)
    print(f"  {len(new_pools)} new pools found")
    pool_addrs.update(new_pools)

    return list(pool_addrs)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  BASE CHAIN VOLUME-SPIKE STRATEGY — BACKTEST (30 days)")
    print("=" * 60)
    print(f"  Sleeping threshold : 24h vol ≤ ${LOW_VOL_MAX:,}")
    print(f"  Spike threshold    : 24h vol ≥ ${SPIKE_MIN:,}")
    print(f"  Min sleeping days  : {MIN_SLEEP_DAYS}")
    print("=" * 60 + "\n")

    pools = build_pool_universe()
    print(f"\nTotal unique pools to test: {len(pools)}\n")

    all_signals = []
    checked = errors = 0

    for i, pool in enumerate(pools):
        sys.stdout.write(f"\r  Checking pool {i+1}/{len(pools)} ...")
        sys.stdout.flush()
        try:
            candles = daily_candles(pool, limit=30)
            if not candles:
                continue
            signals = find_signals(candles)
            for s in signals:
                s["pool"] = pool
                all_signals.append(s)
            checked += 1
            time.sleep(0.35)
        except Exception:
            errors += 1

    print(f"\r  Done. Checked: {checked}  Errors: {errors}          \n")

    if not all_signals:
        print("No signals found in the last 30 days for this token universe.")
        print("Consider running the bot longer to build a larger token pool.")
        return

    # Sort signals by date
    all_signals.sort(key=lambda s: s["date"])

    print(f"{'DATE':<12} {'SLEEP':>5} {'BASELINE VOL':>13} {'SPIKE VOL':>12} {'MULT':>6}  {'DAY Δ':>7}  {'NEXT DAY Δ':>10}")
    print("-" * 78)
    for s in all_signals:
        mult     = f"{s['multiple']:,.0f}x" if s["multiple"] else "  ∞"
        next_d   = f"{s['next_day_chg_pct']:+.1f}%" if s["next_day_chg_pct"] is not None else "    n/a"
        print(
            f"{s['date']:<12} {s['sleeping_days']:>5}d "
            f"${s['baseline_vol']:>11,.0f} "
            f"${s['spike_vol']:>10,.0f} "
            f"{mult:>6}  "
            f"{s['day_chg_pct']:>+6.1f}%  "
            f"{next_d:>10}"
        )

    print("\n" + "=" * 60)
    print(f"  SUMMARY")
    print("=" * 60)
    print(f"  Signals detected       : {len(all_signals)}")

    day_changes  = [s["day_chg_pct"] for s in all_signals]
    next_changes = [s["next_day_chg_pct"] for s in all_signals if s["next_day_chg_pct"] is not None]

    print(f"  Avg price Δ — signal day  : {sum(day_changes)/len(day_changes):+.1f}%")
    if next_changes:
        print(f"  Avg price Δ — next day    : {sum(next_changes)/len(next_changes):+.1f}%")

    winners     = [s for s in all_signals if s["day_chg_pct"] > 0]
    big_winners = [s for s in all_signals if s["day_chg_pct"] > 20]
    print(f"  Positive on signal day    : {len(winners)}/{len(all_signals)}  ({100*len(winners)/len(all_signals):.0f}%)")
    print(f"  >20% gain on signal day   : {len(big_winners)}/{len(all_signals)}  ({100*len(big_winners)/len(all_signals):.0f}%)")

    if all_signals:
        best = max(all_signals, key=lambda s: s["day_chg_pct"])
        print(f"\n  Best signal: {best['date']}  "
              f"${best['baseline_vol']:,.0f} → ${best['spike_vol']:,.0f}  "
              f"({best['day_chg_pct']:+.1f}% on the day)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
