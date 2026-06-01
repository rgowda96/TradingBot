#!/usr/bin/env python3
"""
backtest_universal.py — Exhaustive signal backtest across EVERY pool on every chain
=====================================================================================

Unlike backtest.py (which only checks 14 handpicked known winners), this engine:

  1. Fetches ALL pools from GeckoTerminal across Base, Solana, Ethereum,
     Arbitrum, Optimism, BNB, Polygon — thousands of pools per chain.
  2. For each pool, fetches daily OHLCV history (up to 1000 days).
  3. Simulates the bot's signal detection day-by-day on the historical data.
  4. Records every signal that would have fired: type, day, entry price,
     max returns at 7d / 30d / 90d.
  5. Persists results to backtest_db.json — accumulates across runs.
  6. Checkpoint system: resumes exactly where it left off if interrupted.
  7. Generates a ranked report: which signal types predict the best outcomes,
     which feature patterns separate winners from rugs.

Run modes:
    python backtest_universal.py              # process next batch, update DB
    python backtest_universal.py --report     # just regenerate the report
    python backtest_universal.py --status     # show DB coverage stats
    python backtest_universal.py --reset      # clear checkpoint (restart from scratch)
    python backtest_universal.py --network base --max 500   # tune per-run scope

The daily cron runs this with default settings. Over days and weeks it accumulates
coverage across every chain until every known pool has been evaluated.
"""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from statistics import mean, median

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GT_BASE      = "https://api.geckoterminal.com/api/v2"
GT_HEADERS   = {"Accept": "application/json;version=20230302"}

DB_FILE         = "backtest_db.json"
CHECKPOINT_FILE = "backtest_checkpoint.json"
REPORT_FILE     = "backtest_universal_report.md"

# Networks to scan (GeckoTerminal slugs)
ALL_NETWORKS = ["base", "solana", "eth", "arbitrum", "optimism", "bsc", "polygon_pos"]

# Per-run limits — keeps each run under ~10 minutes
DEFAULT_MAX_POOLS  = 300    # pool OHLCV fetches per run
DEFAULT_POOLS_PAGE = 20     # GT returns 20 pools per page

# Pool filtering — skip noise
MIN_POOL_AGE_DAYS  = 14     # must have at least 14 days of history
MIN_TOTAL_VOLUME   = 10_000 # USD — skip ghost pools
MIN_DAYS_OF_DATA   = 14     # need enough candles to simulate

# Signal thresholds (matching bot.py config)
LOW_VOL_DAILY        = 2_100   # $2.1K/day ≈ $50K/24h quiet baseline
SLEEPING_GIANT_DAYS  = 30      # consecutive quiet days before wakeup
SLEEPING_GIANT_MULT  = 5.0     # wakeup day vol / quiet avg
SURGE_MULT           = 8.0     # volume surge multiplier
SURGE_MIN_VOL        = 5_000   # minimum daily vol to trigger surge
BREAKOUT_PCT         = 0.05    # 5% above 30d high
SLOW_BUILD_GROWTH    = 0.30    # 30% week-over-week txn/vol growth
SLOW_BUILD_WEEKS     = 2       # consecutive weeks of growth required
CONTINUATION_PCT     = 0.50    # 50% above last alert level

# Outcome thresholds
RETURN_WIN_30D = 1.0    # +100% in 30d = win
RETURN_MOON_30D = 4.0   # +400% in 30d = moon

# Rate limiting — GeckoTerminal free tier ~30 req/min
REQUEST_DELAY = 2.0    # seconds between API calls

# Stablecoin / LP token filter
SKIP_SYMBOLS = {
    "USDT","USDC","DAI","BUSD","FRAX","TUSD","USDP","GUSD","LUSD","CRVUSD",
    "WETH","WBTC","WBNB","WMATIC","WSOL","WAVAX","WFTM",
    "ETH","BTC","BNB","MATIC","SOL","AVAX","FTM",
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {
        "version":   2,
        "created":   time.time(),
        "pools":     {},   # key = "network:pool_address"
        "total_processed": 0,
    }


def save_db(db):
    tmp = DB_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(db, fh)    # no indent — keeps file compact
    os.replace(tmp, DB_FILE)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as fh:
                return json.load(fh)
        except Exception:
            pass
    # network → next page to fetch
    return {net: 1 for net in ALL_NETWORKS}


def save_checkpoint(cp):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cp, fh, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# GeckoTerminal API
# ---------------------------------------------------------------------------

def _get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=GT_HEADERS, params=params, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 65))
                print(f"      [GT] rate-limited — sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (404, 422):
                return None
            if r.status_code != 200:
                time.sleep(5)
                continue
            return r.json()
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(5)
    return None


def fetch_pools_page(network, page, sort="h24_volume_usd_desc"):
    """Fetch one page (20 pools) from GT pool list."""
    url  = f"{GT_BASE}/networks/{network}/pools"
    data = _get(url, params={"page": page, "sort": sort})
    if not data:
        return []
    return data.get("data") or []


def fetch_pool_ohlcv(network, pool_address, days=365):
    """Fetch daily OHLCV for a pool.

    Returns list of [timestamp_ms, open, high, low, close, volume_usd]
    sorted oldest-first.  Max 1000 candles.
    """
    limit = min(days, 1000)
    url   = f"{GT_BASE}/networks/{network}/pools/{pool_address}/ohlcv/day"
    data  = _get(url, params={"limit": limit, "currency": "usd", "token": "base"})
    if not data:
        return []
    raw = (data.get("data") or {}).get("attributes", {}).get("ohlcv_list") or []
    if not raw:
        return []
    # GT returns newest-first — reverse to oldest-first
    candles = sorted(raw, key=lambda c: c[0])
    return candles   # [ts_ms, o, h, l, c, vol]


def pool_key(network, address):
    return f"{network}:{address.lower()}"


# ---------------------------------------------------------------------------
# Signal detection on daily OHLCV
# ---------------------------------------------------------------------------

def detect_signals(candles):
    """Simulate signal detection day-by-day on historical daily candles.

    Returns a list of signal dicts:
        {type, day_index, ts, price, volume, context}

    Multiple signals can fire on the same day; only the earliest per type
    is kept (to measure lead time accurately).
    """
    n = len(candles)
    if n < MIN_DAYS_OF_DATA:
        return []

    closes  = [c[4] for c in candles]
    vols    = [c[5] for c in candles]  # daily USD volume
    highs   = [c[2] for c in candles]
    signals = []

    def _sig(stype, i, ctx=None):
        return {
            "type":      stype,
            "day_index": i,
            "ts":        candles[i][0] / 1000,
            "price":     closes[i],
            "volume":    vols[i],
            "context":   ctx or {},
        }

    # ── Track state ──────────────────────────────────────────────────────────
    last_continuation_price = None  # for momentum continuation tracking
    seen_types = set()              # one signal per type (first occurrence)

    slow_build_weeks_ok = 0         # consecutive weeks of vol growth

    for i in range(SLEEPING_GIANT_DAYS, n):

        # Quiet-period rolling average (last SLEEPING_GIANT_DAYS days)
        lookback_vols = vols[max(0, i - SLEEPING_GIANT_DAYS):i]
        quiet_avg     = mean(lookback_vols) if lookback_vols else 0

        # ── Sleeping giant ──────────────────────────────────────────────────
        if "sleeping_giant_wakeup" not in seen_types:
            all_quiet = all(v < LOW_VOL_DAILY * 2 for v in lookback_vols)
            if all_quiet and quiet_avg > 0 and vols[i] >= SLEEPING_GIANT_MULT * quiet_avg:
                signals.append(_sig("sleeping_giant_wakeup", i, {
                    "wake_mult":      round(vols[i] / quiet_avg, 1),
                    "quiet_avg_vol":  round(quiet_avg),
                    "sleep_days":     SLEEPING_GIANT_DAYS,
                }))
                seen_types.add("sleeping_giant_wakeup")

        # ── Volume surge (8× 7-day avg) ─────────────────────────────────────
        if "volume_surge" not in seen_types:
            w7_vols  = vols[max(0, i - 7):i]
            w7_avg   = mean(w7_vols) if w7_vols else 0
            if (w7_avg > 0 and vols[i] >= SURGE_MULT * w7_avg
                    and vols[i] >= SURGE_MIN_VOL):
                signals.append(_sig("volume_surge", i, {
                    "surge_ratio": round(vols[i] / w7_avg, 1),
                    "w7_avg_vol":  round(w7_avg),
                }))
                seen_types.add("volume_surge")

        # ── Resistance breakout ─────────────────────────────────────────────
        if "resistance_breakout" not in seen_types:
            prev30_highs = highs[max(0, i - 30):i]
            if prev30_highs:
                ceiling = max(prev30_highs)
                if ceiling > 0 and closes[i] > ceiling * (1 + BREAKOUT_PCT):
                    signals.append(_sig("resistance_breakout", i, {
                        "ceiling":       ceiling,
                        "breakout_pct":  round((closes[i] / ceiling - 1) * 100, 1),
                    }))
                    seen_types.add("resistance_breakout")

        # ── Slow build accumulation ─────────────────────────────────────────
        if "slow_build_accumulation" not in seen_types and i >= 7:
            w_now  = sum(vols[max(0, i - 7):i])
            w_prev = sum(vols[max(0, i - 14):i - 7])
            if w_prev > 0 and (w_now - w_prev) / w_prev >= SLOW_BUILD_GROWTH:
                slow_build_weeks_ok += 1
                if slow_build_weeks_ok >= SLOW_BUILD_WEEKS:
                    signals.append(_sig("slow_build_accumulation", i, {
                        "vol_growth_pct": round((w_now - w_prev) / w_prev * 100, 1),
                        "weeks":          slow_build_weeks_ok,
                    }))
                    seen_types.add("slow_build_accumulation")
            else:
                slow_build_weeks_ok = 0

        # ── Momentum continuation ───────────────────────────────────────────
        # Fires 50%+ above the last breakout/giant/surge reference price
        if last_continuation_price is None and signals:
            last_continuation_price = signals[-1]["price"]
        if last_continuation_price and last_continuation_price > 0:
            if closes[i] >= last_continuation_price * (1 + CONTINUATION_PCT):
                signals.append(_sig("momentum_continuation", i, {
                    "prev_reference": last_continuation_price,
                    "gain_pct": round((closes[i] / last_continuation_price - 1) * 100, 1),
                }))
                last_continuation_price = closes[i]  # advance reference

    return signals


# ---------------------------------------------------------------------------
# Outcome measurement
# ---------------------------------------------------------------------------

def measure_outcomes(candles, signal_day):
    """Measure max returns after a signal at signal_day index.

    Returns dict with:
        max_7d, max_30d, max_90d — max % return achievable within each window
        final_7d, final_30d      — return at exactly day+7, day+30
        rugged                   — True if price dropped >90% from signal
    """
    n = len(candles)
    entry = candles[signal_day][4]  # close at signal
    if entry <= 0:
        return None

    def max_ret(start, end):
        window = candles[start:min(end, n)]
        if not window:
            return 0.0
        return max(c[2] for c in window) / entry - 1  # use high, not close

    def final_ret(offset):
        idx = signal_day + offset
        if idx >= n:
            return None
        return candles[idx][4] / entry - 1

    # Rugged = price at day+30 is <10% of entry (or data ends early with drop)
    final_30 = final_ret(30)
    rugged   = final_30 is not None and final_30 < -0.90

    return {
        "entry_price": entry,
        "max_7d":      round(max_ret(signal_day + 1, signal_day + 8)  * 100, 1),
        "max_30d":     round(max_ret(signal_day + 1, signal_day + 31) * 100, 1),
        "max_90d":     round(max_ret(signal_day + 1, signal_day + 91) * 100, 1),
        "final_7d":    round(final_ret(7)  * 100, 1) if final_ret(7)  is not None else None,
        "final_30d":   round(final_ret(30) * 100, 1) if final_ret(30) is not None else None,
        "rugged":      rugged,
    }


def classify_outcome(outcomes):
    """Classify overall token outcome from the best signal's returns."""
    if not outcomes:
        return "no_signal"
    best = max(outcomes, key=lambda o: o.get("max_30d", 0))
    r30  = best.get("max_30d", 0)
    if best.get("rugged"):
        return "rugged"
    if r30 >= 900:
        return "10x_plus"
    if r30 >= 300:
        return "5x"
    if r30 >= 100:
        return "2x"
    if r30 >= 20:
        return "winner"
    if r30 >= -20:
        return "flat"
    return "loser"


# ---------------------------------------------------------------------------
# Pool processing
# ---------------------------------------------------------------------------

def is_stablecoin_or_lp(pool_data):
    """Return True if this pool should be skipped (stablecoin, LP, wrapped)."""
    attrs = pool_data.get("attributes") or {}
    name  = (attrs.get("name") or "").upper()
    for skip in SKIP_SYMBOLS:
        if skip in name:
            return True
    return False


def process_pool(network, pool_data, db, verbose=True):
    """Fetch OHLCV and run simulation for one pool.

    Returns True if the pool was processed, False if skipped.
    """
    attrs   = pool_data.get("attributes") or {}
    address = attrs.get("address") or (pool_data.get("id") or "").split("_")[-1]
    if not address:
        return False

    pk = pool_key(network, address)
    if pk in db["pools"]:
        return False   # already done

    # Skip stablecoins and LP tokens
    if is_stablecoin_or_lp(pool_data):
        db["pools"][pk] = {"skipped": "stablecoin_or_lp"}
        return False

    # Basic volume filter — skip ghost pools
    h24_vol = float(attrs.get("volume_usd", {}).get("h24", 0) or 0)
    reserve = float(attrs.get("reserve_in_usd") or 0)
    if h24_vol < 100 and reserve < 5_000:
        db["pools"][pk] = {"skipped": "no_volume"}
        return False

    name   = attrs.get("name", "?")
    base_t = (attrs.get("base_token_price_usd") or "0")

    if verbose:
        print(f"  [{network}] {name[:40]:<40} {address[:10]}... ", end="", flush=True)

    time.sleep(REQUEST_DELAY)
    candles = fetch_pool_ohlcv(network, address, days=730)

    if len(candles) < MIN_DAYS_OF_DATA:
        db["pools"][pk] = {"skipped": "insufficient_history", "candle_count": len(candles)}
        if verbose:
            print(f"skip ({len(candles)} candles)")
        return False

    # Filter by total historical volume
    total_vol = sum(c[5] for c in candles)
    if total_vol < MIN_TOTAL_VOLUME:
        db["pools"][pk] = {"skipped": "low_total_volume", "total_vol": round(total_vol)}
        if verbose:
            print(f"skip (${total_vol:,.0f} total vol)")
        return False

    # Run signal detection
    signals = detect_signals(candles)

    # Measure outcomes for each signal
    signal_results = []
    for sig in signals:
        out = measure_outcomes(candles, sig["day_index"])
        if out:
            signal_results.append({**sig, **out})

    # Find earliest signal across all types
    earliest = min(signal_results, key=lambda s: s["day_index"]) if signal_results else None
    outcome  = classify_outcome(signal_results)

    # Token metadata
    base_token_rel = ((pool_data.get("relationships") or {})
                      .get("base_token") or {}).get("data") or {}
    token_id       = base_token_rel.get("id", "")

    record = {
        "network":         network,
        "address":         address,
        "name":            name,
        "token_id":        token_id,
        "processed_at":    time.time(),
        "candle_count":    len(candles),
        "total_vol_usd":   round(total_vol),
        "first_candle_ts": candles[0][0] / 1000 if candles else None,
        "last_candle_ts":  candles[-1][0] / 1000 if candles else None,
        "signal_count":    len(signal_results),
        "signals":         signal_results,
        "earliest_signal": earliest,
        "outcome":         outcome,
    }
    db["pools"][pk] = record
    db["total_processed"] = db.get("total_processed", 0) + 1

    if verbose:
        sig_txt = (f"{earliest['type']} day {earliest['day_index']} "
                   f"→ 30d +{earliest['max_30d']:.0f}%") if earliest else "no signal"
        print(f"{outcome:12s}  {sig_txt}")

    return True


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(networks=None, max_pools=DEFAULT_MAX_POOLS, verbose=True):
    """Process the next batch of pools, updating DB and checkpoint."""
    networks   = networks or ALL_NETWORKS
    db         = load_db()
    checkpoint = load_checkpoint()

    processed  = 0
    exhausted  = set()   # networks with no more pages to fetch

    print(f"[backtest_universal] DB has {len(db['pools'])} pools, "
          f"target +{max_pools} this run", flush=True)

    # Round-robin across networks so coverage is even
    while processed < max_pools and len(exhausted) < len(networks):
        for net in networks:
            if processed >= max_pools:
                break
            if net in exhausted:
                continue

            page = checkpoint.get(net, 1)
            if verbose:
                print(f"\n  == {net} page {page} ==", flush=True)

            time.sleep(REQUEST_DELAY)
            pools = fetch_pools_page(net, page)

            if not pools:
                exhausted.add(net)
                # Wrap around — start over this network next run
                checkpoint[net] = 1
                if verbose:
                    print(f"  [wrap] {net} exhausted at page {page} — resetting to page 1")
                continue

            for pool_data in pools:
                if processed >= max_pools:
                    break
                ok = process_pool(net, pool_data, db, verbose=verbose)
                if ok:
                    processed += 1

            checkpoint[net] = page + 1
            save_checkpoint(checkpoint)
            save_db(db)

    print(f"\n[backtest_universal] Processed {processed} new pools this run | "
          f"DB total: {len(db['pools'])}")
    return db


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _pct_fmt(v):
    if v is None:
        return "—"
    return f"{v:+.0f}%"


def generate_report(db, save_path=REPORT_FILE):
    """Generate a comprehensive markdown report from the accumulated DB."""
    pools      = [v for v in db["pools"].values() if isinstance(v, dict) and not v.get("skipped")]
    now        = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_db   = len(db["pools"])
    processed  = len(pools)
    skipped    = total_db - processed

    # Outcome distribution
    outcome_counts = {}
    for p in pools:
        o = p.get("outcome", "no_signal")
        outcome_counts[o] = outcome_counts.get(o, 0) + 1

    # Per-signal-type stats
    all_signals = []
    for p in pools:
        for sig in (p.get("signals") or []):
            all_signals.append({**sig, "outcome": p.get("outcome"), "network": p.get("network")})

    sig_stats = {}
    for sig in all_signals:
        st = sig.get("type", "?")
        sig_stats.setdefault(st, []).append(sig)

    # Earliest signal distribution (day_index buckets)
    earliest_days = []
    for p in pools:
        e = p.get("earliest_signal")
        if e:
            earliest_days.append(e["day_index"])

    lines = [
        f"# 📡 Universal Backtest Report — All Chains, All Pools",
        f"**Generated:** {now}",
        f"**Total pools in DB:** {total_db:,}  ({processed:,} analysed, {skipped:,} skipped)",
        f"**Networks covered:** {', '.join(ALL_NETWORKS)}",
        f"",
        "---",
        "",
        "## 📊 Outcome Distribution",
        "",
        "| Outcome | Count | % of analysed |",
        "|---------|-------|--------------|",
    ]
    for o, cnt in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        pct = cnt / processed * 100 if processed else 0
        emoji = {"10x_plus": "🚀", "5x": "🟢", "2x": "🟡", "winner": "🟡",
                 "flat": "⬜", "loser": "🔴", "rugged": "💀", "no_signal": "⬜"}.get(o, "")
        lines.append(f"| {emoji} {o} | {cnt:,} | {pct:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 🎯 Signal Type Performance",
        "",
        "Win rate = % of signals where max_30d >= +100% (2× return).",
        "",
        "| Signal Type | Fires | Win% | Avg 30d | Med 30d | Best 30d | Avg Lead Day |",
        "|-------------|-------|------|---------|---------|----------|-------------|",
    ]
    for st, sigs in sorted(sig_stats.items(), key=lambda x: -len(x[1])):
        rets  = [s.get("max_30d", 0) for s in sigs if s.get("max_30d") is not None]
        wins  = sum(1 for r in rets if r >= 100)
        wr    = wins / len(rets) * 100 if rets else 0
        avg_r = mean(rets) if rets else 0
        med_r = median(rets) if rets else 0
        best  = max(rets) if rets else 0
        days  = [s.get("day_index", 0) for s in sigs]
        avg_d = mean(days) if days else 0
        lines.append(
            f"| `{st}` | {len(sigs):,} | {wr:.0f}% | "
            f"{avg_r:+.0f}% | {med_r:+.0f}% | {best:+.0f}% | {avg_d:.0f}d |"
        )

    # Earliest signal timing
    if earliest_days:
        lines += [
            "",
            "---",
            "",
            "## ⏱ Lead Time Distribution (earliest signal per pool)",
            "",
            "How many days into the token's life does the FIRST signal fire?",
            "Earlier = more lead time before the crowd.",
            "",
            "| Window | Pools caught |",
            "|--------|-------------|",
        ]
        buckets = [(7, "≤ 7 days"), (14, "≤ 14 days"), (30, "≤ 30 days"),
                   (60, "≤ 60 days"), (90, "≤ 90 days"), (180, "≤ 180 days")]
        for cutoff, label in buckets:
            cnt = sum(1 for d in earliest_days if d <= cutoff)
            pct = cnt / len(earliest_days) * 100
            lines.append(f"| {label} | {cnt:,} ({pct:.0f}%) |")

    # Top 20 best signals by max_30d return
    lines += [
        "",
        "---",
        "",
        "## 🏆 Top 20 Best Signals Found",
        "",
        "| Token | Network | Signal | Signal Day | 30d Max | 90d Max |",
        "|-------|---------|--------|-----------|---------|---------|",
    ]
    flat_sigs = []
    for p in pools:
        for sig in (p.get("signals") or []):
            flat_sigs.append({
                "name":    p.get("name", "?"),
                "network": p.get("network", "?"),
                **sig
            })
    flat_sigs.sort(key=lambda s: s.get("max_30d", 0), reverse=True)
    for s in flat_sigs[:20]:
        lines.append(
            f"| {s['name'][:30]} | {s['network']} | `{s['type']}` | "
            f"day {s['day_index']} | {_pct_fmt(s.get('max_30d'))} | "
            f"{_pct_fmt(s.get('max_90d'))} |"
        )

    # Worst signals (rugged)
    rugged_pools = [p for p in pools if p.get("outcome") == "rugged" and p.get("earliest_signal")]
    lines += [
        "",
        "---",
        "",
        "## 💀 False Positive Patterns (signal fired → rugged)",
        "",
        f"Total rugged tokens that had a signal: {len(rugged_pools)}",
        "",
        "| Token | Network | Signal Type | Signal Day | Final 30d |",
        "|-------|---------|------------|-----------|----------|",
    ]
    for p in sorted(rugged_pools, key=lambda x: x.get("earliest_signal", {}).get("day_index", 0))[:15]:
        e = p["earliest_signal"]
        lines.append(
            f"| {p['name'][:30]} | {p['network']} | `{e['type']}` | "
            f"day {e['day_index']} | {_pct_fmt(e.get('final_30d'))} |"
        )

    # Per-network breakdown
    net_counts = {}
    for p in pools:
        n = p.get("network", "?")
        net_counts.setdefault(n, {"total": 0, "with_signal": 0, "winners": 0})
        net_counts[n]["total"] += 1
        if p.get("signal_count", 0) > 0:
            net_counts[n]["with_signal"] += 1
        if p.get("outcome") in ("10x_plus", "5x", "2x", "winner"):
            net_counts[n]["winners"] += 1

    lines += [
        "",
        "---",
        "",
        "## 🌐 Coverage by Network",
        "",
        "| Network | Pools analysed | With signal | Winners |",
        "|---------|---------------|------------|---------|",
    ]
    for net, stats in sorted(net_counts.items(), key=lambda x: -x[1]["total"]):
        lines.append(
            f"| {net} | {stats['total']:,} | "
            f"{stats['with_signal']:,} ({stats['with_signal']/stats['total']*100:.0f}%) | "
            f"{stats['winners']:,} ({stats['winners']/stats['total']*100:.0f}%) |"
        )

    lines += [
        "",
        "---",
        "",
        "## 🔧 Actionable Improvements",
        "",
    ]

    # Compute signal-level win rates and give recommendations
    recs = []
    for st, sigs in sig_stats.items():
        rets = [s.get("max_30d", 0) for s in sigs if s.get("max_30d") is not None]
        if len(rets) < 10:
            continue
        wr = sum(1 for r in rets if r >= 100) / len(rets)
        if wr < 0.25:
            recs.append(f"**{st}** — win rate only {wr:.0%} across {len(rets)} signals. "
                        f"Consider raising conviction floor or adding quality filter.")
        elif wr > 0.60:
            recs.append(f"**{st}** — win rate {wr:.0%} — consider lowering conviction "
                        f"floor to catch more of these.")

    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("*Not enough data yet for recommendations. Keep running daily.*")

    lines += [
        "",
        "---",
        f"*Generated by backtest_universal.py — covers every pool on {', '.join(ALL_NETWORKS)}*",
        f"*Run `python backtest_universal.py` daily to expand coverage.*",
    ]

    content = "\n".join(lines)
    with open(save_path, "w") as fh:
        fh.write(content)
    print(f"[backtest_universal] Report written to {save_path}")
    return content


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def print_status(db):
    pools     = [v for v in db["pools"].values() if isinstance(v, dict)]
    analysed  = [v for v in pools if not v.get("skipped")]
    skipped   = [v for v in pools if v.get("skipped")]
    cp        = load_checkpoint()

    print(f"\n{'='*60}")
    print(f"backtest_universal DB status")
    print(f"{'='*60}")
    print(f"Total entries:  {len(pools):,}")
    print(f"  Analysed:     {len(analysed):,}")
    print(f"  Skipped:      {len(skipped):,}")
    print(f"\nCheckpoint (next page per network):")
    for net, page in cp.items():
        net_count = sum(1 for p in analysed if p.get("network") == net)
        print(f"  {net:<15}  page {page:<5}  ({net_count:,} analysed)")

    outcome_counts = {}
    for p in analysed:
        o = p.get("outcome", "?")
        outcome_counts[o] = outcome_counts.get(o, 0) + 1
    print(f"\nOutcome distribution:")
    for o, cnt in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {o:<20}  {cnt:,}")

    signals = [s for p in analysed for s in (p.get("signals") or [])]
    print(f"\nTotal signals detected: {len(signals):,}")
    sig_counts = {}
    for s in signals:
        sig_counts[s["type"]] = sig_counts.get(s["type"], 0) + 1
    for st, cnt in sorted(sig_counts.items(), key=lambda x: -x[1]):
        print(f"  {st:<35}  {cnt:,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Universal pool backtest")
    parser.add_argument("--max",      type=int, default=DEFAULT_MAX_POOLS,
                        help="Max pools to process this run")
    parser.add_argument("--network",  type=str, default=None,
                        help="Only process this network (e.g. base)")
    parser.add_argument("--report",   action="store_true",
                        help="Only regenerate report, don't fetch new pools")
    parser.add_argument("--status",   action="store_true",
                        help="Print DB coverage stats and exit")
    parser.add_argument("--reset",    action="store_true",
                        help="Clear checkpoint and restart from page 1")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-pool output")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        print("[backtest_universal] Checkpoint cleared — will restart from page 1")
        return

    if args.status:
        db = load_db()
        print_status(db)
        return

    if args.report:
        db = load_db()
        generate_report(db)
        return

    networks = [args.network] if args.network else ALL_NETWORKS
    db = run_batch(networks=networks, max_pools=args.max, verbose=not args.quiet)
    generate_report(db)


if __name__ == "__main__":
    main()
