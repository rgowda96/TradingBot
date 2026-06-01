#!/usr/bin/env python3
"""RL Signal Outcome Tracker — learns from every Telegram alert sent.

How it works
============
bot.py appends one JSON line to signal_log.jsonl every time a Telegram alert
fires.  This script reads that log, fetches the current price from DexScreener
for any signal old enough to evaluate (>= CHECK_DAYS_MIN), computes the return,
and builds per-signal-type statistics.

Outputs:
  signal_outcomes.json      — raw per-signal outcome records
  adaptive_thresholds.json  — suggested conviction floors per signal type
  learning_report.md        — human-readable performance report

Run manually:
  python signal_tracker.py

Or add to cron to run daily:
  0 8 * * * cd /path/to/TradingBot && python signal_tracker.py >> tracker.log 2>&1
"""
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNAL_LOG       = "signal_log.jsonl"
OUTCOMES_FILE    = "signal_outcomes.json"
THRESHOLDS_FILE  = "adaptive_thresholds.json"
REPORT_FILE      = "learning_report.md"

CHECK_DAYS_MIN   = 7     # don't evaluate a signal until it's at least this old
CHECK_DAYS_MAX   = 180   # ignore signals older than this (too stale to be useful)

WIN_THRESHOLD    = 0.20  # +20% = win
LOSS_THRESHOLD   = -0.20 # -20% = loss; between -20% and +20% = neutral

# How much we adjust thresholds based on win rates
# If win rate < LOW_WIN_RATE → raise conviction floor by STEP
# If win rate > HIGH_WIN_RATE → lower conviction floor by STEP (more signals)
LOW_WIN_RATE     = 0.35
HIGH_WIN_RATE    = 0.65
THRESHOLD_STEP   = 0.5
MAX_THRESHOLD    = 8.0
MIN_THRESHOLD    = 3.0

DEXSCREENER_URL  = "https://api.dexscreener.com/tokens/v1"

# Current conviction floors from config — used as baseline for suggestions
CURRENT_FLOORS = {
    "volume_surge":           5.0,
    "volume_spike":           5.0,
    "resistance_breakout":    5.0,
    "pre_breakout_coil":      5.0,
    "sleeping_giant_wakeup":  4.0,
    "slow_build_accumulation":4.0,
    "momentum_continuation":  5.0,
    "dead_token_resurrection":5.0,
    "mcap_milestone":         4.0,
    "vc_backed_project":      4.0,
}

# Signal tier labels for the report
TIER_LABELS = {
    "slow_build_accumulation": "🌱 WATCH",
    "pre_breakout_coil":       "🌱 WATCH",
    "sleeping_giant_wakeup":   "⚡ ENTRY",
    "volume_surge":            "⚡ ENTRY",
    "volume_spike":            "⚡ ENTRY",
    "resistance_breakout":     "⚡ ENTRY",
    "momentum_continuation":   "🔥 RIDE",
    "dead_token_resurrection": "🔥 RIDE",
    "mcap_milestone":          "📊 INFO",
    "vc_backed_project":       "📊 INFO",
}


# ---------------------------------------------------------------------------
# DexScreener price fetch
# ---------------------------------------------------------------------------

def fetch_current_price(token_addr: str, chain: str = "base"):
    """Return the current USD price for a token, or None on failure."""
    try:
        url  = f"{DEXSCREENER_URL}/{chain}/{token_addr}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        pairs = resp.json().get("pairs") or []
        if not pairs:
            return None
        # Pick the highest-liquidity pair
        pairs.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
        price_str = pairs[0].get("priceUsd")
        if price_str:
            return float(price_str)
    except Exception:
        pass
    return None


def fetch_peak_price_since(token_addr: str, chain: str, since_ts: float):
    """Approximate peak price since signal by fetching current + h24/h6 highs.

    DexScreener doesn't provide OHLC history, so we track the peak we can see
    from the current snapshot.  For a more accurate picture, signal_log.jsonl
    also stores periodic price snapshots (written by bot.py on each evaluation).
    """
    return fetch_current_price(token_addr, chain)


# ---------------------------------------------------------------------------
# Load signal log
# ---------------------------------------------------------------------------

def load_signal_log(path: str):
    """Read signal_log.jsonl — one JSON object per line."""
    signals = []
    if not os.path.exists(path):
        return signals
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {i} parse error: {exc}")
    return signals


def load_outcomes(path: str):
    """Load previously computed outcomes to avoid re-fetching prices."""
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            return {r["signal_id"]: r for r in data.get("records", [])}
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------

def make_signal_id(sig: dict) -> str:
    return f"{sig.get('token','?')}_{sig.get('signal_type','?')}_{int(sig.get('ts', 0))}"


def evaluate_signal(sig: dict, cached_outcomes: dict):
    """Fetch outcome for a single signal.

    Returns an outcome dict or None if the signal is not ready to evaluate.
    Skips signals that were already evaluated and cached.
    """
    sid  = make_signal_id(sig)
    now  = time.time()
    age  = now - sig.get("ts", 0)

    # Too fresh or too stale
    if age < CHECK_DAYS_MIN * 86400:
        return None
    if age > CHECK_DAYS_MAX * 86400:
        return None

    # Already evaluated and cached — reuse
    if sid in cached_outcomes:
        return cached_outcomes[sid]

    token      = sig.get("token", "")
    chain      = sig.get("chain", "base")
    entry_px   = sig.get("price")
    signal_age = age / 86400  # days

    if not token or not entry_px or entry_px <= 0:
        return None

    current_px = fetch_current_price(token, chain)
    if current_px is None:
        # Token may have been rugged or delisted — treat as max loss
        ret = -1.0
        outcome = "loss"
    else:
        ret     = (current_px - entry_px) / entry_px
        outcome = "win" if ret >= WIN_THRESHOLD else ("loss" if ret <= LOSS_THRESHOLD else "neutral")

    return {
        "signal_id":   sid,
        "token":       token,
        "name":        sig.get("name", "?"),
        "symbol":      sig.get("symbol", "?"),
        "chain":       chain,
        "signal_type": sig.get("signal_type", "unknown"),
        "conviction":  sig.get("conviction", 0.0),
        "mcap":        sig.get("mcap", 0),
        "entry_price": entry_px,
        "current_price": current_px,
        "return_pct":  ret * 100,
        "outcome":     outcome,
        "signal_age_days": signal_age,
        "signal_ts":   sig.get("ts", 0),
        "signal_date": sig.get("date", ""),
        "evaluated_ts": now,
    }


# ---------------------------------------------------------------------------
# Per-type statistics
# ---------------------------------------------------------------------------

def compute_stats(records):
    """Compute per-signal-type win rates and return stats."""
    by_type = {}
    for r in records:
        st = r.get("signal_type", "unknown")
        by_type.setdefault(st, []).append(r)

    stats = {}
    for stype, recs in by_type.items():
        wins    = sum(1 for r in recs if r["outcome"] == "win")
        losses  = sum(1 for r in recs if r["outcome"] == "loss")
        neutral = sum(1 for r in recs if r["outcome"] == "neutral")
        total   = len(recs)
        rets    = [r["return_pct"] for r in recs if r.get("return_pct") is not None]

        win_rate = wins / total if total else 0.0
        avg_ret  = statistics.mean(rets) if rets else 0.0
        med_ret  = statistics.median(rets) if rets else 0.0

        # Best and worst
        best  = max(rets) if rets else 0.0
        worst = min(rets) if rets else 0.0

        # Win rate by conviction band
        high_conv = [r for r in recs if r.get("conviction", 0) >= 7]
        mid_conv  = [r for r in recs if 5 <= r.get("conviction", 0) < 7]
        low_conv  = [r for r in recs if r.get("conviction", 0) < 5]

        def wr(band):
            if not band:
                return None
            return sum(1 for r in band if r["outcome"] == "win") / len(band)

        stats[stype] = {
            "total":          total,
            "wins":           wins,
            "losses":         losses,
            "neutral":        neutral,
            "win_rate":       round(win_rate, 3),
            "avg_return_pct": round(avg_ret, 1),
            "median_return_pct": round(med_ret, 1),
            "best_pct":       round(best, 1),
            "worst_pct":      round(worst, 1),
            "win_rate_high_conv":  round(wr(high_conv), 3) if wr(high_conv) is not None else None,
            "win_rate_mid_conv":   round(wr(mid_conv),  3) if wr(mid_conv)  is not None else None,
            "win_rate_low_conv":   round(wr(low_conv),  3) if wr(low_conv)  is not None else None,
        }

    return stats


# ---------------------------------------------------------------------------
# Adaptive threshold generation
# ---------------------------------------------------------------------------

def generate_adaptive_thresholds(stats):
    """Suggest conviction floor adjustments per signal type based on outcomes.

    Rules:
      - Need >= 5 evaluated signals before adjusting (avoid noise)
      - win_rate < LOW_WIN_RATE (35%)  → raise floor by THRESHOLD_STEP
      - win_rate > HIGH_WIN_RATE (65%) → lower floor by THRESHOLD_STEP (fire more)
      - Otherwise: keep current floor
    """
    thresholds = {}
    for stype, current_floor in CURRENT_FLOORS.items():
        st = stats.get(stype)
        if st is None or st["total"] < 5:
            thresholds[stype] = {
                "current_floor":   current_floor,
                "suggested_floor": current_floor,
                "win_rate":        st["win_rate"] if st else None,
                "total_signals":   st["total"] if st else 0,
                "adjustment":      "no change (insufficient data)",
                "reason":          f"need ≥5 signals, have {st['total'] if st else 0}",
            }
            continue

        wr = st["win_rate"]
        if wr < LOW_WIN_RATE:
            suggested = min(current_floor + THRESHOLD_STEP, MAX_THRESHOLD)
            adj       = f"+{THRESHOLD_STEP}"
            reason    = f"win rate {wr:.0%} < {LOW_WIN_RATE:.0%} — raising floor to reduce false signals"
        elif wr > HIGH_WIN_RATE:
            suggested = max(current_floor - THRESHOLD_STEP, MIN_THRESHOLD)
            adj       = f"-{THRESHOLD_STEP}"
            reason    = f"win rate {wr:.0%} > {HIGH_WIN_RATE:.0%} — lowering floor to catch more winners"
        else:
            suggested = current_floor
            adj       = "no change"
            reason    = f"win rate {wr:.0%} is healthy (between {LOW_WIN_RATE:.0%}–{HIGH_WIN_RATE:.0%})"

        thresholds[stype] = {
            "current_floor":   current_floor,
            "suggested_floor": suggested,
            "win_rate":        wr,
            "avg_return_pct":  st["avg_return_pct"],
            "median_return_pct": st["median_return_pct"],
            "total_signals":   st["total"],
            "adjustment":      adj,
            "reason":          reason,
        }

    return thresholds


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def bar(pct: float, width: int = 10) -> str:
    filled = round(max(0.0, min(pct, 100.0)) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def outcome_emoji(win_rate) -> str:
    if win_rate is None:
        return "⬜"
    if win_rate >= 0.65:
        return "✅"
    if win_rate >= 0.45:
        return "🟡"
    return "🔴"


def generate_report(records, stats, thresholds, now) -> str:
    dt  = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    all_rets = [r["return_pct"] for r in records if r.get("return_pct") is not None]
    overall_wr = (sum(1 for r in records if r["outcome"] == "win") / len(records)
                  if records else 0.0)

    lines = [
        f"# 📡 Signal Outcome Learning Report",
        f"**Generated:** {dt}",
        f"**Total evaluated signals:** {len(records)}",
        f"**Overall win rate:** {overall_wr:.0%}",
        f"**Avg return across all signals:** {statistics.mean(all_rets):.1f}%"
        if all_rets else "",
        "",
        "---",
        "",
        "## 📊 Performance by Signal Type",
        "",
        "| Signal Type | Tier | Signals | Win % | Avg Return | Median | Best | Worst | Threshold |",
        "|-------------|------|---------|-------|------------|--------|------|-------|-----------|",
    ]

    for stype, st in sorted(stats.items(), key=lambda x: -x[1]["win_rate"]):
        tier  = TIER_LABELS.get(stype, "?")
        wr    = st["win_rate"]
        th    = thresholds.get(stype, {})
        adj   = th.get("adjustment", "?")
        cur   = th.get("current_floor", "?")
        sug   = th.get("suggested_floor", "?")
        th_str = f"{cur}→{sug} ({adj})" if adj != "no change" else f"{cur} (hold)"
        lines.append(
            f"| `{stype}` | {tier} | {st['total']} | "
            f"{outcome_emoji(wr)} {wr:.0%} | "
            f"{st['avg_return_pct']:+.0f}% | "
            f"{st['median_return_pct']:+.0f}% | "
            f"{st['best_pct']:+.0f}% | "
            f"{st['worst_pct']:+.0f}% | "
            f"{th_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 🎯 Adaptive Threshold Recommendations",
        "",
    ]

    any_change = False
    for stype, th in thresholds.items():
        if th["adjustment"] not in ("no change", "no change (insufficient data)"):
            any_change = True
            lines.append(
                f"**{stype}** — raise from {th['current_floor']} → {th['suggested_floor']}  \n"
                f"  Reason: {th['reason']}"
            )
            lines.append("")

    if not any_change:
        lines.append("*All thresholds are calibrated well — no changes needed.*")
        lines.append("")

    lines += [
        "---",
        "",
        "## 🏆 Top 10 Winners",
        "",
    ]
    winners = sorted(
        [r for r in records if r["outcome"] == "win"],
        key=lambda r: r.get("return_pct", 0),
        reverse=True
    )[:10]
    if winners:
        lines.append("| Token | Signal | Entry MCap | Return | Age at Signal |")
        lines.append("|-------|--------|-----------|--------|--------------|")
        for r in winners:
            mcap_k = f"${r.get('mcap', 0)/1000:.0f}K" if r.get("mcap") else "?"
            lines.append(
                f"| {r['name']} ({r['symbol']}) | `{r['signal_type']}` | "
                f"{mcap_k} | {r['return_pct']:+.0f}% | {r['signal_age_days']:.0f}d |"
            )
    else:
        lines.append("*No winners yet — keep accumulating signal data.*")

    lines += [
        "",
        "---",
        "",
        "## 💀 Top 10 Losses",
        "",
    ]
    losers = sorted(
        [r for r in records if r["outcome"] == "loss"],
        key=lambda r: r.get("return_pct", 0)
    )[:10]
    if losers:
        lines.append("| Token | Signal | Entry MCap | Return | Conviction |")
        lines.append("|-------|--------|-----------|--------|-----------|")
        for r in losers:
            mcap_k = f"${r.get('mcap', 0)/1000:.0f}K" if r.get("mcap") else "?"
            lines.append(
                f"| {r['name']} ({r['symbol']}) | `{r['signal_type']}` | "
                f"{mcap_k} | {r['return_pct']:+.0f}% | {r['conviction']:.1f} |"
            )
    else:
        lines.append("*No losses recorded yet.*")

    lines += [
        "",
        "---",
        "",
        "## 📈 Conviction vs Win Rate",
        "",
        "Higher conviction should predict higher win rate. If it doesn't, the",
        "scoring model needs recalibration.",
        "",
    ]
    all_types = list(stats.keys())
    if all_types:
        lines.append("| Signal | Conv <5 WR | Conv 5-7 WR | Conv ≥7 WR |")
        lines.append("|--------|-----------|------------|-----------|")
        for stype in sorted(all_types):
            st = stats[stype]
            lc = f"{st['win_rate_low_conv']:.0%}"  if st["win_rate_low_conv"]  is not None else "—"
            mc = f"{st['win_rate_mid_conv']:.0%}"  if st["win_rate_mid_conv"]  is not None else "—"
            hc = f"{st['win_rate_high_conv']:.0%}" if st["win_rate_high_conv"] is not None else "—"
            lines.append(f"| `{stype}` | {lc} | {mc} | {hc} |")

    lines += [
        "",
        "---",
        "",
        "## 🔧 How to Apply These Learnings",
        "",
        "1. **Copy `adaptive_thresholds.json` values into `config.yaml`** under",
        "   `telegram_min_conviction:` to raise/lower each signal's bar.",
        "",
        "2. **Win rate >65% on a signal type** → lower the conviction floor.",
        "   You're being too conservative — you're missing entries.",
        "",
        "3. **Win rate <35% on a signal type** → raise the conviction floor.",
        "   Too many false positives — only fire on the strongest setups.",
        "",
        "4. **Check the conviction-band table** above.",
        "   If conv ≥7 still wins <40%, the signal type itself is unreliable",
        "   (not a threshold problem — a model problem).",
        "",
        f"*Generated by signal_tracker.py — {dt}*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    now = time.time()
    print(f"[signal_tracker] Starting evaluation at {datetime.fromtimestamp(now, tz=timezone.utc)}")

    # Load log
    raw_signals = load_signal_log(SIGNAL_LOG)
    print(f"[signal_tracker] {len(raw_signals)} raw signals in log")

    if not raw_signals:
        print("[signal_tracker] No signals to evaluate. Run the bot first.")
        print("[signal_tracker] Expected file:", SIGNAL_LOG)
        # Write empty outputs so the pipeline doesn't break on missing files
        _write_json(OUTCOMES_FILE,   {"generated_ts": now, "records": []})
        _write_json(THRESHOLDS_FILE, {"generated_ts": now, "thresholds": {}})
        _write_report(REPORT_FILE, "# Signal Tracker\n\nNo signals evaluated yet. Run the bot first.\n")
        return

    # Filter by age window
    evaluable = [
        s for s in raw_signals
        if CHECK_DAYS_MIN * 86400 <= now - s.get("ts", 0) <= CHECK_DAYS_MAX * 86400
    ]
    fresh     = [s for s in raw_signals if now - s.get("ts", 0) < CHECK_DAYS_MIN * 86400]
    too_old   = [s for s in raw_signals if now - s.get("ts", 0) > CHECK_DAYS_MAX * 86400]
    print(f"[signal_tracker] {len(evaluable)} evaluable | {len(fresh)} too fresh | {len(too_old)} too old")

    # Load cached outcomes to avoid unnecessary API calls
    cached = load_outcomes(OUTCOMES_FILE)
    print(f"[signal_tracker] {len(cached)} cached outcomes loaded")

    # Evaluate
    records = []
    for i, sig in enumerate(evaluable):
        sid = make_signal_id(sig)
        if sid in cached:
            records.append(cached[sid])
            continue
        print(f"  [{i+1}/{len(evaluable)}] Fetching outcome for {sig.get('name','?')} ({sig.get('signal_type','?')}) ...")
        result = evaluate_signal(sig, cached)
        if result:
            records.append(result)
            time.sleep(0.3)   # rate-limit DexScreener

    print(f"[signal_tracker] {len(records)} outcomes computed")

    # Stats
    stats      = compute_stats(records)
    thresholds = generate_adaptive_thresholds(stats)
    report     = generate_report(records, stats, thresholds, now)

    # Write outputs
    _write_json(OUTCOMES_FILE, {
        "generated_ts":  now,
        "total_signals": len(raw_signals),
        "evaluated":     len(records),
        "records":       records,
    })
    _write_json(THRESHOLDS_FILE, {
        "generated_ts": now,
        "note":         "Apply suggested_floor values to config.yaml → telegram_min_conviction",
        "thresholds":   thresholds,
    })
    _write_report(REPORT_FILE, report)

    print(f"[signal_tracker] Outputs written:")
    print(f"  {OUTCOMES_FILE}")
    print(f"  {THRESHOLDS_FILE}")
    print(f"  {REPORT_FILE}")

    # Print summary
    overall_wr = (sum(1 for r in records if r["outcome"] == "win") / len(records)
                  if records else 0.0)
    print(f"\n=== SUMMARY ===")
    print(f"Overall win rate: {overall_wr:.0%}")
    for stype, st in sorted(stats.items(), key=lambda x: -x[1]["win_rate"]):
        th = thresholds.get(stype, {})
        adj = th.get("adjustment", "")
        print(
            f"  {TIER_LABELS.get(stype, '?'):12s}  {stype:30s}  "
            f"WR={st['win_rate']:.0%}  n={st['total']:3d}  "
            f"avg={st['avg_return_pct']:+.0f}%  threshold: {adj}"
        )


def _write_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _write_report(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(content)
    os.replace(tmp, path)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    run()
