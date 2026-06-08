"""
mistake_tracker.py — Trade mistake classifier and improvement signal generator.

Classifies every closed trade, identifies what went wrong, records it to
mistakes.json, and provides aggregated adjustment signals per strategy.
"""
import json
import os
import math
from datetime import datetime, timezone, timedelta
from typing import Optional


MISTAKES_FILE = "mistakes.json"

# ─── Mistake types ────────────────────────────────────────────────────────────
MISTAKE_TYPES = [
    None,               # No mistake — trade was fine
    "regime_mismatch",  # Opened a trend trade in ranging market, etc.
    "overlevered",      # Leverage too high relative to volatility
    "chased_entry",     # Entered after a big move; high RSI/low RSI extremes
    "poor_rr",          # Risk/reward < 1.0
    "held_too_long",    # Held past optimal exit (bars_held > threshold)
    "cut_too_early",    # Trade closed at tiny gain then reversed favourably
    "fee_drag",         # Fee/funding ate most of the gross PnL
    "liquidated",       # Hit liquidation price
    "low_conviction_loss",  # Small loss — noise trade, not a clear mistake
]

# Regime ↔ strategy family compatibility
_REGIME_COMPAT = {
    "STRONG_BULL":  {"good": ["ema_trend","supertrend","macd","obv","volume_breakout","momentum"],
                     "bad":  ["bb","stoch","mean_revert","rsi_reverse"]},
    "WEAK_BULL":    {"good": ["ema_trend","supertrend","macd","momentum"],
                     "bad":  ["mean_revert"]},
    "CHOPPY_BULL":  {"good": ["bb","stoch","engulf","mean_revert"],
                     "bad":  ["ema_trend","supertrend","macd"]},
    "STRONG_BEAR":  {"good": ["ema_trend","supertrend","macd","obv","momentum"],
                     "bad":  ["bb","stoch","mean_revert","rsi_reverse"]},
    "WEAK_BEAR":    {"good": ["ema_trend","supertrend","macd","momentum"],
                     "bad":  ["mean_revert"]},
    "CHOPPY_BEAR":  {"good": ["bb","stoch","engulf","mean_revert"],
                     "bad":  ["ema_trend","supertrend","macd"]},
    "RANGING":      {"good": ["bb","stoch","rsi","engulf","mean_revert","vwap"],
                     "bad":  ["ema_trend","supertrend","macd","obv"]},
    "VOLATILE":     {"good": ["bb","breakout","stoch","volume_breakout"],
                     "bad":  ["ema_trend"]},
}


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _load_mistakes() -> list:
    if os.path.exists(MISTAKES_FILE):
        try:
            with open(MISTAKES_FILE) as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _save_mistakes(records: list):
    with open(MISTAKES_FILE, "w") as f:
        json.dump(records, f, indent=2, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ─── Regime mismatch check ────────────────────────────────────────────────────

def _is_regime_mismatch(strategy_name: str, regime: str,
                         direction: int = 0) -> bool:
    """
    Returns True if:
    1. Strategy family is known-bad for this regime, OR
    2. Trade direction opposes the regime bias
       (long in STRONG_BEAR/WEAK_BEAR, short in STRONG_BULL/WEAK_BULL)
    """
    # Direction vs regime bias check
    bear_regimes = {"STRONG_BEAR", "WEAK_BEAR"}
    bull_regimes = {"STRONG_BULL", "WEAK_BULL"}
    if direction == 1 and regime in bear_regimes:
        return True
    if direction == -1 and regime in bull_regimes:
        return True

    # Strategy-family check
    compat = _REGIME_COMPAT.get(regime, {})
    bad    = compat.get("bad", [])
    sn     = strategy_name.lower()
    for bad_kw in bad:
        if bad_kw in sn:
            return True
    return False


# ─── Core classifier ─────────────────────────────────────────────────────────

def classify_trade(trade: dict, regime: str, regime_conf: float) -> dict:
    """
    Classify a closed trade dict and return a classification record.

    trade fields expected:
      pnl_usd, pnl_pct, gross_pnl, total_costs_usd, leverage,
      bars_held, exit_reason, was_liquidated, strategy, key,
      entry_px, exit_px, sl_px, tp_px, direction, margin_return_pct
    """
    pnl_usd     = float(trade.get("pnl_usd", 0))
    pnl_pct     = float(trade.get("pnl_pct", 0))
    gross_pnl   = float(trade.get("gross_pnl", pnl_usd))
    fees        = float(trade.get("total_costs_usd", 0))
    leverage    = float(trade.get("leverage", 3.0))
    bars_held   = int(trade.get("bars_held", 0))
    exit_reason = str(trade.get("exit_reason", ""))
    liquidated  = bool(trade.get("was_liquidated", False))
    strategy    = str(trade.get("strategy", ""))
    entry_px    = float(trade.get("entry_px", 1))
    exit_px     = float(trade.get("exit_px", 1))
    sl_px       = float(trade.get("sl_px", 0))
    tp_px       = float(trade.get("tp_px", 0))
    direction   = int(trade.get("direction", 1))

    # Derived
    is_loss = pnl_usd < 0
    sl_dist = abs(entry_px - sl_px) if sl_px else abs(entry_px * 0.02)
    tp_dist = abs(tp_px - entry_px) if tp_px else sl_dist * 2
    rr      = tp_dist / (sl_dist + 1e-9)
    fee_ratio = abs(fees) / (abs(gross_pnl) + 1e-9) if abs(gross_pnl) > 1e-9 else 0.0

    mistake_type = None
    severity     = 0
    lesson       = "Trade executed correctly."
    adjustment   = {"leverage_mult": 1.0, "risk_mult": 1.0,
                    "skip_regime": None, "cooldown_bars": 0}

    # ── Priority-ordered classification ──────────────────────────────────────

    if liquidated:
        mistake_type = "liquidated"
        severity     = 3
        lesson       = (f"Position was liquidated. Leverage {leverage:.1f}x was too high "
                        f"for the volatility — reduce leverage or widen SL.")
        adjustment   = {"leverage_mult": 0.4, "risk_mult": 0.5,
                        "skip_regime": regime, "cooldown_bars": 48}

    elif _is_regime_mismatch(strategy, regime, direction) and is_loss:
        mistake_type = "regime_mismatch"
        severity     = 2
        dir_str      = "LONG" if direction == 1 else "SHORT"
        lesson       = (f"{dir_str} {strategy} in {regime} (conf={regime_conf:.0f}%) "
                        f"is a regime mismatch — trade with the bias or sit out.")
        adjustment   = {"leverage_mult": 0.7, "risk_mult": 0.7,
                        "skip_regime": regime, "cooldown_bars": 4}

    elif leverage > 10 and is_loss:
        mistake_type = "overlevered"
        severity     = 2
        lesson       = (f"Leverage {leverage:.1f}x amplified the loss. "
                        f"In regime {regime}, cap leverage lower.")
        adjustment   = {"leverage_mult": 0.6, "risk_mult": 0.8,
                        "skip_regime": None, "cooldown_bars": 2}

    elif rr < 1.0 and is_loss:
        mistake_type = "poor_rr"
        severity     = 1
        lesson       = (f"RR={rr:.2f} < 1.0. The SL was too close or TP too tight. "
                        f"Widen TP or tighten SL.")
        adjustment   = {"leverage_mult": 0.9, "risk_mult": 0.85,
                        "skip_regime": None, "cooldown_bars": 0}

    elif bars_held > 80 and is_loss:
        mistake_type = "held_too_long"
        severity     = 2
        lesson       = (f"Held {bars_held} bars before closing at a loss. "
                        f"Time-based exit or trailing SL would have helped.")
        adjustment   = {"leverage_mult": 0.9, "risk_mult": 0.9,
                        "skip_regime": None, "cooldown_bars": 2}

    elif exit_reason in ("max_hold", "time_exit") and pnl_pct > 0 and pnl_pct < 0.3:
        mistake_type = "cut_too_early"
        severity     = 1
        lesson       = (f"Trade exited at +{pnl_pct:.2f}% via {exit_reason} "
                        f"but had potential for more. Extend hold window.")
        adjustment   = {"leverage_mult": 1.0, "risk_mult": 1.0,
                        "skip_regime": None, "cooldown_bars": 0}

    elif fee_ratio > 0.5 and is_loss:
        mistake_type = "fee_drag"
        severity     = 1
        lesson       = (f"Fees/funding (${fees:.3f}) consumed >{fee_ratio*100:.0f}% "
                        f"of gross PnL. Reduce leverage or hold longer.")
        adjustment   = {"leverage_mult": 0.85, "risk_mult": 0.9,
                        "skip_regime": None, "cooldown_bars": 0}

    elif is_loss and abs(pnl_pct) < 0.5:
        mistake_type = "low_conviction_loss"
        severity     = 0
        lesson       = "Small noise loss — no clear mistake, monitor for pattern."
        adjustment   = {"leverage_mult": 1.0, "risk_mult": 1.0,
                        "skip_regime": None, "cooldown_bars": 0}

    return {
        "mistake_type": mistake_type,
        "severity":     severity,
        "lesson":       lesson,
        "adjustment":   adjustment,
    }


# ─── Record a trade ───────────────────────────────────────────────────────────

def record_trade(trade: dict, regime: str, regime_conf: float):
    """Classify and persistently record a closed trade."""
    classification = classify_trade(trade, regime, regime_conf)

    record = {
        "ts":          _now_iso(),
        "date":        _today_str(),
        "key":         trade.get("key", ""),
        "strategy":    trade.get("strategy", ""),
        "tf":          trade.get("tf", ""),
        "regime":      regime,
        "regime_conf": round(regime_conf, 1),
        "pnl_usd":     round(float(trade.get("pnl_usd", 0)), 4),
        "pnl_pct":     round(float(trade.get("pnl_pct", 0)), 3),
        "leverage":    float(trade.get("leverage", 3.0)),
        "bars_held":   int(trade.get("bars_held", 0)),
        "exit_reason": trade.get("exit_reason", ""),
        "was_win":     float(trade.get("pnl_usd", 0)) > 0,
        **classification,
    }

    records = _load_mistakes()
    records.append(record)
    _save_mistakes(records)
    return record


# ─── Aggregated adjustments per strategy ──────────────────────────────────────

def get_adjustments(strategy_key: str, lookback: int = 50) -> dict:
    """
    Aggregate recent mistakes for a given strategy key.
    Returns: {leverage_mult, risk_mult, skip_regimes: list, n_mistakes}
    """
    records = _load_mistakes()
    recent  = [r for r in records if r.get("key") == strategy_key][-lookback:]

    if not recent:
        return {"leverage_mult": 1.0, "risk_mult": 1.0,
                "skip_regimes": [], "n_mistakes": 0}

    n_mistakes  = sum(1 for r in recent if r.get("mistake_type") is not None)
    total       = len(recent)

    # Weighted average of multipliers (weight by severity)
    lev_mults  = []
    risk_mults = []
    skip_regimes: dict = {}

    for r in recent:
        adj = r.get("adjustment", {})
        sev = r.get("severity", 0)
        w   = 1 + sev  # higher severity = more weight
        lev_mults.append(adj.get("leverage_mult", 1.0) * w)
        risk_mults.append(adj.get("risk_mult", 1.0) * w)
        sr = adj.get("skip_regime")
        if sr:
            skip_regimes[sr] = skip_regimes.get(sr, 0) + 1

    total_w = sum(1 + r.get("severity", 0) for r in recent)
    lev_m  = sum(lev_mults) / (total_w + 1e-9)
    risk_m = sum(risk_mults) / (total_w + 1e-9)

    # Only skip a regime if it triggered skip in >20% of recent trades
    skip = [r for r, cnt in skip_regimes.items() if cnt / total > 0.20]

    # Clamp
    lev_m  = max(0.3, min(lev_m, 1.5))
    risk_m = max(0.3, min(risk_m, 1.3))

    return {
        "leverage_mult": round(lev_m, 3),
        "risk_mult":     round(risk_m, 3),
        "skip_regimes":  skip,
        "n_mistakes":    n_mistakes,
    }


# ─── Daily report ─────────────────────────────────────────────────────────────

def daily_mistake_report() -> str:
    records = _load_mistakes()
    today   = _today_str()
    today_r = [r for r in records if r.get("date") == today]

    if not today_r:
        return f"[MistakeTracker] No trades recorded today ({today})."

    wins   = sum(1 for r in today_r if r.get("was_win"))
    losses = len(today_r) - wins
    wr     = wins / len(today_r) * 100 if today_r else 0

    type_counts: dict = {}
    for r in today_r:
        mt = r.get("mistake_type") or "none"
        type_counts[mt] = type_counts.get(mt, 0) + 1

    sev3 = [r for r in today_r if r.get("severity", 0) == 3]

    lines = [
        f"[MistakeTracker] Daily Report — {today}",
        f"  Trades: {len(today_r)}  |  Wins: {wins}  Losses: {losses}  |  WR: {wr:.1f}%",
        f"  Mistake breakdown:",
    ]
    for mt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {mt:30s} × {cnt}")
    if sev3:
        lines.append(f"  CRITICAL (sev=3): {len(sev3)} trade(s)")
        for r in sev3:
            lines.append(f"    [{r['key']}] {r['lesson']}")
    return "\n".join(lines)


# ─── Regime accuracy ──────────────────────────────────────────────────────────

def get_regime_accuracy(lookback: int = 100) -> dict:
    """What % of trades in each regime were profitable (last N records)."""
    records = _load_mistakes()
    recent  = records[-lookback:]

    regime_stats: dict = {}
    for r in recent:
        reg = r.get("regime", "UNKNOWN")
        if reg not in regime_stats:
            regime_stats[reg] = {"wins": 0, "total": 0}
        regime_stats[reg]["total"] += 1
        if r.get("was_win"):
            regime_stats[reg]["wins"] += 1

    result = {}
    for reg, s in regime_stats.items():
        result[reg] = {
            "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0.0,
            "n":        s["total"],
        }
    return result


# ─── Worst recurring patterns ─────────────────────────────────────────────────

def get_worst_patterns() -> list:
    """Return top 5 recurring (non-None) mistake types by frequency × avg_severity."""
    records = _load_mistakes()[-200:]

    pattern: dict = {}
    for r in records:
        mt  = r.get("mistake_type")
        sev = r.get("severity", 0)
        if mt is None:
            continue
        if mt not in pattern:
            pattern[mt] = {"count": 0, "total_sev": 0, "total_loss": 0.0}
        pattern[mt]["count"]     += 1
        pattern[mt]["total_sev"] += sev
        pattern[mt]["total_loss"]+= min(float(r.get("pnl_usd", 0)), 0)

    scored = []
    for mt, stats in pattern.items():
        avg_sev = stats["total_sev"] / stats["count"]
        score   = stats["count"] * (1 + avg_sev)
        scored.append({
            "mistake_type": mt,
            "count":        stats["count"],
            "avg_severity": round(avg_sev, 2),
            "total_loss":   round(stats["total_loss"], 2),
            "score":        round(score, 2),
        })

    scored.sort(key=lambda x: -x["score"])
    return scored[:5]


# ─── Consecutive loss check ───────────────────────────────────────────────────

def get_consecutive_losses(strategy_key: str) -> int:
    """Return number of consecutive losses at the end of this strategy's history."""
    records = _load_mistakes()
    hist    = [r for r in records if r.get("key") == strategy_key]
    if not hist:
        return 0
    count = 0
    for r in reversed(hist):
        if not r.get("was_win", True):
            count += 1
        else:
            break
    return count


# ─── Simple self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_trade = {
        "key": "1d|ema_trend", "strategy": "ema_trend", "tf": "1d",
        "pnl_usd": -55.0, "pnl_pct": -1.2, "gross_pnl": -50.0,
        "total_costs_usd": 5.0, "leverage": 3.0, "bars_held": 12,
        "exit_reason": "sl_hit", "was_liquidated": False,
        "entry_px": 2000.0, "exit_px": 1975.0, "sl_px": 1975.0, "tp_px": 2050.0,
        "direction": 1, "margin_return_pct": -3.6,
    }
    c = classify_trade(test_trade, "RANGING", 75.0)
    print("Classification:", json.dumps(c, indent=2))
