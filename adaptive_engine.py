"""
adaptive_engine.py — 1%-per-day continuous improvement engine.

Tracks per-strategy performance across regimes, adjusts leverage/risk,
runs mini-backtests on top candidates, and manages cooldowns.

State persisted to adaptive_state.json.
"""
import json
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from regime_detector import RegimeResult, detect_regime_fast

ADAPTIVE_STATE_FILE = "adaptive_state.json"

# ─── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_STRAT_ENTRY = {
    "sharpe_7d":        0.0,
    "sharpe_30d":       0.0,
    "win_rate_7d":      0.5,
    "n_trades_7d":      0,
    "n_trades_30d":     0,
    "regime_fit_score": 0.5,
    "lev_mult":         1.0,
    "risk_mult":        1.0,
    "active":           True,
    "cooldown_until":   None,   # ISO string or None
    "consecutive_losses": 0,
    "last_updated":     None,
}

_DEFAULT_REGIME_ENTRY = {
    "best_strategies":  [],
    "avoid_strategies": [],
    "avg_return":       0.0,
    "n_trades":         0,
}

_DEFAULT_GLOBAL = {
    "current_leverage":  3.0,
    "max_positions":     6,
    "heat_budget":       0.10,
    "improvement_score": 0.0,
    "daily_pnl_history": [],   # list of {date, pnl_pct}
    "last_improvement_run": None,
}


# ─── State I/O ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_state(state_file: str) -> dict:
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "strategy_scores":   {},
        "regime_performance":{},
        "global_params":     _DEFAULT_GLOBAL.copy(),
        "version":           1,
        "created_at":        _now_iso(),
    }


def _save_state(state: dict, state_file: str):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Statistics helpers ───────────────────────────────────────────────────────

def _sharpe(returns: list) -> float:
    if len(returns) < 3:
        return 0.0
    r  = np.array(returns, dtype=float)
    mu = r.mean()
    sd = r.std() + 1e-9
    return float(mu / sd * math.sqrt(252))


def _win_rate(returns: list) -> float:
    if not returns:
        return 0.5
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def _filter_window(trades: list, days: int) -> list:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = []
    for t in trades:
        ts_str = t.get("closed_at") or t.get("opened_at") or ""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= cutoff:
                result.append(t)
        except Exception:
            result.append(t)  # include if we can't parse date
    return result


# ─── Mini-backtest (vectorised, no external deps) ─────────────────────────────

def _mini_backtest(df: pd.DataFrame, signal_fn, risk_pct: float = 0.01,
                   fee_pct: float = 0.001, sl_mult: float = 1.5,
                   tp_mult: float = 3.0) -> dict:
    """
    Run a fast vectorised backtest on df using signal_fn.
    Returns {sharpe, win_rate, n_trades, total_return_pct}
    """
    try:
        sig = signal_fn(df)
    except Exception:
        return {"sharpe": 0.0, "win_rate": 0.5, "n_trades": 0, "total_return_pct": 0.0}

    if sig is None or len(sig) < 10:
        return {"sharpe": 0.0, "win_rate": 0.5, "n_trades": 0, "total_return_pct": 0.0}

    close = df["close"].values
    sig_v = sig.values if hasattr(sig, "values") else np.array(sig)

    # Detect ATR for SL/TP
    atr_col = None
    for col in ["atr_14", "atr", "ATR"]:
        if col in df.columns:
            atr_col = df[col].values
            break
    if atr_col is None:
        atr_col = np.full(len(close), close.mean() * 0.02)

    returns = []
    in_trade = False
    entry_px = sl = tp = direction = 0.0

    for i in range(1, len(close)):
        if in_trade:
            px = close[i]
            if direction == 1:
                if px <= sl or px >= tp or sig_v[i] == -1:
                    pnl = (px - entry_px) / entry_px - 2 * fee_pct
                    returns.append(pnl)
                    in_trade = False
            else:
                if px >= sl or px <= tp or sig_v[i] == 1:
                    pnl = (entry_px - px) / entry_px - 2 * fee_pct
                    returns.append(pnl)
                    in_trade = False
        else:
            cur_sig  = int(sig_v[i])
            prev_sig = int(sig_v[i - 1])
            if cur_sig != 0 and cur_sig != prev_sig:
                entry_px  = close[i] * (1 + fee_pct * cur_sig * 0.1)  # slippage
                atr_here  = float(atr_col[i])
                sl        = entry_px - cur_sig * sl_mult * atr_here
                tp        = entry_px + cur_sig * tp_mult * atr_here
                direction = cur_sig
                in_trade  = True

    if not returns:
        return {"sharpe": 0.0, "win_rate": 0.5, "n_trades": 0, "total_return_pct": 0.0}

    return {
        "sharpe":           _sharpe(returns),
        "win_rate":         _win_rate(returns),
        "n_trades":         len(returns),
        "total_return_pct": float(np.prod([1 + r for r in returns]) - 1) * 100,
    }


# ─── AdaptiveEngine ──────────────────────────────────────────────────────────

class AdaptiveEngine:

    def __init__(self, state_file: str = ADAPTIVE_STATE_FILE):
        self.state_file = state_file
        self._state     = _load_state(state_file)
        self._ensure_defaults()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_defaults(self):
        self._state.setdefault("strategy_scores", {})
        self._state.setdefault("regime_performance", {})
        gp = self._state.setdefault("global_params", {})
        for k, v in _DEFAULT_GLOBAL.items():
            gp.setdefault(k, v)

    def _get_strat(self, key: str) -> dict:
        ss = self._state["strategy_scores"]
        if key not in ss:
            ss[key] = _DEFAULT_STRAT_ENTRY.copy()
        return ss[key]

    def _save(self):
        _save_state(self._state, self.state_file)

    def _is_on_cooldown(self, key: str) -> bool:
        entry = self._get_strat(key)
        cd    = entry.get("cooldown_until")
        if cd is None:
            return False
        try:
            until = datetime.fromisoformat(cd.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) < until
        except Exception:
            return False

    def _set_cooldown(self, key: str, hours: int):
        entry = self._get_strat(key)
        until = datetime.now(timezone.utc) + timedelta(hours=hours)
        entry["cooldown_until"] = until.isoformat()

    # ── Update scores from closed trades ─────────────────────────────────────

    def update(self, closed_trades: list, current_regime: RegimeResult):
        """
        Called daily (or whenever new trades close).
        Re-scores all strategies, rotates parameters.
        """
        if not closed_trades:
            return

        # Group trades by strategy key
        by_key: dict = {}
        for t in closed_trades:
            k = t.get("key", "")
            by_key.setdefault(k, []).append(t)

        for key, trades in by_key.items():
            entry   = self._get_strat(key)
            t7d     = _filter_window(trades, 7)
            t30d    = _filter_window(trades, 30)

            r7d     = [t.get("pnl_pct", 0) for t in t7d]
            r30d    = [t.get("pnl_pct", 0) for t in t30d]

            entry["sharpe_7d"]    = _sharpe(r7d)
            entry["sharpe_30d"]   = _sharpe(r30d)
            entry["win_rate_7d"]  = _win_rate(r7d)
            entry["n_trades_7d"]  = len(t7d)
            entry["n_trades_30d"] = len(t30d)
            entry["last_updated"] = _now_iso()

            # Consecutive losses
            cons_losses = 0
            for t in reversed(trades):
                if t.get("pnl_usd", 0) < 0:
                    cons_losses += 1
                else:
                    break
            entry["consecutive_losses"] = cons_losses

            # Cooldown: 3+ consecutive losses → 24h cooldown
            if cons_losses >= 3 and not self._is_on_cooldown(key):
                self._set_cooldown(key, 24)

            # Leverage and risk multipliers
            entry["lev_mult"]  = self._calc_lev_mult(entry)
            entry["risk_mult"] = self._calc_risk_mult(entry)

            # Regime-fit score
            entry["regime_fit_score"] = self._regime_fit_score(
                key, current_regime)

        # Update regime performance
        self._update_regime_perf(closed_trades, current_regime.regime)

        # Global params: improvement cycle
        self._update_global_params(closed_trades)

        self._save()

    def _calc_lev_mult(self, entry: dict) -> float:
        """Sharpe-driven leverage multiplier: [0.5, 1.5]."""
        # Weighted Sharpe: 2× recent, 1× longer
        sh7  = entry.get("sharpe_7d", 0.0)
        sh30 = entry.get("sharpe_30d", 0.0)
        n30  = entry.get("n_trades_30d", 0)

        # Data-starved bonus: don't reduce too fast with little data
        if n30 < 5:
            return 1.0

        composite = (sh7 * 2 + sh30) / 3.0
        if composite >= 1.5:   return 1.5
        if composite >= 1.0:   return 1.3
        if composite >= 0.5:   return 1.1
        if composite >= 0.0:   return 1.0
        if composite >= -0.5:  return 0.8
        if composite >= -1.0:  return 0.65
        return 0.5

    def _calc_risk_mult(self, entry: dict) -> float:
        """Win-rate-driven risk multiplier: [0.5, 1.3]."""
        wr = entry.get("win_rate_7d", 0.5)
        n  = entry.get("n_trades_7d", 0)
        if n < 3:
            return 1.0
        if wr >= 0.65:   return 1.3
        if wr >= 0.55:   return 1.1
        if wr >= 0.45:   return 1.0
        if wr >= 0.35:   return 0.8
        return 0.5

    def _regime_fit_score(self, key: str, regime: RegimeResult) -> float:
        """Score 0-1 for how well this strategy has performed in current regime."""
        rp   = self._state["regime_performance"].get(regime.regime, {})
        best = rp.get("best_strategies", [])
        avd  = rp.get("avoid_strategies", [])
        if key in best: return 1.0
        if key in avd:  return 0.0
        return 0.5

    def _update_regime_perf(self, trades: list, regime: str):
        rp = self._state["regime_performance"]
        if regime not in rp:
            rp[regime] = _DEFAULT_REGIME_ENTRY.copy()

        regime_trades = [t for t in trades if t.get("regime") == regime]
        if not regime_trades:
            return

        by_key: dict = {}
        for t in regime_trades:
            k = t.get("key", "")
            by_key.setdefault(k, []).append(t.get("pnl_pct", 0))

        key_avg = {k: float(np.mean(v)) for k, v in by_key.items()}
        sorted_keys = sorted(key_avg, key=lambda k: -key_avg[k])

        top_n = max(5, len(sorted_keys) // 3)
        rp[regime]["best_strategies"]  = sorted_keys[:top_n]
        rp[regime]["avoid_strategies"] = sorted_keys[-top_n:]
        all_ret = [t.get("pnl_pct", 0) for t in regime_trades]
        rp[regime]["avg_return"] = round(float(np.mean(all_ret)), 4)
        rp[regime]["n_trades"]   = rp[regime].get("n_trades", 0) + len(regime_trades)

    def _update_global_params(self, trades: list):
        gp    = self._state["global_params"]
        today = _today_str()

        daily_returns = [t.get("pnl_pct", 0) for t in trades]
        if not daily_returns:
            return

        today_pnl = float(np.sum(daily_returns))
        history   = gp.get("daily_pnl_history", [])

        # Upsert today's entry
        today_entry = next((e for e in history if e.get("date") == today), None)
        if today_entry:
            today_entry["pnl_pct"] = round(today_pnl, 4)
        else:
            history.append({"date": today, "pnl_pct": round(today_pnl, 4)})

        # Keep last 30 days
        history = sorted(history, key=lambda x: x["date"])[-30:]
        gp["daily_pnl_history"] = history

        # Improvement score: 7d delta vs prior 7d
        recent7  = [e["pnl_pct"] for e in history[-7:]]
        prior7   = [e["pnl_pct"] for e in history[-14:-7]]
        avg_r    = float(np.mean(recent7)) if recent7 else 0.0
        avg_p    = float(np.mean(prior7))  if prior7  else 0.0
        imp_score= avg_r - avg_p
        gp["improvement_score"] = round(imp_score, 4)

        # 1% per day: adjust max_positions and heat_budget
        if imp_score > 0.001:   # improving
            gp["max_positions"] = min(gp.get("max_positions", 6) + 1, 12)
            gp["heat_budget"]   = min(gp.get("heat_budget", 0.10) + 0.002, 0.15)
        elif imp_score < -0.005:  # declining
            gp["max_positions"] = max(gp.get("max_positions", 6) - 1, 3)
            gp["heat_budget"]   = max(gp.get("heat_budget", 0.10) - 0.003, 0.05)

    # ── Per-strategy params ───────────────────────────────────────────────────

    def get_strategy_params(self, key: str,
                            regime: RegimeResult) -> dict:
        """
        Returns {leverage_mult, risk_mult, active, sl_adjust, tp_adjust}
        for this strategy + regime combo.
        """
        entry = self._get_strat(key)
        on_cd = self._is_on_cooldown(key)

        # Check if this regime is in skip list (from mistake_tracker)
        # (caller can feed in adjustments; we just apply our own state here)
        rp    = self._state["regime_performance"].get(regime.regime, {})
        avoid = rp.get("avoid_strategies", [])
        in_avoid = key in avoid

        active = entry.get("active", True) and not on_cd and not in_avoid

        lev_m  = entry.get("lev_mult", 1.0)
        risk_m = entry.get("risk_mult", 1.0)

        # Regime confidence scaling: reduce aggression in low-confidence regimes
        conf_scale = regime.confidence / 100.0
        if conf_scale < 0.5:
            lev_m  = max(lev_m * conf_scale * 1.5, 0.5)
            risk_m = max(risk_m * conf_scale * 1.5, 0.5)

        # SL/TP adjustments based on vol regime
        sl_adj = 1.0
        tp_adj = 1.0
        if regime.vol_regime == "EXTREME":
            sl_adj = 1.4
            tp_adj = 1.2
        elif regime.vol_regime == "HIGH":
            sl_adj = 1.2
            tp_adj = 1.1
        elif regime.vol_regime == "LOW":
            sl_adj = 0.8
            tp_adj = 0.9

        return {
            "leverage_mult": round(min(lev_m, 1.5), 3),
            "risk_mult":     round(min(risk_m, 1.3), 3),
            "active":        active,
            "sl_adjust":     sl_adj,
            "tp_adjust":     tp_adj,
            "on_cooldown":   on_cd,
        }

    # ── Rank strategies ───────────────────────────────────────────────────────

    def rank_strategies(self, regime: RegimeResult) -> List[Tuple[str, float]]:
        """
        Returns list of (key, score) sorted descending for current regime.
        Score = weighted Sharpe * regime_fit * (1 if active else 0)
        """
        ss      = self._state["strategy_scores"]
        rp      = self._state["regime_performance"].get(regime.regime, {})
        best    = set(rp.get("best_strategies", []))
        avoid   = set(rp.get("avoid_strategies", []))

        ranked = []
        for key, entry in ss.items():
            if self._is_on_cooldown(key):
                continue

            sh7     = entry.get("sharpe_7d", 0.0)
            sh30    = entry.get("sharpe_30d", 0.0)
            n30     = entry.get("n_trades_30d", 0)

            # Data-starved bonus (don't kill unexplored strategies)
            if n30 < 5:
                data_bonus = 0.3
            else:
                data_bonus = 0.0

            composite = (sh7 * 2 + sh30) / 3.0 + data_bonus

            # Regime fit multiplier
            if key in best:   fit = 1.5
            elif key in avoid: fit = 0.2
            else:              fit = 1.0

            score = composite * fit
            ranked.append((key, round(score, 4)))

        ranked.sort(key=lambda x: -x[1])
        return ranked

    # ── Daily improvement cycle ───────────────────────────────────────────────

    def daily_improvement_cycle(self, df_dict: dict,
                                 strategy_registry: dict,
                                 current_regime: RegimeResult):
        """
        Runs mini-backtests on top candidate strategies in the current regime,
        updates scores, logs the improvement delta.

        df_dict:           {tf: DataFrame}  — candle data per timeframe
        strategy_registry: {key: fn}        — strategy name → signal function
        current_regime:    RegimeResult
        """
        gp = self._state["global_params"]
        gp["last_improvement_run"] = _now_iso()

        ranked   = self.rank_strategies(current_regime)
        top_keys = [k for k, _ in ranked[:15]]  # test top 15

        updated = 0
        for key in top_keys:
            if key not in strategy_registry:
                continue
            # Pick best timeframe for this key
            tf = key.split("|")[0] if "|" in key else "1d"
            df = df_dict.get(tf)
            if df is None or len(df) < 50:
                continue

            fn  = strategy_registry[key]
            res = _mini_backtest(df, fn)

            entry = self._get_strat(key)
            # Blend mini-backtest sharpe into sharpe_7d (EMA blend)
            old_sh7 = entry.get("sharpe_7d", 0.0)
            entry["sharpe_7d"] = round(old_sh7 * 0.7 + res["sharpe"] * 0.3, 4)
            entry["lev_mult"]  = self._calc_lev_mult(entry)
            entry["risk_mult"] = self._calc_risk_mult(entry)
            updated += 1

        self._save()

        # Log improvement delta
        imp = self.improvement_score()
        direction = "↑ IMPROVING" if imp > 0 else "↓ DECLINING"
        print(f"[AdaptiveEngine] daily_improvement_cycle done — "
              f"{updated} strategies re-scored. "
              f"7d-vs-prior delta={imp:+.4f} {direction}")

    # ── Improvement score ─────────────────────────────────────────────────────

    def improvement_score(self) -> float:
        """7-day rolling P&L delta vs prior 7d (positive = improving)."""
        return self._state["global_params"].get("improvement_score", 0.0)

    # ── Formatted report ─────────────────────────────────────────────────────

    def report(self) -> str:
        gp    = self._state["global_params"]
        ss    = self._state["strategy_scores"]
        imp   = self.improvement_score()

        active   = sum(1 for e in ss.values() if e.get("active", True) and
                       not self._is_on_cooldown_entry(e))
        on_cd    = sum(1 for e in ss.values() if self._is_on_cooldown_entry(e))
        total    = len(ss)

        lines = [
            "[AdaptiveEngine] Report",
            f"  Improvement score  : {imp:+.4f}  "
            f"({'improving' if imp > 0 else 'declining'})",
            f"  Strategies tracked : {total}  |  active={active}  cooldown={on_cd}",
            f"  Max positions      : {gp.get('max_positions', 6)}",
            f"  Heat budget        : {gp.get('heat_budget', 0.10)*100:.1f}%",
            f"  Current leverage   : {gp.get('current_leverage', 3.0):.1f}x",
            "  Top 5 strategies by score:",
        ]

        from regime_detector import RegimeResult as RR
        dummy_regime = RR()
        ranked = self.rank_strategies(dummy_regime)
        for k, score in ranked[:5]:
            entry = self._get_strat(k)
            lines.append(
                f"    {k:42s}  score={score:+.3f}  "
                f"sh7={entry.get('sharpe_7d',0):.2f}  "
                f"wr={entry.get('win_rate_7d',0.5)*100:.0f}%  "
                f"lev={entry.get('lev_mult',1):.2f}x"
            )

        hist = gp.get("daily_pnl_history", [])
        if hist:
            lines.append("  Daily P&L (last 7 days):")
            for e in hist[-7:]:
                bar = "█" * int(abs(e["pnl_pct"]) * 10)
                sign = "+" if e["pnl_pct"] >= 0 else ""
                lines.append(f"    {e['date']}  {sign}{e['pnl_pct']:.3f}%  {bar}")

        return "\n".join(lines)

    def _is_on_cooldown_entry(self, entry: dict) -> bool:
        cd = entry.get("cooldown_until")
        if cd is None:
            return False
        try:
            until = datetime.fromisoformat(cd.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) < until
        except Exception:
            return False

    # ── Persist helper (callable externally) ─────────────────────────────────

    def save(self):
        self._save()


# ─── Module-level convenience helpers ─────────────────────────────────────────

_engine_singleton: Optional[AdaptiveEngine] = None


def get_engine(state_file: str = ADAPTIVE_STATE_FILE) -> AdaptiveEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = AdaptiveEngine(state_file)
    return _engine_singleton


# ─── Simple self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from regime_detector import RegimeResult
    eng = AdaptiveEngine()

    dummy_trades = [
        {"key": "1d|ema_trend", "strategy": "ema_trend", "pnl_pct": 1.2,
         "pnl_usd": 120, "regime": "STRONG_BULL", "closed_at": "2026-06-03T12:00:00+00:00"},
        {"key": "1d|ema_trend", "strategy": "ema_trend", "pnl_pct": -0.8,
         "pnl_usd": -80, "regime": "STRONG_BULL", "closed_at": "2026-06-03T13:00:00+00:00"},
        {"key": "4h|supertrend", "strategy": "supertrend", "pnl_pct": 2.1,
         "pnl_usd": 210, "regime": "STRONG_BULL", "closed_at": "2026-06-03T14:00:00+00:00"},
    ]
    regime = RegimeResult(regime="STRONG_BULL", confidence=80.0)
    eng.update(dummy_trades, regime)
    print(eng.report())
    params = eng.get_strategy_params("1d|ema_trend", regime)
    print("Params for 1d|ema_trend:", params)
