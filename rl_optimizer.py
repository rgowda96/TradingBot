"""
RL Engine v2 — capital-maximising, fully adaptive.

Mechanisms:
  1. EMA Bandit         — recency-weighted UCB per strategy; hot strategies
                          get more capital, cold ones shrink toward 0.
  2. Q-Learner          — tunes (sl_mult, tp_mult) per (regime, strategy).
                          Reward = log-return (Kelly-optimal signal).
  3. Kelly Sizing       — position size from per-strategy win-rate + avg RR
                          estimated from recent closed trades (rolling 30).
  4. Portfolio Heat     — total open risk capped at MAX_HEAT of equity.
  5. Regime Gate        — trend strategies only fire in trending regimes;
                          mean-reversion only in ranging/volatile.
  6. Strategy Culling   — disable strategies with Sharpe < CULL_SHARPE after
                          MIN_CULL_TRADES; re-enable after conditions change.
  7. Epsilon Decay      — exploration reduces as sample size grows.
"""
import json, math, os, numpy as np
from datetime import datetime, timezone

RL_STATE_FILE  = "rl_state.json"
MAX_HEAT       = 0.10    # max 10% total equity at risk simultaneously
BASE_RISK      = 0.01    # 1% baseline
MAX_RISK       = 0.025   # 2.5% cap per trade
MIN_RISK       = 0.002   # 0.2% floor
KELLY_FRACTION = 0.25    # fractional Kelly (safety factor)
CULL_SHARPE    = -1.0    # disable strategy below this Sharpe after...
MIN_CULL_TRADES= 15      # ...this many trades
EMA_DECAY      = 0.92    # EMA decay for bandit recency weighting
UCB_C          = 1.2
ALPHA          = 0.20    # Q-learning rate
GAMMA          = 0.85    # discount
ROLLING_WINDOW = 30      # trades used for Kelly / Sharpe estimates

SL_OPTIONS = [0.8, 1.2, 1.5, 2.0, 2.5, 3.0]
TP_OPTIONS = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
ACTIONS    = [(sl, tp) for sl in SL_OPTIONS for tp in TP_OPTIONS]  # 36 combos

REGIMES = ["trending_up", "trending_down", "ranging", "volatile",
           "STRONG_BULL", "WEAK_BULL", "CHOPPY_BULL",
           "STRONG_BEAR", "WEAK_BEAR", "CHOPPY_BEAR",
           "RANGING", "VOLATILE"]

# Regime suitability: which strategy types work in which regimes
REGIME_SUITABILITY = {
    # Legacy regime names
    "trending_up":   ["ema_trend","supertrend","macd","golden","chandelier","triple_screen",
                      "vol_mom","obv","squeeze"],
    "trending_down": ["ema_trend","supertrend","macd","golden","chandelier","triple_screen",
                      "vol_mom","obv","squeeze"],
    "ranging":       ["rsi","bb","stoch","engulf","mean_revert","squeeze","vwap","st_rsi"],
    "volatile":      ["bb","stoch","engulf","mean_revert","vwap","squeeze"],
    # New 8-regime names from regime_detector.py
    "STRONG_BULL":   ["ema_trend","supertrend","macd","chandelier","triple_screen",
                      "squeeze","obv","vol_mom"],
    "WEAK_BULL":     ["macd","chandelier","triple_screen","bb","squeeze","obv"],
    "CHOPPY_BULL":   ["bb","mean_revert","vwap","st_rsi","rsi","squeeze"],
    "STRONG_BEAR":   ["ema_trend","supertrend","macd","chandelier","triple_screen",
                      "squeeze","obv","vol_mom","bb_rsi","st_rsi"],
    "WEAK_BEAR":     ["macd","chandelier","triple_screen","bb","squeeze","obv","st_rsi"],
    "CHOPPY_BEAR":   ["bb","mean_revert","vwap","st_rsi","rsi","squeeze"],
    "RANGING":       ["rsi","bb","stoch","engulf","mean_revert","squeeze","vwap","st_rsi"],
    "VOLATILE":      ["bb","stoch","engulf","mean_revert","vwap","squeeze","bb_rsi"],
}


def _now():
    return datetime.now(timezone.utc).isoformat()


# ── State I/O ──────────────────────────────────────────────────────────────────

def _default_state(strategy_keys):
    return {
        "version":   2,
        "bandit":    {sk: {"n": 0, "ema_reward": 0.0, "total_reward": 0.0,
                           "disabled": False, "disable_reason": ""}
                      for sk in strategy_keys},
        "q_table":   {r: {sk: [0.0]*len(ACTIONS) for sk in strategy_keys}
                      for r in REGIMES},
        "act_counts":{r: {sk: [0]*len(ACTIONS)   for sk in strategy_keys}
                      for r in REGIMES},
        "total_pulls": 0,
        "trade_history": {},   # sk → list of last ROLLING_WINDOW {"pnl_pct","won"}
        "created_at": _now(),
    }


def load_rl(strategy_keys):
    state = _default_state(strategy_keys)
    if os.path.exists(RL_STATE_FILE):
        try:
            with open(RL_STATE_FILE) as f:
                saved = json.load(f)
            # Merge: keep existing learned values, add any new keys
            for sk in strategy_keys:
                if sk in saved.get("bandit", {}):
                    state["bandit"][sk] = saved["bandit"][sk]
                for r in REGIMES:
                    if (r in saved.get("q_table", {}) and
                            sk in saved["q_table"][r] and
                            len(saved["q_table"][r][sk]) == len(ACTIONS)):
                        state["q_table"][r][sk]    = saved["q_table"][r][sk]
                        state["act_counts"][r][sk] = saved.get("act_counts", {}).get(
                            r, {}).get(sk, [0]*len(ACTIONS))
                if sk in saved.get("trade_history", {}):
                    state["trade_history"][sk] = saved["trade_history"][sk]
            state["total_pulls"] = saved.get("total_pulls", 0)
        except Exception:
            pass
    return state


def save_rl(state):
    with open(RL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Regime detection ───────────────────────────────────────────────────────────

def detect_regime(df) -> str:
    close = df["close"]
    atr   = df["atr_14"]
    n     = min(20, len(close))
    ret   = (close.iloc[-1] / close.iloc[-n] - 1) * 100
    norm  = atr.iloc[-1] / atr.tail(n).mean() if atr.tail(n).mean() > 0 else 1.0

    if norm > 1.6:
        return "volatile"
    if ret >  3.0:
        return "trending_up"
    if ret < -3.0:
        return "trending_down"
    return "ranging"


def strategy_fits_regime(strategy_name: str, regime: str) -> bool:
    """Return True if this strategy type is appropriate for the current regime."""
    suitable = REGIME_SUITABILITY.get(regime, [])
    name_lower = strategy_name.lower()
    return any(kw in name_lower for kw in suitable)


# ── Kelly position sizing ──────────────────────────────────────────────────────

def kelly_risk(state, strategy_key, base_risk=BASE_RISK) -> float:
    """
    Compute fractional Kelly risk from recent trade history.
    Falls back to base_risk when sample is small.
    """
    hist = state["trade_history"].get(strategy_key, [])
    if len(hist) < 5:
        return base_risk   # not enough data yet

    recent = hist[-ROLLING_WINDOW:]
    wins   = [t for t in recent if t["won"]]
    losses = [t for t in recent if not t["won"]]
    n      = len(recent)

    win_rate = len(wins) / n
    avg_win  = np.mean([t["pnl_pct"] for t in wins])  if wins   else 0.01
    avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 0.01

    if avg_loss == 0:
        return base_risk

    rr        = avg_win / avg_loss        # reward-to-risk ratio
    kelly_f   = win_rate - (1 - win_rate) / rr if rr > 0 else 0
    kelly_f   = max(0, kelly_f) * KELLY_FRACTION  # fractional Kelly

    # Scale by bandit quality (high-confidence strategies get closer to Kelly)
    b = state["bandit"].get(strategy_key, {})
    confidence = min(1.0, b.get("n", 0) / 20)
    blended = base_risk + confidence * (kelly_f - base_risk)

    return float(np.clip(blended, MIN_RISK, MAX_RISK))


# ── Portfolio heat ─────────────────────────────────────────────────────────────

def current_heat(open_positions: dict, equity: float) -> float:
    """Sum of all open risk amounts as fraction of equity."""
    total_risk = 0.0
    for pos in open_positions.values():
        sl_dist   = abs(pos["entry_px"] - pos["sl_px"])
        total_risk += pos["qty"] * sl_dist
    return total_risk / equity if equity > 0 else 0.0


def heat_budget(open_positions: dict, equity: float) -> float:
    """Remaining risk budget as fraction of equity."""
    return max(0.0, MAX_HEAT - current_heat(open_positions, equity))


# ── UCB1 + EMA Bandit ─────────────────────────────────────────────────────────

def ucb_score(state, strategy_key) -> float:
    b     = state["bandit"].get(strategy_key, {"n": 0, "ema_reward": 0.0})
    n     = b.get("n", 0)
    total = state.get("total_pulls", 0)
    if n == 0 or total == 0:
        return float("inf")
    ema_r = b.get("ema_reward", 0.0)
    return ema_r + UCB_C * math.sqrt(math.log(total + 1) / n)


def get_risk_multiplier(state, strategy_key) -> float:
    """
    Maps UCB score to a risk multiplier in [0.3, 2.0].
    Disabled strategies return 0.
    """
    b = state["bandit"].get(strategy_key, {})
    if b.get("disabled", False):
        return 0.0

    scores = {}
    for sk, bv in state["bandit"].items():
        if not bv.get("disabled", False):
            scores[sk] = ucb_score(state, sk)

    if not scores:
        return 1.0

    finite = {k: v for k, v in scores.items() if v != float("inf")}
    if not finite:
        return 1.0

    sk_score = scores.get(strategy_key, 1.0)
    if sk_score == float("inf"):
        return 1.3   # unexplored → slightly above baseline

    mn, mx = min(finite.values()), max(finite.values())
    if mx == mn:
        return 1.0
    norm = (sk_score - mn) / (mx - mn)
    return 0.3 + 1.7 * norm   # [0.3, 2.0]


def update_bandit(state, strategy_key, log_ret: float):
    """
    Update bandit with log-return reward (Kelly-optimal).
    log_ret = ln(exit_px / entry_px) * direction
    """
    b = state["bandit"].setdefault(strategy_key,
        {"n": 0, "ema_reward": 0.0, "total_reward": 0.0,
         "disabled": False, "disable_reason": ""})
    n_prev = b["n"]
    b["n"]            += 1
    b["total_reward"] += log_ret
    # EMA update: new observations weighted more
    if n_prev == 0:
        b["ema_reward"] = log_ret
    else:
        b["ema_reward"] = EMA_DECAY * b["ema_reward"] + (1 - EMA_DECAY) * log_ret

    state["total_pulls"] = state.get("total_pulls", 0) + 1

    # Record in rolling trade history
    hist = state["trade_history"].setdefault(strategy_key, [])
    hist.append({"pnl_pct": log_ret, "won": log_ret > 0})
    if len(hist) > ROLLING_WINDOW * 2:
        state["trade_history"][strategy_key] = hist[-ROLLING_WINDOW:]

    # Auto-cull check
    _maybe_cull(state, strategy_key)


def _maybe_cull(state, sk):
    """Disable a strategy if its recent Sharpe is terrible."""
    hist = state["trade_history"].get(sk, [])
    b    = state["bandit"].get(sk, {})
    if len(hist) < MIN_CULL_TRADES:
        return
    recent = [t["pnl_pct"] for t in hist[-MIN_CULL_TRADES:]]
    arr    = np.array(recent)
    sharpe = arr.mean() / (arr.std() + 1e-9) * math.sqrt(252)
    if sharpe < CULL_SHARPE and not b.get("disabled", False):
        b["disabled"]       = True
        b["disable_reason"] = f"Sharpe={sharpe:.2f} < {CULL_SHARPE} (culled)"
    elif sharpe > 0.0 and b.get("disabled", False):
        # Re-enable if it recovers
        b["disabled"]       = False
        b["disable_reason"] = ""


# ── Q-Learner ──────────────────────────────────────────────────────────────────

def _epsilon(state, strategy_key) -> float:
    """Decay epsilon as we gather more data (less exploration over time)."""
    n = state["bandit"].get(strategy_key, {}).get("n", 0)
    return max(0.05, 0.30 * math.exp(-n / 50))


def choose_action(state, regime, strategy_key):
    """Returns (sl_mult, tp_mult, action_idx) using epsilon-greedy Q-learning."""
    q_vals = state["q_table"].get(regime, {}).get(strategy_key, [0.0]*len(ACTIONS))
    eps    = _epsilon(state, strategy_key)

    if np.random.random() < eps:
        idx = np.random.randint(len(ACTIONS))
    else:
        idx = int(np.argmax(q_vals))

    sl, tp = ACTIONS[idx]
    return sl, tp, idx


def update_q(state, regime, strategy_key, action_idx, log_ret, next_regime):
    """TD(0) Q-learning update using log-return as reward."""
    qt = state["q_table"]
    if regime not in qt:
        qt[regime] = {}
    if strategy_key not in qt[regime]:
        qt[regime][strategy_key] = [0.0]*len(ACTIONS)

    q_curr = qt[regime][strategy_key][action_idx]

    next_q  = qt.get(next_regime, {}).get(strategy_key, [0.0]*len(ACTIONS))
    q_max   = max(next_q) if next_q else 0.0

    qt[regime][strategy_key][action_idx] = (
        q_curr + ALPHA * (log_ret + GAMMA * q_max - q_curr)
    )

    ac = state["act_counts"]
    if regime not in ac:
        ac[regime] = {}
    if strategy_key not in ac[regime]:
        ac[regime][strategy_key] = [0]*len(ACTIONS)
    ac[regime][strategy_key][action_idx] += 1


# ── Best known params for a strategy (greedy, no exploration) ─────────────────

def best_params(state, regime, strategy_key):
    """Return the best-known (sl, tp) for this strategy in this regime."""
    q_vals = state["q_table"].get(regime, {}).get(strategy_key, [0.0]*len(ACTIONS))
    idx    = int(np.argmax(q_vals))
    return ACTIONS[idx][0], ACTIONS[idx][1], idx


# ── Summary ────────────────────────────────────────────────────────────────────

def rl_summary(state) -> str:
    lines = ["=== RL STATE SUMMARY ==="]
    lines.append(f"  Total pulls : {state.get('total_pulls', 0)}")

    bandit = state.get("bandit", {})
    rows = [(sk, b) for sk, b in bandit.items() if b.get("n", 0) > 0]
    rows.sort(key=lambda x: -x[1].get("ema_reward", 0))

    active   = [(sk, b) for sk, b in rows if not b.get("disabled")]
    disabled = [(sk, b) for sk, b in rows if b.get("disabled")]

    lines.append(f"\n  Active strategies ({len(active)}):")
    for sk, b in active[:15]:
        kelly = kelly_risk(state, sk)
        lines.append(f"    {sk:45s}  ema_r={b['ema_reward']:+.4f}  "
                     f"n={b['n']:3d}  kelly_risk={kelly*100:.2f}%")

    if disabled:
        lines.append(f"\n  Culled strategies ({len(disabled)}):")
        for sk, b in disabled[:5]:
            lines.append(f"    {sk:45s}  reason: {b.get('disable_reason','')}")

    # Best Q-learned params per regime
    lines.append("\n  Best SL/TP params learned (top strategy per regime):")
    for regime in REGIMES:
        qt = state["q_table"].get(regime, {})
        if not qt:
            continue
        best_sk = max(qt, key=lambda sk: max(qt[sk]) if qt[sk] else 0, default=None)
        if best_sk:
            sl, tp, _ = best_params(state, regime, best_sk)
            lines.append(f"    {regime:15s}  {best_sk:40s}  SL={sl}×ATR  TP={tp}×ATR")

    return "\n".join(lines)
