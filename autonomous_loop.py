"""
Autonomous ETH Paper Trader v3 — capital-maximising RL loop with adaptive intelligence.

Capital growth is THE objective. Every decision runs through RL + adaptive engine:
  - Position size      = fractional Kelly (from per-strategy win-rate + RR)
  - SL/TP params       = Q-learner tuned per (regime, strategy)
  - Strategy alloc     = EMA-UCB bandit (recency-weighted)
  - Portfolio heat     = hard cap at adaptive heat_budget (default 10%)
  - Regime detection   = multi-timeframe (1d/4h/1h) → 8-regime classification
  - Adaptive engine    = 1%-per-day improvement, auto-scales leverage/risk
  - Mistake tracker    = classifies every loser, feeds corrections back
  - Strategy culling   = auto-disable chronically losing strategies
  - Compounding        = risk$ scales with equity (never resets to flat $)

Runs for 7 days, restarts via watchdog.py on crash.
"""
import ccxt
import pandas as pd
import numpy as np
import json
import os
import time
import math
import traceback
from datetime import datetime, timezone, timedelta

from indicators   import add_indicators
from strategies   import STRATEGIES
from exchange_sim import ExchangeSim
from rl_optimizer import (
    load_rl, save_rl, detect_regime, choose_action, update_q,
    update_bandit, kelly_risk, get_risk_multiplier,
    current_heat, heat_budget, strategy_fits_regime,
    rl_summary, best_params, ACTIONS,
)
from strategy_expander import (
    run_discovery, load_active_strategies, build_candidates,
)

# ── New adaptive modules ───────────────────────────────────────────────────────
from regime_detector import (
    detect_regime_full, detect_regime_fast,
    get_strategy_fit, RegimeResult,
)
from mistake_tracker import (
    record_trade as mt_record_trade,
    get_adjustments as mt_get_adjustments,
    daily_mistake_report,
    get_regime_accuracy,
    get_worst_patterns,
    get_consecutive_losses,
)
from adaptive_engine import AdaptiveEngine

# ─────────────────────────────────────────────────────────────────────────────
SYMBOL        = "ETH/USDT"
CAPITAL       = 10_000.0
FEE_PCT       = 0.001
MAX_HOLD_BARS = 50
CANDLES       = 400
STATE_FILE    = "paper_trades.json"
LOG_FILE      = "paper_trade_log.txt"
DASHBOARD     = "/tmp/eth-dashboard"
RUN_DAYS      = 7
EXPAND_EVERY  = 4  * 3600
REPORT_EVERY  = 24 * 3600
CYCLE_SECS    = 60

REFRESH = {"1d": 3600, "4h": 900, "1h": 300, "15m": 120, "5m": 60}

# Timeframes used for multi-tf regime detection
REGIME_TFS = ["1d", "4h", "1h"]

BASELINE = [
    ("1d",  "obv_divergence",  STRATEGIES["obv_divergence"]),
    ("1d",  "engulf_reversal", STRATEGIES["engulf_reversal"]),
    ("1d",  "stoch_rsi_cross", STRATEGIES["stoch_rsi_cross"]),
    ("1d",  "macd_momentum",   STRATEGIES["macd_momentum"]),
    ("1d",  "bb_breakout",     STRATEGIES["bb_breakout"]),
    ("1d",  "supertrend",      STRATEGIES["supertrend"]),
    ("1d",  "ema_trend",       STRATEGIES["ema_trend"]),
    ("1d",  "volume_breakout", STRATEGIES["volume_breakout"]),
    ("4h",  "macd_momentum",   STRATEGIES["macd_momentum"]),
    ("4h",  "volume_breakout", STRATEGIES["volume_breakout"]),
    ("4h",  "stoch_rsi_cross", STRATEGIES["stoch_rsi_cross"]),
    ("4h",  "bb_breakout",     STRATEGIES["bb_breakout"]),
    ("4h",  "supertrend",      STRATEGIES["supertrend"]),
    ("4h",  "ema_trend",       STRATEGIES["ema_trend"]),
    ("1h",  "supertrend",      STRATEGIES["supertrend"]),
    ("1h",  "macd_momentum",   STRATEGIES["macd_momentum"]),
    ("1h",  "stoch_rsi_cross", STRATEGIES["stoch_rsi_cross"]),
    ("1h",  "ema_trend",       STRATEGIES["ema_trend"]),
    ("15m", "supertrend",      STRATEGIES["supertrend"]),
    ("15m", "stoch_rsi_cross", STRATEGIES["stoch_rsi_cross"]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now():   return datetime.now(timezone.utc).isoformat()
def _nowdt(): return datetime.now(timezone.utc)


def _log(msg):
    line = f"[{_now()}] {msg}"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    # Also print to stdout only if it's a terminal (not redirected to log file)
    import sys
    if sys.stdout.isatty():
        print(line, flush=True)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "equity":     CAPITAL,
        "positions":  {},
        "closed":     [],
        "started_at": _now(),
        "cycles":     0,
        "regime":     {},  # serialised RegimeResult
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    _push_dashboard(state)


def _push_dashboard(state):
    os.makedirs(DASHBOARD, exist_ok=True)
    log_tail = ""
    try:
        with open(LOG_FILE) as f:
            log_tail = "".join(f.readlines()[-200:])
    except Exception:
        pass

    # Enrich regime dict with strategy_fit for dashboard
    reg = state.get("regime", {})
    if reg and "regime" in reg:
        try:
            from regime_detector import get_strategy_fit, RegimeResult
            rr = RegimeResult(**{k: v for k, v in reg.items()
                                  if k in RegimeResult.__dataclass_fields__})
            reg["strategy_fit"] = get_strategy_fit(rr)
        except Exception:
            pass

    # Enrich with adaptive state for dashboard
    try:
        adp_file = "adaptive_state.json"
        if os.path.exists(adp_file):
            with open(adp_file) as f:
                state["adaptive_state"] = json.load(f)
    except Exception:
        pass

    js = (f"window.__ETH_STATE__ = {json.dumps(state, default=str)};\n"
          f"window.__ETH_LOG__   = {json.dumps(log_tail)};\n")
    with open(os.path.join(DASHBOARD, "data.js"), "w") as f:
        f.write(js)


_TF_PARQUET = {
    "1d":  "data/ETH_USDT_1d_full.parquet",
    "4h":  "data/ETH_USDT_4h_full.parquet",
    "1h":  "data/ETH_USDT_1h_full.parquet",
    "15m": "data/ETH_USDT_15m_full.parquet",
    "1w":  "data/ETH_USDT_1w_full.parquet",
}

def _refresh_parquet(tf):
    """Re-fetch latest 600 bars from Binance into the local parquet file.
    Runs via subprocess so the background loop isn't blocked by DNS issues."""
    import subprocess
    script = f"""
import ccxt, pandas as pd, warnings
warnings.filterwarnings('ignore')
ex = ccxt.binance({{'enableRateLimit': True}})
ex.load_markets()
raw = ex.fetch_ohlcv('ETH/USDT', '{tf}', limit=600)
df  = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df.set_index('timestamp', inplace=True)
df.to_parquet('{_TF_PARQUET[tf]}')
print(df['close'].iloc[-1])
"""
    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def fetch_df(exchange, tf):
    """Load OHLCV from local parquet (always works). Background refresh happens
    separately via _refresh_parquet."""
    parquet = _TF_PARQUET.get(tf)
    if parquet and os.path.exists(parquet):
        df = pd.read_parquet(parquet).tail(CANDLES)
        df = add_indicators(df.dropna()).dropna()
        if len(df) >= 50:
            return df
    raise ValueError(f"No data for {tf} — run data refresh first")


# ── Per-strategy performance rolling stats ─────────────────────────────────────

def _rolling_sharpe(state_rl, sk, window=20):
    hist = state_rl["trade_history"].get(sk, [])
    if len(hist) < 5:
        return None
    r = np.array([t["pnl_pct"] for t in hist[-window:]])
    return float(r.mean() / (r.std() + 1e-9) * math.sqrt(252))


# ── Core: process one strategy on one bar ─────────────────────────────────────

def process(state, rl, key, tf, sname, df, px,
            regime_result: RegimeResult, adaptive: AdaptiveEngine):
    pos_map = state["positions"]
    equity  = state["equity"]

    strat_fn = _fn_registry.get(key)
    if strat_fn is None:
        return

    try:
        signal = strat_fn(df)
    except Exception:
        return

    last_sig = int(signal.iloc[-1])
    atr      = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else px * 0.02
    regime   = detect_regime(df)   # RL regime (legacy string)
    avg_vol  = float(df["volume"].tail(20).mean()) if "volume" in df.columns else 0.0

    # Adaptive params for this strategy
    adp_params = adaptive.get_strategy_params(key, regime_result)

    # Get (or create) this strategy's ExchangeSim instance
    sim = _get_sim(key, tf, equity)

    # ── Manage open position via ExchangeSim ──────────────────────────────
    if key in pos_map:
        pos  = pos_map[key]

        # Restore sim position if loop restarted without ExchangeSim state
        if sim.position is None:
            _restore_sim_position(sim, pos, px)

        pos["bars_held"] = pos.get("bars_held", 0) + 1
        bars = pos["bars_held"]

        high_bar = float(df["high"].iloc[-1])
        low_bar  = float(df["low"].iloc[-1])

        result = sim.update(
            bar_idx    = bars,
            high       = high_bar,
            low        = low_bar,
            close      = px,
            signal     = last_sig,
            tf         = tf,
            era_bias   = regime,
            avg_volume = avg_vol,
        )

        # Update live snapshot
        snap = sim.position_snapshot(px, tf)
        if snap:
            pos.update({
                "leverage":         snap["leverage"],
                "liq_price":        snap["liq_price"],
                "bankruptcy_price": snap["bankruptcy_price"],
                "margin_ratio_pct": snap["margin_ratio_pct"],
                "dist_to_liq_pct":  snap["dist_to_liq_pct"],
                "unrealized_pnl":   snap["unrealized_pnl"],
                "unrealized_pct":   snap["unrealized_pct"],
                "initial_margin":   snap["initial_margin"],
                "bars_held":        bars,
            })

        if result is not None:
            # Trade closed
            direction = result.direction
            entry_px  = result.entry_px
            exit_px   = result.exit_px
            net_pnl   = result.net_pnl
            reason    = result.exit_reason

            pnl_pct     = result.unlev_return_pct
            log_ret_net = direction * math.log(exit_px / entry_px) if entry_px > 0 else 0
            log_ret_net -= result.total_cost / max(result.notional_entry, 1)

            state["equity"] = round(sim.equity, 4)

            trade = {
                "key":               key,
                "tf":                tf,
                "strategy":          sname,
                "direction":         direction,
                "entry_px":          entry_px,
                "exit_px":           round(exit_px, 4),
                "qty":               result.qty,
                "leverage":          result.leverage,
                "notional":          result.notional_entry,
                "initial_margin":    result.initial_margin,
                "liq_price":         result.liq_price,
                "bankruptcy_price":  result.bankruptcy_price,
                "gross_pnl":         round(result.gross_pnl, 4),
                "entry_fee":         result.entry_fee,
                "exit_fee":          result.exit_fee,
                "funding_paid":      result.funding_paid,
                "total_costs_usd":   result.total_cost,
                "pnl_usd":           round(net_pnl, 4),
                "pnl_pct":           round(pnl_pct, 3),
                "margin_return_pct": result.margin_return_pct,
                "log_ret":           round(log_ret_net, 6),
                "bars_held":         result.bars_held,
                "exit_reason":       reason,
                "was_liquidated":    result.was_liquidated,
                "final_margin_ratio":result.final_margin_ratio,
                "regime":            pos.get("regime", "unknown"),
                "regime_full":       regime_result.regime,
                "regime_conf":       regime_result.confidence,
                "action_idx":        pos.get("action_idx", 0),
                "opened_at":         pos["opened_at"],
                "closed_at":         _now(),
                "sl_px":             pos.get("sl_px", 0),
                "tp_px":             pos.get("tp_px", 0),
            }
            # Classify mistake and embed in trade record
            try:
                from mistake_tracker import classify_trade as _ct
                mc = _ct(trade, regime_result.regime, regime_result.confidence)
                trade["mistake_type"] = mc["mistake_type"]
                trade["mistake_lesson"] = mc["lesson"]
                trade["mistake_severity"] = mc["severity"]
            except Exception:
                trade["mistake_type"] = None

            state["closed"].append(trade)
            state.setdefault("_new_closed_this_cycle", [])
            state["_new_closed_this_cycle"].append(trade)
            state.setdefault("total_fees_paid", 0)
            state["total_fees_paid"] = round(
                state["total_fees_paid"] + result.total_cost, 4)
            state.setdefault("n_liquidations", 0)
            if result.was_liquidated:
                state["n_liquidations"] += 1
            del pos_map[key]

            # ── Mistake tracker: classify and record ──────────────────────
            try:
                mt_record_trade(trade, regime_result.regime, regime_result.confidence)
            except Exception as e:
                _log(f"[MistakeTracker] record error: {e}")

            # ── RL updates ────────────────────────────────────────────────
            update_bandit(rl, key, log_ret_net)
            update_q(rl, pos.get("regime", "ranging"), key,
                     pos.get("action_idx", 0), log_ret_net, regime)

            emoji = "💥" if result.was_liquidated else ("✅" if net_pnl > 0 else "❌")
            liq_tag = " ⚡LIQUIDATED" if result.was_liquidated else ""
            _log(f"{emoji} CLOSED  {key:38s} "
                 f"{'LONG' if direction==1 else 'SHORT'}  "
                 f"${entry_px:.2f}→${exit_px:.2f}  "
                 f"net={net_pnl:+.2f}$ ({pnl_pct:+.2f}%)  "
                 f"margin_ret={result.margin_return_pct:+.1f}%  "
                 f"lev={result.leverage:.1f}x  "
                 f"fees=${result.total_cost:.3f}  fund=${result.funding_paid:.3f}  "
                 f"{reason}{liq_tag}  equity=${state['equity']:,.2f}")

    # ── Open new position ──────────────────────────────────────────────────
    if key not in pos_map and last_sig != 0:
        prev_sig = int(signal.iloc[-2]) if len(signal) > 1 else 0
        if last_sig == prev_sig:
            return   # no fresh signal

        # Adaptive engine gate
        if not adp_params.get("active", True):
            return  # on cooldown or poor regime fit

        # Regime gate (RL legacy)
        if not strategy_fits_regime(sname, regime):
            return

        # Consecutive losses gate (per mistake tracker)
        cons_losses = get_consecutive_losses(key)
        if cons_losses >= 5:
            return   # very hot losing streak — stop

        # Bandit gate
        mult = get_risk_multiplier(rl, key)
        if mult == 0.0:
            return

        # Portfolio heat gate
        gp = adaptive._state.get("global_params", {})
        max_heat = gp.get("heat_budget", 0.10)
        budget = heat_budget(pos_map, state["equity"])
        if budget < 0.002:
            return

        # Check max open positions
        max_pos = gp.get("max_positions", 6)
        if len(pos_map) >= max_pos:
            return

        # Kelly + adaptive risk multiplier
        k_risk    = kelly_risk(rl, key)
        mt_adj    = mt_get_adjustments(key, lookback=50)
        risk_mult = min(adp_params.get("risk_mult", 1.0),
                        mt_adj.get("risk_mult", 1.0))
        lev_mult  = min(adp_params.get("leverage_mult", 1.0),
                        mt_adj.get("leverage_mult", 1.0))
        risk_pct  = min(k_risk * mult * risk_mult, budget, 0.025)
        risk_pct  = max(risk_pct, 0.002)
        sim.risk_pct = risk_pct

        # Check regime-based skip
        skip_regimes = mt_adj.get("skip_regimes", [])
        if regime_result.regime in skip_regimes:
            return

        # Q-learner: choose SL/TP
        sl_mult, tp_mult, action_idx = choose_action(rl, regime, key)
        if atr <= 0:
            return

        # Vol-adjusted SL/TP via adaptive engine
        sl_adj = adp_params.get("sl_adjust", 1.0)
        tp_adj = adp_params.get("tp_adjust", 1.0)
        sl_px  = round(px - last_sig * sl_mult * atr * sl_adj, 4)
        tp_px  = round(px + last_sig * tp_mult * atr * tp_adj, 4)

        # Override leverage via adaptive engine
        base_lev = LEVERAGE_TF.get(tf, LEVERAGE)
        adj_lev  = base_lev * lev_mult
        sim_orig_lev = sim.leverage
        sim.leverage = min(adj_lev, 20.0)

        open_info = sim.open(
            direction  = last_sig,
            entry_px   = px,
            sl_px      = sl_px,
            tp_px      = tp_px,
            bar_idx    = 0,
            avg_volume = avg_vol,
            era_bias   = regime,
        )
        sim.leverage = sim_orig_lev  # restore

        if "error" in open_info:
            return

        pos_map[key] = {
            "direction":        last_sig,
            "entry_px":         round(px, 4),
            "sl_px":            sl_px,
            "tp_px":            tp_px,
            "qty":              open_info.get("qty", 0),
            "leverage":         open_info.get("leverage", f"{LEVERAGE_TF.get(tf,3)}x"),
            "notional":         open_info.get("notional", 0),
            "initial_margin":   open_info.get("initial_margin", 0),
            "liq_price":        open_info.get("liq_price", 0),
            "bankruptcy_price": open_info.get("bankruptcy_price", 0),
            "dist_to_liq_pct":  open_info.get("dist_to_liq_pct", 0),
            "maint_margin_rate":open_info.get("maint_margin_rate", "N/A"),
            "maint_margin":     open_info.get("maint_margin", 0),
            "margin_ratio_pct": round(open_info.get("maint_margin",0) /
                                      max(open_info.get("initial_margin",1),1) * 100, 2),
            "unrealized_pnl":   0.0,
            "unrealized_pct":   0.0,
            "bars_held":        0,
            "opened_at":        _now(),
            "equity_at_open":   round(state["equity"], 4),
            "regime":           regime,
            "regime_full":      regime_result.regime,
            "regime_conf":      regime_result.confidence,
            "action_idx":       action_idx,
            "sl_mult":          sl_mult,
            "tp_mult":          tp_mult,
            "risk_pct":         round(risk_pct * 100, 3),
            "lev_adj":          round(lev_mult, 3),
        }

        d = "LONG" if last_sig == 1 else "SHORT"
        _log(f"📈 OPENED  {key:38s} {d}  "
             f"${px:.2f}  SL=${sl_px:.2f}({sl_mult:.1f}×)  "
             f"TP=${tp_px:.2f}({tp_mult:.1f}×)  "
             f"qty={open_info.get('qty',0):.4f}  "
             f"lev={open_info.get('leverage','?')}(adj={lev_mult:.2f}x)  "
             f"notional=${open_info.get('notional',0):,.0f}  "
             f"margin=${open_info.get('initial_margin',0):,.2f}  "
             f"liq=${open_info.get('liq_price',0):.2f} "
             f"(dist={open_info.get('dist_to_liq_pct',0):.1f}%)  "
             f"entry_cost=${open_info.get('total_entry_cost',0):.3f}  "
             f"regime={regime_result.regime}({regime_result.confidence:.0f}%)  "
             f"equity=${state['equity']:,.2f}")


# ── Fn registry (populated at runtime) ───────────────────────────────────────
_fn_registry = {}

# ── ExchangeSim instances — one per strategy slot ────────────────────────────
_sims: dict = {}

LEVERAGE    = 3.0
LEVERAGE_TF = {"1w": 2.0, "1d": 3.0, "4h": 5.0, "1h": 5.0, "15m": 3.0, "5m": 2.0}


def _restore_sim_position(sim: ExchangeSim, pos: dict, px: float):
    """If sim has no open position but state says there is one, reconstruct it."""
    if sim.position is not None:
        return
    if not pos or not pos.get("entry_px"):
        return
    from exchange_sim import Position, get_bracket
    entry = float(pos["entry_px"])
    direction = int(pos.get("direction", 1))
    liq = float(pos.get("liq_price") or 0)
    bank = float(pos.get("bankruptcy_price") or 0)
    notional = float(pos.get("notional") or 0)
    lev = float(str(pos.get("leverage","3x")).replace("x","")) if pos.get("leverage") else 3.0
    margin = float(pos.get("initial_margin") or (notional / max(lev, 1)))
    mmr = float(pos.get("maint_margin_rate") or 0.004)
    if notional <= 0:
        notional = margin * lev
    qty = notional / entry if entry > 0 else 0
    if qty <= 0 or entry <= 0:
        return
    if liq <= 0:
        liq = sim._liq_price(direction, entry, lev, mmr)
    if bank <= 0:
        bank = sim._bankruptcy_price(direction, entry, lev)
    from exchange_sim import Position as _Pos
    sim.position = _Pos(
        direction=direction, entry_px=entry, qty=qty, leverage=lev,
        notional=notional, initial_margin=margin, maint_margin_rate=mmr,
        maint_margin=notional * mmr, liq_price=liq, bankruptcy_price=bank,
        sl_px=float(pos.get("sl_px", 0)), tp_px=float(pos.get("tp_px", 0)),
        entry_bar=0, era_bias=pos.get("regime", "mixed"),
    )
    # Adjust sim equity to account for margin already committed
    sim.equity = max(float(pos.get("equity_at_open", sim.equity)) - margin, 0)


def _get_sim(key: str, tf: str, equity: float) -> ExchangeSim:
    if key not in _sims:
        lev = LEVERAGE_TF.get(tf, LEVERAGE)
        _sims[key] = ExchangeSim(
            capital      = equity,
            leverage     = lev,
            risk_pct     = 0.01,
            max_leverage = 20.0,
            margin_mode  = "isolated",
        )
    return _sims[key]


def _build_registry(baseline, discovered):
    global _fn_registry
    _fn_registry = {}
    for tf, sn, fn in baseline:
        _fn_registry[f"{tf}|{sn}"] = fn
    for tf, sn, fn in discovered:
        _fn_registry[f"{tf}|{sn}"] = fn
    active = set(_fn_registry.keys())
    for k in list(_sims.keys()):
        if k not in active:
            del _sims[k]


# ── Multi-timeframe regime detection ─────────────────────────────────────────

def _detect_regime_mtf(tf_cache: dict) -> RegimeResult:
    """Run multi-timeframe regime detection over cached candle data."""
    tf_dfs = {tf: tf_cache[tf] for tf in REGIME_TFS if tf in tf_cache}
    if not tf_dfs:
        # Fallback to single tf
        if tf_cache:
            tf0 = next(iter(tf_cache))
            return detect_regime_fast(tf_cache[tf0], tf0)
        return RegimeResult()
    return detect_regime_full(tf_dfs)


# ── Daily report ──────────────────────────────────────────────────────────────

def daily_report(state, rl, regime_result: RegimeResult,
                 adaptive: AdaptiveEngine):
    closed = state["closed"]
    eq     = state["equity"]
    wins   = [t for t in closed if t["pnl_usd"] > 0]
    loses  = [t for t in closed if t["pnl_usd"] <= 0]
    pnl    = sum(t["pnl_usd"] for t in closed)
    wr     = len(wins) / len(closed) * 100 if closed else 0
    gw     = sum(t["pnl_usd"] for t in wins)
    gl     = abs(sum(t["pnl_usd"] for t in loses))
    pf     = gw / gl if gl > 0 else float("inf")
    ret    = (eq / CAPITAL - 1) * 100

    by_strat: dict = {}
    for t in closed:
        by_strat.setdefault(t["key"], {"pnl": 0, "n": 0, "wins": 0})
        by_strat[t["key"]]["pnl"]  += t["pnl_usd"]
        by_strat[t["key"]]["n"]    += 1
        by_strat[t["key"]]["wins"] += 1 if t["pnl_usd"] > 0 else 0

    _log("=" * 72)
    _log(f"  📊 DAILY REPORT  |  {_now()[:10]}")
    _log(f"  Equity      : ${eq:>10,.2f}  (started ${CAPITAL:,.2f})")
    _log(f"  Return      : {ret:>+8.2f}%")
    total_fees = state.get("total_fees_paid", 0)
    gross_pnl  = sum(t.get("gross_pnl", t["pnl_usd"]) for t in closed)
    _log(f"  Gross PnL   : ${gross_pnl:>+10.2f}")
    _log(f"  Total Fees  : ${-total_fees:>+10.2f}  "
         f"(taker+maker+spread+slip+funding+borrow)")
    _log(f"  Net PnL     : ${pnl:>+10.2f}")
    n_liqs  = state.get("n_liquidations", 0)
    avg_lev = (sum(t.get("leverage", 3.0) for t in closed) / len(closed)) if closed else 0
    if closed:
        _log(f"  Trades      : {len(closed)} total  |  {len(wins)}W / {len(loses)}L  "
             f"|  WR={wr:.1f}%  |  PF={pf:.2f}  "
             f"|  avg_fee/trade=${total_fees/len(closed):.3f}")
    else:
        _log("  Trades      : 0")
    _log(f"  Leverage    : {avg_lev:.1f}x avg  |  Liquidations: {n_liqs}")
    _log(f"  Open now    : {len(state['positions'])}")

    # ── Regime section ────────────────────────────────────────────────────
    _log(f"  ─ Regime ─")
    _log(f"  Current     : {regime_result.regime} ({regime_result.confidence:.0f}%)  "
         f"| vol={regime_result.vol_regime}  ADX={regime_result.adx:.1f}  "
         f"htf={regime_result.htf_bias}")
    regime_acc = get_regime_accuracy(lookback=100)
    for reg, stat in sorted(regime_acc.items(), key=lambda x: -x[1]["n"]):
        _log(f"    {reg:20s}  WR={stat['win_rate']:.1f}%  n={stat['n']}")

    # ── Mistake section ───────────────────────────────────────────────────
    _log("  ─ Mistakes ─")
    _log(daily_mistake_report())
    worst = get_worst_patterns()
    if worst:
        _log("  Worst patterns:")
        for p in worst:
            _log(f"    {p['mistake_type']:30s}  ×{p['count']}  "
                 f"avg_sev={p['avg_severity']:.1f}  "
                 f"total_loss=${p['total_loss']:.2f}")

    # ── Adaptive engine section ───────────────────────────────────────────
    _log("  ─ Adaptive Engine ─")
    _log(adaptive.report())

    _log("  ─ Top strategies by cumulative P&L ─")
    rows = sorted(by_strat.items(), key=lambda x: -x[1]["pnl"])
    for k, v in rows[:12]:
        wr_k = v["wins"] / v["n"] * 100 if v["n"] else 0
        sh   = _rolling_sharpe(rl, k)
        sh_s = f"{sh:.2f}" if sh is not None else "N/A"
        _log(f"    {k:42s}  P&L={v['pnl']:>+8.2f}$  "
             f"n={v['n']:3d}  WR={wr_k:.0f}%  Sharpe={sh_s}")
    _log(rl_summary(rl))
    _log("=" * 72)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    # ── Single-instance lock ──────────────────────────────────────────────────
    import atexit
    _PID_FILE = "loop.pid"
    if os.path.exists(_PID_FILE):
        try:
            old_pid = int(open(_PID_FILE).read().strip())
            # Check if that PID is actually still running
            os.kill(old_pid, 0)
            print(f"[GUARD] Another instance already running (PID {old_pid}). Exiting.")
            return
        except (ProcessLookupError, ValueError):
            pass   # stale PID file — proceed
    with open(_PID_FILE, "w") as _pf:
        _pf.write(str(os.getpid()))
    atexit.register(lambda: os.remove(_PID_FILE) if os.path.exists(_PID_FILE) else None)

    _log("=" * 72)
    _log("  🚀 AUTONOMOUS ETH PAPER TRADER v3  —  CAPITAL MAXIMISATION MODE")
    _log(f"  Run window  : {_nowdt().date()} → {(_nowdt()+timedelta(days=RUN_DAYS)).date()}")
    _log(f"  Objective   : Maximise equity growth via RL + adaptive intelligence")
    _log("=" * 72)

    exchange    = None   # unused — all fetches go through local parquet + subprocess
    state       = load_state()
    end_time    = _nowdt() + timedelta(days=RUN_DAYS)

    all_keys    = [f"{tf}|{sn}" for tf, sn, _ in BASELINE]
    rl          = load_rl(all_keys)
    adaptive    = AdaptiveEngine()

    discovered  = []
    _build_registry(BASELINE, discovered)

    # Parquet refresh timing (seconds) — short TFs refresh more often
    _DATA_REFRESH = {"1d": 3600, "4h": 900, "1h": 300, "15m": 120, "1w": 86400}
    _last_data_refresh: dict = {}

    tf_cache:   dict = {}
    last_fetch: dict = {}
    last_expand  = 0
    last_report  = time.time()
    last_improve = 0.0
    cycle        = state.get("cycles", 0)

    # Initial regime from cached state or default
    last_regime  = RegimeResult()
    if state.get("regime") and isinstance(state["regime"], dict):
        try:
            r = state["regime"]
            last_regime = RegimeResult(**{k: v for k, v in r.items()
                                          if k in RegimeResult.__dataclass_fields__})
        except Exception:
            last_regime = RegimeResult()

    try:
        while _nowdt() < end_time:
            cycle += 1
            state["cycles"] = cycle
            ts = time.time()

            # ── Refresh candles from local parquet ───────────────────────────
            tfs = set(tf for tf, _, _ in BASELINE)
            if discovered:
                tfs |= set(tf for tf, _, _ in discovered)
            tfs |= set(REGIME_TFS)

            for tf in sorted(tfs):
                # Refresh in-memory cache from parquet
                if ts - last_fetch.get(tf, 0) >= REFRESH.get(tf, 300):
                    try:
                        tf_cache[tf]   = fetch_df(exchange, tf)
                        last_fetch[tf] = ts
                    except Exception as e:
                        _log(f"  [DataCache] {tf}: {e}")

                # Background parquet refresh from Binance (subprocess)
                refresh_secs = _DATA_REFRESH.get(tf, 900)
                if ts - _last_data_refresh.get(tf, 0) >= refresh_secs:
                    try:
                        new_px = _refresh_parquet(tf)
                        _last_data_refresh[tf] = ts
                        if new_px:
                            # Immediately reload from updated parquet
                            tf_cache[tf]   = fetch_df(exchange, tf)
                            last_fetch[tf] = ts
                    except Exception:
                        pass

            # ── Live price ───────────────────────────────────────────────────
            # Primary: last close from the fastest cached TF (15m or 1h)
            _px_tf = next((t for t in ["15m","1h","4h","1d"] if t in tf_cache), None)
            if _px_tf:
                px = float(tf_cache[_px_tf]["close"].iloc[-1])
                state["last_price"] = round(px, 2)
            elif tf_cache:
                px = float(next(iter(tf_cache.values()))["close"].iloc[-1])
                state["last_price"] = round(px, 2)
            else:
                # No data at all yet — wait
                time.sleep(CYCLE_SECS)
                continue

            # ── Multi-timeframe regime detection ─────────────────────────────
            try:
                regime_result = _detect_regime_mtf(tf_cache)
                last_regime   = regime_result
                state["regime"] = regime_result.to_dict()
            except Exception as e:
                _log(f"  [RegimeDetect] error: {e}")
                regime_result = last_regime

            # ── Process all strategies ───────────────────────────────────────
            watch = list(BASELINE) + [
                (tf, sn, fn) for tf, sn, fn in discovered
                if f"{tf}|{sn}" not in {f"{t}|{s}" for t, s, _ in BASELINE}
            ]
            for tf, sn, fn in watch:
                if tf not in tf_cache:
                    continue
                try:
                    process(state, rl, f"{tf}|{sn}", tf, sn,
                            tf_cache[tf], px, regime_result, adaptive)
                except Exception:
                    pass

            # ── Adaptive engine update (after each cycle, for new closed trades) ─
            newly_closed = state.get("_new_closed_this_cycle", [])
            if newly_closed:
                try:
                    adaptive.update(newly_closed, regime_result)
                except Exception as e:
                    _log(f"  [AdaptiveEngine] update error: {e}")
            state["_new_closed_this_cycle"] = []

            # ── Persist ──────────────────────────────────────────────────────
            state["current_px"] = float(px)
            state["last_updated"] = _now()
            save_state(state)
            save_rl(rl)

            # ── Strategy expansion (every 4h) ────────────────────────────────
            if ts - last_expand >= EXPAND_EVERY:
                _log("[Expander] Discovery run starting...")
                try:
                    run_discovery(log_fn=_log)
                    discovered = load_active_strategies()
                    new_keys   = [f"{tf}|{sn}" for tf, sn, _ in discovered]
                    rl         = load_rl(list(set(all_keys + new_keys)))
                    _build_registry(BASELINE, discovered)
                    _log(f"[Expander] Watch list: {len(BASELINE)} baseline "
                         f"+ {len(discovered)} discovered = "
                         f"{len(BASELINE)+len(discovered)} total")
                except Exception as e:
                    _log(f"[Expander] ERROR: {e}")
                last_expand = ts

            # ── Daily improvement cycle (every 24h) ──────────────────────────
            if ts - last_improve >= REPORT_EVERY:
                try:
                    adaptive.daily_improvement_cycle(
                        tf_cache,
                        _fn_registry,
                        regime_result,
                    )
                    adaptive.update(state["closed"], regime_result)
                except Exception as e:
                    _log(f"  [AdaptiveEngine] improvement_cycle error: {e}")
                last_improve = ts

            # ── Daily report ─────────────────────────────────────────────────
            if ts - last_report >= REPORT_EVERY:
                daily_report(state, rl, regime_result, adaptive)
                last_report = ts

            # ── Cycle log every 10 cycles ────────────────────────────────────
            if cycle % 10 == 0:
                eq     = state["equity"]
                unr    = sum(pos.get("unrealized_pnl", 0)
                             for pos in state["positions"].values())
                heat   = current_heat(state["positions"], eq)
                days_left = max(0, (end_time - _nowdt()).days)
                n_liqs = state.get("n_liquidations", 0)
                imp    = adaptive.improvement_score()
                gp     = adaptive._state.get("global_params", {})
                lev_adj = gp.get("current_leverage", LEVERAGE)
                _log(f"[#{cycle:05d}] eq=${eq:,.2f}  unreal~${unr:+.2f}  "
                     f"heat={heat*100:.1f}%  open={len(state['positions'])}  "
                     f"closed={len(state['closed'])}  liqs={n_liqs}  "
                     f"px=${px:.2f}  days_left={days_left}  "
                     f"regime={regime_result.regime}({regime_result.confidence:.0f}%)  "
                     f"lev_adj={lev_adj:.1f}x  "
                     f"heat_budget={gp.get('heat_budget',0.10)*100:.1f}%  "
                     f"imp={imp:+.4f}")

                # Regime log line
                _log(f"[REGIME] {regime_result.regime} ({regime_result.confidence:.0f}%) "
                     f"| vol={regime_result.vol_regime} "
                     f"| ADX={regime_result.adx:.1f} "
                     f"| htf={regime_result.htf_bias} "
                     f"| struct={regime_result.market_structure} "
                     f"| RSI={regime_result.rsi:.1f}")

            time.sleep(CYCLE_SECS)

    except KeyboardInterrupt:
        _log("=== Stopped by user ===")
    except Exception as e:
        _log(f"=== FATAL: {e}\n{traceback.format_exc()}")
    finally:
        save_state(state)
        save_rl(rl)
        adaptive.save()
        daily_report(state, rl, last_regime, adaptive)
        _log("=== Loop ended ===")


if __name__ == "__main__":
    run()
