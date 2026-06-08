"""
BRUTAL BACKTEST ENGINE — 35+ metrics per strategy
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from backtest_engine import backtest

BARS_PER_YEAR = {"1w": 52, "1d": 365, "4h": 365*6, "1h": 365*24, "15m": 365*96}

def _drawdown_series(equity):
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak

def _max_dd_duration(equity):
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    max_dur = curr = 0
    for u in underwater:
        curr = curr + 1 if u else 0
        max_dur = max(max_dur, curr)
    return int(max_dur)

def _monthly_wr(equity_series):
    if len(equity_series) < 30: return None
    m = equity_series.resample("ME").last()
    pct = m.pct_change().dropna()
    if len(pct) < 3: return None
    return round((pct > 0).mean() * 100, 1)

def _sortino(bar_rets, bpy):
    if len(bar_rets) < 5: return None
    downside = bar_rets[bar_rets < 0]
    if len(downside) < 2: return None
    dsd = np.std(downside, ddof=1) * np.sqrt(bpy)
    if dsd == 0: return None
    return round(np.mean(bar_rets) * bpy / dsd, 3)

def _ulcer(equity):
    if len(equity) < 5: return None
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return round(float(np.sqrt(np.mean(dd**2))), 4)

def _tail_ratio(trade_rets):
    if len(trade_rets) < 10: return None
    p95 = np.percentile(trade_rets, 95)
    p05 = np.percentile(trade_rets, 5)
    if p05 >= 0: return None
    return round(abs(p95 / p05), 3)

def _tstat(trade_rets):
    if len(trade_rets) < 5: return None, None
    t, p = sp_stats.ttest_1samp(trade_rets, 0)
    return round(float(t), 3), round(float(p), 4)

def _monte_carlo(trade_rets, n_sim=2000):
    if len(trade_rets) < 5: return None
    finals = []
    for _ in range(n_sim):
        perm = np.random.permutation(trade_rets)
        equity = 10000 * np.prod(1 + perm / 100)
        finals.append((equity / 10000 - 1) * 100)
    return round(float(np.percentile(finals, 5)), 2)

def _max_consec(mask):
    max_c = curr = 0
    for v in mask:
        curr = curr + 1 if v else 0
        max_c = max(max_c, curr)
    return int(max_c)

def _fee_breakeven(trade_rets, avg_cost):
    if len(trade_rets) < 5 or not avg_cost or avg_cost == 0: return None
    for mult in np.arange(1.0, 20.5, 0.5):
        adj = [r - avg_cost * (mult - 1) for r in trade_rets]
        if np.mean(adj) <= 0:
            return round(float(mult), 1)
    return ">20x"

def _walk_forward(df, fn, tf, split=0.6):
    try:
        n = len(df)
        cut = int(n * split)
        if cut < 25 or n - cut < 15: return None
        df_is, df_oos = df.iloc[:cut], df.iloc[cut:]
        sig_is  = fn(df_is)
        sig_oos = fn(df_oos)
        r_is  = backtest(df_is,  sig_is,  tf=tf)["stats"]
        r_oos = backtest(df_oos, sig_oos, tf=tf)["stats"]
        sh_is  = r_is.get("sharpe")
        sh_oos = r_oos.get("sharpe")
        if sh_is in (None, "N/A") or sh_oos in (None, "N/A"): return None
        sh_is, sh_oos = float(sh_is), float(sh_oos)
        if sh_is == 0: return None
        return round(sh_oos / sh_is, 3)
    except Exception:
        return None


def brutal_stats(df, fn, tf="1d"):
    """
    fn: the strategy function (takes df, returns signal series).
    Runs the full backtest + computes 35+ metrics.
    Returns a flat dict.
    """
    bpy = BARS_PER_YEAR.get(tf, 365)

    try:
        sig    = fn(df)
        result = backtest(df, sig, tf=tf)
    except Exception as e:
        return {"n_trades": 0, "error": str(e)[:80]}

    s      = result["stats"]
    trades = result.get("trades")   # DataFrame
    equity = result.get("equity")   # DataFrame

    n = int(s.get("n_trades", 0))
    if n < 3:
        return {"n_trades": n, "error": "too_few_trades"}

    # ── Extract equity curve ──────────────────────────────────────────────
    if equity is not None and isinstance(equity, pd.DataFrame) and len(equity) > 0:
        eq_col = [c for c in equity.columns if "equity" in c.lower() or "value" in c.lower()]
        if eq_col:
            equity_arr = equity[eq_col[0]].values.astype(float)
        else:
            equity_arr = equity.iloc[:, 0].values.astype(float)
        equity_series = pd.Series(equity_arr, index=equity.index if hasattr(equity, "index") else range(len(equity_arr)))
    else:
        equity_arr    = np.array([10000.0])
        equity_series = pd.Series(equity_arr)

    bar_rets = np.diff(equity_arr) / equity_arr[:-1] if len(equity_arr) > 1 else np.array([0.0])

    # ── Extract trade returns ─────────────────────────────────────────────
    trade_rets  = []
    wins_mask   = []
    losses_mask = []
    hold_bars   = []
    gross_rets  = []

    if trades is not None and isinstance(trades, pd.DataFrame) and len(trades) > 0:
        for _, t in trades.iterrows():
            net_pct   = float(t.get("pnl_pct", 0) or 0)
            gross_pct = float(t.get("pnl_pct", 0) or 0)  # gross not separately stored
            cost_pct  = float(t.get("cost_pct", 0) or 0)
            trade_rets.append(net_pct)
            gross_rets.append(net_pct + cost_pct)
            wins_mask.append(net_pct > 0)
            losses_mask.append(net_pct < 0)
            hold_bars.append(int(t.get("bars_held", 1) or 1))

    trade_rets_arr = np.array(trade_rets) if trade_rets else np.array([0.0])

    # ── Base metrics from stats dict ─────────────────────────────────────
    def safe(v): return float(v) if v not in (None, "N/A") else None

    sharpe_f = safe(s.get("sharpe"))
    pf_f     = safe(s.get("profit_factor"))
    ret_f    = safe(s.get("total_return_pct")) or 0.0
    wr_f     = safe(s.get("win_rate_pct")) or 0.0
    dd_f     = safe(s.get("max_drawdown_pct")) or 0.0
    avg_cost = safe(s.get("avg_cost_pct")) or 0.0
    total_costs = safe(s.get("total_costs_usd")) or 0.0

    # ── Derived metrics ───────────────────────────────────────────────────
    years   = len(df) / bpy
    ann_ret = ((1 + ret_f/100) ** (1/max(years, 0.1)) - 1) * 100 if years > 0.1 else ret_f

    wins   = [r for r in trade_rets if r > 0]
    losses = [r for r in trade_rets if r < 0]
    avg_win  = round(float(np.mean(wins)), 3)   if wins   else 0.0
    avg_loss = round(float(np.mean(losses)), 3) if losses else 0.0
    payoff   = round(avg_win / abs(avg_loss), 3) if avg_loss != 0 else None

    sortino   = _sortino(bar_rets, bpy)
    calmar    = round(ann_ret / abs(dd_f), 3) if dd_f != 0 else None
    ulcer     = _ulcer(equity_arr)
    dd_dur    = _max_dd_duration(equity_arr)
    mon_wr    = _monthly_wr(equity_series)
    tail      = _tail_ratio(trade_rets_arr)
    t_stat, p_val = _tstat(trade_rets_arr)
    mc_5th    = _monte_carlo(trade_rets_arr)
    fee_be    = _fee_breakeven(trade_rets, avg_cost)
    recovery  = round(ret_f / abs(dd_f), 3) if dd_f != 0 else None

    max_cw = _max_consec(wins_mask)
    max_cl = _max_consec(losses_mask)

    in_market = sum(hold_bars)
    pct_in    = round(in_market / len(df) * 100, 1) if len(df) > 0 else None
    avg_hold  = round(float(np.mean(hold_bars)), 1) if hold_bars else None

    total_gross_pnl = sum(gross_rets) * 100  # approximate
    fee_drag = round(abs(total_costs) / max(abs(ret_f * 100), 1) * 100, 1) if ret_f != 0 else None

    best_trade  = round(float(max(trade_rets)), 2) if trade_rets else None
    worst_trade = round(float(min(trade_rets)), 2) if trade_rets else None

    # Walk-forward (uses original fn, not pre-computed signal)
    wf_eff = _walk_forward(df, fn, tf)

    return {
        # Basics
        "n_trades":          n,
        "n_long":            int(s.get("long_trades", 0) or 0),
        "n_short":           int(s.get("short_trades", 0) or 0),
        # Returns
        "total_return":      round(ret_f, 2),
        "ann_return":        round(ann_ret, 2),
        "win_rate":          round(wr_f, 1),
        "monthly_wr":        mon_wr,
        "best_trade":        best_trade,
        "worst_trade":       worst_trade,
        "avg_win":           avg_win,
        "avg_loss":          avg_loss,
        "payoff_ratio":      payoff,
        # Risk-adjusted
        "sharpe":            round(sharpe_f, 3) if sharpe_f is not None else None,
        "sortino":           sortino,
        "calmar":            calmar,
        "profit_factor":     round(pf_f, 3)    if pf_f    is not None else None,
        # Drawdown
        "max_dd":            round(dd_f, 2),
        "max_dd_duration":   dd_dur,
        "recovery_factor":   recovery,
        "ulcer_index":       ulcer,
        # Sequence
        "max_consec_wins":   max_cw,
        "max_consec_losses": max_cl,
        # Quality
        "tail_ratio":        tail,
        "t_stat":            t_stat,
        "p_value":           p_val,
        "significant":       bool(p_val is not None and p_val < 0.05),
        # Robustness
        "mc_5th_pct":        mc_5th,
        "fee_breakeven_x":   fee_be,
        "wf_efficiency":     wf_eff,
        # Activity
        "pct_time_in":       pct_in,
        "avg_hold_bars":     avg_hold,
        # Fees
        "avg_cost_pct":      round(avg_cost, 4),
        "fee_drag_pct":      fee_drag,
        "total_costs_usd":   round(total_costs, 2),
        # Exit mix
        "exit_reasons":      s.get("exit_reasons", {}),
    }
