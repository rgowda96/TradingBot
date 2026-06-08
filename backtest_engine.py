"""
Vectorised backtesting engine — Binance perpetual futures simulation.

Uses ExchangeSim for exact cost/liquidation/leverage modelling:
  ✓ Leverage (1x–20x, bracket-capped)
  ✓ Taker/Maker fees (0.05% / 0.02%)
  ✓ Spread + slippage (size-proportional)
  ✓ Funding every 8h
  ✓ Liquidation price (Binance isolated-margin formula)
  ✓ Auto-liquidation when mark price hits liq_price
"""
import numpy as np
import pandas as pd
from exchange_sim import ExchangeSim


def backtest(
    df:              pd.DataFrame,
    signal:          pd.Series,
    atr_sl_mult:     float = 1.5,
    atr_tp_mult:     float = 3.0,
    risk_pct:        float = 0.01,
    initial_capital: float = 10_000,
    max_hold_bars:   int   = 50,
    tf:              str   = "1d",
    leverage:        float = 3.0,
    era_bias:        str   = "mixed",
) -> dict:
    """
    Returns dict: {"equity": DataFrame, "trades": DataFrame, "stats": dict}.
    signal: +1 long, -1 short, 0 flat.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr_14"].values
    vols   = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    dates  = df.index

    sim    = ExchangeSim(
        capital=initial_capital,
        leverage=leverage,
        risk_pct=risk_pct,
        max_leverage=20.0,
        margin_mode="isolated",
    )

    equity_curve = []   # {time, equity}
    trades       = []

    in_trade     = False
    direction    = 0
    entry_bar    = 0

    for i in range(1, len(df)):
        px      = closes[i]
        avg_vol = float(np.mean(vols[max(0, i - 20):i])) if i > 0 else 0.0

        if in_trade:
            # Let ExchangeSim handle SL/TP/liq/funding/max_hold
            result = sim.update(
                bar_idx   = i,
                high      = highs[i],
                low       = lows[i],
                close     = px,
                signal    = int(signal.iloc[i - 1]),
                tf        = tf,
                era_bias  = era_bias,
                avg_volume= avg_vol,
            )

            if result is not None:
                # Trade was closed
                pnl_pct_unl = result.unlev_return_pct  # unlevered % of notional
                trades.append({
                    "entry_time":        dates[entry_bar],
                    "exit_time":         dates[i],
                    "direction":         result.direction,
                    "entry_px":          result.entry_px,
                    "exit_px":           result.exit_px,
                    "qty":               result.qty,
                    "leverage":          result.leverage,
                    "notional":          result.notional_entry,
                    "initial_margin":    result.initial_margin,
                    "liq_price":         result.liq_price,
                    "bankruptcy_price":  result.bankruptcy_price,
                    "gross_pnl":         result.gross_pnl,
                    "entry_fee":         result.entry_fee,
                    "exit_fee":          result.exit_fee,
                    "entry_spread":      result.entry_spread,
                    "entry_slippage":    result.entry_slippage,
                    "exit_slippage":     result.exit_slippage,
                    "funding_paid":      result.funding_paid,
                    "costs":             result.total_cost,
                    "cost_pct":          result.total_cost / result.notional_entry * 100 if result.notional_entry > 0 else 0,
                    "net_pnl":           result.net_pnl,
                    "pnl":               result.net_pnl,
                    "pnl_pct":           pnl_pct_unl,
                    "margin_return_pct": result.margin_return_pct,
                    "bars_held":         result.bars_held,
                    "exit_reason":       result.exit_reason.lower(),
                    "was_liquidated":    result.was_liquidated,
                    "final_margin_ratio":result.final_margin_ratio,
                    "maint_margin_rate": result.maint_margin_rate,
                })
                in_trade = False

        # Record equity (use sim's running equity + unrealized if in trade)
        snap = sim.position_snapshot(px, tf) if in_trade else None
        eq   = (snap["margin_balance"] + sim.equity) if snap else sim.equity
        equity_curve.append({"time": dates[i], "equity": eq})

        # Open new trade
        if not in_trade:
            sig = int(signal.iloc[i - 1])
            if sig != 0 and not np.isnan(atrs[i]):
                atr_val    = atrs[i]
                sl_dist    = atr_sl_mult * atr_val
                tp_dist    = atr_tp_mult * atr_val
                sl_px      = round(px - sig * sl_dist, 4)
                tp_px      = round(px + sig * tp_dist, 4)

                open_info  = sim.open(
                    direction  = sig,
                    entry_px   = px,
                    sl_px      = sl_px,
                    tp_px      = tp_px,
                    bar_idx    = i,
                    avg_volume = avg_vol,
                    era_bias   = era_bias,
                )
                if "error" not in open_info:
                    direction = sig
                    entry_bar = i
                    in_trade  = True

    # Close any lingering position at last bar
    if in_trade and len(df) > 1:
        last_i = len(df) - 1
        result = sim.update(
            bar_idx    = last_i + 1,
            high       = highs[-1],
            low        = lows[-1],
            close      = closes[-1],
            signal     = 0,
            tf         = tf,
            era_bias   = era_bias,
            avg_volume = float(np.mean(vols[-20:])),
        )
        if result is not None:
            pnl_pct_unl = result.unlev_return_pct
            trades.append({
                "entry_time":        dates[entry_bar],
                "exit_time":         dates[-1],
                "direction":         result.direction,
                "entry_px":          result.entry_px,
                "exit_px":           result.exit_px,
                "qty":               result.qty,
                "leverage":          result.leverage,
                "notional":          result.notional_entry,
                "initial_margin":    result.initial_margin,
                "liq_price":         result.liq_price,
                "bankruptcy_price":  result.bankruptcy_price,
                "gross_pnl":         result.gross_pnl,
                "entry_fee":         result.entry_fee,
                "exit_fee":          result.exit_fee,
                "entry_spread":      result.entry_spread,
                "entry_slippage":    result.entry_slippage,
                "exit_slippage":     result.exit_slippage,
                "funding_paid":      result.funding_paid,
                "costs":             result.total_cost,
                "cost_pct":          result.total_cost / result.notional_entry * 100 if result.notional_entry > 0 else 0,
                "net_pnl":           result.net_pnl,
                "pnl":               result.net_pnl,
                "pnl_pct":           pnl_pct_unl,
                "margin_return_pct": result.margin_return_pct,
                "bars_held":         result.bars_held,
                "exit_reason":       result.exit_reason.lower(),
                "was_liquidated":    result.was_liquidated,
                "final_margin_ratio":result.final_margin_ratio,
                "maint_margin_rate": result.maint_margin_rate,
            })

    eq_df = pd.DataFrame(equity_curve).set_index("time")
    tr_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    stats = _compute_stats(eq_df, tr_df, initial_capital, leverage)
    return {"equity": eq_df, "trades": tr_df, "stats": stats}


def _compute_stats(eq_df, tr_df, initial_capital, leverage):
    final_equity = eq_df["equity"].iloc[-1] if len(eq_df) else initial_capital
    total_return = (final_equity / initial_capital - 1) * 100

    if tr_df.empty:
        return {"total_return_pct": total_return, "n_trades": 0,
                "leverage": leverage}

    wins  = tr_df[tr_df["pnl"] > 0]
    loses = tr_df[tr_df["pnl"] <= 0]
    win_rate = len(wins) / len(tr_df) * 100

    avg_win  = wins["pnl_pct"].mean()  if len(wins)  else 0
    avg_loss = loses["pnl_pct"].mean() if len(loses) else 0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    # Drawdown
    equity_vals = eq_df["equity"].values
    roll_max    = np.maximum.accumulate(equity_vals)
    dd          = (equity_vals - roll_max) / roll_max * 100
    max_dd      = dd.min()

    # Sharpe (bar-level)
    bar_rets = eq_df["equity"].pct_change().dropna()
    sharpe   = (bar_rets.mean() / bar_rets.std() * np.sqrt(252)
                if bar_rets.std() > 0 else float("nan"))

    # Profit factor
    gross_win  = wins["pnl"].sum()       if len(wins)  else 0
    gross_loss = abs(loses["pnl"].sum()) if len(loses) else 1
    pf         = gross_win / gross_loss  if gross_loss > 0 else float("nan")

    # Cost analysis
    total_costs  = tr_df["costs"].sum()      if "costs"    in tr_df.columns else 0
    avg_cost     = tr_df["cost_pct"].mean()  if "cost_pct" in tr_df.columns else 0
    total_fund   = tr_df["funding_paid"].sum() if "funding_paid" in tr_df.columns else 0
    n_liqs       = int(tr_df["was_liquidated"].sum()) if "was_liquidated" in tr_df.columns else 0
    avg_lev      = float(tr_df["leverage"].mean())    if "leverage"       in tr_df.columns else leverage

    expectancy  = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    def _fmt(v):
        return round(v, 2) if not (isinstance(v, float) and (
            v != v or v == float("inf") or v == float("-inf"))) else "N/A"

    return {
        "total_return_pct":   _fmt(total_return),
        "n_trades":            len(tr_df),
        "win_rate_pct":        _fmt(win_rate),
        "avg_win_pct":         _fmt(avg_win),
        "avg_loss_pct":        _fmt(avg_loss),
        "risk_reward":         _fmt(rr),
        "profit_factor":       _fmt(pf),
        "expectancy_pct":      _fmt(expectancy),
        "max_drawdown_pct":    _fmt(max_dd),
        "sharpe":              _fmt(sharpe),
        "total_costs_usd":     _fmt(total_costs),
        "avg_cost_pct":        _fmt(avg_cost),
        "total_funding_usd":   _fmt(total_fund),
        "n_liquidations":      n_liqs,
        "avg_leverage":        _fmt(avg_lev),
        "exit_reasons":        tr_df["exit_reason"].value_counts().to_dict(),
        "final_equity":        round(final_equity, 2),
        "long_trades":         int((tr_df["direction"] == 1).sum()),
        "short_trades":        int((tr_df["direction"] == -1).sum()),
        "leverage_used":       leverage,
    }
