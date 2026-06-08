"""
Live signal monitor — fetches latest bars, runs the best-performing strategies,
prints actionable signals every N seconds.

Best edges from backtest (Sharpe > 0):
  1d  obv_divergence    Sharpe=0.64  WR=52%  PF=1.47
  1d  engulf_reversal   Sharpe=0.46  WR=64%  PF=2.1
  1d  stoch_rsi_cross   Sharpe=0.16  WR=49%  PF=1.08
  1d  macd_momentum     Sharpe=0.15  WR=44%  PF=1.12
"""
import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timezone

from indicators import add_indicators
from strategies import STRATEGIES

SYMBOL = "ETH/USDT"
WATCH_TF_STRATEGIES = [
    ("1d", "obv_divergence"),
    ("1d", "engulf_reversal"),
    ("1d", "stoch_rsi_cross"),
    ("1d", "macd_momentum"),
    ("1d", "bb_breakout"),
    ("4h", "macd_momentum"),
    ("4h", "volume_breakout"),
    ("4h", "stoch_rsi_cross"),
    ("1h", "supertrend"),
]
CANDLES_PER_TF = 300  # enough for indicators
REFRESH_SECS  = 60    # check every minute


def fetch_df(exchange, tf, limit=300):
    raw = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.dropna()
    return df


def get_signal_name(val):
    if val == 1:
        return "🟢 LONG"
    elif val == -1:
        return "🔴 SHORT"
    return "⚪ FLAT"


def run_monitor(once=False):
    exchange = ccxt.binance({"enableRateLimit": True})
    iteration = 0

    while True:
        iteration += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'─'*65}")
        print(f"  ETH/USDT  Signal Monitor  |  {now}  |  iteration {iteration}")
        print(f"{'─'*65}")

        active_signals = []

        for tf, strat_name in WATCH_TF_STRATEGIES:
            try:
                df = fetch_df(exchange, tf, CANDLES_PER_TF)
                df = add_indicators(df)
                df = df.dropna()
                if len(df) < 30:
                    continue

                strat_fn = STRATEGIES[strat_name]
                signal = strat_fn(df)
                last_sig = int(signal.iloc[-1])
                prev_sig = int(signal.iloc[-2])

                # Current price context
                px      = df["close"].iloc[-1]
                atr     = df["atr_14"].iloc[-1]
                sl_long = round(px - 1.5 * atr, 2)
                tp_long = round(px + 3.0 * atr, 2)
                sl_short= round(px + 1.5 * atr, 2)
                tp_short= round(px - 3.0 * atr, 2)

                changed = last_sig != prev_sig and last_sig != 0

                status = get_signal_name(last_sig)
                change_marker = " ← NEW" if changed else ""
                print(f"  {tf:4s} | {strat_name:22s} | {status}{change_marker}")

                if last_sig == 1:
                    print(f"        Entry ~${px:,.2f}  SL=${sl_long:,.2f}  TP=${tp_long:,.2f}  "
                          f"RR=1:{3/1.5:.1f}  ATR=${atr:,.2f}")
                elif last_sig == -1:
                    print(f"        Entry ~${px:,.2f}  SL=${sl_short:,.2f}  TP=${tp_short:,.2f}  "
                          f"RR=1:{3/1.5:.1f}  ATR=${atr:,.2f}")

                if changed:
                    active_signals.append((tf, strat_name, last_sig, px, sl_long if last_sig==1 else sl_short,
                                           tp_long if last_sig==1 else tp_short))

            except Exception as e:
                print(f"  {tf:4s} | {strat_name:22s} | ERROR: {e}")

        if active_signals:
            print(f"\n  ▶ NEW SIGNALS THIS BAR:")
            for tf, strat, sig, px, sl, tp in active_signals:
                d = "LONG" if sig == 1 else "SHORT"
                print(f"    {d} on {tf} {strat}  entry≈${px:,.2f}  SL=${sl:,.2f}  TP=${tp:,.2f}")

        print(f"\n  Current ETH Price: ${exchange.fetch_ticker(SYMBOL)['last']:,.2f}")

        if once:
            break

        print(f"\n  Next refresh in {REFRESH_SECS}s ... (Ctrl+C to stop)")
        time.sleep(REFRESH_SECS)


if __name__ == "__main__":
    run_monitor(once=False)
