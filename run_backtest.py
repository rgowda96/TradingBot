"""
Main runner: loads data, runs all strategies on all timeframes, prints ranked results,
saves CSV summary and equity-curve plots.
"""
import os
import glob
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from indicators import add_indicators
from strategies import STRATEGIES
from backtest_engine import backtest

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ATR_SL  = 1.5
ATR_TP  = 3.0
RISK_PCT = 0.01
CAPITAL = 10_000


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna()
    df = add_indicators(df)
    df = df.dropna()
    return df


def run_all():
    data_files = sorted(glob.glob("data/ETH_USDT_*.parquet"))
    if not data_files:
        print("No data files found. Run fetch_data.py first.")
        return

    summary_rows = []

    for fpath in data_files:
        tf = os.path.basename(fpath).replace("ETH_USDT_", "").replace(".parquet", "")
        print(f"\n{'='*60}")
        print(f"  Timeframe: {tf}  ({fpath})")
        print(f"{'='*60}")

        try:
            df = load_and_prepare(fpath)
        except Exception as e:
            print(f"  ERROR loading {fpath}: {e}")
            continue

        print(f"  Bars: {len(df)}  |  {df.index[0].date()} → {df.index[-1].date()}")

        fig, axes = plt.subplots(len(STRATEGIES), 1, figsize=(14, 4 * len(STRATEGIES)), sharex=False)
        if len(STRATEGIES) == 1:
            axes = [axes]

        for ax_idx, (name, strat_fn) in enumerate(STRATEGIES.items()):
            try:
                signal = strat_fn(df)
                result = backtest(
                    df, signal,
                    atr_sl_mult=ATR_SL,
                    atr_tp_mult=ATR_TP,
                    risk_pct=RISK_PCT,
                    initial_capital=CAPITAL,
                )
                s = result["stats"]
                row = {"timeframe": tf, "strategy": name, **s}
                summary_rows.append(row)

                ret  = s["total_return_pct"]
                wr   = s.get("win_rate_pct", "N/A")
                pf   = s.get("profit_factor", "N/A")
                dd   = s.get("max_drawdown_pct", "N/A")
                shar = s.get("sharpe", "N/A")
                n    = s["n_trades"]

                print(f"  [{name:22s}]  Return={ret:+.1f}%  WR={wr}%  PF={pf}  "
                      f"Sharpe={shar}  MaxDD={dd}%  Trades={n}")

                # Plot equity curve
                eq = result["equity"]
                ax = axes[ax_idx]
                ax.plot(eq.index, eq["equity"], linewidth=1)
                ax.axhline(CAPITAL, color="grey", linewidth=0.5, linestyle="--")
                ax.set_title(f"{name} | {tf} | Return={ret:+.1f}% | WR={wr}% | Trades={n}",
                             fontsize=8)
                ax.set_ylabel("Equity $")
                ax.tick_params(axis="x", labelsize=6)

            except Exception as e:
                print(f"  [{name:22s}]  ERROR: {e}")

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"equity_{tf}.png")
        plt.savefig(plot_path, dpi=100)
        plt.close()
        print(f"  → Equity curves saved: {plot_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not summary_rows:
        print("No results to summarise.")
        return

    sumdf = pd.DataFrame(summary_rows)
    sumdf.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

    print(f"\n{'='*70}")
    print("  TOP 20 EDGES  (ranked by Sharpe, min 10 trades)")
    print(f"{'='*70}")

    ranked = sumdf[sumdf["n_trades"] >= 10].copy()
    ranked["sharpe_num"] = pd.to_numeric(ranked["sharpe"], errors="coerce")
    ranked = ranked.sort_values("sharpe_num", ascending=False).head(20)

    for _, r in ranked.iterrows():
        print(f"  {r['timeframe']:5s}  {r['strategy']:22s}  "
              f"Return={r['total_return_pct']:+6.1f}%  "
              f"WR={r['win_rate_pct']:4.1f}%  "
              f"PF={r['profit_factor']}  "
              f"Sharpe={r['sharpe']}  "
              f"MaxDD={r['max_drawdown_pct']:5.1f}%  "
              f"Trades={int(r['n_trades'])}")

    print(f"\n  Full results -> {RESULTS_DIR}/summary.csv")


if __name__ == "__main__":
    run_all()
