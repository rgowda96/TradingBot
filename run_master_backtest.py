"""
Run all 65 strategies × 7 timeframes = up to 455 backtests.
Outputs: results/master_results.json + results/master_report.html
"""
import os, json, glob, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from indicators            import add_indicators
from backtest_engine       import backtest
from master_strategy_library import STRATEGY_REGISTRY, CATEGORY_DESCRIPTIONS

TIMEFRAMES = ["1w","1d","4h","1h","15m","5m","1m"]
DATA_DIR   = "data"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def load_df(tf):
    path = os.path.join(DATA_DIR, f"ETH_USDT_{tf}.parquet")
    if not os.path.exists(path): return None
    df = pd.read_parquet(path).dropna()
    df = add_indicators(df)
    return df.dropna()

def run_all():
    print(f"\n{'═'*65}")
    print(f"  MASTER ETH BACKTEST — {len(STRATEGY_REGISTRY)} strategies × {len(TIMEFRAMES)} timeframes")
    print(f"{'═'*65}\n")

    tf_data = {}
    for tf in TIMEFRAMES:
        df = load_df(tf)
        if df is not None:
            tf_data[tf] = df
            print(f"  Loaded {tf:4s}: {len(df):4d} bars  {df.index[0].date()} → {df.index[-1].date()}")
        else:
            print(f"  Skipped {tf}: no data file")

    print()
    all_results = []
    total = len(STRATEGY_REGISTRY) * len(tf_data)
    done  = 0

    for strat_id, (category, fn) in STRATEGY_REGISTRY.items():
        for tf, df in tf_data.items():
            done += 1
            try:
                sig    = fn(df)
                result = backtest(df, sig, tf=tf)
                s      = result["stats"]
                n      = s.get("n_trades", 0)

                sharpe = s.get("sharpe", "N/A")
                pf     = s.get("profit_factor", "N/A")
                ret    = s.get("total_return_pct", 0)
                wr     = s.get("win_rate_pct", 0)
                dd     = s.get("max_drawdown_pct", 0)
                expct  = s.get("expectancy_pct", 0)
                cost   = s.get("avg_cost_pct", 0)

                sharpe_f = float(sharpe) if sharpe not in ("N/A", None) else None
                pf_f     = float(pf)     if pf     not in ("N/A", None) else None

                row = {
                    "id":            strat_id,
                    "category":      category,
                    "timeframe":     tf,
                    "n_trades":      n,
                    "total_return":  round(float(ret), 2)   if ret  not in ("N/A",None) else None,
                    "win_rate":      round(float(wr), 1)    if wr   not in ("N/A",None) else None,
                    "profit_factor": round(pf_f, 3)         if pf_f is not None else None,
                    "sharpe":        round(sharpe_f, 3)     if sharpe_f is not None else None,
                    "max_dd":        round(float(dd), 2)    if dd   not in ("N/A",None) else None,
                    "expectancy":    round(float(expct), 3) if expct not in ("N/A",None) else None,
                    "avg_cost_pct":  round(float(cost), 4)  if cost not in ("N/A",None) else None,
                    "long_trades":   s.get("long_trades", 0),
                    "short_trades":  s.get("short_trades", 0),
                    "exit_reasons":  s.get("exit_reasons", {}),
                }
                all_results.append(row)

                if n >= 5:
                    pf_str = f"{pf_f:.2f}" if pf_f else "N/A"
                    sh_str = f"{sharpe_f:.2f}" if sharpe_f else "N/A"
                    mark   = "★" if (sharpe_f and sharpe_f > 0.5 and pf_f and pf_f > 1.2) else " "
                    print(f"  {mark} [{done:3d}/{total}] {tf:4s} {strat_id:35s} "
                          f"n={n:4d}  ret={ret:+6.1f}%  wr={wr:5.1f}%  "
                          f"pf={pf_str:6s}  sharpe={sh_str:6s}  dd={dd:6.1f}%")

            except Exception as e:
                all_results.append({
                    "id": strat_id, "category": category,
                    "timeframe": tf, "n_trades": 0,
                    "error": str(e)[:80],
                })

    # Save JSON
    json_path = os.path.join(OUT_DIR, "master_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    valid = [r for r in all_results
             if r.get("n_trades",0) >= 10
             and r.get("sharpe") is not None
             and r.get("profit_factor") is not None]

    ranked = sorted(valid, key=lambda r: r["sharpe"], reverse=True)

    print(f"\n{'═'*85}")
    print(f"  TOP 30 EDGES  (ranked by Sharpe, min 10 trades, after all fees)")
    print(f"{'═'*85}")
    print(f"  {'#':>3}  {'TF':4}  {'Strategy':35}  {'Cat':22}  "
          f"{'Ret%':>7}  {'WR%':>6}  {'PF':>5}  {'Sharpe':>6}  {'MaxDD%':>7}  {'Trades':>6}")
    print(f"  {'─'*3}  {'─'*4}  {'─'*35}  {'─'*22}  "
          f"{'─'*7}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*6}")

    for i, r in enumerate(ranked[:30], 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] else "N/A"
        print(f"  {i:>3}  {r['timeframe']:4}  {r['id']:35}  "
              f"{r['category']:22}  "
              f"{r['total_return']:>+7.1f}%  "
              f"{r['win_rate']:>6.1f}%  "
              f"{pf_str:>5}  "
              f"{r['sharpe']:>6.2f}  "
              f"{r['max_dd']:>+7.1f}%  "
              f"{r['n_trades']:>6}")

    print(f"\n  Results saved → {json_path}")
    return all_results

if __name__ == "__main__":
    results = run_all()
