"""
Strategy Expander — discovers new parameter combos and hybrid strategies,
backtests them, and adds top performers to the active strategy pool.
Runs every 4 hours as part of the autonomous loop.
"""
import ccxt
import pandas as pd
import numpy as np
import json
import os
import itertools
from datetime import datetime, timezone

from indicators import add_indicators
from backtest_engine import backtest

DISCOVERY_FILE  = "discovered_strategies.json"
ACTIVE_FILE     = "active_strategies.json"
MIN_TRADES      = 8
MIN_SHARPE      = 0.0
MIN_WIN_RATE    = 40.0
MIN_PROFIT_FACTOR = 0.95

TIMEFRAMES = ["1d", "4h", "1h", "15m"]
SYMBOL = "ETH/USDT"


# ── Dynamic strategy generators ───────────────────────────────────────────────

def make_rsi_trend(ob=70, os_=30, ema_fast=9, ema_slow=21):
    def _strat(df):
        fast = df[f"ema_{ema_fast}"] if f"ema_{ema_fast}" in df.columns else df["ema_9"]
        slow = df[f"ema_{ema_slow}"] if f"ema_{ema_slow}" in df.columns else df["ema_21"]
        sig  = pd.Series(0, index=df.index)
        sig[(df["rsi_14"] < os_) & (fast > slow)] = 1
        sig[(df["rsi_14"] > ob)  & (fast < slow)] = -1
        return sig
    _strat.__name__ = f"rsi_trend_ob{ob}_os{os_}_e{ema_fast}/{ema_slow}"
    return _strat


def make_macd_rsi(rsi_bull=55, rsi_bear=45):
    def _strat(df):
        hist    = df["macd_hist"]
        flip_up = (hist > 0) & (hist.shift(1) <= 0)
        flip_dn = (hist < 0) & (hist.shift(1) >= 0)
        sig = pd.Series(0, index=df.index)
        sig[flip_up & (df["rsi_14"] > rsi_bull)] = 1
        sig[flip_dn & (df["rsi_14"] < rsi_bear)] = -1
        return sig
    _strat.__name__ = f"macd_rsi_b{rsi_bull}_be{rsi_bear}"
    return _strat


def make_bb_rsi(rsi_os=40, rsi_ob=60):
    def _strat(df):
        sig = pd.Series(0, index=df.index)
        sig[(df["close"] < df["bb_lower_20"]) & (df["rsi_14"] < rsi_os)] = 1
        sig[(df["close"] > df["bb_upper_20"]) & (df["rsi_14"] > rsi_ob)] = -1
        return sig
    _strat.__name__ = f"bb_rsi_os{rsi_os}_ob{rsi_ob}"
    return _strat


def make_vol_momentum(vol_mult=1.5, roc_thresh=2.0):
    def _strat(df):
        sig = pd.Series(0, index=df.index)
        big_vol  = df["volume_ratio"] > vol_mult
        bull_roc = df["roc_9"] > roc_thresh
        bear_roc = df["roc_9"] < -roc_thresh
        sig[big_vol & bull_roc & (df["close"] > df["ema_50"])] = 1
        sig[big_vol & bear_roc & (df["close"] < df["ema_50"])] = -1
        return sig
    _strat.__name__ = f"vol_mom_vm{vol_mult}_roc{roc_thresh}"
    return _strat


def make_supertrend_rsi(rsi_min=40, rsi_max=60):
    def _strat(df):
        sig = pd.Series(0, index=df.index)
        sig[(df["supertrend_dir"] == 1)  & (df["rsi_14"] > rsi_min)] = 1
        sig[(df["supertrend_dir"] == -1) & (df["rsi_14"] < rsi_max)] = -1
        return sig
    _strat.__name__ = f"st_rsi_min{rsi_min}_max{rsi_max}"
    return _strat


def make_squeeze_momentum():
    """Squeeze Momentum: fire when squeeze releases with strong OBV."""
    def _strat(df):
        prev_sq = df["squeeze"].shift(1)
        obv_up  = df["obv"] > df["obv_ema_20"]
        obv_dn  = df["obv"] < df["obv_ema_20"]
        sig = pd.Series(0, index=df.index)
        sig[(~df["squeeze"]) & prev_sq & obv_up]  = 1
        sig[(~df["squeeze"]) & prev_sq & obv_dn]  = -1
        return sig
    _strat.__name__ = "squeeze_obv_momentum"
    return _strat


def make_triple_screen(fast=9, mid=21, slow=50):
    """Elder triple-screen: slow trend + mid momentum + fast entry."""
    def _strat(df):
        e_fast = df.get(f"ema_{fast}", df["ema_9"])
        e_mid  = df.get(f"ema_{mid}",  df["ema_21"])
        e_slow = df.get(f"ema_{slow}", df["ema_50"])
        bull = (e_slow.diff() > 0) & (df["rsi_14"] < 60) & (e_fast > e_mid)
        bear = (e_slow.diff() < 0) & (df["rsi_14"] > 40) & (e_fast < e_mid)
        sig = pd.Series(0, index=df.index)
        sig[bull] = 1
        sig[bear] = -1
        return sig
    _strat.__name__ = f"triple_screen_{fast}/{mid}/{slow}"
    return _strat


def make_mean_revert_vwap(atr_mult=1.5):
    """Price deviates from VWAP by N*ATR → mean revert."""
    def _strat(df):
        diff  = df["close"] - df["vwap_20"]
        thresh = atr_mult * df["atr_14"]
        sig = pd.Series(0, index=df.index)
        sig[(diff < -thresh) & (df["rsi_14"] < 50)] = 1
        sig[(diff >  thresh) & (df["rsi_14"] > 50)] = -1
        return sig
    _strat.__name__ = f"mean_revert_vwap_a{atr_mult}"
    return _strat


def make_chandelier_exit(atr_mult=3.0):
    """Long when price > chandelier stop, short when below."""
    def _strat(df):
        stop_long  = df["high"].rolling(22).max() - atr_mult * df["atr_14"]
        stop_short = df["low"].rolling(22).min()  + atr_mult * df["atr_14"]
        sig = pd.Series(0, index=df.index)
        sig[df["close"] > stop_long]  = 1
        sig[df["close"] < stop_short] = -1
        return sig
    _strat.__name__ = f"chandelier_{atr_mult}"
    return _strat


# ── Build candidate grid ──────────────────────────────────────────────────────

def build_candidates():
    cands = []

    # RSI trend combos
    for ob, os_ in [(65,35),(70,30),(75,25)]:
        cands.append(make_rsi_trend(ob, os_, 9, 21))
        cands.append(make_rsi_trend(ob, os_, 9, 50))

    # MACD × RSI
    for rb, rbe in [(50,50),(55,45),(60,40)]:
        cands.append(make_macd_rsi(rb, rbe))

    # BB × RSI
    for os_, ob in [(35,65),(40,60),(45,55)]:
        cands.append(make_bb_rsi(os_, ob))

    # Volume momentum
    for vm, roc in [(1.5,1.5),(2.0,2.0),(1.5,3.0)]:
        cands.append(make_vol_momentum(vm, roc))

    # Supertrend × RSI
    for mn, mx in [(40,60),(45,55),(35,65)]:
        cands.append(make_supertrend_rsi(mn, mx))

    # One-off specials
    cands.append(make_squeeze_momentum())
    for f,m,s in [(9,21,50),(9,21,200)]:
        cands.append(make_triple_screen(f,m,s))
    for a in [1.0,1.5,2.0]:
        cands.append(make_mean_revert_vwap(a))
    for a in [2.5,3.0,3.5]:
        cands.append(make_chandelier_exit(a))

    return cands


# ── Fetch + prepare ───────────────────────────────────────────────────────────

def fetch_and_prep(exchange, tf, limit=500):
    raw = exchange.fetch_ohlcv(SYMBOL, tf, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.dropna()
    df = add_indicators(df)
    return df.dropna()


# ── Main discovery run ────────────────────────────────────────────────────────

def run_discovery(log_fn=print):
    exchange  = ccxt.binance({"enableRateLimit": True})
    candidates = build_candidates()
    discovered = load_discovered()
    new_count  = 0

    log_fn(f"[Expander] Running {len(candidates)} candidate strategies × {len(TIMEFRAMES)} TFs")

    for tf in TIMEFRAMES:
        try:
            df = fetch_and_prep(exchange, tf)
        except Exception as e:
            log_fn(f"[Expander] fetch error {tf}: {e}")
            continue

        for strat_fn in candidates:
            name = strat_fn.__name__
            key  = f"{tf}|{name}"
            try:
                signal = strat_fn(df)
                res    = backtest(df, signal)
                s      = res["stats"]

                if (s["n_trades"] >= MIN_TRADES and
                    s.get("win_rate_pct", 0) >= MIN_WIN_RATE):

                    sharpe = s.get("sharpe", -99)
                    pf     = s.get("profit_factor", 0)
                    if sharpe == "N/A": sharpe = -99
                    if pf     == "N/A": pf = 0
                    try:
                        sharpe = float(sharpe)
                        pf     = float(pf)
                    except:
                        continue

                    if sharpe >= MIN_SHARPE and pf >= MIN_PROFIT_FACTOR:
                        entry = {
                            "key": key, "tf": tf, "strategy": name,
                            "sharpe": round(sharpe, 3),
                            "profit_factor": round(pf, 3),
                            "win_rate": round(s.get("win_rate_pct", 0), 1),
                            "total_return": round(s.get("total_return_pct", 0), 2),
                            "n_trades": s["n_trades"],
                            "max_dd": round(s.get("max_drawdown_pct", 0), 2),
                            "discovered_at": datetime.now(timezone.utc).isoformat(),
                        }
                        discovered[key] = entry
                        new_count += 1
                        log_fn(f"[Expander] ✨ NEW EDGE: {key}  Sharpe={sharpe:.2f}  "
                               f"PF={pf:.2f}  WR={s.get('win_rate_pct',0):.1f}%  "
                               f"Trades={s['n_trades']}")
            except Exception as e:
                pass   # silent — many combos will fail on short series

    save_discovered(discovered)
    rebuild_active(discovered, log_fn)
    log_fn(f"[Expander] Done. {new_count} new edges found. "
           f"Total discovered: {len(discovered)}")
    return new_count


def load_discovered():
    if os.path.exists(DISCOVERY_FILE):
        with open(DISCOVERY_FILE) as f:
            return json.load(f)
    return {}


def save_discovered(d):
    with open(DISCOVERY_FILE, "w") as f:
        json.dump(d, f, indent=2)


def rebuild_active(discovered, log_fn=print):
    """Rank all discovered strategies by Sharpe and write active list."""
    rows = list(discovered.values())
    rows.sort(key=lambda r: r.get("sharpe", -99), reverse=True)
    top = rows[:30]   # keep best 30

    active = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": top,
    }
    with open(ACTIVE_FILE, "w") as f:
        json.dump(active, f, indent=2)

    log_fn(f"[Expander] Active pool rebuilt — top {len(top)} strategies:")
    for r in top[:10]:
        log_fn(f"  {r['tf']:4s} {r['strategy']:40s}  "
               f"Sharpe={r['sharpe']:.2f}  PF={r['profit_factor']:.2f}  "
               f"WR={r['win_rate']:.1f}%")


def load_active_strategies():
    """Returns list of (tf, strategy_name, strat_fn) to watch."""
    if not os.path.exists(ACTIVE_FILE):
        return []
    with open(ACTIVE_FILE) as f:
        active = json.load(f)
    # Map name → function
    fn_map = {fn.__name__: fn for fn in build_candidates()}
    result = []
    for r in active.get("strategies", []):
        fn = fn_map.get(r["strategy"])
        if fn:
            result.append((r["tf"], r["strategy"], fn))
    return result


if __name__ == "__main__":
    run_discovery()
