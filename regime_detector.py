"""
regime_detector.py — Multi-timeframe regime detection for ETH/USDT perpetual futures.

Defines 8 regimes: STRONG_BULL, WEAK_BULL, CHOPPY_BULL, STRONG_BEAR, WEAK_BEAR,
CHOPPY_BEAR, RANGING, VOLATILE

Uses: EMA slope/alignment, ADX, ATR percentile, OBV slope, RSI zone, price vs MAs,
      HH/HL vs LH/LL structure, multi-timeframe alignment.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


# ─── Constants ────────────────────────────────────────────────────────────────

REGIMES = [
    "STRONG_BULL", "WEAK_BULL", "CHOPPY_BULL",
    "STRONG_BEAR", "WEAK_BEAR", "CHOPPY_BEAR",
    "RANGING", "VOLATILE",
]

VOL_REGIMES   = ["LOW", "NORMAL", "HIGH", "EXTREME"]
STRUCTURES    = ["UPTREND", "DOWNTREND", "RANGING"]
HTF_BIASES    = ["BULL", "BEAR", "NEUTRAL"]

# Strategy family weights per regime  (1.0 = neutral, >1 = prefer, <1 = avoid)
_STRATEGY_FIT: Dict[str, Dict[str, float]] = {
    "STRONG_BULL":  {"trend_following":2.0,"mean_reversion":0.4,"breakout":1.5,"momentum":1.8,"volume":1.2,"neutral":0.8},
    "WEAK_BULL":    {"trend_following":1.4,"mean_reversion":0.8,"breakout":1.2,"momentum":1.2,"volume":1.0,"neutral":1.0},
    "CHOPPY_BULL":  {"trend_following":0.6,"mean_reversion":1.6,"breakout":0.7,"momentum":0.7,"volume":0.9,"neutral":1.3},
    "STRONG_BEAR":  {"trend_following":2.0,"mean_reversion":0.4,"breakout":1.5,"momentum":1.8,"volume":1.2,"neutral":0.8},
    "WEAK_BEAR":    {"trend_following":1.4,"mean_reversion":0.8,"breakout":1.2,"momentum":1.2,"volume":1.0,"neutral":1.0},
    "CHOPPY_BEAR":  {"trend_following":0.6,"mean_reversion":1.6,"breakout":0.7,"momentum":0.7,"volume":0.9,"neutral":1.3},
    "RANGING":      {"trend_following":0.3,"mean_reversion":2.0,"breakout":0.8,"momentum":0.5,"volume":0.8,"neutral":1.5},
    "VOLATILE":     {"trend_following":0.7,"mean_reversion":1.2,"breakout":1.8,"momentum":1.0,"volume":1.4,"neutral":0.9},
}


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    regime:           str   = "RANGING"
    confidence:       float = 50.0          # 0–100
    volatility_pct:   float = 0.0           # ATR / price * 100
    trend_strength:   float = 20.0          # ADX value
    momentum:         float = 0.0           # (RSI - 50) / 50, range -1..1
    vol_regime:       str   = "NORMAL"      # LOW / NORMAL / HIGH / EXTREME
    market_structure: str   = "RANGING"     # UPTREND / DOWNTREND / RANGING
    htf_bias:         str   = "NEUTRAL"     # BULL / BEAR / NEUTRAL (from daily)
    adx:              float = 20.0
    rsi:              float = 50.0
    atr_pct:          float = 0.0
    ema_aligned:      int   = 0             # +1 bull stack, -1 bear stack, 0 mixed
    signals_raw:      dict  = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ─── Low-level indicator helpers ──────────────────────────────────────────────

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _rma(series: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (used in ATR/ADX)."""
    alpha = 1.0 / n
    return series.ewm(alpha=alpha, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = np.where((high - prev_high) > (prev_low - low),
                        np.maximum(high - prev_high, 0.0), 0.0)
    dm_minus = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0.0), 0.0)

    atr14   = _rma(tr, period)
    dp14    = _rma(pd.Series(dm_plus,  index=df.index), period)
    dm14    = _rma(pd.Series(dm_minus, index=df.index), period)

    di_plus  = 100 * dp14 / (atr14 + 1e-9)
    di_minus = 100 * dm14 / (atr14 + 1e-9)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx      = _rma(dx, period)
    return adx


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0)
    down  = (-delta).clip(lower=0)
    rs    = _rma(up, period) / (_rma(down, period) + 1e-9)
    return 100 - 100 / (1 + rs)


def _obv_slope(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return normalised OBV slope over lookback bars."""
    if "volume" not in df.columns or len(df) < lookback + 1:
        return 0.0
    close  = df["close"].values
    volume = df["volume"].values
    sign   = np.sign(np.diff(close))
    sign   = np.concatenate([[0], sign])
    obv    = np.cumsum(sign * volume)
    tail   = obv[-lookback:]
    x      = np.arange(len(tail), dtype=float)
    if tail.std() < 1e-9:
        return 0.0
    slope  = float(np.polyfit(x, tail, 1)[0])
    # normalise by mean absolute obv
    return slope / (np.abs(tail).mean() + 1e-9)


def _hh_hl_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """Detect HH/HL (uptrend), LH/LL (downtrend), or mixed (ranging)."""
    if len(df) < lookback:
        return "RANGING"
    highs  = df["high"].tail(lookback).values
    lows   = df["low"].tail(lookback).values
    n      = len(highs)
    # Use 5-bar swing pivots
    chunk  = max(2, n // 4)
    h1, h2, h3 = highs[:chunk].max(), highs[chunk:2*chunk].max(), highs[2*chunk:].max()
    l1, l2, l3 = lows[:chunk].min(),  lows[chunk:2*chunk].min(),  lows[2*chunk:].min()

    bull_h = h3 > h2 > h1
    bull_l = l3 > l2 > l1
    bear_h = h3 < h2 < h1
    bear_l = l3 < l2 < l1

    if bull_h and bull_l:
        return "UPTREND"
    if bear_h and bear_l:
        return "DOWNTREND"
    if bull_h or bull_l:
        return "UPTREND"
    if bear_h or bear_l:
        return "DOWNTREND"
    return "RANGING"


def _atr_percentile(df: pd.DataFrame, window: int = 252) -> float:
    """Return current ATR as percentile rank over last `window` bars (0–100)."""
    atr_series = _compute_atr(df, 14)
    if len(atr_series) < 10:
        return 50.0
    tail = atr_series.dropna().tail(window)
    cur  = float(tail.iloc[-1])
    pct  = float((tail < cur).sum() / len(tail) * 100)
    return pct


def _vol_regime_label(atr_pct_rank: float) -> str:
    if atr_pct_rank < 25:   return "LOW"
    if atr_pct_rank < 60:   return "NORMAL"
    if atr_pct_rank < 85:   return "HIGH"
    return "EXTREME"


# ─── Single-df signal extractor ───────────────────────────────────────────────

def _extract_signals(df: pd.DataFrame) -> dict:
    """Extract all regime signals from a single OHLCV dataframe."""
    if len(df) < 50:
        return {}

    close = df["close"]
    px    = float(close.iloc[-1])

    # EMAs
    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema200 = _ema(close, 200) if len(close) >= 200 else _ema(close, min(len(close)-1, 100))

    e20, e50, e200 = float(ema20.iloc[-1]), float(ema50.iloc[-1]), float(ema200.iloc[-1])
    e20p = float(ema20.iloc[-5]) if len(ema20) >= 5 else e20
    e50p = float(ema50.iloc[-5]) if len(ema50) >= 5 else e50

    ema20_slope  = (e20 - e20p) / (e20p + 1e-9) * 100
    ema50_slope  = (e50 - e50p) / (e50p + 1e-9) * 100

    # EMA alignment: +1 = bull stack (20>50>200), -1 = bear stack, 0 = mixed
    if e20 > e50 > e200:
        ema_aligned = 1
    elif e20 < e50 < e200:
        ema_aligned = -1
    else:
        ema_aligned = 0

    # Price vs MAs
    px_above_20  = px > e20
    px_above_50  = px > e50
    px_above_200 = px > e200

    # ADX
    adx_series = _compute_adx(df, 14)
    adx        = float(adx_series.iloc[-1]) if not adx_series.isna().all() else 20.0

    # ATR
    atr_series = _compute_atr(df, 14)
    atr        = float(atr_series.iloc[-1]) if not atr_series.isna().all() else px * 0.02
    atr_pct    = atr / px * 100
    atr_rank   = _atr_percentile(df, 252)

    # RSI
    rsi_series = _compute_rsi(close, 14)
    rsi        = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0

    # OBV slope
    obv_slope = _obv_slope(df, 20)

    # Market structure
    structure = _hh_hl_structure(df, 20)

    return {
        "px":           px,
        "ema20":        e20,
        "ema50":        e50,
        "ema200":       e200,
        "ema20_slope":  ema20_slope,
        "ema50_slope":  ema50_slope,
        "ema_aligned":  ema_aligned,
        "px_above_20":  px_above_20,
        "px_above_50":  px_above_50,
        "px_above_200": px_above_200,
        "adx":          adx,
        "atr":          atr,
        "atr_pct":      atr_pct,
        "atr_rank":     atr_rank,
        "rsi":          rsi,
        "obv_slope":    obv_slope,
        "structure":    structure,
    }


# ─── Regime decision from signals ─────────────────────────────────────────────

def _signals_to_regime(sig: dict, htf_bias: str = "NEUTRAL") -> tuple:
    """
    Returns (regime: str, confidence: float) given extracted signals.
    """
    if not sig:
        return "RANGING", 40.0

    adx       = sig.get("adx", 20.0)
    rsi       = sig.get("rsi", 50.0)
    atr_rank  = sig.get("atr_rank", 50.0)
    ema_aln   = sig.get("ema_aligned", 0)
    e20s      = sig.get("ema20_slope", 0.0)
    e50s      = sig.get("ema50_slope", 0.0)
    structure = sig.get("structure", "RANGING")
    obv_sl    = sig.get("obv_slope", 0.0)
    px_a20    = sig.get("px_above_20", True)
    px_a50    = sig.get("px_above_50", True)
    px_a200   = sig.get("px_above_200", True)

    # Extreme volatility → VOLATILE (overrides most)
    if atr_rank >= 85 and adx < 30:
        conf = min(50.0 + atr_rank * 0.4, 92.0)
        return "VOLATILE", conf

    # Strong bull
    if (ema_aln == 1 and adx >= 25 and rsi >= 55 and
            structure == "UPTREND" and e20s > 0.05 and px_a200):
        conf = 50 + (adx - 25) * 1.2 + (rsi - 55) * 0.4 + (int(obv_sl > 0) * 8)
        return "STRONG_BULL", min(conf, 95.0)

    # Strong bear
    if (ema_aln == -1 and adx >= 25 and rsi <= 45 and
            structure == "DOWNTREND" and e20s < -0.05 and not px_a200):
        conf = 50 + (adx - 25) * 1.2 + (45 - rsi) * 0.4 + (int(obv_sl < 0) * 8)
        return "STRONG_BEAR", min(conf, 95.0)

    # Weak bull
    if (px_a50 and e20s > 0 and rsi >= 50 and adx >= 18):
        conf = 40 + (adx - 18) * 0.8 + (rsi - 50) * 0.3
        return "WEAK_BULL", min(conf, 80.0)

    # Weak bear
    if (not px_a50 and e20s < 0 and rsi <= 50 and adx >= 18):
        conf = 40 + (adx - 18) * 0.8 + (50 - rsi) * 0.3
        return "WEAK_BEAR", min(conf, 80.0)

    # Ranging
    if adx < 20 and abs(e20s) < 0.05 and structure == "RANGING":
        conf = 55 + (20 - adx) * 1.5
        return "RANGING", min(conf, 85.0)

    # Choppy bull (mild upside but messy)
    if (px_a20 and adx < 25 and rsi > 48):
        conf = 42 + (rsi - 48) * 0.5
        return "CHOPPY_BULL", min(conf, 72.0)

    # Choppy bear
    if (not px_a20 and adx < 25 and rsi < 52):
        conf = 42 + (52 - rsi) * 0.5
        return "CHOPPY_BEAR", min(conf, 72.0)

    # Default ranging
    return "RANGING", 45.0


# ─── HTF bias from daily signals ──────────────────────────────────────────────

def _htf_bias(sig: dict) -> str:
    if not sig:
        return "NEUTRAL"
    ema_aln = sig.get("ema_aligned", 0)
    rsi     = sig.get("rsi", 50.0)
    adx     = sig.get("adx", 20.0)
    e20s    = sig.get("ema20_slope", 0.0)

    bull_points = (
        int(ema_aln == 1) * 3 +
        int(rsi > 55) * 2 +
        int(e20s > 0.05) * 2 +
        int(sig.get("px_above_200", False))
    )
    bear_points = (
        int(ema_aln == -1) * 3 +
        int(rsi < 45) * 2 +
        int(e20s < -0.05) * 2 +
        int(not sig.get("px_above_200", True))
    )
    if bull_points >= 5:   return "BULL"
    if bear_points >= 5:   return "BEAR"
    return "NEUTRAL"


# ─── Multi-timeframe regime blend ─────────────────────────────────────────────

def _blend_regimes(results: dict) -> tuple:
    """
    results: {tf: (regime, confidence, sig_dict)}
    Weights: 1d=3, 4h=2, 1h=1
    Returns (regime, blended_confidence, htf_bias, combined_sig)
    """
    TF_WEIGHT = {"1d": 3, "4h": 2, "1h": 1, "1w": 4, "15m": 0.5, "5m": 0.3}

    # Highest-timeframe bias
    for htf in ["1w", "1d", "4h"]:
        if htf in results:
            htf_sig  = results[htf][2]
            htf_bias = _htf_bias(htf_sig)
            break
    else:
        htf_bias = "NEUTRAL"

    if not results:
        return "RANGING", 40.0, "NEUTRAL", {}

    regime_votes: dict = {}
    for tf, (regime, conf, sig) in results.items():
        w = TF_WEIGHT.get(tf, 1)
        regime_votes[regime] = regime_votes.get(regime, 0.0) + conf * w

    best_regime = max(regime_votes, key=lambda r: regime_votes[r])
    total_weight = sum(TF_WEIGHT.get(tf, 1) for tf in results)
    blended_conf = regime_votes[best_regime] / (total_weight * 100) * 100
    blended_conf = min(max(blended_conf, 20.0), 96.0)

    # Pick representative signals (prefer 1d > 4h > 1h)
    combined_sig = {}
    for tf in ["1d", "4h", "1h", "1w", "15m", "5m"]:
        if tf in results:
            combined_sig = results[tf][2]
            break

    return best_regime, blended_conf, htf_bias, combined_sig


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_regime_fast(df: pd.DataFrame, tf: str = "1d") -> RegimeResult:
    """Fast single-df regime detection."""
    if df is None or len(df) < 20:
        return RegimeResult()

    try:
        sig      = _extract_signals(df)
        regime, conf = _signals_to_regime(sig)
        htf_bias = _htf_bias(sig)

        px      = sig.get("px", 1.0)
        atr_pct = sig.get("atr_pct", 2.0)
        atr_rank= sig.get("atr_rank", 50.0)
        adx     = sig.get("adx", 20.0)
        rsi     = sig.get("rsi", 50.0)
        struct  = sig.get("structure", "RANGING")
        ema_aln = sig.get("ema_aligned", 0)

        return RegimeResult(
            regime           = regime,
            confidence       = round(conf, 1),
            volatility_pct   = round(atr_pct, 3),
            trend_strength   = round(adx, 1),
            momentum         = round((rsi - 50) / 50.0, 3),
            vol_regime       = _vol_regime_label(atr_rank),
            market_structure = struct,
            htf_bias         = htf_bias,
            adx              = round(adx, 1),
            rsi              = round(rsi, 1),
            atr_pct          = round(atr_pct, 3),
            ema_aligned      = ema_aln,
            signals_raw      = {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in sig.items()},
        )
    except Exception:
        return RegimeResult()


def detect_regime_full(tf_dfs: Dict[str, pd.DataFrame]) -> RegimeResult:
    """
    Multi-timeframe regime detection.
    tf_dfs: dict of {tf_string: DataFrame}  e.g. {"1d": df1d, "4h": df4h, "1h": df1h}
    """
    if not tf_dfs:
        return RegimeResult()

    results = {}
    for tf, df in tf_dfs.items():
        if df is None or len(df) < 20:
            continue
        try:
            sig    = _extract_signals(df)
            regime, conf = _signals_to_regime(sig)
            results[tf] = (regime, conf, sig)
        except Exception:
            continue

    if not results:
        return RegimeResult()

    regime, conf, htf_bias, sig = _blend_regimes(results)

    px       = sig.get("px", 1.0)
    atr_pct  = sig.get("atr_pct", 2.0)
    atr_rank = sig.get("atr_rank", 50.0)
    adx      = sig.get("adx", 20.0)
    rsi      = sig.get("rsi", 50.0)
    struct   = sig.get("structure", "RANGING")
    ema_aln  = sig.get("ema_aligned", 0)

    return RegimeResult(
        regime           = regime,
        confidence       = round(conf, 1),
        volatility_pct   = round(atr_pct, 3),
        trend_strength   = round(adx, 1),
        momentum         = round((rsi - 50) / 50.0, 3),
        vol_regime       = _vol_regime_label(atr_rank),
        market_structure = struct,
        htf_bias         = htf_bias,
        adx              = round(adx, 1),
        rsi              = round(rsi, 1),
        atr_pct          = round(atr_pct, 3),
        ema_aligned      = ema_aln,
        signals_raw      = {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in sig.items()},
    )


def get_strategy_fit(regime_result: RegimeResult) -> Dict[str, float]:
    """
    Returns weights (0.0–2.0) for each strategy family given the regime.
    Families: trend_following, mean_reversion, breakout, momentum, volume, neutral
    """
    base = _STRATEGY_FIT.get(regime_result.regime, _STRATEGY_FIT["RANGING"]).copy()

    # Confidence scaling: low confidence → blend toward neutral (1.0)
    conf_scale = regime_result.confidence / 100.0
    for k in base:
        base[k] = 1.0 + (base[k] - 1.0) * conf_scale

    # Volatility adjustments
    if regime_result.vol_regime == "EXTREME":
        base["breakout"]      = min(base["breakout"] * 1.3, 2.0)
        base["trend_following"] = max(base["trend_following"] * 0.7, 0.3)
    elif regime_result.vol_regime == "LOW":
        base["mean_reversion"] = min(base["mean_reversion"] * 1.2, 2.0)
        base["breakout"]       = max(base["breakout"] * 0.8, 0.3)

    return {k: round(v, 3) for k, v in base.items()}


# ─── Simple self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    t = np.arange(n)
    price = 2000 + t * 2.0 + np.cumsum(np.random.randn(n) * 30)
    df = pd.DataFrame({
        "open":   price - np.abs(np.random.randn(n) * 5),
        "high":   price + np.abs(np.random.randn(n) * 15),
        "low":    price - np.abs(np.random.randn(n) * 15),
        "close":  price,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    r = detect_regime_fast(df, "1d")
    print(f"Regime: {r.regime}  conf={r.confidence}%  vol={r.vol_regime}  htf={r.htf_bias}")
    fit = get_strategy_fit(r)
    print("Strategy fit:", fit)
