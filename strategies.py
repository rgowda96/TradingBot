"""
Strategy definitions.  Each strategy returns a Series of signals:
  +1 = long entry,  -1 = short entry,  0 = flat / exit
"""
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA Trend Follow  (9/21 cross + price above 200 EMA)
# ─────────────────────────────────────────────────────────────────────────────
def ema_trend(df):
    bull = (df["ema_9"] > df["ema_21"]) & (df["close"] > df["ema_200"])
    bear = (df["ema_9"] < df["ema_21"]) & (df["close"] < df["ema_200"])
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 2. RSI Mean Reversion  (oversold / overbought with trend filter)
# ─────────────────────────────────────────────────────────────────────────────
def rsi_reversion(df, ob=70, os=30):
    sig = pd.Series(0, index=df.index)
    sig[(df["rsi_14"] < os) & (df["close"] > df["ema_200"])] = 1
    sig[(df["rsi_14"] > ob) & (df["close"] < df["ema_200"])] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 3. MACD Momentum  (histogram flip + EMA200 trend filter)
# ─────────────────────────────────────────────────────────────────────────────
def macd_momentum(df):
    hist = df["macd_hist"]
    flip_up   = (hist > 0) & (hist.shift(1) <= 0)
    flip_down = (hist < 0) & (hist.shift(1) >= 0)
    sig = pd.Series(0, index=df.index)
    sig[flip_up   & (df["close"] > df["ema_200"])] = 1
    sig[flip_down & (df["close"] < df["ema_200"])] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bollinger Band Breakout  (squeeze release)
# ─────────────────────────────────────────────────────────────────────────────
def bb_breakout(df):
    prev_squeeze = df["squeeze"].shift(1)
    sig = pd.Series(0, index=df.index)
    # Squeeze released, price breaks above upper band
    sig[(~df["squeeze"]) & prev_squeeze & (df["close"] > df["bb_upper_20"])] = 1
    sig[(~df["squeeze"]) & prev_squeeze & (df["close"] < df["bb_lower_20"])] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Supertrend System
# ─────────────────────────────────────────────────────────────────────────────
def supertrend_system(df):
    sig = pd.Series(0, index=df.index)
    sig[df["supertrend_dir"] == 1] = 1
    sig[df["supertrend_dir"] == -1] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stoch RSI Crossover (K crosses D in extreme zones)
# ─────────────────────────────────────────────────────────────────────────────
def stoch_rsi_cross(df):
    k, d = df["stoch_rsi_k"], df["stoch_rsi_d"]
    cross_up   = (k > d) & (k.shift(1) <= d.shift(1))
    cross_down = (k < d) & (k.shift(1) >= d.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (d < 25)] = 1
    sig[cross_down & (d > 75)] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Engulfing Candle  (price-action reversal)
# ─────────────────────────────────────────────────────────────────────────────
def engulf_reversal(df):
    sig = pd.Series(0, index=df.index)
    sig[df["engulf_bull"] & (df["rsi_14"] < 45)] = 1
    sig[df["engulf_bear"] & (df["rsi_14"] > 55)] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Volume Breakout  (volume spike + directional candle)
# ─────────────────────────────────────────────────────────────────────────────
def volume_breakout(df, vol_mult=2.0):
    big_vol = df["volume_ratio"] > vol_mult
    bull_candle = df["close"] > df["open"]
    bear_candle = df["close"] < df["open"]
    sig = pd.Series(0, index=df.index)
    sig[big_vol & bull_candle & (df["close"] > df["ema_21"])] = 1
    sig[big_vol & bear_candle & (df["close"] < df["ema_21"])] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Golden / Death Cross  (50/200 EMA cross)
# ─────────────────────────────────────────────────────────────────────────────
def golden_death_cross(df):
    sig = pd.Series(0, index=df.index)
    sig[df["golden_cross"] == 1] = 1
    sig[df["death_cross"]  == 1] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 10. OBV Divergence  (price vs OBV trend disagreement at extremes)
# ─────────────────────────────────────────────────────────────────────────────
def obv_divergence(df, lookback=14):
    c_lo = df["close"].rolling(lookback).apply(lambda x: x.argmin())
    obv_lo = df["obv"].rolling(lookback).apply(lambda x: x.argmin())
    c_hi = df["close"].rolling(lookback).apply(lambda x: x.argmax())
    obv_hi = df["obv"].rolling(lookback).apply(lambda x: x.argmax())

    # Bullish divergence: price makes lower low, OBV makes higher low
    bull_div = (c_lo == lookback - 1) & (obv_lo != lookback - 1)
    # Bearish divergence: price makes higher high, OBV makes lower high
    bear_div = (c_hi == lookback - 1) & (obv_hi != lookback - 1)

    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = {
    "ema_trend":         ema_trend,
    "rsi_reversion":     rsi_reversion,
    "macd_momentum":     macd_momentum,
    "bb_breakout":       bb_breakout,
    "supertrend":        supertrend_system,
    "stoch_rsi_cross":   stoch_rsi_cross,
    "engulf_reversal":   engulf_reversal,
    "volume_breakout":   volume_breakout,
    "golden_death_cross": golden_death_cross,
    "obv_divergence":    obv_divergence,
}
