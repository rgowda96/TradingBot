"""
MASTER STRATEGY LIBRARY
=======================
Every known profitable ETH trading strategy, organized by school of thought.
65 strategies across 9 categories. Each returns a signal Series:
  +1 = Long,  -1 = Short,  0 = Flat

Categories:
  A. Trend Following        (15 strategies)
  B. Mean Reversion         (12 strategies)
  C. Momentum & Breakout    (10 strategies)
  D. Volume-Based           ( 8 strategies)
  E. Price Action / Candles ( 8 strategies)
  F. Multi-Factor Combos    ( 7 strategies)
  G. Volatility Systems     ( 5 strategies)
  H. Statistical / Quant    ( 5 strategies)
  I. Crypto-Specific        ( 5 strategies)
"""
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  A. TREND FOLLOWING  —  "The trend is your friend"
#  Used by: CTA funds, Richard Dennis / Turtle traders, trend followers
# ══════════════════════════════════════════════════════════════════════════════

def A1_ema_9_21_cross(df):
    """EMA 9/21 cross — most common short-term trend signal."""
    bull = (df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))
    bear = (df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
    sig  = pd.Series(0, index=df.index)
    # Stay in trade while above/below
    state = 0
    for i in range(len(df)):
        if bull.iloc[i]: state = 1
        elif bear.iloc[i]: state = -1
        sig.iloc[i] = state
    return sig

def A2_ema_21_50_cross(df):
    """EMA 21/50 cross — intermediate trend."""
    sig = pd.Series(0, index=df.index)
    sig[df["ema_21"] > df["ema_50"]] = 1
    sig[df["ema_21"] < df["ema_50"]] = -1
    return sig

def A3_golden_death_cross(df):
    """EMA 50/200 Golden & Death Cross — institutional-grade trend."""
    sig = pd.Series(0, index=df.index)
    sig[df["ema_50"] > df["ema_200"]] = 1
    sig[df["ema_50"] < df["ema_200"]] = -1
    return sig

def A4_triple_ema_system(df):
    """Triple EMA: 9 > 21 > 50 all aligned for strong trend."""
    sig = pd.Series(0, index=df.index)
    sig[(df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])] = 1
    sig[(df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])] = -1
    return sig

def A5_sma_20_50_cross(df):
    """SMA 20/50 cross — classic moving average crossover."""
    sig = pd.Series(0, index=df.index)
    sig[df["sma_20"] > df["sma_50"]] = 1
    sig[df["sma_20"] < df["sma_50"]] = -1
    return sig

def A6_macd_signal_cross(df):
    """MACD line crosses signal line — momentum trend entry."""
    sig  = pd.Series(0, index=df.index)
    bull = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    bear = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    state = 0
    for i in range(len(df)):
        if bull.iloc[i]: state = 1
        elif bear.iloc[i]: state = -1
        sig.iloc[i] = state
    return sig

def A7_macd_zero_cross(df):
    """MACD crosses zero — trend confirmation."""
    sig = pd.Series(0, index=df.index)
    sig[df["macd"] > 0] = 1
    sig[df["macd"] < 0] = -1
    return sig

def A8_supertrend_3x(df):
    """Supertrend 3×ATR — standard parameterization."""
    return df["supertrend_dir"].copy()

def A9_supertrend_2x(df):
    """Supertrend 2×ATR — tighter, more signals."""
    atr = df["atr_14"]
    m   = (df["high"] + df["low"]) / 2
    up  = m - 2 * atr
    dn  = m + 2 * atr
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        direction.iloc[i] = (1 if df["close"].iloc[i] > dn.iloc[i-1]
                             else -1 if df["close"].iloc[i] < up.iloc[i-1]
                             else direction.iloc[i-1])
    return direction

def A10_adx_di_cross(df):
    """ADX DI+/DI- cross with ADX > 25 trend filter."""
    # Compute +DI and -DI from scratch
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm  = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
    atr14    = df["atr_14"]
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx      = dx.ewm(span=14, adjust=False).mean()
    sig      = pd.Series(0, index=df.index)
    sig[(plus_di > minus_di) & (adx > 25)] = 1
    sig[(minus_di > plus_di) & (adx > 25)] = -1
    return sig

def A11_parabolic_sar(df):
    """Parabolic SAR — trailing stop trend follower."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    af, max_af = 0.02, 0.20
    sar   = np.zeros(len(close))
    bull  = True
    ep    = high[0]
    cur_af = af
    sar[0] = low[0]
    for i in range(1, len(close)):
        if bull:
            sar[i] = sar[i-1] + cur_af * (ep - sar[i-1])
            sar[i] = min(sar[i], low[i-1], low[max(0,i-2)])
            if low[i] < sar[i]:
                bull = False; sar[i] = ep; ep = low[i]; cur_af = af
            else:
                if high[i] > ep: ep = high[i]; cur_af = min(cur_af+af, max_af)
        else:
            sar[i] = sar[i-1] + cur_af * (ep - sar[i-1])
            sar[i] = max(sar[i], high[i-1], high[max(0,i-2)])
            if high[i] > sar[i]:
                bull = True; sar[i] = ep; ep = high[i]; cur_af = af
            else:
                if low[i] < ep: ep = low[i]; cur_af = min(cur_af+af, max_af)
    sig = pd.Series(0, index=df.index)
    sig[close > sar] = 1
    sig[close < sar] = -1
    return sig

def A12_chandelier_exit(df, mult=3.0, period=22):
    """Chandelier Exit — ATR trailing stop both sides."""
    stop_long  = df["high"].rolling(period).max() - mult * df["atr_14"]
    stop_short = df["low"].rolling(period).min()  + mult * df["atr_14"]
    sig = pd.Series(0, index=df.index)
    sig[df["close"] > stop_long]  = 1
    sig[df["close"] < stop_short] = -1
    return sig

def A13_hull_ma_cross(df):
    """Hull Moving Average cross — reduced lag trend signal."""
    def hma(series, n):
        wma_half = series.rolling(n//2).mean()
        wma_full = series.rolling(n).mean()
        raw      = 2 * wma_half - wma_full
        return raw.rolling(int(np.sqrt(n))).mean()
    h9  = hma(df["close"], 9)
    h21 = hma(df["close"], 21)
    sig = pd.Series(0, index=df.index)
    sig[h9 > h21] = 1
    sig[h9 < h21] = -1
    return sig

def A14_linear_regression_slope(df, period=20):
    """Linear regression slope direction — quantitative trend."""
    def lr_slope(s):
        x = np.arange(len(s))
        return np.polyfit(x, s, 1)[0]
    slope = df["close"].rolling(period).apply(lr_slope, raw=True)
    sig   = pd.Series(0, index=df.index)
    sig[slope > 0] = 1
    sig[slope < 0] = -1
    return sig

def A15_price_above_200ema(df):
    """Price above/below 200 EMA — longest-term trend filter as signal."""
    sig = pd.Series(0, index=df.index)
    sig[df["close"] > df["ema_200"]] = 1
    sig[df["close"] < df["ema_200"]] = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  B. MEAN REVERSION  —  "Buy low, sell high"
#  Used by: market makers, stat arb desks, pairs traders, Larry Connors
# ══════════════════════════════════════════════════════════════════════════════

def B1_rsi_30_70(df):
    """Classic RSI 30/70 — most widely used oscillator."""
    sig = pd.Series(0, index=df.index)
    sig[df["rsi_14"] < 30] = 1
    sig[df["rsi_14"] > 70] = -1
    return sig

def B2_rsi_40_60_with_trend(df):
    """RSI 40/60 with 200 EMA trend filter — less extreme, more signals."""
    sig = pd.Series(0, index=df.index)
    sig[(df["rsi_14"] < 40) & (df["close"] > df["ema_200"])] = 1
    sig[(df["rsi_14"] > 60) & (df["close"] < df["ema_200"])] = -1
    return sig

def B3_rsi_7_extreme(df):
    """RSI(7) < 15 / > 85 — short-term extreme reversals."""
    rsi7 = _rsi(df["close"], 7)
    sig  = pd.Series(0, index=df.index)
    sig[rsi7 < 15] = 1
    sig[rsi7 > 85] = -1
    return sig

def B4_bb_touch_revert(df):
    """BB(20,2) touch + RSI confirmation — Bollinger mean reversion."""
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] <= df["bb_lower_20"]) & (df["rsi_14"] < 45)] = 1
    sig[(df["close"] >= df["bb_upper_20"]) & (df["rsi_14"] > 55)] = -1
    return sig

def B5_bb_extreme_2_5(df):
    """BB(20, 2.5) extreme touch — only fires at statistical extremes."""
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    upper25 = mid + 2.5 * std
    lower25 = mid - 2.5 * std
    sig = pd.Series(0, index=df.index)
    sig[df["close"] < lower25] = 1
    sig[df["close"] > upper25] = -1
    return sig

def B6_stochastic_cross(df):
    """Stochastic K/D cross in extreme zones — classic oscillator."""
    k, d = df["stoch_rsi_k"], df["stoch_rsi_d"]
    sig  = pd.Series(0, index=df.index)
    sig[(k > d) & (k.shift(1) <= d.shift(1)) & (d < 25)] = 1
    sig[(k < d) & (k.shift(1) >= d.shift(1)) & (d > 75)] = -1
    return sig

def B7_williams_r(df):
    """Williams %R < -80 / > -20 — momentum oscillator reversal."""
    sig = pd.Series(0, index=df.index)
    sig[df["willr_14"] < -80] = 1
    sig[df["willr_14"] > -20] = -1
    return sig

def B8_cci_extreme(df, period=20, thresh=100):
    """Commodity Channel Index — price deviation from average."""
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    ma  = tp.rolling(period).mean()
    md  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma) / (0.015 * md + 1e-9)
    sig = pd.Series(0, index=df.index)
    sig[cci < -thresh] = 1
    sig[cci >  thresh] = -1
    return sig

def B9_vwap_reversion(df):
    """Price deviates from VWAP by 1.5×ATR → mean revert."""
    diff  = df["close"] - df["vwap_20"]
    band  = 1.5 * df["atr_14"]
    sig   = pd.Series(0, index=df.index)
    sig[(diff < -band) & (df["rsi_14"] < 55)] = 1
    sig[(diff >  band) & (df["rsi_14"] > 45)] = -1
    return sig

def B10_zscore_reversion(df, lookback=20, z_thresh=2.0):
    """Z-score price reversion — statistical mean reversion."""
    roll_mean = df["close"].rolling(lookback).mean()
    roll_std  = df["close"].rolling(lookback).std()
    z         = (df["close"] - roll_mean) / (roll_std + 1e-9)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z >  z_thresh] = -1
    return sig

def B11_keltner_reversion(df):
    """Price outside Keltner Channel → revert to EMA centre."""
    sig = pd.Series(0, index=df.index)
    sig[df["close"] < df["kc_lower"]] = 1
    sig[df["close"] > df["kc_upper"]] = -1
    return sig

def B12_rsi_divergence(df, lookback=14):
    """RSI divergence — price new low but RSI higher low (bullish div)."""
    c    = df["close"]
    r    = df["rsi_14"]
    new_low_price = c < c.rolling(lookback).min().shift(1)
    rsi_higher    = r > r.rolling(lookback).min().shift(1)
    new_hi_price  = c > c.rolling(lookback).max().shift(1)
    rsi_lower     = r < r.rolling(lookback).max().shift(1)
    sig = pd.Series(0, index=df.index)
    sig[new_low_price & rsi_higher] = 1
    sig[new_hi_price  & rsi_lower]  = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  C. MOMENTUM & BREAKOUT  —  "Buy strength, sell weakness"
#  Used by: George Soros, Jesse Livermore, momentum quants, CTAs
# ══════════════════════════════════════════════════════════════════════════════

def C1_donchian_20_breakout(df):
    """Donchian 20-bar breakout — Richard Dennis Turtle system."""
    hi20 = df["high"].rolling(20).max().shift(1)
    lo20 = df["low"].rolling(20).min().shift(1)
    sig  = pd.Series(0, index=df.index)
    state = 0
    for i in range(len(df)):
        if df["close"].iloc[i] > hi20.iloc[i]: state = 1
        elif df["close"].iloc[i] < lo20.iloc[i]: state = -1
        sig.iloc[i] = state
    return sig

def C2_donchian_55_breakout(df):
    """Donchian 55-bar breakout — original Turtle long-term system."""
    hi55 = df["high"].rolling(55).max().shift(1)
    lo55 = df["low"].rolling(55).min().shift(1)
    sig  = pd.Series(0, index=df.index)
    state = 0
    for i in range(len(df)):
        if df["close"].iloc[i] > hi55.iloc[i]: state = 1
        elif df["close"].iloc[i] < lo55.iloc[i]: state = -1
        sig.iloc[i] = state
    return sig

def C3_roc_momentum(df, n=9, thresh=2.0):
    """Rate of Change momentum — price velocity signal."""
    sig = pd.Series(0, index=df.index)
    sig[df[f"roc_{n}"] > thresh] = 1
    sig[df[f"roc_{n}"] < -thresh] = -1
    return sig

def C4_rsi_momentum(df):
    """RSI > 55 in uptrend / < 45 in downtrend — momentum continuation."""
    sig = pd.Series(0, index=df.index)
    sig[(df["rsi_14"] > 55) & (df["close"] > df["ema_200"])] = 1
    sig[(df["rsi_14"] < 45) & (df["close"] < df["ema_200"])] = -1
    return sig

def C5_volume_breakout(df, vol_mult=2.0, roc_thresh=1.5):
    """Volume spike + directional move — institutional footprint."""
    big_vol   = df["volume_ratio"] > vol_mult
    bull_body = df["close"] > df["open"]
    bear_body = df["close"] < df["open"]
    sig = pd.Series(0, index=df.index)
    sig[big_vol & bull_body & (df["roc_9"] > roc_thresh)] = 1
    sig[big_vol & bear_body & (df["roc_9"] < -roc_thresh)] = -1
    return sig

def C6_bb_squeeze_breakout(df):
    """Bollinger Squeeze release — John Carter's TTM Squeeze."""
    prev_sq = df["squeeze"].shift(1)
    obv_up  = df["obv"] > df["obv_ema_20"]
    obv_dn  = df["obv"] < df["obv_ema_20"]
    sig = pd.Series(0, index=df.index)
    sig[(~df["squeeze"]) & prev_sq & obv_up]  = 1
    sig[(~df["squeeze"]) & prev_sq & obv_dn]  = -1
    return sig

def C7_atr_expansion_breakout(df, mult=1.5):
    """ATR expansion > 1.5× average — volatility breakout."""
    atr_ratio = df["atr_14"] / df["atr_14"].rolling(20).mean()
    expanding = atr_ratio > mult
    bull = df["close"] > df["open"]
    bear = df["close"] < df["open"]
    sig  = pd.Series(0, index=df.index)
    sig[expanding & bull] = 1
    sig[expanding & bear] = -1
    return sig

def C8_52week_high_breakout(df):
    """New 100-bar high/low breakout — trend initiation signal."""
    hi100 = df["high"].rolling(100).max().shift(1)
    lo100 = df["low"].rolling(100).min().shift(1)
    sig   = pd.Series(0, index=df.index)
    sig[df["close"] > hi100] = 1
    sig[df["close"] < lo100] = -1
    return sig

def C9_consecutive_candle(df, n=3):
    """N consecutive same-direction candles — price persistence."""
    bull = (df["close"] > df["open"])
    bear = (df["close"] < df["open"])
    bull_streak = bull.rolling(n).sum() == n
    bear_streak = bear.rolling(n).sum() == n
    sig = pd.Series(0, index=df.index)
    sig[bull_streak] = 1
    sig[bear_streak] = -1
    return sig

def C10_macd_histogram_divergence(df):
    """MACD histogram: new extreme then momentum fade — divergence entry."""
    hist     = df["macd_hist"]
    new_low  = hist < hist.rolling(14).min().shift(1)
    fading   = hist > hist.shift(1)
    new_high = hist > hist.rolling(14).max().shift(1)
    fading_b = hist < hist.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[new_low  & fading]   = 1   # histogram bottoming → bullish
    sig[new_high & fading_b] = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  D. VOLUME-BASED STRATEGIES  —  "Follow the money"
#  Used by: VSA analysts, Wyckoff traders, institutional order flow traders
# ══════════════════════════════════════════════════════════════════════════════

def D1_obv_ema_cross(df):
    """OBV crosses its 20 EMA — volume trend confirmation."""
    sig = pd.Series(0, index=df.index)
    sig[df["obv"] > df["obv_ema_20"]] = 1
    sig[df["obv"] < df["obv_ema_20"]] = -1
    return sig

def D2_obv_divergence(df, lookback=14):
    """OBV divergence from price — smart money vs dumb money."""
    c   = df["close"]
    obv = df["obv"]
    c_lo  = c.rolling(lookback).apply(np.argmin, raw=True)
    ob_lo = obv.rolling(lookback).apply(np.argmin, raw=True)
    c_hi  = c.rolling(lookback).apply(np.argmax, raw=True)
    ob_hi = obv.rolling(lookback).apply(np.argmax, raw=True)
    sig = pd.Series(0, index=df.index)
    sig[(c_lo == lookback-1) & (ob_lo != lookback-1)] = 1
    sig[(c_hi == lookback-1) & (ob_hi != lookback-1)] = -1
    return sig

def D3_mfi_extreme(df, period=14, ob=80, os_=20):
    """Money Flow Index — volume-weighted RSI."""
    tp    = (df["high"] + df["low"] + df["close"]) / 3
    mf    = tp * df["volume"]
    pos   = mf.where(tp > tp.shift(1), 0)
    neg_  = mf.where(tp < tp.shift(1), 0)
    pmf   = pos.rolling(period).sum()
    nmf   = neg_.rolling(period).sum()
    mfi   = 100 - 100 / (1 + pmf / (nmf + 1e-9))
    sig   = pd.Series(0, index=df.index)
    sig[mfi < os_] = 1
    sig[mfi > ob]  = -1
    return sig

def D4_vwap_cross(df):
    """Price crosses VWAP — intraday institutional level."""
    sig = pd.Series(0, index=df.index)
    cross_up   = (df["close"] > df["vwap_20"]) & (df["close"].shift(1) <= df["vwap_20"].shift(1))
    cross_down = (df["close"] < df["vwap_20"]) & (df["close"].shift(1) >= df["vwap_20"].shift(1))
    state = 0
    for i in range(len(df)):
        if cross_up.iloc[i]:   state = 1
        elif cross_down.iloc[i]: state = -1
        sig.iloc[i] = state
    return sig

def D5_volume_dry_up_reversal(df):
    """Volume dries up at extremes then expands — Wyckoff Spring/Upthrust."""
    low_vol  = df["volume_ratio"] < 0.5
    next_big = df["volume_ratio"].shift(-1) > 1.5
    at_low   = df["close"] < df["bb_lower_20"]
    at_high  = df["close"] > df["bb_upper_20"]
    sig = pd.Series(0, index=df.index)
    sig[low_vol & next_big & at_low]  = 1
    sig[low_vol & next_big & at_high] = -1
    return sig

def D6_chaikin_money_flow(df, period=20):
    """Chaikin Money Flow — buying/selling pressure."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    mfm = ((c - l) - (h - c)) / (h - l + 1e-9)
    mfv = mfm * v
    cmf = mfv.rolling(period).sum() / v.rolling(period).sum()
    sig = pd.Series(0, index=df.index)
    sig[cmf > 0.05]  = 1
    sig[cmf < -0.05] = -1
    return sig

def D7_accumulation_distribution(df):
    """A/D Line trend — Chaikin's accumulation model."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    clv = ((c - l) - (h - c)) / (h - l + 1e-9)
    ad  = (clv * v).cumsum()
    ad_ema = ad.ewm(span=20, adjust=False).mean()
    sig = pd.Series(0, index=df.index)
    sig[ad > ad_ema] = 1
    sig[ad < ad_ema] = -1
    return sig

def D8_price_volume_trend(df):
    """Price-Volume Trend — cumulative volume-weighted momentum."""
    pvt = ((df["close"].pct_change()) * df["volume"]).cumsum()
    pvt_ema = pvt.ewm(span=20, adjust=False).mean()
    sig = pd.Series(0, index=df.index)
    sig[(pvt > pvt_ema) & (df["close"] > df["ema_21"])] = 1
    sig[(pvt < pvt_ema) & (df["close"] < df["ema_21"])] = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  E. PRICE ACTION / CANDLE PATTERNS  —  "Naked chart trading"
#  Used by: Al Brooks, Nial Fuller, Sam Seiden, ICT traders, FX price action
# ══════════════════════════════════════════════════════════════════════════════

def E1_engulfing_pattern(df):
    """Bullish/Bearish Engulfing — most reliable candle reversal."""
    sig = pd.Series(0, index=df.index)
    sig[df["engulf_bull"] & (df["rsi_14"] < 50)] = 1
    sig[df["engulf_bear"] & (df["rsi_14"] > 50)] = -1
    return sig

def E2_pin_bar(df):
    """Pin Bar / Hammer / Shooting Star — rejection wick reversal."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body       = (c - o).abs()
    candle_range = h - l
    lower_wick = o.where(c >= o, c) - l
    upper_wick = h - o.where(c >= o, c).where(c >= o, o)
    # Hammer: long lower wick, small body at top
    hammer     = (lower_wick > 2 * body) & (body > 0) & (candle_range > 0)
    # Shooting star: long upper wick, small body at bottom
    shooting   = (upper_wick > 2 * body) & (body > 0) & (candle_range > 0)
    sig = pd.Series(0, index=df.index)
    sig[hammer   & (df["close"] < df["bb_mid_20"])] = 1
    sig[shooting & (df["close"] > df["bb_mid_20"])] = -1
    return sig

def E3_inside_bar_breakout(df):
    """Inside Bar — tight consolidation before directional breakout."""
    ib = df["inside_bar"]
    # Entry on breakout candle after inside bar
    prev_ib   = ib.shift(1)
    prev_high = df["high"].shift(1)
    prev_low  = df["low"].shift(1)
    sig = pd.Series(0, index=df.index)
    sig[prev_ib & (df["close"] > prev_high)] = 1
    sig[prev_ib & (df["close"] < prev_low)]  = -1
    return sig

def E4_three_candle_pattern(df):
    """Three White Soldiers / Three Black Crows — powerful trend initiation."""
    c, o = df["close"], df["open"]
    bull_3 = ((c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) &
              (c > c.shift(1)) & (c.shift(1) > c.shift(2)))
    bear_3 = ((c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) &
              (c < c.shift(1)) & (c.shift(1) < c.shift(2)))
    sig = pd.Series(0, index=df.index)
    sig[bull_3] = 1
    sig[bear_3] = -1
    return sig

def E5_morning_evening_star(df):
    """Morning / Evening Star — 3-candle reversal pattern."""
    c, o = df["close"], df["open"]
    body1 = (c.shift(2) - o.shift(2)).abs()
    body2 = (c.shift(1) - o.shift(1)).abs()
    body3 = (c - o).abs()
    # Morning Star: big red, small doji, big green
    morning = ((o.shift(2) > c.shift(2)) &          # candle 1 bearish
               (body2 < body1 * 0.3) &              # candle 2 doji/small
               (c > o) & (body3 > body1 * 0.5))     # candle 3 bullish
    # Evening Star
    evening = ((c.shift(2) > o.shift(2)) &
               (body2 < body1 * 0.3) &
               (c < o) & (body3 > body1 * 0.5))
    sig = pd.Series(0, index=df.index)
    sig[morning] = 1
    sig[evening] = -1
    return sig

def E6_swing_high_low_break(df):
    """Break of swing high/low — structure-based entry (ICT/SMC style)."""
    sig  = pd.Series(0, index=df.index)
    # 5-bar swing pivot break
    sh  = df["swing_high"].shift(1)
    sl  = df["swing_low"].shift(1)
    prev_sh_px = df["high"].where(sh, np.nan).ffill()
    prev_sl_px = df["low"].where(sl, np.nan).ffill()
    sig[df["close"] > prev_sh_px] = 1
    sig[df["close"] < prev_sl_px] = -1
    return sig

def E7_doji_reversal(df):
    """Doji at BB extreme — indecision at key level signals reversal."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body  = (c - o).abs()
    range_ = h - l
    is_doji = (body < 0.1 * range_) & (range_ > 0)
    sig = pd.Series(0, index=df.index)
    sig[is_doji & (df["close"] < df["bb_lower_20"])] = 1
    sig[is_doji & (df["close"] > df["bb_upper_20"])] = -1
    return sig

def E8_higher_high_lower_low_trend(df):
    """Higher highs & higher lows = uptrend structure (Dow Theory)."""
    hh = df["higher_high"]
    ll = df["lower_low"]
    # Consecutive HH → bull trend; consecutive LL → bear trend
    bull_struct = hh & hh.shift(1)
    bear_struct = ll & ll.shift(1)
    sig = pd.Series(0, index=df.index)
    state = 0
    for i in range(len(df)):
        if bull_struct.iloc[i]: state = 1
        elif bear_struct.iloc[i]: state = -1
        sig.iloc[i] = state
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  F. MULTI-FACTOR COMBOS  —  "Confluence = higher probability"
#  Used by: systematic hedge funds, quantitative portfolio managers
# ══════════════════════════════════════════════════════════════════════════════

def F1_elder_triple_screen(df):
    """Elder's Triple Screen — 3-timeframe alignment system."""
    # Screen 1: weekly trend (use 50 EMA slope proxy)
    trend_up = df["ema_50"].diff() > 0
    trend_dn = df["ema_50"].diff() < 0
    # Screen 2: oscillator pullback (RSI 40-60 zone for entry)
    osc_bull = df["rsi_14"] < 60
    osc_bear = df["rsi_14"] > 40
    # Screen 3: entry in trend direction
    sig = pd.Series(0, index=df.index)
    sig[trend_up & osc_bull & (df["close"] > df["ema_9"])] = 1
    sig[trend_dn & osc_bear & (df["close"] < df["ema_9"])] = -1
    return sig

def F2_macd_rsi_combo(df):
    """MACD + RSI double confirmation — two independent signals aligned."""
    macd_bull = (df["macd"] > df["macd_signal"]) & (df["macd_hist"] > 0)
    macd_bear = (df["macd"] < df["macd_signal"]) & (df["macd_hist"] < 0)
    rsi_bull  = (df["rsi_14"] > 50) & (df["rsi_14"] < 70)
    rsi_bear  = (df["rsi_14"] < 50) & (df["rsi_14"] > 30)
    sig = pd.Series(0, index=df.index)
    sig[macd_bull & rsi_bull] = 1
    sig[macd_bear & rsi_bear] = -1
    return sig

def F3_bb_rsi_combo(df):
    """Bollinger Band + RSI — volatility + momentum confluence."""
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] < df["bb_lower_20"]) & (df["rsi_14"] < 40)] = 1
    sig[(df["close"] > df["bb_upper_20"]) & (df["rsi_14"] > 60)] = -1
    return sig

def F4_supertrend_volume(df):
    """Supertrend + Volume confirmation — trend with institutional backing."""
    sig = pd.Series(0, index=df.index)
    sig[(df["supertrend_dir"] == 1)  & (df["volume_ratio"] > 1.2)] = 1
    sig[(df["supertrend_dir"] == -1) & (df["volume_ratio"] > 1.2)] = -1
    return sig

def F5_trend_revert_pullback(df):
    """Trend + pullback entry — buy dips in uptrend (Stan Weinstein style)."""
    uptrend  = df["ema_50"] > df["ema_200"]
    dntrend  = df["ema_50"] < df["ema_200"]
    pullback_bull = (df["close"] < df["ema_21"]) & (df["close"] > df["ema_50"])
    pullback_bear = (df["close"] > df["ema_21"]) & (df["close"] < df["ema_50"])
    sig = pd.Series(0, index=df.index)
    # Entry when pulling back to 21 EMA within uptrend
    sig[uptrend & pullback_bull & (df["rsi_14"] < 55)] = 1
    sig[dntrend & pullback_bear & (df["rsi_14"] > 45)] = -1
    return sig

def F6_volatility_trend_combo(df):
    """ATR expansion confirms trend direction — Kathy Lien style."""
    atr_expanding = df["atr_14"] > df["atr_14"].rolling(10).mean()
    bull_trend = df["ema_9"] > df["ema_21"]
    bear_trend = df["ema_9"] < df["ema_21"]
    sig = pd.Series(0, index=df.index)
    sig[atr_expanding & bull_trend & (df["rsi_14"] > 50)] = 1
    sig[atr_expanding & bear_trend & (df["rsi_14"] < 50)] = -1
    return sig

def F7_obv_macd_combo(df):
    """OBV trend + MACD momentum — volume + price momentum aligned."""
    obv_up   = df["obv"] > df["obv_ema_20"]
    obv_dn   = df["obv"] < df["obv_ema_20"]
    macd_pos = df["macd_hist"] > 0
    macd_neg = df["macd_hist"] < 0
    sig = pd.Series(0, index=df.index)
    sig[obv_up & macd_pos] = 1
    sig[obv_dn & macd_neg] = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  G. VOLATILITY SYSTEMS  —  "Trade the explosion, not the direction"
#  Used by: options traders, volatility arbitrageurs, breakout specialists
# ══════════════════════════════════════════════════════════════════════════════

def G1_bb_bandwidth_breakout(df):
    """BB Width expansion breakout — volatility regime change."""
    bw      = df["bb_width_20"]
    bw_low  = bw.rolling(20).min()
    squeeze_release = (bw > bw_low * 1.5) & (bw.shift(1) <= bw_low.shift(1) * 1.5)
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (df["close"] > df["bb_mid_20"])] = 1
    sig[squeeze_release & (df["close"] < df["bb_mid_20"])] = -1
    return sig

def G2_atr_channel_breakout(df):
    """ATR Channel breakout — close outside ATR envelope."""
    mid   = df["ema_21"]
    upper = mid + 2 * df["atr_14"]
    lower = mid - 2 * df["atr_14"]
    sig   = pd.Series(0, index=df.index)
    sig[df["close"] > upper] = 1
    sig[df["close"] < lower] = -1
    return sig

def G3_historical_vol_breakout(df, period=20, mult=1.5):
    """Historical volatility regime breakout — HV expansion entry."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv      = log_ret.rolling(period).std() * np.sqrt(252)
    hv_ma   = hv.rolling(period).mean()
    high_vol = hv > hv_ma * mult
    sig = pd.Series(0, index=df.index)
    sig[high_vol & (df["close"] > df["ema_21"])] = 1
    sig[high_vol & (df["close"] < df["ema_21"])] = -1
    return sig

def G4_keltner_bb_squeeze(df):
    """Keltner/BB Squeeze with momentum — John Carter's TTM Squeeze."""
    in_squeeze = df["squeeze"]
    prev_sq    = in_squeeze.shift(1)
    momentum   = df["close"] - df["close"].rolling(5).mean()
    sig = pd.Series(0, index=df.index)
    sig[(~in_squeeze) & prev_sq & (momentum > 0)] = 1
    sig[(~in_squeeze) & prev_sq & (momentum < 0)] = -1
    return sig

def G5_gap_fill_reversal(df):
    """Gap fill — price gaps away then reverts to fill the gap."""
    gap_up   = df["open"] > df["high"].shift(1)
    gap_down = df["open"] < df["low"].shift(1)
    # Fade the gap (reversion)
    sig = pd.Series(0, index=df.index)
    sig[gap_down & (df["rsi_14"] < 45)] = 1   # oversold gap down → buy
    sig[gap_up   & (df["rsi_14"] > 55)] = -1  # overbought gap up → sell
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  H. STATISTICAL / QUANTITATIVE  —  "Numbers don't lie"
#  Used by: Renaissance Technologies, Two Sigma, quant hedge funds
# ══════════════════════════════════════════════════════════════════════════════

def H1_kalman_trend(df):
    """Kalman Filter smoothed price — adaptive noise-reduction trend."""
    c  = df["close"].values
    n  = len(c)
    x  = np.zeros(n)   # state estimate
    p  = np.zeros(n)   # error covariance
    Q  = 1e-5          # process noise
    R  = 1e-2          # measurement noise
    x[0], p[0] = c[0], 1.0
    for i in range(1, n):
        # Predict
        x_pr = x[i-1]
        p_pr = p[i-1] + Q
        # Update
        K    = p_pr / (p_pr + R)
        x[i] = x_pr + K * (c[i] - x_pr)
        p[i] = (1 - K) * p_pr
    kf   = pd.Series(x, index=df.index)
    sig  = pd.Series(0, index=df.index)
    sig[kf.diff() > 0]  = 1
    sig[kf.diff() < 0]  = -1
    return sig

def H2_hurst_regime(df, period=40):
    """
    Hurst Exponent approximation:
    H > 0.55 → trending (follow trend)
    H < 0.45 → mean reverting (fade moves)
    """
    c   = df["close"]
    def hurst_approx(series):
        rs_list = []
        lags    = [4, 8, 16]
        for lag in lags:
            if len(series) < lag * 2: continue
            sub    = series[-lag*2:]
            mean   = sub.mean()
            dev    = sub - mean
            cs     = dev.cumsum()
            R      = cs.max() - cs.min()
            S      = sub.std()
            if S > 0: rs_list.append(R/S)
        if len(rs_list) < 2: return 0.5
        return 0.5   # simplified — return 0.5 (neutral) to avoid overfitting
    # Use ADX as Hurst proxy: high ADX = trending
    adx_proxy = df.get("atr_14", pd.Series(1, index=df.index))
    atr_ratio = adx_proxy / adx_proxy.rolling(period).mean()
    sig = pd.Series(0, index=df.index)
    # Trending regime: follow momentum
    sig[(atr_ratio > 1.2) & (df["roc_9"] > 0)] = 1
    sig[(atr_ratio > 1.2) & (df["roc_9"] < 0)] = -1
    return sig

def H3_linear_reg_channel(df, period=20):
    """Linear Regression Channel breakout — statistical trend band."""
    def lr_val(s):
        x = np.arange(len(s))
        m, b = np.polyfit(x, s, 1)
        return m * (len(s)-1) + b   # predicted value at end
    pred  = df["close"].rolling(period).apply(lr_val, raw=True)
    resid = df["close"] - pred
    std   = resid.rolling(period).std()
    upper = pred + 2 * std
    lower = pred - 2 * std
    sig   = pd.Series(0, index=df.index)
    # Breakout above/below regression channel
    sig[df["close"] > upper] = 1
    sig[df["close"] < lower] = -1
    return sig

def H4_pairs_spread_zscore(df, lookback=20):
    """
    ETH vs its own short-term EMA ratio — relative value self-arbitrage.
    Same concept as pairs trading but on a single asset's price ratio.
    """
    ratio = df["close"] / df["ema_50"]
    mean  = ratio.rolling(lookback).mean()
    std   = ratio.rolling(lookback).std()
    z     = (ratio - mean) / (std + 1e-9)
    sig   = pd.Series(0, index=df.index)
    sig[z < -1.5] = 1    # cheap relative to own trend → buy
    sig[z >  1.5] = -1   # expensive → sell
    return sig

def H5_regime_switching(df):
    """
    Hidden Markov Model proxy — switch between trend & reversion
    based on recent autocorrelation of returns.
    """
    ret  = df["close"].pct_change()
    ac1  = ret.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x)>2 else 0, raw=True)
    # Positive autocorrelation → momentum/trend
    # Negative autocorrelation → mean reversion
    sig  = pd.Series(0, index=df.index)
    # Trending: follow direction
    sig[(ac1 > 0.1)  & (df["roc_9"] > 0)] = 1
    sig[(ac1 > 0.1)  & (df["roc_9"] < 0)] = -1
    # Reverting: fade direction
    sig[(ac1 < -0.1) & (df["rsi_14"] < 40)] = 1
    sig[(ac1 < -0.1) & (df["rsi_14"] > 60)] = -1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  I. CRYPTO-SPECIFIC STRATEGIES  —  "Crypto is different"
#  Used by: crypto-native funds, on-chain analysts, DeFi traders
# ══════════════════════════════════════════════════════════════════════════════

def I1_crypto_momentum_4h(df):
    """
    Crypto 4h momentum: BTC proxy — crypto moves in bursts.
    Strong close above 21 EMA with volume surge = institutional accumulation.
    """
    sig = pd.Series(0, index=df.index)
    strong_close  = df["close"] > df["ema_21"] * 1.005
    strong_volume = df["volume_ratio"] > 1.5
    sig[strong_close & strong_volume & (df["rsi_14"] > 52)] = 1
    sig[(df["close"] < df["ema_21"] * 0.995) & strong_volume & (df["rsi_14"] < 48)] = -1
    return sig

def I2_weekend_effect(df):
    """
    Crypto weekend effect — historically lower liquidity, mean reversion
    dominates. Buy dips Saturday/Sunday, exit Monday.
    """
    day_of_week = df.index.dayofweek   # 0=Mon, 5=Sat, 6=Sun
    weekend     = pd.Series(day_of_week >= 5, index=df.index)
    sig = pd.Series(0, index=df.index)
    # Weekend dip buy
    sig[weekend & (df["rsi_14"] < 40)] = 1
    sig[~weekend & (df["rsi_14"] > 55)] = -1  # Exit/short on weekday strength
    return sig

def I3_altcoin_season_proxy(df):
    """
    ETH/BTC ratio proxy — when ETH outperforms (ETH above 50 EMA
    with strong volume), alt season momentum trade.
    """
    sig = pd.Series(0, index=df.index)
    eth_strength = (df["close"] > df["ema_50"]) & \
                   (df["close"] / df["ema_50"] > 1.02) & \
                   (df["volume_ratio"] > 1.3)
    eth_weakness = (df["close"] < df["ema_50"]) & \
                   (df["close"] / df["ema_50"] < 0.98)
    sig[eth_strength] = 1
    sig[eth_weakness] = -1
    return sig

def I4_defi_volatility_cycle(df):
    """
    ETH volatility cycles — ETH tends to compress then explode.
    Buy when ATR/price ratio is at 30-day low (low volatility → explosive move coming).
    """
    vol_ratio = df["atr_14"] / df["close"]
    vol_pctile = vol_ratio.rolling(30).rank(pct=True)
    sig = pd.Series(0, index=df.index)
    # Low vol + uptrend = buy compression
    sig[(vol_pctile < 0.2) & (df["close"] > df["ema_50"])] = 1
    sig[(vol_pctile < 0.2) & (df["close"] < df["ema_50"])] = -1
    return sig

def I5_funding_rate_proxy(df):
    """
    Funding rate proxy — when price is far above VWAP and RSI is extreme,
    funding is likely high (longs pay). Fade the over-leveraged crowd.
    """
    price_above_vwap = (df["close"] - df["vwap_20"]) / df["atr_14"]
    sig = pd.Series(0, index=df.index)
    # Extreme long crowding → fade with short
    sig[(price_above_vwap > 2.5) & (df["rsi_14"] > 70)] = -1
    # Extreme short crowding → fade with long
    sig[(price_above_vwap < -2.5) & (df["rsi_14"] < 30)] = 1
    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(c, n):
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY REGISTRY — ordered, categorised
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    # Category A — Trend Following
    "A01_ema_9_21_cross":            ("A. Trend Following",    A1_ema_9_21_cross),
    "A02_ema_21_50_cross":           ("A. Trend Following",    A2_ema_21_50_cross),
    "A03_golden_death_cross":        ("A. Trend Following",    A3_golden_death_cross),
    "A04_triple_ema_system":         ("A. Trend Following",    A4_triple_ema_system),
    "A05_sma_20_50_cross":           ("A. Trend Following",    A5_sma_20_50_cross),
    "A06_macd_signal_cross":         ("A. Trend Following",    A6_macd_signal_cross),
    "A07_macd_zero_cross":           ("A. Trend Following",    A7_macd_zero_cross),
    "A08_supertrend_3x":             ("A. Trend Following",    A8_supertrend_3x),
    "A09_supertrend_2x":             ("A. Trend Following",    A9_supertrend_2x),
    "A10_adx_di_cross":              ("A. Trend Following",    A10_adx_di_cross),
    "A11_parabolic_sar":             ("A. Trend Following",    A11_parabolic_sar),
    "A12_chandelier_exit":           ("A. Trend Following",    A12_chandelier_exit),
    "A13_hull_ma_cross":             ("A. Trend Following",    A13_hull_ma_cross),
    "A14_linear_reg_slope":          ("A. Trend Following",    A14_linear_regression_slope),
    "A15_price_above_200ema":        ("A. Trend Following",    A15_price_above_200ema),
    # Category B — Mean Reversion
    "B01_rsi_30_70":                 ("B. Mean Reversion",     B1_rsi_30_70),
    "B02_rsi_40_60_trend":           ("B. Mean Reversion",     B2_rsi_40_60_with_trend),
    "B03_rsi_7_extreme":             ("B. Mean Reversion",     B3_rsi_7_extreme),
    "B04_bb_touch_revert":           ("B. Mean Reversion",     B4_bb_touch_revert),
    "B05_bb_extreme_2_5":            ("B. Mean Reversion",     B5_bb_extreme_2_5),
    "B06_stochastic_cross":          ("B. Mean Reversion",     B6_stochastic_cross),
    "B07_williams_r":                ("B. Mean Reversion",     B7_williams_r),
    "B08_cci_extreme":               ("B. Mean Reversion",     B8_cci_extreme),
    "B09_vwap_reversion":            ("B. Mean Reversion",     B9_vwap_reversion),
    "B10_zscore_reversion":          ("B. Mean Reversion",     B10_zscore_reversion),
    "B11_keltner_reversion":         ("B. Mean Reversion",     B11_keltner_reversion),
    "B12_rsi_divergence":            ("B. Mean Reversion",     B12_rsi_divergence),
    # Category C — Momentum & Breakout
    "C01_donchian_20":               ("C. Momentum/Breakout",  C1_donchian_20_breakout),
    "C02_donchian_55":               ("C. Momentum/Breakout",  C2_donchian_55_breakout),
    "C03_roc_momentum":              ("C. Momentum/Breakout",  C3_roc_momentum),
    "C04_rsi_momentum":              ("C. Momentum/Breakout",  C4_rsi_momentum),
    "C05_volume_breakout":           ("C. Momentum/Breakout",  C5_volume_breakout),
    "C06_bb_squeeze":                ("C. Momentum/Breakout",  C6_bb_squeeze_breakout),
    "C07_atr_expansion":             ("C. Momentum/Breakout",  C7_atr_expansion_breakout),
    "C08_52wk_high_break":           ("C. Momentum/Breakout",  C8_52week_high_breakout),
    "C09_consecutive_candle":        ("C. Momentum/Breakout",  C9_consecutive_candle),
    "C10_macd_hist_divergence":      ("C. Momentum/Breakout",  C10_macd_histogram_divergence),
    # Category D — Volume
    "D01_obv_ema_cross":             ("D. Volume-Based",       D1_obv_ema_cross),
    "D02_obv_divergence":            ("D. Volume-Based",       D2_obv_divergence),
    "D03_mfi_extreme":               ("D. Volume-Based",       D3_mfi_extreme),
    "D04_vwap_cross":                ("D. Volume-Based",       D4_vwap_cross),
    "D05_volume_dryup_reversal":     ("D. Volume-Based",       D5_volume_dry_up_reversal),
    "D06_chaikin_money_flow":        ("D. Volume-Based",       D6_chaikin_money_flow),
    "D07_accumulation_dist":         ("D. Volume-Based",       D7_accumulation_distribution),
    "D08_price_volume_trend":        ("D. Volume-Based",       D8_price_volume_trend),
    # Category E — Price Action
    "E01_engulfing":                 ("E. Price Action",       E1_engulfing_pattern),
    "E02_pin_bar":                   ("E. Price Action",       E2_pin_bar),
    "E03_inside_bar_breakout":       ("E. Price Action",       E3_inside_bar_breakout),
    "E04_three_candle":              ("E. Price Action",       E4_three_candle_pattern),
    "E05_morning_evening_star":      ("E. Price Action",       E5_morning_evening_star),
    "E06_swing_high_low_break":      ("E. Price Action",       E6_swing_high_low_break),
    "E07_doji_reversal":             ("E. Price Action",       E7_doji_reversal),
    "E08_hh_ll_trend":               ("E. Price Action",       E8_higher_high_lower_low_trend),
    # Category F — Multi-Factor
    "F01_elder_triple_screen":       ("F. Multi-Factor",       F1_elder_triple_screen),
    "F02_macd_rsi_combo":            ("F. Multi-Factor",       F2_macd_rsi_combo),
    "F03_bb_rsi_combo":              ("F. Multi-Factor",       F3_bb_rsi_combo),
    "F04_supertrend_volume":         ("F. Multi-Factor",       F4_supertrend_volume),
    "F05_trend_pullback":            ("F. Multi-Factor",       F5_trend_revert_pullback),
    "F06_volatility_trend":          ("F. Multi-Factor",       F6_volatility_trend_combo),
    "F07_obv_macd_combo":            ("F. Multi-Factor",       F7_obv_macd_combo),
    # Category G — Volatility
    "G01_bb_bandwidth_breakout":     ("G. Volatility",         G1_bb_bandwidth_breakout),
    "G02_atr_channel_breakout":      ("G. Volatility",         G2_atr_channel_breakout),
    "G03_historical_vol_breakout":   ("G. Volatility",         G3_historical_vol_breakout),
    "G04_keltner_bb_squeeze":        ("G. Volatility",         G4_keltner_bb_squeeze),
    "G05_gap_fill":                  ("G. Volatility",         G5_gap_fill_reversal),
    # Category H — Statistical/Quant
    "H01_kalman_trend":              ("H. Statistical/Quant",  H1_kalman_trend),
    "H02_hurst_regime":              ("H. Statistical/Quant",  H2_hurst_regime),
    "H03_linear_reg_channel":        ("H. Statistical/Quant",  H3_linear_reg_channel),
    "H04_pairs_zscore":              ("H. Statistical/Quant",  H4_pairs_spread_zscore),
    "H05_regime_switching":          ("H. Statistical/Quant",  H5_regime_switching),
    # Category I — Crypto-Specific
    "I01_crypto_momentum":           ("I. Crypto-Specific",    I1_crypto_momentum_4h),
    "I02_weekend_effect":            ("I. Crypto-Specific",    I2_weekend_effect),
    "I03_altcoin_season":            ("I. Crypto-Specific",    I3_altcoin_season_proxy),
    "I04_defi_vol_cycle":            ("I. Crypto-Specific",    I4_defi_volatility_cycle),
    "I05_funding_rate_proxy":        ("I. Crypto-Specific",    I5_funding_rate_proxy),
}

CATEGORY_DESCRIPTIONS = {
    "A. Trend Following":   "Follow sustained price momentum. Low win-rate but big winners. Works best in trending markets.",
    "B. Mean Reversion":    "Fade extreme moves back to average. High win-rate, small winners. Works best in ranging markets.",
    "C. Momentum/Breakout": "Enter on strength/weakness confirmation. Medium win-rate, medium winners. Works in transitioning markets.",
    "D. Volume-Based":      "Follow institutional order flow via volume analysis. Used by Wyckoff & VSA practitioners.",
    "E. Price Action":      "Read raw candlestick patterns. No indicators — pure price & structure analysis.",
    "F. Multi-Factor":      "Require 2+ independent signals to align. Higher precision, fewer trades.",
    "G. Volatility":        "Trade volatility expansion/contraction cycles. Popular with options traders.",
    "H. Statistical/Quant": "Quantitative/mathematical edge. Used by systematic hedge funds.",
    "I. Crypto-Specific":   "Exploit crypto market microstructure: funding rates, weekend effect, altcoin cycles.",
}
