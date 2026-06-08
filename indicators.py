"""
Add all technical indicators to a DataFrame.
"""
import pandas as pd
import numpy as np


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # ── Trend ────────────────────────────────────────────────────────────────
    for n in [9, 21, 50, 200]:
        df[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()
        df[f"sma_{n}"] = c.rolling(n).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Supertrend (ATR-based)
    atr14 = _atr(h, lo, c, 14)
    df["atr_14"] = atr14
    m = (h + lo) / 2
    upper = m + 3 * atr14
    lower = m - 3 * atr14
    df["supertrend_up"] = lower
    df["supertrend_dn"] = upper
    df["supertrend_dir"] = _supertrend_dir(c, lower, upper)

    # ── Momentum ─────────────────────────────────────────────────────────────
    # RSI
    for n in [14, 21]:
        df[f"rsi_{n}"] = _rsi(c, n)

    # Stochastic RSI
    rsi14 = df["rsi_14"]
    stoch_k, stoch_d = _stoch_rsi(rsi14, 14, 3, 3)
    df["stoch_rsi_k"] = stoch_k
    df["stoch_rsi_d"] = stoch_d

    # Williams %R
    df["willr_14"] = _willr(h, lo, c, 14)

    # Rate of Change
    for n in [9, 21]:
        df[f"roc_{n}"] = c.pct_change(n) * 100

    # ── Volatility ────────────────────────────────────────────────────────────
    # Bollinger Bands
    for n, k in [(20, 2.0)]:
        mid = c.rolling(n).mean()
        std = c.rolling(n).std()
        df[f"bb_upper_{n}"] = mid + k * std
        df[f"bb_mid_{n}"] = mid
        df[f"bb_lower_{n}"] = mid - k * std
        df[f"bb_width_{n}"] = (df[f"bb_upper_{n}"] - df[f"bb_lower_{n}"]) / mid
        df[f"bb_pct_{n}"] = (c - df[f"bb_lower_{n}"]) / (df[f"bb_upper_{n}"] - df[f"bb_lower_{n}"])

    # ATR (multiple periods)
    for n in [7, 14, 21]:
        df[f"atr_{n}"] = _atr(h, lo, c, n)

    # Keltner Channels
    ema20 = c.ewm(span=20, adjust=False).mean()
    atr20 = _atr(h, lo, c, 20)
    df["kc_upper"] = ema20 + 1.5 * atr20
    df["kc_lower"] = ema20 - 1.5 * atr20

    # Squeeze (BB inside KC)
    df["squeeze"] = (df["bb_upper_20"] < df["kc_upper"]) & (df["bb_lower_20"] > df["kc_lower"])

    # ── Volume ────────────────────────────────────────────────────────────────
    df["volume_sma_20"] = v.rolling(20).mean()
    df["volume_ratio"] = v / df["volume_sma_20"]

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()

    # VWAP (rolling 20-bar approximation)
    tp = (h + lo + c) / 3
    df["vwap_20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum()

    # ── Structure / Price Action ──────────────────────────────────────────────
    df["higher_high"] = h > h.shift(1)
    df["lower_low"] = lo < lo.shift(1)
    df["inside_bar"] = (h <= h.shift(1)) & (lo >= lo.shift(1))
    df["engulf_bull"] = (c > df["open"]) & (df["open"].shift(1) > c.shift(1)) & \
                        (c > df["open"].shift(1)) & (df["open"] < c.shift(1))
    df["engulf_bear"] = (c < df["open"]) & (df["open"].shift(1) < c.shift(1)) & \
                        (c < df["open"].shift(1)) & (df["open"] > c.shift(1))

    # Swing highs / lows (5-bar pivot)
    df["swing_high"] = (h == h.rolling(5, center=True).max())
    df["swing_low"]  = (lo == lo.rolling(5, center=True).min())

    # ── Derived signals ───────────────────────────────────────────────────────
    df["above_ema200"] = (c > df["ema_200"]).astype(int)
    df["golden_cross"]  = ((df["ema_50"] > df["ema_200"]) &
                           (df["ema_50"].shift(1) <= df["ema_200"].shift(1))).astype(int)
    df["death_cross"]   = ((df["ema_50"] < df["ema_200"]) &
                           (df["ema_50"].shift(1) >= df["ema_200"].shift(1))).astype(int)

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(h, lo, c, n):
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _rsi(c, n):
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _stoch_rsi(rsi, n, k, d):
    lo = rsi.rolling(n).min()
    hi = rsi.rolling(n).max()
    k_raw = 100 * (rsi - lo) / (hi - lo + 1e-9)
    k_smooth = k_raw.rolling(k).mean()
    d_smooth = k_smooth.rolling(d).mean()
    return k_smooth, d_smooth


def _willr(h, lo, c, n):
    hh = h.rolling(n).max()
    ll = lo.rolling(n).min()
    return -100 * (hh - c) / (hh - ll + 1e-9)


def _supertrend_dir(c, lower, upper):
    direction = pd.Series(1, index=c.index)
    for i in range(1, len(c)):
        if c.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return direction
