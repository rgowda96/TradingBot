"""
EXTENDED STRATEGY LIBRARY — 180+ parameterised variants
Covers every major family with multiple parameter sets so we find
the actual best configuration, not just one arbitrary choice.
"""
import pandas as pd
import numpy as np

def _safe(s):
    return s.fillna(0).replace([np.inf, -np.inf], 0)

# ──────────────────────────────────────────────────────────────────────────────
# TREND FOLLOWING FAMILY  (EMA, MACD, Supertrend, ADX variants)
# ──────────────────────────────────────────────────────────────────────────────

def make_ema_cross(fast, slow):
    def fn(df):
        e_f = df['close'].ewm(span=fast, adjust=False).mean()
        e_s = df['close'].ewm(span=slow, adjust=False).mean()
        sig = pd.Series(0, index=df.index)
        sig[e_f > e_s] = 1
        sig[e_f < e_s] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"EMA_{fast}_{slow}_cross"
    return fn

def make_triple_ema(f, m, s):
    def fn(df):
        ef = df['close'].ewm(span=f, adjust=False).mean()
        em = df['close'].ewm(span=m, adjust=False).mean()
        es = df['close'].ewm(span=s, adjust=False).mean()
        sig = pd.Series(0, index=df.index)
        sig[(ef > em) & (em > es)] = 1
        sig[(ef < em) & (em < es)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"TRIPLE_EMA_{f}_{m}_{s}"
    return fn

def make_macd_variant(fast, slow, signal):
    def fn(df):
        ema_f = df['close'].ewm(span=fast, adjust=False).mean()
        ema_s = df['close'].ewm(span=slow, adjust=False).mean()
        macd  = ema_f - ema_s
        sig_l = macd.ewm(span=signal, adjust=False).mean()
        hist  = macd - sig_l
        sig   = pd.Series(0, index=df.index)
        sig[hist > 0] = 1
        sig[hist < 0] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"MACD_{fast}_{slow}_{signal}"
    return fn

def make_supertrend(atr_mult, atr_period=14):
    def fn(df):
        h, l, c = df['high'], df['low'], df['close']
        tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        mid = (h + l) / 2
        up  = mid - atr_mult * atr
        dn  = mid + atr_mult * atr
        trend = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if c.iloc[i] > dn.iloc[i-1]:
                trend.iloc[i] = 1
            elif c.iloc[i] < up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        return trend.shift(1).fillna(0)
    fn.__name__ = f"SUPERTREND_{atr_mult}x_{atr_period}"
    return fn

def make_adx_trend(adx_thresh, di_period=14):
    def fn(df):
        h, l, c = df['high'], df['low'], df['close']
        tr    = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(di_period).mean()
        dm_p  = (h - h.shift()).clip(lower=0)
        dm_n  = (l.shift() - l).clip(lower=0)
        di_p  = 100 * dm_p.rolling(di_period).mean() / atr14.replace(0, np.nan)
        di_n  = 100 * dm_n.rolling(di_period).mean() / atr14.replace(0, np.nan)
        dx    = (100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan))
        adx   = dx.rolling(di_period).mean()
        sig   = pd.Series(0, index=df.index)
        sig[(adx > adx_thresh) & (di_p > di_n)] = 1
        sig[(adx > adx_thresh) & (di_n > di_p)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ADX_{adx_thresh}_DI{di_period}"
    return fn

def make_donchian(period):
    def fn(df):
        high_max = df['high'].rolling(period).max()
        low_min  = df['low'].rolling(period).min()
        sig = pd.Series(0, index=df.index)
        sig[df['close'] >= high_max.shift(1)] = 1
        sig[df['close'] <= low_min.shift(1)]  = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"DONCHIAN_{period}"
    return fn

def make_price_above_ma(period, ma_type="ema"):
    def fn(df):
        ma = df['close'].ewm(span=period, adjust=False).mean() if ma_type=="ema" else df['close'].rolling(period).mean()
        sig = pd.Series(0, index=df.index)
        sig[df['close'] > ma] = 1
        sig[df['close'] < ma] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"PRICE_ABOVE_{ma_type.upper()}{period}"
    return fn

def make_chandelier(atr_mult, period=22):
    def fn(df):
        h, l, c = df['high'], df['low'], df['close']
        tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        long_stop  = df['high'].rolling(period).max() - atr_mult * atr
        short_stop = df['low'].rolling(period).min()  + atr_mult * atr
        sig = pd.Series(0, index=df.index)
        sig[c > long_stop.shift(1)]  = 1
        sig[c < short_stop.shift(1)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"CHANDELIER_{atr_mult}x_{period}"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# MEAN REVERSION FAMILY
# ──────────────────────────────────────────────────────────────────────────────

def make_rsi_reversion(ob, os_, period=14):
    def fn(df):
        delta = df['close'].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        sig   = pd.Series(0, index=df.index)
        sig[rsi < os_] = 1
        sig[rsi > ob]  = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"RSI_REV_{os_}_{ob}_p{period}"
    return fn

def make_bb_reversion(std_dev, period=20):
    def fn(df):
        ma  = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        ub  = ma + std_dev * std
        lb  = ma - std_dev * std
        sig = pd.Series(0, index=df.index)
        sig[df['close'] < lb] = 1
        sig[df['close'] > ub] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"BB_REV_{std_dev}std_p{period}"
    return fn

def make_zscore_reversion(period, thresh):
    def fn(df):
        ma   = df['close'].rolling(period).mean()
        std  = df['close'].rolling(period).std()
        z    = (df['close'] - ma) / std.replace(0, np.nan)
        sig  = pd.Series(0, index=df.index)
        sig[z < -thresh] = 1
        sig[z >  thresh] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ZSCORE_REV_p{period}_t{thresh}"
    return fn

def make_vwap_reversion(atr_mult):
    def fn(df):
        tp   = (df['high'] + df['low'] + df['close']) / 3
        cum_tp_vol = (tp * df['volume']).cumsum()
        cum_vol    = df['volume'].cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        tr   = pd.concat([df['high']-df['low'],
                          (df['high']-df['close'].shift()).abs(),
                          (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        atr  = tr.rolling(14).mean()
        sig  = pd.Series(0, index=df.index)
        sig[df['close'] < vwap - atr_mult*atr] = 1
        sig[df['close'] > vwap + atr_mult*atr] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"VWAP_REV_{atr_mult}xATR"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# MOMENTUM / BREAKOUT FAMILY
# ──────────────────────────────────────────────────────────────────────────────

def make_rsi_momentum(bull_thresh, bear_thresh, period=14):
    def fn(df):
        delta = df['close'].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        sig   = pd.Series(0, index=df.index)
        sig[rsi > bull_thresh] = 1
        sig[rsi < bear_thresh] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"RSI_MOM_{bull_thresh}_{bear_thresh}_p{period}"
    return fn

def make_roc_momentum(period, thresh):
    def fn(df):
        roc = df['close'].pct_change(period) * 100
        sig = pd.Series(0, index=df.index)
        sig[roc >  thresh] = 1
        sig[roc < -thresh] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ROC_MOM_p{period}_t{thresh}"
    return fn

def make_vol_breakout(vol_mult, lookback):
    def fn(df):
        avg_vol = df['volume'].rolling(lookback).mean()
        atr     = (df['high'] - df['low']).rolling(lookback).mean()
        ret     = df['close'].pct_change()
        sig     = pd.Series(0, index=df.index)
        cond    = df['volume'] > vol_mult * avg_vol
        sig[cond & (ret > 0)] = 1
        sig[cond & (ret < 0)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"VOL_BREAK_{vol_mult}x_p{lookback}"
    return fn

def make_52wk_break(lookback):
    def fn(df):
        high_n = df['high'].rolling(lookback).max()
        low_n  = df['low'].rolling(lookback).min()
        sig    = pd.Series(0, index=df.index)
        sig[df['close'] >= high_n.shift(1)] = 1
        sig[df['close'] <= low_n.shift(1)]  = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"NPERIOD_BREAK_{lookback}"
    return fn

def make_keltner_breakout(atr_mult, period=20):
    def fn(df):
        ma  = df['close'].ewm(span=period, adjust=False).mean()
        tr  = pd.concat([df['high']-df['low'],
                         (df['high']-df['close'].shift()).abs(),
                         (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        ub  = ma + atr_mult * atr
        lb  = ma - atr_mult * atr
        sig = pd.Series(0, index=df.index)
        sig[df['close'] > ub] = 1
        sig[df['close'] < lb] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"KELTNER_BREAK_{atr_mult}x_p{period}"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# VOLUME FAMILY
# ──────────────────────────────────────────────────────────────────────────────

def make_obv_ma(period):
    def fn(df):
        direction = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv  = (df['volume'] * direction).cumsum()
        obv_ma = obv.rolling(period).mean()
        sig  = pd.Series(0, index=df.index)
        sig[obv > obv_ma] = 1
        sig[obv < obv_ma] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"OBV_MA{period}"
    return fn

def make_mfi_extreme(ob, os_, period=14):
    def fn(df):
        tp  = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']
        pos = rmf.where(tp > tp.shift(1), 0).rolling(period).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(period).sum()
        mfr = pos / neg.replace(0, np.nan)
        mfi = 100 - 100 / (1 + mfr)
        sig = pd.Series(0, index=df.index)
        sig[mfi < os_] = 1
        sig[mfi > ob]  = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"MFI_{os_}_{ob}_p{period}"
    return fn

def make_pvt_trend(period):
    def fn(df):
        pvt = ((df['close'] - df['close'].shift()) / df['close'].shift() * df['volume']).cumsum()
        pvt_ma = pvt.rolling(period).mean()
        sig = pd.Series(0, index=df.index)
        sig[pvt > pvt_ma] = 1
        sig[pvt < pvt_ma] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"PVT_MA{period}"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# MULTI-FACTOR COMBOS
# ──────────────────────────────────────────────────────────────────────────────

def make_ema_rsi_combo(ema_fast, ema_slow, rsi_bull, rsi_bear, rsi_period=14):
    def fn(df):
        ef = df['close'].ewm(span=ema_fast, adjust=False).mean()
        es = df['close'].ewm(span=ema_slow, adjust=False).mean()
        delta = df['close'].diff()
        gain  = delta.clip(lower=0).rolling(rsi_period).mean()
        loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rsi   = 100 - 100 / (1 + gain/loss.replace(0,np.nan))
        sig   = pd.Series(0, index=df.index)
        sig[(ef > es) & (rsi > rsi_bull)] = 1
        sig[(ef < es) & (rsi < rsi_bear)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"EMA{ema_fast}{ema_slow}_RSI{rsi_bull}{rsi_bear}"
    return fn

def make_macd_rsi_filter(macd_fast, macd_slow, macd_sig, rsi_bull, rsi_bear):
    def fn(df):
        ef   = df['close'].ewm(span=macd_fast, adjust=False).mean()
        es   = df['close'].ewm(span=macd_slow, adjust=False).mean()
        macd = ef - es
        sl   = macd.ewm(span=macd_sig, adjust=False).mean()
        hist = macd - sl
        delta= df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi  = 100 - 100 / (1 + gain/loss.replace(0,np.nan))
        sig  = pd.Series(0, index=df.index)
        sig[(hist > 0) & (rsi > rsi_bull)] = 1
        sig[(hist < 0) & (rsi < rsi_bear)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"MACD{macd_fast}{macd_slow}_RSI{rsi_bull}{rsi_bear}"
    return fn

def make_supertrend_rsi(atr_mult, rsi_min, rsi_max, atr_period=14):
    def fn(df):
        h, l, c = df['high'], df['low'], df['close']
        tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        mid = (h + l) / 2
        up  = mid - atr_mult * atr
        dn  = mid + atr_mult * atr
        trend = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            trend.iloc[i] = 1 if c.iloc[i] > dn.iloc[i-1] else (-1 if c.iloc[i] < up.iloc[i-1] else trend.iloc[i-1])
        delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - 100 / (1 + gain/loss.replace(0,np.nan))
        sig   = pd.Series(0, index=df.index)
        sig[(trend == 1) & (rsi >= rsi_min) & (rsi <= rsi_max)] = 1
        sig[(trend ==-1) & (rsi >= rsi_min) & (rsi <= rsi_max)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ST{atr_mult}_RSI{rsi_min}_{rsi_max}"
    return fn

def make_bb_rsi_combo(bb_std, rsi_os, rsi_ob, period=20):
    def fn(df):
        ma  = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        ub  = ma + bb_std * std; lb = ma - bb_std * std
        delta= df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi  = 100 - 100 / (1 + gain/loss.replace(0,np.nan))
        sig  = pd.Series(0, index=df.index)
        sig[(df['close'] < lb) & (rsi < rsi_os)] = 1
        sig[(df['close'] > ub) & (rsi > rsi_ob)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"BB{bb_std}_RSI{rsi_os}_{rsi_ob}"
    return fn

def make_trend_volume_combo(ema_period, vol_mult):
    def fn(df):
        ema = df['close'].ewm(span=ema_period, adjust=False).mean()
        avg_vol = df['volume'].rolling(20).mean()
        trending_up   = df['close'] > ema
        trending_down = df['close'] < ema
        high_vol      = df['volume'] > vol_mult * avg_vol
        ret = df['close'].pct_change()
        sig = pd.Series(0, index=df.index)
        sig[trending_up   & high_vol & (ret > 0)] = 1
        sig[trending_down & high_vol & (ret < 0)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"TREND_VOL_EMA{ema_period}_{vol_mult}x"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY FAMILY
# ──────────────────────────────────────────────────────────────────────────────

def make_atr_breakout(atr_mult, period=14):
    def fn(df):
        tr  = pd.concat([df['high']-df['low'],
                         (df['high']-df['close'].shift()).abs(),
                         (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        prev_c = df['close'].shift(1)
        sig = pd.Series(0, index=df.index)
        sig[df['close'] > prev_c + atr_mult * atr] = 1
        sig[df['close'] < prev_c - atr_mult * atr] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ATR_BREAK_{atr_mult}x_p{period}"
    return fn

def make_bb_squeeze_breakout(squeeze_thresh, period=20):
    def fn(df):
        ma  = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        bw  = (2 * std) / ma.replace(0, np.nan)
        low_bw = bw < squeeze_thresh
        ret = df['close'].pct_change()
        sig = pd.Series(0, index=df.index)
        sig[low_bw.shift(1) & (ret > 0.005)] = 1
        sig[low_bw.shift(1) & (ret < -0.005)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"BB_SQZ_{squeeze_thresh}_p{period}"
    return fn

def make_historical_vol_regime(hv_period, vol_thresh):
    def fn(df):
        log_ret  = np.log(df['close'] / df['close'].shift(1))
        hv       = log_ret.rolling(hv_period).std() * np.sqrt(252)
        hv_ma    = hv.rolling(hv_period*2).mean()
        high_vol = hv > vol_thresh * hv_ma
        ret = df['close'].pct_change()
        sig = pd.Series(0, index=df.index)
        sig[high_vol & (ret > 0)] = 1
        sig[high_vol & (ret < 0)] = -1
        sig[~high_vol] = 0
        return sig.shift(1).fillna(0)
    fn.__name__ = f"HVOL_{hv_period}p_{vol_thresh}x"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# PRICE ACTION FAMILY
# ──────────────────────────────────────────────────────────────────────────────

def make_swing_break(lookback):
    def fn(df):
        swing_high = df['high'].rolling(lookback).max()
        swing_low  = df['low'].rolling(lookback).min()
        sig = pd.Series(0, index=df.index)
        sig[df['close'] > swing_high.shift(1)] = 1
        sig[df['close'] < swing_low.shift(1)]  = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"SWING_BREAK_{lookback}"
    return fn

def make_candle_momentum(n_candles, ret_thresh):
    def fn(df):
        returns = df['close'].pct_change()
        recent  = pd.Series([returns.iloc[max(0,i-n_candles):i].sum()
                             for i in range(len(df))], index=df.index)
        sig = pd.Series(0, index=df.index)
        sig[recent >  ret_thresh] = 1
        sig[recent < -ret_thresh] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"CANDLE_MOM_{n_candles}c_{ret_thresh}"
    return fn

def make_higher_high_trend(period):
    def fn(df):
        hh = df['high'].rolling(period).max() > df['high'].rolling(period).max().shift(period)
        ll = df['low'].rolling(period).min()  < df['low'].rolling(period).min().shift(period)
        sig = pd.Series(0, index=df.index)
        sig[hh & ~ll]  = 1
        sig[~hh & ll]  = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"HH_LL_TREND_{period}"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# STATISTICAL / QUANT FAMILY
# ──────────────────────────────────────────────────────────────────────────────

def make_zscore_momentum(period, entry_thresh, exit_thresh=0.5):
    def fn(df):
        log_ret = np.log(df['close'] / df['close'].shift(1))
        z = (log_ret - log_ret.rolling(period).mean()) / log_ret.rolling(period).std().replace(0, np.nan)
        sig = pd.Series(0, index=df.index)
        sig[z >  entry_thresh] = 1
        sig[z < -entry_thresh] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ZSCORE_MOM_p{period}_t{entry_thresh}"
    return fn

def make_rolling_sharpe(period, thresh):
    def fn(df):
        ret = df['close'].pct_change()
        rs  = ret.rolling(period).mean() / ret.rolling(period).std().replace(0, np.nan)
        sig = pd.Series(0, index=df.index)
        sig[rs >  thresh] = 1
        sig[rs < -thresh] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"ROLL_SHARPE_p{period}_t{thresh}"
    return fn

def make_regime_filter_ema(fast, slow, roc_period, roc_thresh):
    """Only trade when trend is clear (ROC filter)"""
    def fn(df):
        ef  = df['close'].ewm(span=fast, adjust=False).mean()
        es  = df['close'].ewm(span=slow, adjust=False).mean()
        roc = df['close'].pct_change(roc_period) * 100
        sig = pd.Series(0, index=df.index)
        sig[(ef > es) & (roc >  roc_thresh)] = 1
        sig[(ef < es) & (roc < -roc_thresh)] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"EMA{fast}{slow}_ROC{roc_period}_{roc_thresh}"
    return fn

def make_triple_screen(weekly_ema, daily_ema, rsi_trigger):
    """Elder Triple Screen: higher TF trend + lower TF entry"""
    def fn(df):
        w_trend = df['close'].ewm(span=weekly_ema, adjust=False).mean()
        d_trend = df['close'].ewm(span=daily_ema, adjust=False).mean()
        delta   = df['close'].diff()
        gain    = delta.clip(lower=0).rolling(7).mean()
        loss    = (-delta.clip(upper=0)).rolling(7).mean()
        rsi     = 100 - 100 / (1 + gain/loss.replace(0,np.nan))
        sig     = pd.Series(0, index=df.index)
        sig[(df['close'] > w_trend) & (df['close'] > d_trend) & (rsi > rsi_trigger)] = 1
        sig[(df['close'] < w_trend) & (df['close'] < d_trend) & (rsi < (100-rsi_trigger))] = -1
        return sig.shift(1).fillna(0)
    fn.__name__ = f"TRIPLE_SCR_{weekly_ema}_{daily_ema}_RSI{rsi_trigger}"
    return fn

# ──────────────────────────────────────────────────────────────────────────────
# BUILD THE EXTENDED REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

EXTENDED_REGISTRY = {}

def _reg(cat, fn):
    EXTENDED_REGISTRY[fn.__name__] = (cat, fn)

# EMA crosses — 8 variants
for f, s in [(5,13),(8,21),(9,21),(13,34),(21,50),(21,89),(34,89),(50,200)]:
    _reg("X1.Trend/EMA", make_ema_cross(f, s))

# Triple EMA — 4 variants
for f,m,s in [(5,13,34),(8,21,55),(9,21,50),(13,34,89)]:
    _reg("X1.Trend/EMA", make_triple_ema(f,m,s))

# MACD — 6 variants
for fast,slow,sig in [(8,17,9),(10,20,7),(12,26,9),(12,26,5),(6,13,5),(19,39,9)]:
    _reg("X1.Trend/MACD", make_macd_variant(fast,slow,sig))

# Supertrend — 6 variants
for mult in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    _reg("X1.Trend/Supertrend", make_supertrend(mult))

# ADX — 4 variants
for thresh in [20, 25, 30, 35]:
    _reg("X1.Trend/ADX", make_adx_trend(thresh))

# Donchian — 5 variants
for period in [10, 20, 30, 55, 100]:
    _reg("X2.Breakout/Donchian", make_donchian(period))

# Price above MA — 4 variants
for period in [50, 100, 150, 200]:
    _reg("X1.Trend/MA", make_price_above_ma(period))

# Chandelier — 4 variants
for mult in [2.0, 2.5, 3.0, 3.5]:
    _reg("X1.Trend/Chandelier", make_chandelier(mult))

# RSI reversion — 6 variants
for ob,os in [(70,30),(75,25),(80,20),(65,35),(72,28),(78,22)]:
    _reg("X3.MeanRev/RSI", make_rsi_reversion(ob, os))

# BB reversion — 4 variants
for std in [1.5, 2.0, 2.5, 3.0]:
    _reg("X3.MeanRev/BB", make_bb_reversion(std))

# Z-score reversion — 4 variants
for p,t in [(20,1.5),(30,2.0),(50,2.0),(20,2.5)]:
    _reg("X3.MeanRev/ZScore", make_zscore_reversion(p,t))

# VWAP reversion — 3 variants
for mult in [1.0, 1.5, 2.0]:
    _reg("X3.MeanRev/VWAP", make_vwap_reversion(mult))

# RSI momentum — 6 variants
for bull,bear in [(55,45),(60,40),(60,45),(55,40),(65,35),(50,50)]:
    _reg("X2.Momentum/RSI", make_rsi_momentum(bull, bear))

# ROC momentum — 4 variants
for p,t in [(5,2),(10,3),(20,5),(14,4)]:
    _reg("X2.Momentum/ROC", make_roc_momentum(p,t))

# Volume breakout — 4 variants
for mult,lb in [(1.5,20),(2.0,20),(2.0,14),(3.0,20)]:
    _reg("X2.Breakout/Volume", make_vol_breakout(mult,lb))

# N-period breakout — 5 variants
for period in [20, 30, 50, 90, 126]:
    _reg("X2.Breakout/NBar", make_52wk_break(period))

# Keltner breakout — 4 variants
for mult in [1.5, 2.0, 2.5, 3.0]:
    _reg("X2.Breakout/Keltner", make_keltner_breakout(mult))

# OBV MA — 3 variants
for p in [10, 20, 50]:
    _reg("X4.Volume/OBV", make_obv_ma(p))

# MFI — 3 variants
for ob,os in [(80,20),(75,25),(70,30)]:
    _reg("X4.Volume/MFI", make_mfi_extreme(ob,os))

# PVT — 2 variants
for p in [20, 50]:
    _reg("X4.Volume/PVT", make_pvt_trend(p))

# EMA+RSI combos — 6 variants
for ef,es,rb,rbear in [(9,21,55,45),(9,21,60,40),(21,50,55,45),(21,50,60,40),(13,34,55,45),(50,200,55,40)]:
    _reg("X5.MultiFactor/EMA_RSI", make_ema_rsi_combo(ef,es,rb,rbear))

# MACD+RSI — 4 variants
for mf,ms,ms_,rb,rbear in [(12,26,9,55,45),(12,26,9,60,40),(8,17,9,55,45),(12,26,5,55,45)]:
    _reg("X5.MultiFactor/MACD_RSI", make_macd_rsi_filter(mf,ms,ms_,rb,rbear))

# Supertrend+RSI — 4 variants
for mult,rmin,rmax in [(2.0,40,70),(2.5,45,65),(2.0,50,75),(3.0,40,70)]:
    _reg("X5.MultiFactor/ST_RSI", make_supertrend_rsi(mult,rmin,rmax))

# BB+RSI combo — 4 variants
for bstd,ros,rob in [(2.0,30,70),(2.0,35,65),(2.5,25,75),(1.5,35,65)]:
    _reg("X5.MultiFactor/BB_RSI", make_bb_rsi_combo(bstd,ros,rob))

# Trend+Volume — 4 variants
for ep,vm in [(20,1.5),(50,2.0),(20,2.0),(100,1.5)]:
    _reg("X5.MultiFactor/Trend_Vol", make_trend_volume_combo(ep,vm))

# ATR breakout — 4 variants
for mult in [1.0, 1.5, 2.0, 2.5]:
    _reg("X6.Volatility/ATR", make_atr_breakout(mult))

# BB squeeze breakout — 3 variants
for thresh in [0.03, 0.05, 0.08]:
    _reg("X6.Volatility/BB_Squeeze", make_bb_squeeze_breakout(thresh))

# Historical vol regime — 3 variants
for p,t in [(21,1.2),(21,1.5),(14,1.3)]:
    _reg("X6.Volatility/HVol", make_historical_vol_regime(p,t))

# Swing break — 4 variants
for lb in [5, 10, 20, 30]:
    _reg("X7.PriceAction/Swing", make_swing_break(lb))

# Candle momentum — 4 variants
for n,t in [(3,0.02),(5,0.03),(7,0.04),(3,0.03)]:
    _reg("X7.PriceAction/CandleMom", make_candle_momentum(n,t))

# HH/LL trend — 3 variants
for p in [5, 10, 20]:
    _reg("X7.PriceAction/HH_LL", make_higher_high_trend(p))

# Z-score momentum — 4 variants
for p,t in [(20,1.0),(30,1.5),(14,1.0),(50,1.5)]:
    _reg("X8.Quant/ZMom", make_zscore_momentum(p,t))

# Rolling Sharpe — 3 variants
for p,t in [(20,0.5),(30,0.5),(20,1.0)]:
    _reg("X8.Quant/RollSharpe", make_rolling_sharpe(p,t))

# Regime filter EMA — 4 variants
for f,s,rp,rt in [(9,21,14,2),(21,50,20,3),(13,34,10,2),(9,21,20,5)]:
    _reg("X8.Quant/RegimeEMA", make_regime_filter_ema(f,s,rp,rt))

# Triple screen — 3 variants
for we,de,rst in [(52,13,50),(26,9,55),(40,10,52)]:
    _reg("X5.MultiFactor/TripleScreen", make_triple_screen(we,de,rst))

print(f"Extended registry: {len(EXTENDED_REGISTRY)} strategies")
