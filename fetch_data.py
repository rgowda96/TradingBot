"""
Fetch OHLCV data for ETH/USDT across all timeframes from Binance via ccxt.
Saves to data/ as parquet files.
"""
import ccxt
import pandas as pd
import os
import time

TIMEFRAMES = {
    "1m":   500,   # ~8h of 1m
    "5m":   1000,  # ~3.5 days
    "15m":  2000,  # ~20 days
    "1h":   4000,  # ~166 days
    "4h":   4000,  # ~666 days (~1.8y)
    "1d":   2000,  # ~5.5y
    "1w":   500,   # ~10y
}

SYMBOL = "ETH/USDT"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_ohlcv(exchange, symbol, timeframe, limit):
    """Paginate backwards to collect `limit` candles total."""
    all_candles = []
    since = None
    per_page = 1000

    while len(all_candles) < limit:
        fetch_limit = min(per_page, limit - len(all_candles))
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_limit)
        except Exception as e:
            print(f"  Error fetching {timeframe}: {e}")
            break

        if not candles:
            break

        # Prepend older data: next fetch ends just before this batch
        all_candles = candles + all_candles
        since = None  # will re-paginate differently below
        break  # single fetch for now; extend below for deep history

    return all_candles


def fetch_full_history(exchange, symbol, timeframe, max_candles):
    """Fetch up to max_candles going back as far as possible."""
    all_candles = []
    since = None
    per_page = 1000

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=per_page)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            break

        if not candles:
            break

        if since is None:
            all_candles = candles
        else:
            all_candles = all_candles + candles

        if len(candles) < per_page:
            break  # no more data

        since = candles[-1][0] + 1  # next ms after last candle

        if len(all_candles) >= max_candles:
            break

        time.sleep(0.2)  # rate limit

    return all_candles[-max_candles:]


def candles_to_df(candles):
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


if __name__ == "__main__":
    exchange = ccxt.binance({"enableRateLimit": True})

    for tf, max_candles in TIMEFRAMES.items():
        out_path = os.path.join(DATA_DIR, f"ETH_USDT_{tf}.parquet")
        print(f"Fetching {tf} ({max_candles} candles max)...")
        candles = fetch_full_history(exchange, SYMBOL, tf, max_candles)
        if candles:
            df = candles_to_df(candles)
            df.to_parquet(out_path)
            print(f"  Saved {len(df)} rows -> {out_path}  [{df.index[0]} -> {df.index[-1]}]")
        else:
            print(f"  No data returned for {tf}")

    print("\nAll done.")
