"""
Fetch historical OHLCV data for all portfolio constituents.
Sources: Binance (ccxt) for crypto, synthetic for Treasuries/stablecoins.
"""
import os, time as _time, logging, yaml
import numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class BinanceFetcher:
    def __init__(self):
        import ccxt
        self.exchange = ccxt.binance({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol, timeframe="1h", since="2022-01-01", until="2025-12-31"):
        since_ts = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
        until_ts = int(datetime.strptime(until, "%Y-%m-%d").timestamp() * 1000)
        all_candles, current = [], since_ts
        logger.info(f"Fetching {symbol}...")
        while current < until_ts:
            try:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current, limit=1000)
                if not candles: break
                all_candles.extend(candles)
                current = candles[-1][0] + 1
                _time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                logger.error(f"Error: {e}"); _time.sleep(10)
        if not all_candles: return pd.DataFrame()
        df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        logger.info(f"  {symbol}: {len(df)} candles")
        return df

def generate_treasury_series(start, end, freq="1h", base_price=1.0, annual_yield=0.045, name="BUIDL"):
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(idx)
    hourly_yield = annual_yield / (365.25 * 24)
    prices = base_price * (1 + np.arange(n) * hourly_yield) + np.random.normal(0, 0.0001/np.sqrt(24), n).cumsum()
    prices = np.maximum(prices, base_price * 0.99)
    df = pd.DataFrame({"open": prices, "high": prices*1.00005, "low": prices*0.99995,
                        "close": prices, "volume": np.random.uniform(1e6, 5e6, n)}, index=idx)
    logger.info(f"  {name}: {len(df)} synthetic candles")
    return df

def generate_stablecoin_series(start, end, freq="1h", name="USDC"):
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(idx)
    prices = 1.0 + np.random.normal(0, 0.00002, n)
    df = pd.DataFrame({"open": prices, "high": prices+0.00001, "low": prices-0.00001,
                        "close": prices, "volume": np.random.uniform(1e8, 5e8, n)}, index=idx)
    return df

def fetch_all_data(config_path="config.yaml"):
    with open(config_path) as f: config = yaml.safe_load(f)
    start, end = config["data"]["start_date"], config["data"]["end_date"]
    cache_dir = Path("data/cache"); cache_dir.mkdir(parents=True, exist_ok=True)
    fetcher = BinanceFetcher()
    all_data = {}
    symbols = config["data"]["assets"]
    pairs = {"BTC":"BTC/USDT","ETH":"ETH/USDT","SOL":"SOL/USDT","stETH":"STETH/USDT","rETH":"RETH/USDT"}
    treasuries = {"BUIDL": 0.045, "USDY": 0.05}
    for sym in symbols:
        cache_file = cache_dir / f"{sym}_hourly.parquet"
        if cache_file.exists():
            all_data[sym] = pd.read_parquet(cache_file); continue
        if sym in pairs:
            df = fetcher.fetch_ohlcv(pairs[sym], "1h", start, end)
        elif sym in treasuries:
            df = generate_treasury_series(start, end, annual_yield=treasuries[sym], name=sym)
        elif sym == "USDC":
            df = generate_stablecoin_series(start, end, name=sym)
        else:
            df = fetcher.fetch_ohlcv(f"{sym}/USDT", "1h", start, end)
        if not df.empty:
            df.to_parquet(cache_file); all_data[sym] = df
    logger.info(f"Loaded {len(all_data)} assets")
    return all_data

if __name__ == "__main__":
    data = fetch_all_data()
    for s, d in data.items(): print(f"{s}: {len(d)} rows")
