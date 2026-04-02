"""
Fetch historical OHLCV data for all portfolio constituents.

Sources:
  - Primary: Binance via ccxt
  - Fallback: Kraken, Coinbase via ccxt
  - Synthetic: GBM for Treasuries (BUIDL/USDY), depeg-aware for USDC

Includes:
  - Exponential-backoff retries
  - OHLCV validation (no negative prices, high >= low, etc.)
  - Data quality reporting with JSON output
  - SHA-256 cache integrity verification
  - Multiple exchange fallback
"""

import os
import time as _time
import json
import hashlib
import logging
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config values (used when keys are missing from config.yaml)
# ---------------------------------------------------------------------------
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_FALLBACK_EXCHANGES: List[str] = ["kraken", "coinbase"]
_DEFAULT_MIN_COVERAGE_PCT = 95.0
_DEFAULT_MAX_GAP_HOURS = 24

# ---------------------------------------------------------------------------
# Exchange helpers
# ---------------------------------------------------------------------------

def _create_exchange(name: str):
    """Instantiate a ccxt exchange by name with rate limiting enabled."""
    import ccxt
    exchange_cls = getattr(ccxt, name, None)
    if exchange_cls is None:
        raise ValueError(f"Unknown exchange: {name}")
    return exchange_cls({"enableRateLimit": True})


def _symbol_for_exchange(symbol: str, exchange_name: str) -> str:
    """
    Map a generic symbol (e.g. 'BTC') to the pair format each exchange uses.
    Binance and Coinbase use /USDT, Kraken uses /USD for majors.
    """
    base = symbol.upper()
    # Kraken uses USD instead of USDT for majors
    if exchange_name == "kraken":
        return f"{base}/USD"
    return f"{base}/USDT"


# ---------------------------------------------------------------------------
# Cache integrity
# ---------------------------------------------------------------------------

def _compute_file_hash(filepath: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_manifest(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def _save_hash_manifest(cache_dir: Path, manifest: dict):
    manifest_path = cache_dir / "cache_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _verify_cache_integrity(cache_file: Path, manifest: dict) -> bool:
    """Return True if the cached file matches its stored hash."""
    key = cache_file.name
    if key not in manifest:
        return False
    expected = manifest[key]
    actual = _compute_file_hash(cache_file)
    if actual != expected:
        logger.warning(
            f"Cache integrity check FAILED for {key} "
            f"(expected {expected[:12]}... got {actual[:12]}...)"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, dict]:
    """
    Validate OHLCV data and return (cleaned_df, quality_metrics).

    Rules:
      - No negative prices
      - Volume >= 0
      - High >= Low
      - Close within [Low, High]

    Rows violating rules are dropped.
    """
    if df.empty:
        return df, {"rows_dropped": 0, "pct_invalid": 0.0}

    n_before = len(df)
    mask_valid = pd.Series(True, index=df.index)

    # No negative prices
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            neg = df[col] < 0
            if neg.any():
                logger.warning(f"  {symbol}: {neg.sum()} negative {col} values removed")
            mask_valid &= ~neg

    # Volume >= 0
    if "volume" in df.columns:
        neg_vol = df["volume"] < 0
        if neg_vol.any():
            logger.warning(f"  {symbol}: {neg_vol.sum()} negative volume values removed")
        mask_valid &= ~neg_vol

    # High >= Low
    hl_bad = df["high"] < df["low"]
    if hl_bad.any():
        logger.warning(f"  {symbol}: {hl_bad.sum()} rows with high < low removed")
    mask_valid &= ~hl_bad

    # Close within [Low, High]
    close_bad = (df["close"] < df["low"]) | (df["close"] > df["high"])
    if close_bad.any():
        logger.warning(f"  {symbol}: {close_bad.sum()} rows with close outside [low,high] removed")
    mask_valid &= ~close_bad

    df_clean = df[mask_valid].copy()
    n_dropped = n_before - len(df_clean)

    metrics = {
        "rows_before_validation": n_before,
        "rows_dropped": n_dropped,
        "pct_invalid": round(100.0 * n_dropped / n_before, 4) if n_before > 0 else 0.0,
    }

    if n_dropped > 0:
        logger.info(f"  {symbol}: Validation dropped {n_dropped}/{n_before} rows ({metrics['pct_invalid']:.2f}%)")

    return df_clean, metrics


def _compute_quality_metrics(df: pd.DataFrame, symbol: str, freq: str = "1h") -> dict:
    """
    Compute data quality metrics: coverage, gaps, missing %.
    """
    if df.empty:
        return {
            "symbol": symbol,
            "rows": 0,
            "date_range_start": None,
            "date_range_end": None,
            "pct_missing": 100.0,
            "gaps_detected": 0,
            "max_gap_hours": 0,
        }

    expected_freq = pd.Timedelta(freq)
    date_start = df.index.min()
    date_end = df.index.max()
    expected_rows = int((date_end - date_start) / expected_freq) + 1
    actual_rows = len(df)
    pct_missing = round(100.0 * (1 - actual_rows / max(expected_rows, 1)), 4)

    # Detect gaps
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > expected_freq * 1.5]
    max_gap_hours = round(gaps.max().total_seconds() / 3600, 2) if len(gaps) > 0 else 0.0

    return {
        "symbol": symbol,
        "rows": actual_rows,
        "date_range_start": str(date_start),
        "date_range_end": str(date_end),
        "pct_missing": pct_missing,
        "gaps_detected": len(gaps),
        "max_gap_hours": max_gap_hours,
    }


# ---------------------------------------------------------------------------
# Fetcher with retries and fallback
# ---------------------------------------------------------------------------

class MultiExchangeFetcher:
    """
    Fetches OHLCV data from a primary exchange (Binance) with fallback
    to alternative exchanges if the primary fails.
    """

    def __init__(self, fallback_exchanges: Optional[List[str]] = None,
                 retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS):
        self.retry_attempts = retry_attempts
        self.fallback_exchanges = fallback_exchanges or _DEFAULT_FALLBACK_EXCHANGES
        self._exchanges: dict = {}  # lazy-init cache

    def _get_exchange(self, name: str):
        """Lazy-initialise and cache exchange objects."""
        if name not in self._exchanges:
            self._exchanges[name] = _create_exchange(name)
        return self._exchanges[name]

    def fetch_ohlcv(
        self,
        symbol: str,
        pair_override: Optional[str] = None,
        timeframe: str = "1h",
        since: str = "2022-01-01",
        until: str = "2025-12-31",
    ) -> Tuple[pd.DataFrame, str]:
        """
        Fetch OHLCV for *symbol* with retry + fallback.

        Returns (df, exchange_used).  ``exchange_used`` is a string like
        "binance" or "kraken".
        """
        exchange_order = ["binance"] + [
            e for e in self.fallback_exchanges if e != "binance"
        ]

        for exch_name in exchange_order:
            pair = pair_override or _symbol_for_exchange(symbol, exch_name)
            df = self._fetch_from_exchange(exch_name, pair, timeframe, since, until)
            if not df.empty:
                return df, exch_name
            logger.warning(f"  {symbol}: {exch_name} returned no data, trying next exchange...")

        logger.error(f"  {symbol}: All exchanges exhausted — no data")
        return pd.DataFrame(), "none"

    def _fetch_from_exchange(
        self,
        exchange_name: str,
        pair: str,
        timeframe: str,
        since: str,
        until: str,
    ) -> pd.DataFrame:
        """Fetch from a single exchange with exponential-backoff retries."""
        exchange = self._get_exchange(exchange_name)
        since_ts = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
        until_ts = int(datetime.strptime(until, "%Y-%m-%d").timestamp() * 1000)

        all_candles: list = []
        current = since_ts
        logger.info(f"Fetching {pair} from {exchange_name}...")

        while current < until_ts:
            candles = self._fetch_with_retry(exchange, pair, timeframe, current)
            if candles is None or len(candles) == 0:
                break
            all_candles.extend(candles)
            current = candles[-1][0] + 1
            _time.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Trim to requested range
        end_dt = pd.Timestamp(until, tz="UTC")
        df = df[df.index <= end_dt]

        logger.info(f"  {pair} ({exchange_name}): {len(df)} candles")
        return df

    def _fetch_with_retry(self, exchange, pair, timeframe, since_ts) -> Optional[list]:
        """Single page fetch with exponential backoff."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return exchange.fetch_ohlcv(
                    pair, timeframe=timeframe, since=since_ts, limit=1000
                )
            except Exception as e:
                wait = min(2 ** attempt, 60)
                logger.warning(
                    f"    Attempt {attempt}/{self.retry_attempts} failed for {pair}: {e}  "
                    f"— retrying in {wait}s"
                )
                _time.sleep(wait)
        logger.error(f"    All {self.retry_attempts} retries exhausted for {pair}")
        return None


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_treasury_series(
    start: str,
    end: str,
    freq: str = "1h",
    base_price: float = 1.0,
    annual_yield: float = 0.045,
    name: str = "BUIDL",
) -> pd.DataFrame:
    """
    Geometric Brownian Motion with drift = annual_yield and volatility
    calibrated from historical T-bill rate vol (~15 bps annualised).

    Business-day adjustment: volume is reduced to ~10 % on weekends to
    reflect real Treasury trading patterns.
    """
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(idx)

    # GBM parameters
    dt = 1.0 / (365.25 * 24)  # hourly step in years
    mu = annual_yield
    sigma = 0.0015  # ~15 bps annualised — calibrated to T-bill rate vol

    rng = np.random.default_rng(seed=42 + hash(name) % 2**31)
    z = rng.standard_normal(n)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_returns[0] = 0.0  # start exactly at base_price
    prices = base_price * np.exp(np.cumsum(log_returns))

    # Floor: treasury token price should not drop below 99 % of base
    prices = np.maximum(prices, base_price * 0.99)

    # OHLCV from GBM close
    spread = 0.00005 * base_price
    high = prices + rng.uniform(0, spread, n)
    low = prices - rng.uniform(0, spread, n)
    open_ = np.roll(prices, 1)
    open_[0] = base_price

    # Volume: lower on weekends (business day adjustment)
    is_weekend = np.isin(idx.dayofweek, [5, 6])
    vol_base = rng.uniform(1e6, 5e6, n)
    vol_base[is_weekend] *= 0.10

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": vol_base},
        index=idx,
    )

    logger.info(f"  {name}: {len(df)} synthetic GBM candles (yield={annual_yield:.1%})")
    return df


def generate_stablecoin_series(
    start: str,
    end: str,
    freq: str = "1h",
    name: str = "USDC",
) -> pd.DataFrame:
    """
    Generate synthetic stablecoin price series with realistic depeg events.

    Includes the SVB crisis (March 10-13 2023) where USDC dropped to ~$0.87.
    """
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(idx)

    rng = np.random.default_rng(seed=123)
    prices = 1.0 + rng.normal(0, 0.00002, n)

    # --- SVB depeg event: 10 Mar 2023 to 13 Mar 2023 ---
    svb_start = pd.Timestamp("2023-03-10 18:00", tz="UTC")
    svb_trough = pd.Timestamp("2023-03-11 12:00", tz="UTC")
    svb_recovery = pd.Timestamp("2023-03-13 12:00", tz="UTC")
    svb_end = pd.Timestamp("2023-03-14 06:00", tz="UTC")

    for i, ts in enumerate(idx):
        if svb_start <= ts <= svb_trough:
            # Drop phase: linear decline from 1.0 to 0.87
            frac = (ts - svb_start) / (svb_trough - svb_start)
            prices[i] = 1.0 - 0.13 * frac + rng.normal(0, 0.003)
        elif svb_trough < ts <= svb_recovery:
            # Recovery phase: 0.87 back towards 0.98
            frac = (ts - svb_trough) / (svb_recovery - svb_trough)
            prices[i] = 0.87 + 0.11 * frac + rng.normal(0, 0.002)
        elif svb_recovery < ts <= svb_end:
            # Final convergence to 1.0
            frac = (ts - svb_recovery) / (svb_end - svb_recovery)
            prices[i] = 0.98 + 0.02 * frac + rng.normal(0, 0.0005)

    # --- Smaller depeg event: 15 May 2022 (UST contagion) ---
    ust_start = pd.Timestamp("2022-05-12 00:00", tz="UTC")
    ust_trough = pd.Timestamp("2022-05-12 12:00", tz="UTC")
    ust_end = pd.Timestamp("2022-05-14 00:00", tz="UTC")

    for i, ts in enumerate(idx):
        if ust_start <= ts <= ust_trough:
            frac = (ts - ust_start) / (ust_trough - ust_start)
            prices[i] = 1.0 - 0.05 * frac + rng.normal(0, 0.001)
        elif ust_trough < ts <= ust_end:
            frac = (ts - ust_trough) / (ust_end - ust_trough)
            prices[i] = 0.95 + 0.05 * frac + rng.normal(0, 0.0008)

    prices = np.clip(prices, 0.85, 1.02)

    spread = 0.00001
    df = pd.DataFrame(
        {
            "open": np.roll(prices, 1),
            "high": prices + rng.uniform(0, spread, n),
            "low": prices - rng.uniform(0, spread, n),
            "close": prices,
            "volume": rng.uniform(1e8, 5e8, n),
        },
        index=idx,
    )
    df.iloc[0, df.columns.get_loc("open")] = 1.0

    logger.info(f"  {name}: {len(df)} synthetic candles (with depeg events)")
    return df


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

def _print_quality_table(report: List[dict]):
    """Pretty-print data quality summary to logger."""
    header = f"{'Asset':<8} {'Rows':>8} {'Start':>22} {'End':>22} {'Gaps':>5} {'MaxGap(h)':>10} {'Miss%':>7} {'Source':<10}"
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for r in report:
        logger.info(
            f"{r.get('symbol','?'):<8} "
            f"{r.get('rows',0):>8} "
            f"{str(r.get('date_range_start',''))[:22]:>22} "
            f"{str(r.get('date_range_end',''))[:22]:>22} "
            f"{r.get('gaps_detected',0):>5} "
            f"{r.get('max_gap_hours',0):>10} "
            f"{r.get('pct_missing',0):>7.2f} "
            f"{r.get('exchange_source','synthetic'):<10}"
        )
    logger.info(sep)


def _save_quality_report(report: List[dict], cache_dir: Path):
    """Persist quality report as JSON."""
    report_path = cache_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Data quality report saved to {report_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_data(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Fetch (or load from cache) OHLCV data for every asset in config.

    Returns dict: symbol -> DataFrame with OHLCV columns.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    start = data_cfg.get("start_date", "2022-01-01")
    end = data_cfg.get("end_date", "2025-12-31")
    symbols = data_cfg.get("assets", [])
    retry_attempts = data_cfg.get("retry_attempts", _DEFAULT_RETRY_ATTEMPTS)
    fallback_exchanges = data_cfg.get("fallback_exchanges", _DEFAULT_FALLBACK_EXCHANGES)
    validation_cfg = data_cfg.get("validation", {})
    min_coverage = validation_cfg.get("min_coverage_pct", _DEFAULT_MIN_COVERAGE_PCT)
    max_gap = validation_cfg.get("max_gap_hours", _DEFAULT_MAX_GAP_HOURS)

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_hash_manifest(cache_dir)

    fetcher = MultiExchangeFetcher(
        fallback_exchanges=fallback_exchanges,
        retry_attempts=retry_attempts,
    )

    # Known Binance pair overrides
    pair_overrides = {
        "BTC": "BTC/USDT",
        "ETH": "ETH/USDT",
        "SOL": "SOL/USDT",
        "stETH": "STETH/USDT",
        "rETH": "RETH/USDT",
    }
    treasuries = {"BUIDL": 0.045, "USDY": 0.05}

    all_data: Dict[str, pd.DataFrame] = {}
    quality_report: List[dict] = []

    for sym in symbols:
        cache_file = cache_dir / f"{sym}_hourly.parquet"

        # --- Try cache first (with integrity check) ---
        if cache_file.exists():
            if _verify_cache_integrity(cache_file, manifest):
                logger.info(f"  {sym}: loaded from verified cache")
                df = pd.read_parquet(cache_file)
                metrics = _compute_quality_metrics(df, sym)
                metrics["exchange_source"] = "cache"
                quality_report.append(metrics)
                all_data[sym] = df
                continue
            else:
                logger.warning(f"  {sym}: cache integrity failed — re-fetching")

        # --- Fetch fresh data ---
        exchange_used = "synthetic"
        if sym in treasuries:
            df = generate_treasury_series(
                start, end, annual_yield=treasuries[sym], name=sym
            )
        elif sym == "USDC":
            df = generate_stablecoin_series(start, end, name=sym)
        else:
            pair = pair_overrides.get(sym)
            df, exchange_used = fetcher.fetch_ohlcv(
                sym, pair_override=pair, timeframe="1h", since=start, until=end
            )

        if df.empty:
            logger.error(f"  {sym}: NO DATA — skipping")
            quality_report.append(
                {"symbol": sym, "rows": 0, "exchange_source": exchange_used}
            )
            continue

        # --- Validate ---
        df, val_metrics = validate_ohlcv(df, sym)
        if df.empty:
            logger.error(f"  {sym}: All rows invalid after validation — skipping")
            continue

        # --- Quality metrics ---
        metrics = _compute_quality_metrics(df, sym)
        metrics["exchange_source"] = exchange_used
        metrics.update(val_metrics)

        # Warn on low coverage or large gaps
        if metrics["pct_missing"] > (100 - min_coverage):
            logger.warning(
                f"  {sym}: coverage {100 - metrics['pct_missing']:.1f}% "
                f"< required {min_coverage}%"
            )
        if metrics["max_gap_hours"] > max_gap:
            logger.warning(
                f"  {sym}: max gap {metrics['max_gap_hours']:.1f}h "
                f"> threshold {max_gap}h"
            )

        quality_report.append(metrics)

        # --- Cache + manifest ---
        df.to_parquet(cache_file)
        manifest[cache_file.name] = _compute_file_hash(cache_file)
        all_data[sym] = df

    _save_hash_manifest(cache_dir, manifest)

    # --- Quality summary ---
    _print_quality_table(quality_report)
    _save_quality_report(quality_report, cache_dir)

    logger.info(f"Loaded {len(all_data)}/{len(symbols)} assets")
    return all_data


def load_cached_data(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Load previously-fetched OHLCV data from parquet cache.

    This is the lightweight entry point used by preprocess.py so that
    preprocessing does not trigger a network fetch.  Falls back to
    ``fetch_all_data`` if the cache is empty.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    symbols = config.get("data", {}).get("assets", [])
    cache_dir = Path("data/cache")
    manifest = _load_hash_manifest(cache_dir) if cache_dir.exists() else {}

    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        cache_file = cache_dir / f"{sym}_hourly.parquet"
        if cache_file.exists():
            if manifest and not _verify_cache_integrity(cache_file, manifest):
                logger.warning(f"  {sym}: cache integrity failed during load")
            data[sym] = pd.read_parquet(cache_file)

    if not data:
        logger.warning("No cached data found — triggering full fetch")
        return fetch_all_data(config_path)

    logger.info(f"Loaded {len(data)} assets from cache")
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = fetch_all_data()
    for s, d in data.items():
        print(f"{s}: {len(d)} rows")
