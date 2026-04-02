"""
Fetch historical OHLCV data for all portfolio constituents.

Sources:
  - Primary: Binance via ccxt
  - Fallback: Kraken, Coinbase via ccxt
  - Synthetic: GBM for Treasuries (BUIDL/USDY), depeg-aware for USDC

Includes:
  - Adaptive token-bucket rate limiter with latency tracking
  - Retry handler with circuit breaker pattern per exchange
  - Connection pool with health checks and auto-reconnect
  - Parallel fetching via ThreadPoolExecutor with progress tracking
  - Post-fetch data integrity pipeline (gap fill, anomaly detection)
  - Exponential-backoff retries with jitter
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
import random
import yaml
import numpy as np
import pandas as pd
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, List, Tuple

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm not installed
    class tqdm:  # type: ignore[no-redef]
        """Minimal no-op shim when tqdm is unavailable."""
        def __init__(self, iterable=None, **kw):
            self._it = iterable
            self.total = kw.get("total", 0)
        def __iter__(self):
            return iter(self._it or [])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s, refresh=True):
            pass

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
# Adaptive Token Bucket Rate Limiter
# ---------------------------------------------------------------------------

class AdaptiveRateLimiter:
    """
    Thread-safe token-bucket rate limiter that adapts to exchange pressure.

    Tracks recent request latencies; when latency spikes above 2x the
    running average the limiter halves its request rate.  When latency
    returns to normal the rate gradually recovers (10% per successful
    low-latency request).
    """

    def __init__(self, max_rps: float = 10.0, burst: int = 5):
        self._max_rps = max_rps
        self._current_rps = max_rps
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = _time.monotonic()
        self._lock = Lock()

        # Latency tracking (sliding window of last 50 measurements)
        self._latencies: deque = deque(maxlen=50)
        self._latency_lock = Lock()
        self._baseline_latency: Optional[float] = None

    def acquire(self) -> float:
        """
        Block until a token is available.  Returns the time spent waiting
        (in seconds).
        """
        waited = 0.0
        while True:
            with self._lock:
                now = _time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._burst,
                    self._tokens + elapsed * self._current_rps,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited

            # No token available — wait for the next one
            sleep_time = 1.0 / max(self._current_rps, 0.1)
            _time.sleep(sleep_time)
            waited += sleep_time

    def record_latency(self, latency_ms: float):
        """
        Record a request latency and adapt the rate if needed.

        * Spike (>2x average) -> halve rate (floor: 10% of max)
        * Normal               -> increase rate by 10% (cap: max_rps)
        """
        with self._latency_lock:
            self._latencies.append(latency_ms)
            if len(self._latencies) < 5:
                return  # not enough data yet
            avg = sum(self._latencies) / len(self._latencies)
            if self._baseline_latency is None:
                self._baseline_latency = avg

        with self._lock:
            if latency_ms > 2.0 * (self._baseline_latency or avg):
                # Pressure detected — throttle
                self._current_rps = max(
                    self._current_rps * 0.5,
                    self._max_rps * 0.1,
                )
                logger.debug(
                    f"Rate limiter: latency spike ({latency_ms:.0f}ms vs "
                    f"avg {avg:.0f}ms) — throttled to {self._current_rps:.1f} rps"
                )
            else:
                # Normal — recover gradually
                self._current_rps = min(
                    self._current_rps * 1.10,
                    self._max_rps,
                )

    @property
    def current_rps(self) -> float:
        with self._lock:
            return self._current_rps


# ---------------------------------------------------------------------------
# Retry Handler with Circuit Breaker
# ---------------------------------------------------------------------------

class RetryHandler:
    """
    Exponential backoff with jitter and per-exchange circuit breaker.

    After *circuit_breaker_threshold* consecutive failures for a given
    exchange the circuit opens and all requests to that exchange are
    rejected for *circuit_breaker_cooldown* seconds.
    """

    def __init__(
        self,
        max_retries: int = 5,
        circuit_breaker_threshold: int = 10,
        circuit_breaker_cooldown: int = 300,
    ):
        self.max_retries = max_retries
        self.cb_threshold = circuit_breaker_threshold
        self.cb_cooldown = circuit_breaker_cooldown

        # Per-exchange tracking  {name: {"failures": int, "successes": int,
        #                                 "consecutive_fails": int,
        #                                 "open_until": float}}
        self._stats: Dict[str, dict] = {}
        self._lock = Lock()

    def _ensure_exchange(self, name: str):
        if name not in self._stats:
            self._stats[name] = {
                "failures": 0,
                "successes": 0,
                "retries": 0,
                "consecutive_fails": 0,
                "open_until": 0.0,
            }

    def is_circuit_open(self, exchange_name: str) -> bool:
        with self._lock:
            self._ensure_exchange(exchange_name)
            s = self._stats[exchange_name]
            if s["open_until"] > _time.monotonic():
                return True
            if s["open_until"] > 0:
                # Cooldown expired — reset
                s["consecutive_fails"] = 0
                s["open_until"] = 0.0
            return False

    def execute(self, func, *args, exchange_name: str = "unknown", **kwargs):
        """
        Call *func(*args, **kwargs)* with retries and circuit breaker.
        Returns the function result or raises the last exception.
        """
        if self.is_circuit_open(exchange_name):
            raise RuntimeError(
                f"Circuit breaker OPEN for {exchange_name} — skipping"
            )

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self._ensure_exchange(exchange_name)
                    self._stats[exchange_name]["successes"] += 1
                    self._stats[exchange_name]["consecutive_fails"] = 0
                return result
            except Exception as exc:
                last_exc = exc
                with self._lock:
                    self._ensure_exchange(exchange_name)
                    s = self._stats[exchange_name]
                    s["failures"] += 1
                    s["retries"] += 1
                    s["consecutive_fails"] += 1
                    if s["consecutive_fails"] >= self.cb_threshold:
                        s["open_until"] = (
                            _time.monotonic() + self.cb_cooldown
                        )
                        logger.warning(
                            f"Circuit breaker OPENED for {exchange_name} "
                            f"after {s['consecutive_fails']} consecutive "
                            f"failures — cooldown {self.cb_cooldown}s"
                        )
                        raise

                # Exponential backoff with full jitter
                base_wait = min(2 ** attempt, 60)
                wait = random.uniform(0, base_wait)
                logger.warning(
                    f"    Retry {attempt}/{self.max_retries} for "
                    f"{exchange_name}: {exc} — backoff {wait:.1f}s"
                )
                _time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    def get_stats(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._stats.items()}


# ---------------------------------------------------------------------------
# Exchange Connection Pool
# ---------------------------------------------------------------------------

class ExchangeConnectionPool:
    """
    Pre-initialises and reuses ccxt exchange instances.  Provides health
    checks and automatic reconnection on failure.
    """

    def __init__(self, exchanges: Optional[List[str]] = None):
        self._pool: Dict[str, object] = {}
        self._lock = Lock()
        if exchanges:
            for name in exchanges:
                self._init_exchange(name)

    def _init_exchange(self, name: str):
        """Create a fresh ccxt exchange instance."""
        import ccxt
        exchange_cls = getattr(ccxt, name, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown exchange: {name}")
        self._pool[name] = exchange_cls({"enableRateLimit": True})
        logger.debug(f"Connection pool: initialised {name}")

    def get(self, name: str):
        """Return a cached exchange instance, creating one if needed."""
        with self._lock:
            if name not in self._pool:
                self._init_exchange(name)
            return self._pool[name]

    def reconnect(self, name: str):
        """Force-recreate the connection for *name*."""
        with self._lock:
            logger.info(f"Connection pool: reconnecting {name}")
            self._init_exchange(name)
            return self._pool[name]

    def health_check(self) -> Dict[str, bool]:
        """Ping each exchange and return {name: is_healthy}."""
        results: Dict[str, bool] = {}
        for name, exch in list(self._pool.items()):
            try:
                exch.fetch_time()  # type: ignore[union-attr]
                results[name] = True
            except Exception:
                results[name] = False
        return results


# ---------------------------------------------------------------------------
# Streaming Progress Reporter
# ---------------------------------------------------------------------------

@dataclass
class _FetchStats:
    """Mutable counters shared across workers."""
    total_assets: int = 0
    completed: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    data_points: int = 0
    retries: int = 0
    start_time: float = field(default_factory=_time.monotonic)


class FetchProgressReporter:
    """Real-time progress bar + end-of-run summary for fetch_all_data."""

    def __init__(self, total: int):
        self._stats = _FetchStats(total_assets=total)
        self._lock = Lock()
        self._bar = tqdm(
            total=total, desc="Fetching assets", unit="asset",
            dynamic_ncols=True,
        )

    def tick(self, symbol: str, cache_hit: bool = False,
             data_points: int = 0, api_calls: int = 0):
        with self._lock:
            self._stats.completed += 1
            self._stats.data_points += data_points
            self._stats.api_calls += api_calls
            if cache_hit:
                self._stats.cache_hits += 1
        self._bar.set_postfix_str(symbol, refresh=True)
        self._bar.update(1)

    def add_retries(self, n: int = 1):
        with self._lock:
            self._stats.retries += n

    def close(self):
        self._bar.close()
        elapsed = _time.monotonic() - self._stats.start_time
        s = self._stats
        logger.info(
            f"Fetch complete in {elapsed:.1f}s | "
            f"{s.completed}/{s.total_assets} assets | "
            f"cache hits: {s.cache_hits} | "
            f"API calls: {s.api_calls} | "
            f"data points: {s.data_points:,} | "
            f"retries: {s.retries}"
        )

    @property
    def stats(self) -> _FetchStats:
        return self._stats


# ---------------------------------------------------------------------------
# Data Integrity Pipeline
# ---------------------------------------------------------------------------

def verify_data_integrity(
    df: pd.DataFrame, symbol: str, freq: str = "1h",
) -> Tuple[pd.DataFrame, dict]:
    """
    Post-fetch verification and repair.

    1. Verify timestamps are monotonically increasing (sort if not).
    2. Detect gaps: fill small ones (<=3 candles) via interpolation,
       flag large ones (>3 candles).
    3. Detect price anomalies (>20% single-candle move) and flag them.
    4. Compute a completeness score.

    Returns (repaired_df, integrity_report).
    """
    if df.empty:
        return df, {"symbol": symbol, "completeness_score": 0.0,
                     "gaps_filled": 0, "gaps_flagged": 0,
                     "anomalies_flagged": 0}

    report: dict = {"symbol": symbol, "gaps_filled": 0, "gaps_flagged": 0,
                     "anomalies_flagged": 0, "large_gap_locations": [],
                     "anomaly_locations": []}

    # 1. Monotonic timestamps
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        logger.debug(f"  {symbol}: sorted non-monotonic timestamps")

    # 2. Gap detection & repair
    expected_td = pd.Timedelta(freq)
    full_idx = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, tz="UTC",
    )
    missing_idx = full_idx.difference(df.index)

    if len(missing_idx) > 0:
        # Group consecutive missing timestamps into gap runs
        gaps: List[List[pd.Timestamp]] = []
        current_run: List[pd.Timestamp] = [missing_idx[0]]
        for i in range(1, len(missing_idx)):
            if missing_idx[i] - missing_idx[i - 1] <= expected_td * 1.5:
                current_run.append(missing_idx[i])
            else:
                gaps.append(current_run)
                current_run = [missing_idx[i]]
        gaps.append(current_run)

        small_fill_idx = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
        for gap in gaps:
            if len(gap) <= 3:
                small_fill_idx = small_fill_idx.append(
                    pd.DatetimeIndex(gap)
                )
                report["gaps_filled"] += len(gap)
            else:
                report["gaps_flagged"] += 1
                report["large_gap_locations"].append(
                    {"start": str(gap[0]), "end": str(gap[-1]),
                     "missing_candles": len(gap)}
                )

        # Interpolate small gaps
        if len(small_fill_idx) > 0:
            df = df.reindex(df.index.union(small_fill_idx))
            df = df.interpolate(method="time")
            df = df.sort_index()
            logger.debug(
                f"  {symbol}: interpolated {report['gaps_filled']} "
                f"small-gap candles"
            )

    # 3. Price anomaly detection (>20% single-candle move)
    pct_change = df["close"].pct_change().abs()
    anomaly_mask = pct_change > 0.20
    anomaly_count = anomaly_mask.sum()
    if anomaly_count > 0:
        report["anomalies_flagged"] = int(anomaly_count)
        report["anomaly_locations"] = [
            str(ts) for ts in df.index[anomaly_mask][:20]  # cap at 20
        ]
        logger.debug(
            f"  {symbol}: {anomaly_count} price anomalies (>20% move) flagged"
        )

    # 4. Completeness score
    expected_total = len(full_idx) if len(missing_idx) > 0 else len(
        pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    )
    report["completeness_score"] = round(
        100.0 * len(df) / max(expected_total, 1), 2
    )

    return df, report


# ---------------------------------------------------------------------------
# Fetcher with retries and fallback
# ---------------------------------------------------------------------------

class MultiExchangeFetcher:
    """
    Fetches OHLCV data from a primary exchange (Binance) with fallback
    to alternative exchanges if the primary fails.

    Enhanced with adaptive rate limiting, circuit-breaker retries,
    connection pooling, and data integrity verification.
    """

    def __init__(self, fallback_exchanges: Optional[List[str]] = None,
                 retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS):
        self.retry_attempts = retry_attempts
        self.fallback_exchanges = fallback_exchanges or _DEFAULT_FALLBACK_EXCHANGES
        self._exchanges: dict = {}  # legacy lazy-init cache (kept for compat)

        # --- New infrastructure ---
        all_exchanges = ["bybit", "gate", "binanceus", "binance"] + list(self.fallback_exchanges)
        self._pool = ExchangeConnectionPool(all_exchanges)
        self._rate_limiter = AdaptiveRateLimiter(max_rps=10.0, burst=5)
        self._retry_handler = RetryHandler(
            max_retries=retry_attempts,
            circuit_breaker_threshold=10,
            circuit_breaker_cooldown=300,
        )
        self._api_call_count = 0
        self._api_call_lock = Lock()

    def _get_exchange(self, name: str):
        """Return exchange from connection pool (backward-compatible)."""
        return self._pool.get(name)

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
        exchange_order = ["bybit", "gate", "binanceus", "binance"] + [
            e for e in self.fallback_exchanges if e not in ("binance", "bybit", "gate", "binanceus")
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
        """
        Fetch from a single exchange with adaptive rate limiting,
        circuit-breaker retries, and HTTP 429/418 handling.
        """
        if self._retry_handler.is_circuit_open(exchange_name):
            logger.warning(
                f"  {pair}: circuit breaker open for {exchange_name} — skipping"
            )
            return pd.DataFrame()

        exchange = self._get_exchange(exchange_name)
        since_ts = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
        until_ts = int(datetime.strptime(until, "%Y-%m-%d").timestamp() * 1000)

        all_candles: list = []
        current = since_ts
        page_calls = 0
        logger.info(f"Fetching {pair} from {exchange_name}...")

        while current < until_ts:
            # Adaptive rate limiting (replaces fixed sleep)
            self._rate_limiter.acquire()

            t0 = _time.monotonic()
            candles = self._fetch_with_retry(exchange, exchange_name, pair,
                                              timeframe, current)
            latency_ms = (_time.monotonic() - t0) * 1000.0
            self._rate_limiter.record_latency(latency_ms)

            with self._api_call_lock:
                self._api_call_count += 1
            page_calls += 1

            if candles is None or len(candles) == 0:
                break

            # Filter out partial candles beyond our range
            candles = [c for c in candles if c[0] <= until_ts]
            all_candles.extend(candles)
            current = candles[-1][0] + 1

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

        logger.info(
            f"  {pair} ({exchange_name}): {len(df)} candles "
            f"({page_calls} API pages)"
        )
        return df

    def _fetch_with_retry(self, exchange, exchange_name: str, pair: str,
                          timeframe: str, since_ts: int) -> Optional[list]:
        """
        Single page fetch with circuit-breaker retries, jitter backoff,
        and specific HTTP 429 / 418 handling.
        """
        try:
            return self._retry_handler.execute(
                exchange.fetch_ohlcv,
                pair, timeframe, since_ts, 1000,
                exchange_name=exchange_name,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            # HTTP 429 — rate limited: honour Retry-After if present
            if "429" in exc_str or "rate limit" in exc_str:
                retry_after = 30  # default
                # ccxt sometimes embeds the header value in the message
                for token in str(exc).split():
                    if token.isdigit():
                        retry_after = min(int(token), 120)
                        break
                logger.warning(
                    f"    HTTP 429 for {pair} on {exchange_name} — "
                    f"backing off {retry_after}s"
                )
                _time.sleep(retry_after)
                return None
            # HTTP 418 — IP ban: switch exchange immediately
            if "418" in exc_str:
                logger.error(
                    f"    HTTP 418 (IP ban) for {pair} on {exchange_name} "
                    f"— switching exchange"
                )
                return None
            logger.error(
                f"    All retries exhausted for {pair} on {exchange_name}: "
                f"{exc}"
            )
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

def fetch_all_data(config_path: str = "config.yaml",
                   max_workers: int = 4) -> Dict[str, pd.DataFrame]:
    """
    Fetch (or load from cache) OHLCV data for every asset in config.

    Crypto assets are fetched in parallel via ThreadPoolExecutor.
    Synthetic assets (treasuries, stablecoin) are generated synchronously.

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
    manifest_lock = Lock()

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
    synthetic_symbols = set(treasuries.keys()) | {"USDC"}

    all_data: Dict[str, pd.DataFrame] = {}
    quality_report: List[dict] = []
    data_lock = Lock()
    report_lock = Lock()

    progress = FetchProgressReporter(total=len(symbols))

    # ------------------------------------------------------------------
    # Helper: process a single symbol (runs in worker thread or main)
    # ------------------------------------------------------------------
    def _process_symbol(sym: str) -> None:
        cache_file = cache_dir / f"{sym}_hourly.parquet"

        # --- Try cache first (with integrity check) ---
        with manifest_lock:
            cache_valid = (
                cache_file.exists()
                and _verify_cache_integrity(cache_file, manifest)
            )

        if cache_valid:
            logger.info(f"  {sym}: loaded from verified cache")
            df = pd.read_parquet(cache_file)
            metrics = _compute_quality_metrics(df, sym)
            metrics["exchange_source"] = "cache"
            with report_lock:
                quality_report.append(metrics)
            with data_lock:
                all_data[sym] = df
            progress.tick(sym, cache_hit=True, data_points=len(df))
            return

        if cache_file.exists():
            logger.warning(f"  {sym}: cache integrity failed — re-fetching")

        # --- Fetch fresh data ---
        exchange_used = "synthetic"
        api_calls = 0
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
            with fetcher._api_call_lock:
                api_calls = fetcher._api_call_count

        if df.empty:
            logger.error(f"  {sym}: NO DATA — skipping")
            with report_lock:
                quality_report.append(
                    {"symbol": sym, "rows": 0, "exchange_source": exchange_used}
                )
            progress.tick(sym, api_calls=api_calls)
            return

        # --- Validate ---
        df, val_metrics = validate_ohlcv(df, sym)
        if df.empty:
            logger.error(f"  {sym}: All rows invalid after validation — skipping")
            progress.tick(sym, api_calls=api_calls)
            return

        # --- Data integrity pipeline ---
        df, integrity_report = verify_data_integrity(df, sym)

        # --- Quality metrics ---
        metrics = _compute_quality_metrics(df, sym)
        metrics["exchange_source"] = exchange_used
        metrics.update(val_metrics)
        metrics["integrity"] = integrity_report

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

        with report_lock:
            quality_report.append(metrics)

        # --- Cache + manifest ---
        df.to_parquet(cache_file)
        file_hash = _compute_file_hash(cache_file)
        with manifest_lock:
            manifest[cache_file.name] = file_hash

        with data_lock:
            all_data[sym] = df

        progress.tick(sym, data_points=len(df), api_calls=api_calls)

    # ------------------------------------------------------------------
    # Phase 1: Synthetic assets (instant, no parallelisation needed)
    # ------------------------------------------------------------------
    for sym in symbols:
        if sym in synthetic_symbols:
            _process_symbol(sym)

    # ------------------------------------------------------------------
    # Phase 2: Crypto assets in parallel via ThreadPoolExecutor
    # ------------------------------------------------------------------
    crypto_symbols = [s for s in symbols if s not in synthetic_symbols]
    if crypto_symbols:
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_symbol, sym): sym
                    for sym in crypto_symbols
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            f"  {sym}: worker raised {type(exc).__name__}: {exc}"
                        )
                        progress.tick(sym)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt — shutting down workers...")

    progress.close()

    _save_hash_manifest(cache_dir, manifest)

    # --- Retry handler statistics ---
    retry_stats = fetcher._retry_handler.get_stats()
    if retry_stats:
        logger.info(f"Retry handler stats: {retry_stats}")

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
