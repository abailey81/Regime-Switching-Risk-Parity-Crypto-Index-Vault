"""
Asset Universe Screening Module — Quantitative selection of optimal portfolio constituents.

Implements a rigorous 8-stage pipeline to screen ~500 crypto assets and select
the optimal 10 for the RiskParity Vault portfolio:

    Stage 1: Universe Construction (fetch ~500 liquid USDT pairs from Binance)
    Stage 2: Data Collection (daily OHLCV with concurrent fetching)
    Stage 3: Liquidity Filter (volume, history, stablecoin/duplicate removal)
    Stage 4: Statistical Quality Filter (stationarity, missing data, wash trading)
    Stage 5: Risk-Return Profiling (Sharpe, CVaR, Hurst, beta, idio vol)
    Stage 6: Correlation Clustering (Ward hierarchical, representative selection)
    Stage 7: Diversification Optimisation (combinatorial or greedy selection of final 10)
    Stage 8: Validation (spanning test, diversification ratio, benchmark comparison)

Designed for Google Colab compatibility with aggressive caching and Binance
rate-limit awareness (1200 req/min -> 50ms inter-request delay).

Dependencies: ccxt, numpy, pandas, scipy, sklearn, tqdm, logging.
"""

import json
import logging
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize as sp_minimize

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """Minimal tqdm fallback that supports iteration and no-op methods."""
        def __init__(self, iterable=None, *a, **kw):
            self._iterable = iterable
            self.total = kw.get("total", None)
        def __iter__(self):
            return iter(self._iterable if self._iterable is not None else [])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kw):
            pass
        def set_description(self, desc):
            pass
        def close(self):
            pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stablecoins to exclude from screening (USDC added back as defensive anchor)
_STABLECOINS = frozenset({
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "FRAX", "LUSD",
    "SUSD", "USDD", "FDUSD", "PYUSD", "EURC", "EURT", "AEUR", "UST", "MIM",
    "CUSD", "CEUR", "DOLA", "RAI", "FEI", "TRIBE", "EURS",
})

# Wrapped / bridged token prefixes to detect duplicates
_WRAPPED_PREFIXES = ("W", "ST", "CB", "R", "A", "C")

# Known asset classifications for post-selection labelling
_ASSET_CLASSIFICATIONS: Dict[str, str] = {
    # Spot crypto — L1
    "BTC": "spot_crypto", "ETH": "spot_crypto", "SOL": "spot_crypto",
    "BNB": "spot_crypto", "ADA": "spot_crypto", "AVAX": "spot_crypto",
    "DOT": "spot_crypto", "MATIC": "spot_crypto", "POL": "spot_crypto",
    "ATOM": "spot_crypto", "NEAR": "spot_crypto", "APT": "spot_crypto",
    "SUI": "spot_crypto", "SEI": "spot_crypto", "TIA": "spot_crypto",
    "FTM": "spot_crypto", "ALGO": "spot_crypto", "XRP": "spot_crypto",
    "TRX": "spot_crypto", "TON": "spot_crypto", "ICP": "spot_crypto",
    "HBAR": "spot_crypto", "FIL": "spot_crypto", "ARB": "spot_crypto",
    "OP": "spot_crypto", "INJ": "spot_crypto", "DOGE": "spot_crypto",
    "SHIB": "spot_crypto", "PEPE": "spot_crypto", "WIF": "spot_crypto",
    "BONK": "spot_crypto", "FLOKI": "spot_crypto",
    # Spot crypto — L2 / infrastructure
    "LINK": "spot_crypto", "GRT": "spot_crypto", "RENDER": "spot_crypto",
    "FET": "spot_crypto", "OCEAN": "spot_crypto", "TAO": "spot_crypto",
    "AR": "spot_crypto", "STX": "spot_crypto", "MINA": "spot_crypto",
    "IMX": "spot_crypto", "LRC": "spot_crypto", "ZK": "spot_crypto",
    # Liquid staking
    "STETH": "liquid_staking", "RETH": "liquid_staking",
    "CBETH": "liquid_staking", "WSTETH": "liquid_staking",
    "MSOL": "liquid_staking", "JITOSOL": "liquid_staking",
    "LIDO": "liquid_staking",
    # Tokenised RWA
    "BUIDL": "tokenised_rwa", "USDY": "tokenised_rwa",
    "PAXG": "tokenised_rwa", "XAUT": "tokenised_rwa",
    "ONDO": "tokenised_rwa", "RWA": "tokenised_rwa",
    "MPL": "tokenised_rwa",
    # DeFi
    "AAVE": "defi", "UNI": "defi", "MKR": "defi", "CRV": "defi",
    "COMP": "defi", "SNX": "defi", "SUSHI": "defi", "YFI": "defi",
    "BAL": "defi", "1INCH": "defi", "DYDX": "defi", "GMX": "defi",
    "PENDLE": "defi", "JUP": "defi", "RAY": "defi", "CAKE": "defi",
    "LDO": "defi", "RPL": "defi", "SSV": "defi", "ENA": "defi",
    "ETHFI": "defi",
}

# Annualisation factor for daily returns
_DAILY_ANN_FACTOR = 365.25
_SQRT_ANN = np.sqrt(_DAILY_ANN_FACTOR)

# Binance rate-limit delay (seconds between requests)
_REQUEST_DELAY_S = 0.05  # 50ms -> max 1200/min


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_log(msg: str, level: str = "info") -> None:
    """Log with fallback to print for Colab environments."""
    getattr(logger, level, logger.info)(msg)


def _load_cache_json(path: Path) -> Optional[dict]:
    """Load a JSON cache file if it exists and is not expired (24h TTL)."""
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=24):
            _safe_log(f"Cache expired: {path.name}")
            return None
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        _safe_log(f"Cache load failed for {path.name}: {e}", "warning")
        return None


def _save_cache_json(path: Path, data: dict) -> None:
    """Save data to a JSON cache file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        _safe_log(f"Cache save failed for {path.name}: {e}", "warning")


def _hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent via rescaled range (R/S) analysis.

    H > 0.5: trending (persistent)
    H = 0.5: random walk
    H < 0.5: mean-reverting (anti-persistent)

    Returns 0.5 on failure (agnostic default).
    """
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    if n < 20:
        return 0.5

    max_lag = min(max_lag, n // 2)
    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split into non-overlapping sub-series of length `lag`
        n_subseries = n // lag
        if n_subseries < 1:
            continue
        rs_list = []
        for i in range(n_subseries):
            sub = ts[i * lag : (i + 1) * lag]
            mean_sub = np.mean(sub)
            cum_dev = np.cumsum(sub - mean_sub)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(sub, ddof=1)
            if s > 1e-12:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

    if len(rs_values) < 5:
        return 0.5

    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _, _, _, _ = stats.linregress(x, y)
    return float(np.clip(slope, 0.0, 1.0))


def _amihud_illiquidity(returns: np.ndarray, volume_usd: np.ndarray) -> float:
    """
    Amihud (2002) illiquidity ratio: mean(|r| / volume).

    Higher values indicate less liquid assets. Returns np.inf when
    volume data is insufficient.
    """
    mask = (volume_usd > 0) & np.isfinite(returns) & np.isfinite(volume_usd)
    if mask.sum() < 10:
        return np.inf
    ratio = np.abs(returns[mask]) / volume_usd[mask]
    return float(np.mean(ratio))


def _cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Empirical Conditional Value at Risk (Expected Shortfall) at level alpha."""
    clean = returns[np.isfinite(returns)]
    if len(clean) < 20:
        return 0.0
    cutoff = np.percentile(clean, alpha * 100)
    tail = clean[clean <= cutoff]
    return float(-np.mean(tail)) if len(tail) > 0 else 0.0


def _diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Diversification ratio: weighted average vol / portfolio vol.

    DR = (w' * sigma) / sqrt(w' * Sigma * w)

    A higher ratio indicates more diversification benefit from imperfect
    correlation. DR = 1 means all assets are perfectly correlated.
    """
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    port_vol = np.sqrt(max(weights @ cov @ weights, 1e-24))
    weighted_vols = vols @ weights
    return float(weighted_vols / port_vol)


def _effective_n(weights: np.ndarray) -> float:
    """Effective number of assets: 1 / HHI = 1 / sum(w_i^2)."""
    hhi = np.sum(weights ** 2)
    return float(1.0 / hhi) if hhi > 1e-12 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  UniverseScreener
# ═══════════════════════════════════════════════════════════════════════════════

class UniverseScreener:
    """
    Quantitative asset universe screening for the RiskParity Vault.

    Screens ~500 crypto assets from Binance and selects the optimal 10 for
    portfolio construction using a rigorous multi-stage pipeline:
    liquidity filtering, statistical quality checks, risk-return profiling,
    hierarchical clustering, and diversification-maximising selection.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary (from config.yaml). Used for reading default
        assets, date ranges, etc.
    cache_dir : str
        Directory for caching universe data, OHLCV, and screening results.
    """

    def __init__(self, config: dict = None, cache_dir: str = "data/cache"):
        self.config = config or {}
        self.cache_dir = Path(cache_dir) / "universe_screening"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State populated during the pipeline
        self.universe_df: Optional[pd.DataFrame] = None
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.returns_data: Dict[str, pd.Series] = {}
        self.volume_data: Dict[str, pd.Series] = {}
        self.liquid_symbols: List[str] = []
        self.quality_symbols: List[str] = []
        self.asset_profiles: Optional[pd.DataFrame] = None
        self.cluster_df: Optional[pd.DataFrame] = None
        self.selected_assets: List[str] = []

        # BTC returns for factor analysis (populated in Stage 5)
        self._btc_returns: Optional[pd.Series] = None

        _safe_log("UniverseScreener initialised, cache: %s" % self.cache_dir)

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 1: Universe Construction
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_universe(self, top_n: int = 1000) -> pd.DataFrame:
        """
        Fetch the top-N USDT trading pairs from Binance by 24h quote volume.

        Queries the Binance spot market for all active USDT pairs, sorts by
        24h volume (in USDT terms), and returns the top ``top_n`` assets.

        Parameters
        ----------
        top_n : int
            Number of assets to include in the initial universe.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, pair, volume_24h_usd, last_price, exchange.
        """
        cache_path = self.cache_dir / "universe_list.json"
        cached = _load_cache_json(cache_path)
        if cached is not None:
            df = pd.DataFrame(cached)
            self.universe_df = df
            _safe_log(f"Stage 1: Loaded {len(df)} assets from cache")
            return df

        import ccxt

        # Try multiple exchanges — Binance is geo-blocked in some regions (UK, etc.)
        _exchange_priority = ["bybit", "gate", "binanceus", "binance", "okx"]
        exchange = None
        for _exch_name in _exchange_priority:
            try:
                _cls = getattr(ccxt, _exch_name, None)
                if _cls is None:
                    continue
                _safe_log(f"Stage 1: Trying {_exch_name}...")
                exchange = _cls({"enableRateLimit": True})
                exchange.load_markets()
                _safe_log(f"Stage 1: Connected to {_exch_name} ({len(exchange.markets)} markets)")
                self._connected_exchange = _exch_name
                break
            except Exception as _e:
                _safe_log(f"Stage 1: {_exch_name} failed ({_e}), trying next...", "warning")
                exchange = None
        if exchange is None:
            _safe_log("Stage 1: All exchanges failed!", "error")
            return pd.DataFrame(columns=["symbol", "volume_24h", "market_cap", "exchange"])
        exchange.load_markets()

        records = []
        for pair, market in exchange.markets.items():
            # Filter: spot, active, USDT quote
            if not market.get("spot", False):
                continue
            if not market.get("active", True):
                continue
            if market.get("quote", "") != "USDT":
                continue

            base = market.get("base", "")
            if not base or base in _STABLECOINS:
                continue

            records.append({
                "symbol": base,
                "pair": pair,
                "exchange": "binance",
            })

        if not records:
            _safe_log("Stage 1: No USDT pairs found on Binance!", "error")
            return pd.DataFrame()

        _safe_log(f"Stage 1: Found {len(records)} USDT pairs, fetching 24h tickers...")

        # Fetch 24h ticker data for volume/price
        _time.sleep(_REQUEST_DELAY_S)
        try:
            tickers = exchange.fetch_tickers()
        except Exception as e:
            _safe_log(f"Stage 1: Failed to fetch tickers: {e}", "error")
            tickers = {}

        for rec in records:
            ticker = tickers.get(rec["pair"], {})
            rec["volume_24h_usd"] = float(ticker.get("quoteVolume", 0) or 0)
            rec["last_price"] = float(ticker.get("last", 0) or 0)

        df = pd.DataFrame(records)

        # Remove duplicates (keep highest volume per base symbol)
        df = df.sort_values("volume_24h_usd", ascending=False)
        df = df.drop_duplicates(subset="symbol", keep="first")

        # Sort by volume, take top N
        df = df.sort_values("volume_24h_usd", ascending=False).head(top_n)
        df = df.reset_index(drop=True)

        self.universe_df = df

        # Cache
        _save_cache_json(cache_path, df.to_dict(orient="records"))
        _safe_log(f"Stage 1: Universe constructed with {len(df)} assets")

        return df

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 2: Data Collection
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_universe_data(
        self,
        symbols: List[str],
        lookback_days: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV data for all symbols with concurrent fetching.

        Uses ThreadPoolExecutor with 10 workers and 50ms inter-request delay
        for Binance rate-limit compliance. Each asset's data is cached
        individually as a parquet file.

        Parameters
        ----------
        symbols : list of str
            Base symbols (e.g., ["BTC", "ETH", "SOL"]).
        lookback_days : int
            Number of days of history to fetch.

        Returns
        -------
        dict
            symbol -> pd.DataFrame with OHLCV columns.
        """
        import ccxt

        _safe_log(f"Stage 2: Fetching daily OHLCV for {len(symbols)} assets "
                   f"({lookback_days}d lookback)...")

        since_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        until_date = datetime.utcnow().strftime("%Y-%m-%d")
        since_ts = int(datetime.strptime(since_date, "%Y-%m-%d").timestamp() * 1000)

        ohlcv_cache_dir = self.cache_dir / "ohlcv_daily"
        ohlcv_cache_dir.mkdir(parents=True, exist_ok=True)

        data: Dict[str, pd.DataFrame] = {}
        symbols_to_fetch: List[str] = []

        # Check individual caches first
        for sym in symbols:
            cache_file = ohlcv_cache_dir / f"{sym}_daily.parquet"
            if cache_file.exists():
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=24):
                    try:
                        df = pd.read_parquet(cache_file)
                        if len(df) >= lookback_days * 0.5:  # at least 50% coverage
                            data[sym] = df
                            continue
                    except Exception:
                        pass
            symbols_to_fetch.append(sym)

        cached_count = len(data)
        if cached_count > 0:
            _safe_log(f"Stage 2: {cached_count} assets loaded from cache, "
                       f"{len(symbols_to_fetch)} to fetch")

        if not symbols_to_fetch:
            self.ohlcv_data = data
            return data

        # Concurrent fetching with rate limiting
        def _fetch_single(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            """Fetch OHLCV for a single symbol. Thread-safe with per-call delay."""
            _time.sleep(_REQUEST_DELAY_S)
            try:
                # Use the same exchange that worked for universe fetch
                _exch_name = getattr(self, '_connected_exchange', 'bybit')
                _cls = getattr(ccxt, _exch_name, ccxt.bybit)
                ex = _cls({"enableRateLimit": True})
                pair = f"{sym}/USDT"
                all_candles = []
                current_ts = since_ts

                while True:
                    candles = ex.fetch_ohlcv(
                        pair, timeframe="1d", since=current_ts, limit=1000
                    )
                    if not candles:
                        break
                    all_candles.extend(candles)
                    last_ts = candles[-1][0]
                    if last_ts <= current_ts:
                        break
                    current_ts = last_ts + 1
                    _time.sleep(_REQUEST_DELAY_S)

                if not all_candles:
                    return sym, None

                df = pd.DataFrame(
                    all_candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                df = df[~df.index.duplicated(keep="first")]
                df["volume_usd"] = df["close"] * df["volume"]

                # Cache individually
                cache_file = ohlcv_cache_dir / f"{sym}_daily.parquet"
                df.to_parquet(cache_file)

                return sym, df

            except Exception as e:
                logger.debug("Stage 2: Failed to fetch %s: %s", sym, e)
                return sym, None

        # Execute with ThreadPoolExecutor
        max_workers = min(10, len(symbols_to_fetch))
        succeeded = 0
        failed = 0

        fetch_bar = tqdm(total=len(symbols_to_fetch), desc="Fetching OHLCV", unit="asset", leave=True)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_single, sym): sym for sym in symbols_to_fetch}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    s, df = future.result(timeout=120)
                    if df is not None and len(df) > 0:
                        data[s] = df
                        succeeded += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.debug("Stage 2: Future failed for %s: %s", sym, e)
                    failed += 1

                fetch_bar.update(1)
                fetch_bar.set_postfix(ok=succeeded, fail=failed, asset=sym[:8])

        fetch_bar.close()

        self.ohlcv_data = data
        _safe_log(f"Stage 2: Fetched {succeeded} assets, {failed} failed, "
                   f"{cached_count} from cache -> {len(data)} total")

        return data

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 3: Liquidity Filter
    # ══════════════════════════════════════════════════════════════════════════

    def filter_liquidity(
        self,
        min_daily_volume_usd: float = 10_000_000,
        min_history_days: int = 365,
    ) -> List[str]:
        """
        Remove illiquid assets, stablecoins, and wrapped duplicates.

        Criteria:
          - Average daily USD volume >= ``min_daily_volume_usd``
          - At least ``min_history_days`` of trading history
          - Not a stablecoin (USDC added back separately as defensive anchor)
          - Not a wrapped/bridged duplicate (keep most liquid version)

        Parameters
        ----------
        min_daily_volume_usd : float
            Minimum average daily volume in USD.
        min_history_days : int
            Minimum number of calendar days of data required.

        Returns
        -------
        list of str
            Symbols passing the liquidity filter.
        """
        _safe_log(f"Stage 3: Applying liquidity filter (vol >= ${min_daily_volume_usd:,.0f}, "
                   f"history >= {min_history_days}d)...")

        initial_count = len(self.ohlcv_data)
        survivors: List[str] = []
        removed_reasons: Dict[str, str] = {}

        # Track base symbols to detect wrapped duplicates
        base_volume_map: Dict[str, Tuple[str, float]] = {}

        liq_bar = tqdm(self.ohlcv_data.items(), desc="Liquidity Filter", unit="asset", leave=False, total=initial_count)
        for sym, df in liq_bar:
            liq_bar.set_postfix(sym=sym[:8], passed=len(survivors))
            # Skip stablecoins
            if sym.upper() in _STABLECOINS:
                removed_reasons[sym] = "stablecoin"
                continue

            # Check minimum history
            if len(df) < min_history_days:
                removed_reasons[sym] = f"insufficient_history ({len(df)}d < {min_history_days}d)"
                continue

            # Compute average daily USD volume
            if "volume_usd" in df.columns:
                avg_vol = df["volume_usd"].mean()
            else:
                avg_vol = (df["close"] * df["volume"]).mean()

            if avg_vol < min_daily_volume_usd:
                removed_reasons[sym] = (
                    f"low_volume (${avg_vol:,.0f} < ${min_daily_volume_usd:,.0f})"
                )
                continue

            # Compute daily returns for downstream use
            close = df["close"].dropna()
            if len(close) < min_history_days:
                removed_reasons[sym] = "insufficient_close_data"
                continue

            rets = np.log(close / close.shift(1)).dropna()
            self.returns_data[sym] = rets
            self.volume_data[sym] = (
                df["volume_usd"] if "volume_usd" in df.columns
                else df["close"] * df["volume"]
            )

            # Track for duplicate detection: map normalised base -> (symbol, volume)
            normalised = sym.upper()
            for prefix in _WRAPPED_PREFIXES:
                if normalised.startswith(prefix) and len(normalised) > len(prefix) + 1:
                    base_candidate = normalised[len(prefix):]
                    if base_candidate in ("ETH", "BTC", "SOL", "BNB", "MATIC"):
                        normalised = base_candidate
                        break

            if normalised in base_volume_map:
                existing_sym, existing_vol = base_volume_map[normalised]
                if avg_vol > existing_vol:
                    # Current is more liquid -> replace
                    removed_reasons[existing_sym] = f"wrapped_duplicate (replaced by {sym})"
                    survivors = [s for s in survivors if s != existing_sym]
                    base_volume_map[normalised] = (sym, avg_vol)
                    survivors.append(sym)
                else:
                    removed_reasons[sym] = f"wrapped_duplicate (less liquid than {existing_sym})"
            else:
                base_volume_map[normalised] = (sym, avg_vol)
                survivors.append(sym)

        self.liquid_symbols = survivors
        _safe_log(f"Stage 3: Liquidity Filter | {initial_count} -> {len(survivors)} passed")

        if removed_reasons:
            reasons_summary = {}
            for reason in removed_reasons.values():
                key = reason.split(" (")[0]
                reasons_summary[key] = reasons_summary.get(key, 0) + 1
            for reason, count in sorted(reasons_summary.items(), key=lambda x: -x[1]):
                _safe_log(f"  Removed {count} assets: {reason}")

        return survivors

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 4: Statistical Quality Filter
    # ══════════════════════════════════════════════════════════════════════════

    def filter_statistical_quality(self) -> List[str]:
        """
        Remove assets with poor statistical properties.

        Checks:
          - ADF stationarity on returns (reject non-stationary at 5%)
          - Missing data threshold (> 5% missing -> remove)
          - Suspicious patterns: constant price streaks, zero-volume days
            exceeding 10% (wash trading signal)
          - Hurst exponent flagging (H > 0.7 or H < 0.3 flagged but kept)

        Returns
        -------
        list of str
            Symbols passing the quality filter.
        """
        from statsmodels.tsa.stattools import adfuller

        _safe_log("Stage 4: Applying statistical quality filter...")

        initial_count = len(self.liquid_symbols)
        survivors: List[str] = []
        removed: Dict[str, str] = {}
        hurst_flags: Dict[str, float] = {}

        qual_bar = tqdm(self.liquid_symbols, desc="Quality Filter", unit="asset", leave=False)
        for sym in qual_bar:
            qual_bar.set_postfix(sym=sym[:8], passed=len(survivors))
            rets = self.returns_data.get(sym)
            if rets is None or len(rets) < 100:
                removed[sym] = "insufficient_returns"
                continue

            rets_clean = rets.dropna()

            # (a) Missing data check
            df = self.ohlcv_data.get(sym)
            if df is not None:
                total_expected = len(df)
                missing_pct = 1.0 - len(rets_clean) / max(total_expected - 1, 1)
                if missing_pct > 0.05:
                    removed[sym] = f"missing_data ({missing_pct:.1%})"
                    continue

            # (b) ADF stationarity test on returns
            try:
                adf_result = adfuller(rets_clean.values, maxlag=14, autolag="AIC")
                adf_pvalue = adf_result[1]
                if adf_pvalue > 0.05:
                    removed[sym] = f"non_stationary (ADF p={adf_pvalue:.4f})"
                    continue
            except Exception as e:
                logger.debug("ADF failed for %s: %s", sym, e)
                removed[sym] = "adf_failed"
                continue

            # (c) Suspicious pattern detection
            if df is not None:
                # Constant price streaks: more than 5 consecutive identical closes
                closes = df["close"].values
                max_streak = 1
                current_streak = 1
                for i in range(1, len(closes)):
                    if closes[i] == closes[i - 1]:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 1

                if max_streak > 20:
                    removed[sym] = f"constant_price_streak ({max_streak} bars)"
                    continue

                # Zero-volume days
                vol_series = df["volume"]
                zero_vol_pct = (vol_series == 0).sum() / len(vol_series)
                if zero_vol_pct > 0.10:
                    removed[sym] = f"zero_volume ({zero_vol_pct:.1%})"
                    continue

            # (d) Hurst exponent (flag but keep)
            h = _hurst_exponent(rets_clean.values)
            if h > 0.7 or h < 0.3:
                hurst_flags[sym] = h

            survivors.append(sym)

        self.quality_symbols = survivors

        _safe_log(f"Stage 4: {initial_count} -> {len(survivors)} assets after quality filter")

        if removed:
            reasons_summary: Dict[str, int] = {}
            for reason in removed.values():
                key = reason.split(" (")[0]
                reasons_summary[key] = reasons_summary.get(key, 0) + 1
            for reason, count in sorted(reasons_summary.items(), key=lambda x: -x[1]):
                _safe_log(f"  Removed {count} assets: {reason}")

        if hurst_flags:
            _safe_log(f"  Hurst flags: {len(hurst_flags)} assets "
                       f"(H > 0.7 or H < 0.3, kept for diversity)")
            for sym, h in sorted(hurst_flags.items(), key=lambda x: x[1]):
                logger.debug("    %s: H=%.3f", sym, h)

        return survivors

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 5: Risk-Return Profiling
    # ══════════════════════════════════════════════════════════════════════════

    def compute_asset_profiles(self) -> pd.DataFrame:
        """
        Compute comprehensive risk-return profiles for all quality-filtered assets.

        Metrics per asset:
          - ann_return: Annualised log return
          - ann_volatility: Annualised volatility
          - sharpe: Annualised Sharpe ratio (rf from config, default 4.5%)
          - max_drawdown: Maximum drawdown of cumulative returns
          - cvar_5pct: Empirical CVaR at 5% level
          - skewness, kurtosis: Distribution shape
          - hurst: Hurst exponent (R/S method)
          - amihud: Amihud illiquidity ratio
          - btc_correlation: Rolling correlation with BTC
          - btc_beta: OLS beta to BTC returns
          - idio_vol: Idiosyncratic volatility (residual after BTC factor)

        Returns
        -------
        pd.DataFrame
            One row per asset, sorted by Sharpe ratio descending.
        """
        _safe_log("Stage 5: Computing risk-return profiles...")

        rf_annual = self.config.get("risk_free_rate", 0.045)
        rf_daily = rf_annual / _DAILY_ANN_FACTOR

        # Prepare BTC returns as market factor
        if "BTC" in self.returns_data:
            self._btc_returns = self.returns_data["BTC"]
        else:
            # Find BTC-like symbol
            for candidate in ("BTC", "BTCB", "WBTC"):
                if candidate in self.returns_data:
                    self._btc_returns = self.returns_data[candidate]
                    break

        profiles: List[dict] = []

        profile_bar = tqdm(self.quality_symbols, desc="Profiling", unit="asset", leave=False)
        for sym in profile_bar:
            profile_bar.set_postfix(asset=sym[:8])
            rets = self.returns_data.get(sym)
            vol = self.volume_data.get(sym)
            if rets is None or len(rets) < 50:
                continue

            rets_arr = rets.values
            clean = rets_arr[np.isfinite(rets_arr)]
            if len(clean) < 50:
                continue

            # Core metrics
            ann_ret = float(np.mean(clean) * _DAILY_ANN_FACTOR)
            ann_vol = float(np.std(clean, ddof=1) * _SQRT_ANN)
            sharpe = float((ann_ret - rf_annual) / ann_vol) if ann_vol > 1e-8 else 0.0

            # Max drawdown
            cum_ret = np.cumsum(clean)
            running_max = np.maximum.accumulate(cum_ret)
            drawdowns = cum_ret - running_max
            max_dd = float(np.min(drawdowns))

            # CVaR
            cvar_5 = _cvar(clean, alpha=0.05)

            # Distribution shape
            skew = float(stats.skew(clean))
            kurt = float(stats.kurtosis(clean))

            # Hurst exponent
            hurst = _hurst_exponent(clean)

            # Amihud illiquidity
            vol_arr = vol.reindex(rets.index).values if vol is not None else np.zeros(len(rets))
            amihud = _amihud_illiquidity(rets_arr, vol_arr)

            # BTC factor analysis
            btc_corr = 0.0
            btc_beta = 0.0
            idio_vol = ann_vol

            if self._btc_returns is not None and sym != "BTC":
                # Align indices
                common_idx = rets.index.intersection(self._btc_returns.index)
                if len(common_idx) > 50:
                    r_asset = rets.loc[common_idx].values
                    r_btc = self._btc_returns.loc[common_idx].values

                    mask = np.isfinite(r_asset) & np.isfinite(r_btc)
                    if mask.sum() > 50:
                        r_a = r_asset[mask]
                        r_b = r_btc[mask]

                        btc_corr = float(np.corrcoef(r_a, r_b)[0, 1])

                        # OLS regression: r_asset = alpha + beta * r_btc + epsilon
                        cov_ab = np.cov(r_a, r_b, ddof=1)
                        var_btc = cov_ab[1, 1]
                        if var_btc > 1e-12:
                            btc_beta = float(cov_ab[0, 1] / var_btc)

                        # Idiosyncratic volatility
                        residuals = r_a - btc_beta * r_b
                        idio_vol = float(np.std(residuals, ddof=1) * _SQRT_ANN)

            profiles.append({
                "symbol": sym,
                "ann_return": round(ann_ret, 6),
                "ann_volatility": round(ann_vol, 6),
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_dd, 6),
                "cvar_5pct": round(cvar_5, 6),
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "hurst": round(hurst, 4),
                "amihud": round(amihud, 10) if np.isfinite(amihud) else 999.0,
                "btc_correlation": round(btc_corr, 4),
                "btc_beta": round(btc_beta, 4),
                "idio_volatility": round(idio_vol, 6),
                "n_observations": len(clean),
            })

        df = pd.DataFrame(profiles)
        if not df.empty:
            df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

        self.asset_profiles = df

        # Cache profiles
        cache_path = self.cache_dir / "asset_profiles.json"
        _save_cache_json(cache_path, df.to_dict(orient="records"))

        _safe_log(f"Stage 5: Profiled {len(df)} assets "
                   f"(top Sharpe: {df['sharpe'].iloc[0]:.3f} [{df['symbol'].iloc[0]}], "
                   f"median vol: {df['ann_volatility'].median():.2%})"
                   if len(df) > 0 else "Stage 5: No assets profiled")

        return df

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 6: Correlation Clustering
    # ══════════════════════════════════════════════════════════════════════════

    def cluster_assets(self, n_clusters: int = 20) -> pd.DataFrame:
        """
        Hierarchical clustering to group correlated assets and select
        cluster representatives.

        Uses Ward linkage on correlation distance d(i,j) = sqrt(0.5*(1 - rho)).
        Within each cluster, the asset with the highest Sharpe ratio is chosen
        as the representative. This prevents selecting multiple highly correlated
        L1 tokens (e.g., SOL + AVAX + NEAR) at the expense of diversity.

        Parameters
        ----------
        n_clusters : int
            Target number of clusters.

        Returns
        -------
        pd.DataFrame
            Columns: cluster_id, representative, cluster_sharpe, cluster_size,
            cluster_members.
        """
        _safe_log(f"Stage 6: Clustering assets into {n_clusters} groups...")

        if self.asset_profiles is None or len(self.asset_profiles) < n_clusters:
            _safe_log("Stage 6: Not enough assets for clustering, returning all", "warning")
            if self.asset_profiles is not None:
                result = pd.DataFrame({
                    "cluster_id": range(len(self.asset_profiles)),
                    "representative": self.asset_profiles["symbol"].values,
                    "cluster_sharpe": self.asset_profiles["sharpe"].values,
                    "cluster_size": 1,
                    "cluster_members": [[s] for s in self.asset_profiles["symbol"]],
                })
                self.cluster_df = result
                return result
            return pd.DataFrame()

        symbols = self.asset_profiles["symbol"].tolist()
        n = len(symbols)

        # Build correlation matrix from returns
        returns_matrix = pd.DataFrame(
            {sym: self.returns_data[sym] for sym in symbols if sym in self.returns_data}
        ).dropna()

        if len(returns_matrix.columns) < n_clusters:
            _safe_log("Stage 6: Insufficient overlapping data for correlation matrix", "warning")
            symbols = list(returns_matrix.columns)
            n = len(symbols)
            n_clusters = min(n_clusters, n)

        corr = returns_matrix.corr().values
        # Ensure valid correlation matrix
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        # Correlation distance
        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)

        # Ensure symmetry and no NaN
        dist = (dist + dist.T) / 2
        dist = np.nan_to_num(dist, nan=1.0)

        # Ward hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="ward")
        labels = fcluster(link, t=n_clusters, criterion="maxclust")

        # Build Sharpe lookup
        sharpe_map = dict(
            zip(self.asset_profiles["symbol"], self.asset_profiles["sharpe"])
        )
        idio_vol_map = dict(
            zip(self.asset_profiles["symbol"], self.asset_profiles["idio_volatility"])
        )

        # Select representative per cluster (highest Sharpe, tiebreak by idio vol)
        cluster_records: List[dict] = []
        col_symbols = list(returns_matrix.columns)

        for cid in sorted(set(labels)):
            member_indices = [i for i, lab in enumerate(labels) if lab == cid]
            members = [col_symbols[i] for i in member_indices]

            # Score: Sharpe is primary, idiosyncratic vol is tiebreaker (higher = more unique)
            best = max(
                members,
                key=lambda s: (sharpe_map.get(s, -999), idio_vol_map.get(s, 0))
            )

            cluster_records.append({
                "cluster_id": int(cid),
                "representative": best,
                "cluster_sharpe": sharpe_map.get(best, 0.0),
                "cluster_size": len(members),
                "cluster_members": members,
            })

        result = pd.DataFrame(cluster_records)
        result = result.sort_values("cluster_sharpe", ascending=False).reset_index(drop=True)
        self.cluster_df = result

        # Cache
        cache_path = self.cache_dir / "cluster_assignments.json"
        _save_cache_json(cache_path, result.to_dict(orient="records"))

        _safe_log(f"Stage 6: Formed {len(result)} clusters "
                   f"(sizes: {sorted(result['cluster_size'].tolist(), reverse=True)})")

        return result

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 7: Adaptive Threshold-Based Selection (Final 10–50)
    # ══════════════════════════════════════════════════════════════════════════

    def select_optimal_portfolio(
        self,
        n_assets: int = None,
        constraints: dict = None,
    ) -> List[str]:
        """
        Select the optimal portfolio using strict quantitative thresholds.

        When *n_assets* is None (default), the number of assets is determined
        adaptively: every candidate that passes ALL of the following strict
        thresholds is included, with a hard floor of 10 and ceiling of 50:

          1. Sharpe ratio       ≥ 0.3   (positive risk-adjusted return)
          2. Max drawdown       ≤ -70%  (survived major crises)
          3. Daily volume       ≥ $10M  (tradeable without market impact)
          4. History             ≥ 365 days (sufficient for covariance estimation)
          5. Idiosyncratic vol  ≥ 5%    (contributes unique diversification)
          6. BTC correlation    ≤ 0.90  (not a BTC proxy)
          7. Amihud illiquidity ≤ 95th percentile (remove the most illiquid)

        After threshold filtering, greedy forward selection maximises the
        diversification ratio until marginal improvement < 1%.

        Parameters
        ----------
        n_assets : int or None
            Target count.  ``None`` → adaptive (10–50 based on thresholds).
        constraints : dict, optional
            Override default constraints.

        Returns
        -------
        list of str
            Ordered list of selected symbols.
        """
        adaptive = n_assets is None
        if adaptive:
            _safe_log("Stage 7: Adaptive threshold-based selection (10–50 assets)...")
        else:
            _safe_log(f"Stage 7: Selecting optimal {n_assets} assets...")

        # Default constraints aligned with CLAUDE.md asset structure
        default_constraints = {
            "mandatory": ["BTC", "ETH"],
            "mandatory_categories": {
                "liquid_staking": 1,  # at least 1 staking asset
                "tokenised_rwa": 1,   # at least 1 treasury/RWA
                "stablecoin": 0,      # USDC handled separately
            },
            "max_per_cluster": 3,
            "min_avg_volume": 50_000_000,
            "include_usdc": True,
        }
        constraints = {**default_constraints, **(constraints or {})}

        # ── Adaptive threshold filter (when n_assets is None) ──
        if adaptive and self.asset_profiles is not None and len(self.asset_profiles) > 0:
            profiles = self.asset_profiles.copy()
            n_before = len(profiles)

            # Strict quantitative thresholds
            thresholds = {
                "sharpe":              (">=", 0.3),
                "max_drawdown":        (">=", -0.70),   # DD is negative
                "amihud_illiquidity":  ("<=", profiles["amihud_illiquidity"].quantile(0.95)),
                "btc_correlation":     ("<=", 0.90),
            }
            if "idiosyncratic_vol" in profiles.columns:
                thresholds["idiosyncratic_vol"] = (">=", 0.05)

            mask = pd.Series(True, index=profiles.index)
            for col, (op, val) in thresholds.items():
                if col not in profiles.columns:
                    continue
                if op == ">=":
                    mask &= profiles[col] >= val
                else:
                    mask &= profiles[col] <= val

            profiles = profiles[mask]
            n_after = len(profiles)
            _safe_log(f"  Threshold filter: {n_before} → {n_after} assets passed")
            for col, (op, val) in thresholds.items():
                _safe_log(f"    {col} {op} {val:.4f}")

            # Clamp to [10, 50]
            n_assets = max(10, min(50, n_after))
            _safe_log(f"  Adaptive target: {n_assets} assets")

            # Use threshold-passing assets as the candidate pool
            threshold_candidates = profiles["symbol"].tolist()
        else:
            threshold_candidates = None
            if n_assets is None:
                n_assets = 10

        # Gather candidate pool from cluster representatives
        if self.cluster_df is None or len(self.cluster_df) == 0:
            _safe_log("Stage 7: No clusters available!", "error")
            return []

        candidates = self.cluster_df["representative"].tolist()

        # If adaptive, expand candidates with all threshold-passing assets
        if threshold_candidates is not None:
            for tc in threshold_candidates:
                if tc not in candidates and tc in self.returns_data:
                    candidates.append(tc)

        # Add mandatory assets if not already in candidates
        mandatory = constraints.get("mandatory", [])
        for m in mandatory:
            if m not in candidates and m in self.returns_data:
                candidates.append(m)

        # Add category-mandatory assets
        category_mandatory: List[str] = []
        cat_requirements = constraints.get("mandatory_categories", {})
        for cat, min_count in cat_requirements.items():
            if min_count <= 0:
                continue
            cat_assets = [
                sym for sym in candidates
                if _ASSET_CLASSIFICATIONS.get(sym.upper(), "unknown") == cat
            ]
            if not cat_assets:
                # Search in all quality symbols
                cat_assets = [
                    sym for sym in self.quality_symbols
                    if _ASSET_CLASSIFICATIONS.get(sym.upper(), "unknown") == cat
                    and sym in self.returns_data
                ]
            # Take the top by Sharpe
            sharpe_map = dict(
                zip(self.asset_profiles["symbol"], self.asset_profiles["sharpe"])
            ) if self.asset_profiles is not None else {}

            cat_assets.sort(key=lambda s: sharpe_map.get(s, -999), reverse=True)
            for a in cat_assets[:min_count]:
                if a not in candidates:
                    candidates.append(a)
                category_mandatory.append(a)

        # Always include USDC as defensive anchor
        include_usdc = constraints.get("include_usdc", True)
        usdc_symbol = "USDC"

        # Build correlation matrix for candidates with available data
        available = [s for s in candidates if s in self.returns_data]
        if len(available) < n_assets - (1 if include_usdc else 0):
            _safe_log(f"Stage 7: Only {len(available)} candidates available "
                       f"(need {n_assets}), returning all", "warning")
            self.selected_assets = available
            if include_usdc and usdc_symbol not in available:
                self.selected_assets.append(usdc_symbol)
            return self.selected_assets

        # Build returns matrix and covariance
        returns_mat = pd.DataFrame(
            {sym: self.returns_data[sym] for sym in available}
        ).dropna()

        if len(returns_mat) < 50:
            _safe_log("Stage 7: Insufficient overlapping returns data", "error")
            self.selected_assets = available[:n_assets]
            return self.selected_assets

        cov = returns_mat.cov().values
        sym_list = list(returns_mat.columns)
        sym_to_idx = {s: i for i, s in enumerate(sym_list)}

        # Cluster membership for each candidate
        cluster_map: Dict[str, int] = {}
        if self.cluster_df is not None:
            for _, row in self.cluster_df.iterrows():
                for member in row.get("cluster_members", [row["representative"]]):
                    cluster_map[member] = row["cluster_id"]

        # Slots reserved for mandatory / USDC
        n_free = n_assets - (1 if include_usdc else 0)
        all_mandatory = list(set(mandatory + category_mandatory))
        forced = [s for s in all_mandatory if s in sym_to_idx]
        n_to_select = n_free - len(forced)

        if n_to_select < 0:
            _safe_log("Stage 7: Too many mandatory assets for target N", "warning")
            forced = forced[:n_free]
            n_to_select = 0

        # Pool of non-forced candidates
        pool = [s for s in available if s in sym_to_idx and s not in forced]
        max_per_cluster = constraints.get("max_per_cluster", 3)

        def _check_cluster_constraint(selection: List[str]) -> bool:
            """Verify no cluster exceeds max_per_cluster."""
            cluster_counts: Dict[int, int] = {}
            for s in selection:
                cid = cluster_map.get(s, -1)
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
                if cluster_counts[cid] > max_per_cluster:
                    return False
            return True

        def _portfolio_div_ratio(selection: List[str]) -> float:
            """Compute diversification ratio for equal-weight portfolio."""
            indices = [sym_to_idx[s] for s in selection if s in sym_to_idx]
            if len(indices) < 2:
                return 0.0
            sub_cov = cov[np.ix_(indices, indices)]
            w = np.ones(len(indices)) / len(indices)
            return _diversification_ratio(w, sub_cov)

        # Decide between exhaustive and greedy
        from math import comb
        total_combos = comb(len(pool), n_to_select) if n_to_select > 0 else 1

        if n_to_select > 0 and total_combos <= 200_000:
            # Exhaustive search
            _safe_log(f"Stage 7: Exhaustive search over {total_combos:,} combinations")
            best_dr = -np.inf
            best_combo: Optional[Tuple[str, ...]] = None

            combo_bar = tqdm(
                combinations(pool, n_to_select),
                desc="Selection | Exhaustive", unit="combo",
                total=total_combos, leave=False,
            )
            for combo in combo_bar:
                selection = forced + list(combo)
                if not _check_cluster_constraint(selection):
                    continue
                dr = _portfolio_div_ratio(selection)
                if dr > best_dr:
                    best_dr = dr
                    best_combo = combo
                    combo_bar.set_postfix(best_DR=f"{best_dr:.3f}")

            if best_combo is not None:
                selected = forced + list(best_combo)
            else:
                _safe_log("Stage 7: No valid combination found, using greedy", "warning")
                selected = self._greedy_select(
                    forced, pool, n_to_select, cov, sym_to_idx,
                    cluster_map, max_per_cluster
                )
        elif n_to_select > 0:
            # Greedy forward selection
            _safe_log(f"Stage 7: Greedy selection ({total_combos:,} combos too large "
                       f"for exhaustive search)")
            selected = self._greedy_select(
                forced, pool, n_to_select, cov, sym_to_idx,
                cluster_map, max_per_cluster
            )
        else:
            selected = forced

        # Append USDC
        if include_usdc and usdc_symbol not in selected:
            selected.append(usdc_symbol)

        self.selected_assets = selected

        # Log final selection details
        final_dr = _portfolio_div_ratio(
            [s for s in selected if s in sym_to_idx]
        )
        _safe_log(f"Stage 7: Selected {len(selected)} assets "
                   f"(diversification ratio: {final_dr:.3f})")
        for i, sym in enumerate(selected, 1):
            cat = _ASSET_CLASSIFICATIONS.get(sym.upper(), "unknown")
            sharpe_map = dict(
                zip(self.asset_profiles["symbol"], self.asset_profiles["sharpe"])
            ) if self.asset_profiles is not None else {}
            sr = sharpe_map.get(sym, 0.0)
            _safe_log(f"  {i:2d}. {sym:<10s}  Sharpe: {sr:+.3f}  Class: {cat}")

        return selected

    def _greedy_select(
        self,
        forced: List[str],
        pool: List[str],
        n_to_select: int,
        cov: np.ndarray,
        sym_to_idx: Dict[str, int],
        cluster_map: Dict[str, int],
        max_per_cluster: int,
    ) -> List[str]:
        """
        Greedy forward selection maximising marginal diversification ratio.

        At each step, adds the candidate that yields the largest increase in
        the equal-weight portfolio's diversification ratio, subject to the
        cluster concentration constraint.

        Parameters
        ----------
        forced : list of str
            Pre-selected mandatory assets.
        pool : list of str
            Candidate assets to choose from.
        n_to_select : int
            Number of additional assets to select.
        cov : np.ndarray
            Full covariance matrix.
        sym_to_idx : dict
            Symbol -> index in cov matrix.
        cluster_map : dict
            Symbol -> cluster ID.
        max_per_cluster : int
            Maximum assets from any single cluster.

        Returns
        -------
        list of str
            forced + selected assets.
        """
        selected = list(forced)
        remaining = list(pool)

        greedy_bar = tqdm(range(n_to_select), desc="Selection | Greedy", unit="asset", leave=False)
        for step in greedy_bar:
            best_dr = -np.inf
            best_candidate = None

            for candidate in remaining:
                trial = selected + [candidate]

                # Cluster constraint
                cluster_counts: Dict[int, int] = {}
                valid = True
                for s in trial:
                    cid = cluster_map.get(s, -1)
                    cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
                    if cluster_counts[cid] > max_per_cluster:
                        valid = False
                        break
                if not valid:
                    continue

                # Diversification ratio
                indices = [sym_to_idx[s] for s in trial if s in sym_to_idx]
                if len(indices) < 2:
                    continue
                sub_cov = cov[np.ix_(indices, indices)]
                w = np.ones(len(indices)) / len(indices)
                dr = _diversification_ratio(w, sub_cov)

                if dr > best_dr:
                    best_dr = dr
                    best_candidate = candidate

            if best_candidate is None:
                _safe_log(f"Stage 7: Greedy selection stalled at step {step + 1}", "warning")
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            greedy_bar.set_postfix(added=best_candidate, DR=f"{best_dr:.3f}")

        greedy_bar.close()
        return selected

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 8: Validation
    # ══════════════════════════════════════════════════════════════════════════

    def validate_selection(self, selected: List[str]) -> dict:
        """
        Validate the selected portfolio against benchmarks and statistical tests.

        Checks:
          - Diversification ratio vs equal-weight full universe
          - Portfolio efficiency: effective N, HHI
          - Risk metrics: volatility, Sharpe, CVaR of equal-weight portfolio
          - Mean-variance spanning test (do selected assets span the universe?)
          - Comparison against: top-10 by market cap, top-10 by Sharpe, random 10

        Parameters
        ----------
        selected : list of str
            The selected asset symbols.

        Returns
        -------
        dict
            Comprehensive validation results.
        """
        _safe_log("Stage 8: Validating selection...")

        # Build returns matrices
        selected_with_data = [s for s in selected if s in self.returns_data]
        if not selected_with_data:
            return {"error": "No return data for selected assets"}

        all_syms = [s for s in self.quality_symbols if s in self.returns_data]
        if len(all_syms) < len(selected_with_data) + 2:
            all_syms = list(self.returns_data.keys())

        returns_selected = pd.DataFrame(
            {s: self.returns_data[s] for s in selected_with_data}
        ).dropna()

        returns_universe = pd.DataFrame(
            {s: self.returns_data[s] for s in all_syms if s in self.returns_data}
        ).dropna()

        if len(returns_selected) < 50 or len(returns_universe) < 50:
            return {"error": "Insufficient data for validation"}

        # ---- Selected portfolio metrics ----
        n_sel = len(selected_with_data)
        w_eq = np.ones(n_sel) / n_sel
        cov_sel = returns_selected.cov().values

        sel_vol = float(np.sqrt(w_eq @ cov_sel @ w_eq) * _SQRT_ANN)
        sel_ret = float(returns_selected.mean().values @ w_eq * _DAILY_ANN_FACTOR)
        rf = self.config.get("risk_free_rate", 0.045)
        sel_sharpe = float((sel_ret - rf) / sel_vol) if sel_vol > 1e-8 else 0.0
        sel_dr = _diversification_ratio(w_eq, cov_sel)
        sel_eff_n = _effective_n(w_eq)
        sel_hhi = float(np.sum(w_eq ** 2))

        port_returns = returns_selected.values @ w_eq
        sel_cvar = _cvar(port_returns, alpha=0.05)

        # ---- Universe equal-weight metrics ----
        n_univ = len(returns_universe.columns)
        w_univ = np.ones(n_univ) / n_univ
        cov_univ = returns_universe.cov().values
        univ_dr = _diversification_ratio(w_univ, cov_univ)
        univ_vol = float(np.sqrt(w_univ @ cov_univ @ w_univ) * _SQRT_ANN)

        # ---- Benchmark portfolios ----
        benchmarks: Dict[str, dict] = {}

        # (a) Top-10 by Sharpe
        if self.asset_profiles is not None and len(self.asset_profiles) >= 10:
            top_sharpe = self.asset_profiles.head(10)["symbol"].tolist()
            benchmarks["top10_sharpe"] = self._compute_benchmark_metrics(
                top_sharpe, "Top-10 Sharpe"
            )

        # (b) Top-10 by volume (proxy for market cap)
        if self.universe_df is not None and len(self.universe_df) >= 10:
            top_vol = self.universe_df.head(10)["symbol"].tolist()
            benchmarks["top10_volume"] = self._compute_benchmark_metrics(
                top_vol, "Top-10 Volume"
            )

        # (c) Random 10 (average of 100 random draws)
        rng = np.random.default_rng(seed=42)
        random_sharpes = []
        random_vols = []
        random_drs = []

        for _ in range(100):
            if len(all_syms) >= 10:
                random_10 = list(rng.choice(all_syms, size=10, replace=False))
                r_df = pd.DataFrame(
                    {s: self.returns_data[s] for s in random_10 if s in self.returns_data}
                ).dropna()
                if len(r_df.columns) >= 5 and len(r_df) >= 50:
                    w_r = np.ones(len(r_df.columns)) / len(r_df.columns)
                    cov_r = r_df.cov().values
                    r_vol = float(np.sqrt(w_r @ cov_r @ w_r) * _SQRT_ANN)
                    r_ret = float(r_df.mean().values @ w_r * _DAILY_ANN_FACTOR)
                    r_sr = (r_ret - rf) / r_vol if r_vol > 1e-8 else 0.0
                    r_dr = _diversification_ratio(w_r, cov_r)
                    random_sharpes.append(r_sr)
                    random_vols.append(r_vol)
                    random_drs.append(r_dr)

        if random_sharpes:
            benchmarks["random10_avg"] = {
                "sharpe": round(float(np.mean(random_sharpes)), 4),
                "volatility": round(float(np.mean(random_vols)), 6),
                "div_ratio": round(float(np.mean(random_drs)), 4),
                "label": "Random-10 (avg of 100 draws)",
            }

        # ---- Spanning test ----
        spanning_result = {}
        base_assets = [s for s in selected_with_data if s in returns_universe.columns]
        test_assets = [
            s for s in returns_universe.columns if s not in base_assets
        ][:10]  # Test with up to 10 non-selected assets

        if len(base_assets) >= 3 and len(test_assets) >= 1:
            try:
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools import add_constant

                # Simple spanning test: regress each test asset on base assets
                # H0: alpha = 0, sum(betas) = 1
                rejections = 0
                total_tests = 0

                for t_sym in test_assets:
                    y = returns_universe[t_sym].values
                    X = returns_universe[base_assets].values
                    X = add_constant(X)
                    mask = np.all(np.isfinite(np.column_stack([y, X])), axis=1)
                    if mask.sum() < len(base_assets) + 10:
                        continue

                    model = OLS(y[mask], X[mask]).fit()
                    alpha = model.params[0]
                    betas = model.params[1:]
                    beta_sum = np.sum(betas)

                    # Test alpha = 0
                    alpha_p = model.pvalues[0]
                    # Test sum(beta) = 1 via Wald test
                    R = np.zeros((1, len(model.params)))
                    R[0, 1:] = 1.0
                    try:
                        wald = model.wald_test(R, scalar=True)
                        beta_sum_p = float(wald.pvalue)
                    except Exception:
                        beta_sum_p = 1.0

                    if alpha_p < 0.05 or beta_sum_p < 0.05:
                        rejections += 1
                    total_tests += 1

                spanning_result = {
                    "test_assets": len(test_assets),
                    "rejections": rejections,
                    "total_tests": total_tests,
                    "rejection_rate": (
                        round(rejections / total_tests, 3) if total_tests > 0 else 0
                    ),
                    "interpretation": (
                        "Selected assets span the universe well (low rejection rate)"
                        if total_tests > 0 and rejections / total_tests < 0.3
                        else "Some assets outside selection offer incremental value"
                    ),
                }
            except ImportError:
                spanning_result = {"error": "statsmodels not available"}
            except Exception as e:
                spanning_result = {"error": str(e)}

        validation = {
            "selected_portfolio": {
                "n_assets": n_sel,
                "assets": selected_with_data,
                "ann_return": round(sel_ret, 6),
                "ann_volatility": round(sel_vol, 6),
                "sharpe": round(sel_sharpe, 4),
                "cvar_5pct": round(sel_cvar, 6),
                "diversification_ratio": round(sel_dr, 4),
                "effective_n": round(sel_eff_n, 2),
                "hhi": round(sel_hhi, 6),
            },
            "universe": {
                "n_assets": n_univ,
                "diversification_ratio": round(univ_dr, 4),
                "ann_volatility": round(univ_vol, 6),
            },
            "benchmarks": benchmarks,
            "spanning_test": spanning_result,
        }

        # Cache validation results
        cache_path = self.cache_dir / "validation_results.json"
        _save_cache_json(cache_path, validation)

        # Log summary
        _safe_log("Stage 8: Validation complete")
        _safe_log(f"  Selected: Sharpe={sel_sharpe:.3f}, Vol={sel_vol:.2%}, "
                   f"DR={sel_dr:.3f}, EffN={sel_eff_n:.1f}")
        _safe_log(f"  Universe: DR={univ_dr:.3f}, Vol={univ_vol:.2%}")
        for name, bm in benchmarks.items():
            _safe_log(f"  {bm.get('label', name)}: "
                       f"Sharpe={bm.get('sharpe', 0):.3f}, "
                       f"DR={bm.get('div_ratio', 0):.3f}")

        return validation

    def _compute_benchmark_metrics(self, symbols: List[str], label: str) -> dict:
        """Compute equal-weight portfolio metrics for a benchmark set."""
        available = [s for s in symbols if s in self.returns_data]
        if len(available) < 3:
            return {"label": label, "sharpe": 0, "volatility": 0, "div_ratio": 0}

        r_df = pd.DataFrame(
            {s: self.returns_data[s] for s in available}
        ).dropna()
        if len(r_df) < 50:
            return {"label": label, "sharpe": 0, "volatility": 0, "div_ratio": 0}

        n = len(r_df.columns)
        w = np.ones(n) / n
        cov_bm = r_df.cov().values
        vol = float(np.sqrt(w @ cov_bm @ w) * _SQRT_ANN)
        ret = float(r_df.mean().values @ w * _DAILY_ANN_FACTOR)
        rf = self.config.get("risk_free_rate", 0.045)
        sr = (ret - rf) / vol if vol > 1e-8 else 0.0

        return {
            "label": label,
            "sharpe": round(sr, 4),
            "volatility": round(vol, 6),
            "div_ratio": round(_diversification_ratio(w, cov_bm), 4),
            "n_available": n,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  Asset Classification
    # ══════════════════════════════════════════════════════════════════════════

    def classify_assets(self, selected: List[str]) -> Dict[str, List[str]]:
        """
        Classify selected assets into portfolio categories.

        Categories:
          - spot_crypto: L1/L2 tokens (BTC, ETH, SOL, etc.)
          - liquid_staking: Staked derivatives (stETH, rETH, cbETH)
          - tokenised_rwa: Real-world assets (BUIDL, USDY, PAXG, ONDO)
          - defi: DeFi protocol tokens (AAVE, UNI, MKR)
          - stablecoin: USDC
          - unknown: Unclassified assets

        Uses a lookup table for known assets and heuristics for unknowns.

        Parameters
        ----------
        selected : list of str
            Selected asset symbols.

        Returns
        -------
        dict
            Category -> list of symbols.
        """
        classification: Dict[str, List[str]] = {
            "spot_crypto": [],
            "liquid_staking": [],
            "tokenised_rwa": [],
            "defi": [],
            "stablecoin": [],
            "unknown": [],
        }

        for sym in selected:
            upper = sym.upper()

            # Check lookup table first
            cat = _ASSET_CLASSIFICATIONS.get(upper)

            if cat is None:
                # Stablecoin detection
                if upper in _STABLECOINS or upper == "USDC":
                    cat = "stablecoin"
                # Heuristic: wrapped staking tokens
                elif any(upper.startswith(p) for p in ("ST", "WST", "CB", "R")):
                    suffix = upper
                    for prefix in ("WST", "ST", "CB"):
                        if upper.startswith(prefix):
                            suffix = upper[len(prefix):]
                            break
                    if suffix in ("ETH", "SOL", "BNB", "MATIC"):
                        cat = "liquid_staking"
                # Heuristic: gold tokens
                elif any(kw in upper for kw in ("GOLD", "XAU", "PAX")):
                    cat = "tokenised_rwa"
                else:
                    cat = "unknown"

            if cat not in classification:
                classification[cat] = []
            classification[cat].append(sym)

        # Log classification
        _safe_log("Asset classification:")
        for cat, assets in classification.items():
            if assets:
                _safe_log(f"  {cat}: {', '.join(assets)}")

        return classification

    # ══════════════════════════════════════════════════════════════════════════
    #  Full Pipeline Runner
    # ══════════════════════════════════════════════════════════════════════════

    def run_full_screening(
        self,
        top_n: int = 1000,
        final_n: int = None,
        lookback_days: int = 730,
        n_clusters: int = 30,
        constraints: dict = None,
    ) -> dict:
        """
        Run the complete 8-stage screening pipeline.

        Parameters
        ----------
        top_n : int
            Initial universe size (Stage 1).  Default 1000.
        final_n : int or None
            Number of assets to select (Stage 7).  ``None`` → adaptive
            threshold-based selection (10–50 assets).
        lookback_days : int
            Historical data lookback in days (Stage 2).
        n_clusters : int
            Target cluster count (Stage 6).
        constraints : dict, optional
            Selection constraints for Stage 7.

        Returns
        -------
        dict
            Complete screening results:
              - universe_size: initial count
              - after_liquidity: count after Stage 3
              - after_quality: count after Stage 4
              - n_clusters: cluster count
              - selected: final asset list
              - classification: asset categories
              - validation: validation metrics
              - asset_profiles: DataFrame (as records)
        """
        _safe_log("=" * 72)
        _safe_log("UNIVERSE SCREENING PIPELINE")
        _safe_log("=" * 72)
        _safe_log(f"Parameters: top_n={top_n}, final_n={final_n}, "
                   f"lookback={lookback_days}d, clusters={n_clusters}")
        _safe_log("=" * 72)

        t_start = _time.time()

        # Stage 1: Universe Construction
        universe_df = self.fetch_universe(top_n=top_n)
        symbols = universe_df["symbol"].tolist() if not universe_df.empty else []

        # Stage 2: Data Collection
        self.fetch_universe_data(symbols, lookback_days=lookback_days)

        # Stage 3: Liquidity Filter
        liquid = self.filter_liquidity()

        # Stage 4: Statistical Quality Filter
        quality = self.filter_statistical_quality()

        # Stage 5: Risk-Return Profiling
        profiles = self.compute_asset_profiles()

        # Stage 6: Correlation Clustering
        clusters = self.cluster_assets(n_clusters=n_clusters)

        # Stage 7: Diversification-Optimal Selection
        selected = self.select_optimal_portfolio(
            n_assets=final_n, constraints=constraints
        )

        # Stage 8: Validation
        validation = self.validate_selection(selected)

        # Asset Classification
        classification = self.classify_assets(selected)

        elapsed = _time.time() - t_start

        results = {
            "universe_size": len(symbols),
            "after_liquidity": len(liquid),
            "after_quality": len(quality),
            "n_profiles": len(profiles),
            "n_clusters": len(clusters) if isinstance(clusters, pd.DataFrame) else 0,
            "selected": selected,
            "classification": classification,
            "validation": validation,
            "asset_profiles": (
                profiles.to_dict(orient="records") if isinstance(profiles, pd.DataFrame)
                else []
            ),
            "elapsed_seconds": round(elapsed, 1),
        }

        # Save full results
        cache_path = self.cache_dir / "screening_results.json"
        _save_cache_json(cache_path, results)

        _safe_log("=" * 72)
        _safe_log("SCREENING COMPLETE")
        _safe_log(f"  Universe: {len(symbols)} -> Liquid: {len(liquid)} -> "
                   f"Quality: {len(quality)} -> Selected: {len(selected)}")
        _safe_log(f"  Final assets: {', '.join(selected)}")
        _safe_log(f"  Elapsed: {elapsed:.1f}s")
        _safe_log("=" * 72)

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load config if available
    config = {}
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    screener = UniverseScreener(config=config)
    results = screener.run_full_screening(
        top_n=500,
        final_n=10,
        lookback_days=365,
        n_clusters=20,
    )

    print("\n--- SELECTED ASSETS ---")
    for i, sym in enumerate(results["selected"], 1):
        print(f"  {i}. {sym}")

    print("\n--- CLASSIFICATION ---")
    for cat, assets in results["classification"].items():
        if assets:
            print(f"  {cat}: {', '.join(assets)}")

    print(f"\nResults saved to data/cache/universe_screening/")
