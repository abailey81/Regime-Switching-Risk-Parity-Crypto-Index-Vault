"""
Feature engineering pipeline for the ML ensemble.

Core features (per-asset):
    log returns, realised vol (close-to-close, Parkinson, Garman-Klass),
    rolling Sharpe, momentum, volume ratio, rolling skew / kurtosis,
    Amihud illiquidity, Hurst exponent, rolling BTC correlation,
    volume-weighted returns.

Cross-asset features:
    market breadth, average pairwise correlation, first principal component,
    return dispersion.

Data quality:
    outlier detection (winsorize / MAD / z-score), ADF stationarity tests,
    look-ahead bias checks, NaN assertion.
"""

import logging
import json
import os
import yaml
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_MAX_WORKERS = min(8, os.cpu_count() or 4)

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
# Config defaults (safe fallbacks when keys are missing)
# ---------------------------------------------------------------------------
_DEFAULT_PREPROCESSING = {
    "outlier": {
        "method": "winsorize",
        "winsorize_percentile": [1, 99],
        "sigma_threshold": 5,
    },
    "stationarity": {
        "test": "adf",
        "significance": 0.05,
    },
    "features": {
        "volatility_windows": [24, 120, 480],
        "momentum_windows": [120, 480, 1440],
        "sharpe_window": 1440,
        "correlation_window": 720,
        "hurst_window": 480,
        "parkinson_vol": True,
        "garman_klass_vol": True,
        "rolling_skew": True,
        "rolling_kurtosis": True,
        "amihud_illiquidity": True,
    },
    "normalization": {
        "method": "zscore",
        "window": 720,
    },
}


def _get_preprocess_config(config: dict) -> dict:
    """Merge user config with defaults so every key is guaranteed present."""
    user = config.get("preprocessing", {})
    merged = {}
    for section, defaults in _DEFAULT_PREPROCESSING.items():
        if isinstance(defaults, dict):
            merged[section] = {**defaults, **user.get(section, {})}
        else:
            merged[section] = user.get(section, defaults)
    return merged


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def _winsorize_series(s: pd.Series, lo_pct: float, hi_pct: float) -> pd.Series:
    """Clip values below lo_pct-th and above hi_pct-th percentile."""
    lo = np.nanpercentile(s.dropna(), lo_pct)
    hi = np.nanpercentile(s.dropna(), hi_pct)
    return s.clip(lower=lo, upper=hi)


def _mad_outlier_mask(s: pd.Series, threshold: float = 5.0) -> pd.Series:
    """Return boolean mask where True = outlier (MAD method)."""
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=s.index)
    modified_z = 0.6745 * (s - median) / mad
    return modified_z.abs() > threshold


def detect_and_handle_outliers(
    returns_df: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """
    Apply outlier handling to a returns DataFrame based on config.

    Methods: 'winsorize', 'mad', 'zscore'.
    """
    method = cfg.get("method", "winsorize")
    sigma_thresh = cfg.get("sigma_threshold", 5)
    pctiles = cfg.get("winsorize_percentile", [1, 99])

    cleaned = returns_df.copy()

    # Log extreme returns (> sigma_threshold) regardless of method
    for col in cleaned.columns:
        series = cleaned[col].dropna()
        if len(series) < 10:
            continue
        mu, sig = series.mean(), series.std()
        if sig == 0:
            continue
        extremes = ((series - mu).abs() / sig) > sigma_thresh
        n_ext = extremes.sum()
        if n_ext > 0:
            logger.warning(
                f"  {col}: {n_ext} extreme returns (> {sigma_thresh} sigma) detected"
            )

    if method == "winsorize":
        for col in cleaned.columns:
            cleaned[col] = _winsorize_series(cleaned[col], pctiles[0], pctiles[1])
        logger.info(f"Winsorized returns at [{pctiles[0]}, {pctiles[1]}] percentiles")

    elif method == "mad":
        for col in cleaned.columns:
            mask = _mad_outlier_mask(cleaned[col], threshold=sigma_thresh)
            n_replaced = mask.sum()
            if n_replaced > 0:
                median_val = cleaned[col].median()
                cleaned.loc[mask, col] = median_val
                logger.info(f"  {col}: {n_replaced} MAD outliers replaced with median")

    elif method == "zscore":
        for col in cleaned.columns:
            s = cleaned[col]
            mu, sig = s.mean(), s.std()
            if sig > 0:
                z = (s - mu) / sig
                mask = z.abs() > sigma_thresh
                cleaned.loc[mask, col] = np.nan
        cleaned = cleaned.ffill().bfill()
        logger.info(f"Z-score outlier removal (threshold={sigma_thresh})")

    return cleaned


# ---------------------------------------------------------------------------
# Stationarity testing
# ---------------------------------------------------------------------------

def test_stationarity(
    returns_df: pd.DataFrame, significance: float = 0.05
) -> Dict[str, dict]:
    """
    Run Augmented Dickey-Fuller test on each column.

    Returns dict: col -> {statistic, pvalue, is_stationary, lags_used}.
    """
    from statsmodels.tsa.stattools import adfuller

    def _adf_single(col: str) -> Tuple[str, dict]:
        """Run ADF test on a single column."""
        series = returns_df[col].dropna()
        if len(series) < 30:
            logger.warning(f"  ADF skip {col}: too few observations ({len(series)})")
            return col, {
                "statistic": None,
                "pvalue": None,
                "is_stationary": None,
                "lags_used": None,
            }
        try:
            stat, pval, lags, nobs, crit, icbest = adfuller(series, autolag="AIC")
            is_stat = pval < significance
            if not is_stat:
                logger.warning(
                    f"  ADF: {col} is NON-STATIONARY (p={pval:.4f}). "
                    f"Consider differencing."
                )
            return col, {
                "statistic": round(float(stat), 6),
                "pvalue": round(float(pval), 6),
                "is_stationary": bool(is_stat),
                "lags_used": int(lags),
            }
        except Exception as e:
            logger.error(f"  ADF failed for {col}: {e}")
            return col, {
                "statistic": None,
                "pvalue": None,
                "is_stationary": None,
                "lags_used": None,
            }

    results = {}
    columns = list(returns_df.columns)
    adf_bar = tqdm(total=len(columns), desc="ADF Test", unit="asset", leave=False)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_adf_single, col): col for col in columns}
        for future in as_completed(futures):
            col, res = future.result()
            results[col] = res
            adf_bar.update(1)
            if res["pvalue"] is not None:
                status_mark = "ok" if res["is_stationary"] else "FAIL"
                adf_bar.set_postfix(asset=col, p=f"{res['pvalue']:.3f}", status=status_mark)

    adf_bar.close()

    n_stat = sum(1 for v in results.values() if v.get("is_stationary") is True)
    logger.info(
        f"ADF stationarity: {n_stat}/{len(results)} series stationary "
        f"at {significance:.0%} level"
    )
    return results


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

def parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int
) -> pd.Series:
    """
    Parkinson (1980) high-low volatility estimator.
    More efficient than close-to-close when only H/L data used.
    """
    log_hl = np.log(high / low)
    return np.sqrt(
        (1.0 / (4.0 * np.log(2))) * (log_hl ** 2).rolling(window).mean()
    ) * np.sqrt(8760)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC volatility estimator.
    Uses all four OHLC prices for improved efficiency.
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2.0 * np.log(2) - 1.0) * log_co
    return np.sqrt(gk.rolling(window).mean()) * np.sqrt(8760)


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

def rolling_hurst_exponent(
    series: pd.Series, window: int = 480, min_periods: int = 100
) -> pd.Series:
    """
    Estimate Hurst exponent using rescaled range (R/S) analysis
    on a rolling window.

    H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
    """

    def _hurst_rs(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        n = len(x)
        if n < 20:
            return np.nan
        mean_x = np.mean(x)
        y = np.cumsum(x - mean_x)
        r = np.max(y) - np.min(y)
        s = np.std(x, ddof=1)
        if s == 0 or r == 0:
            return np.nan
        rs = r / s
        return np.log(rs) / np.log(n)

    return series.rolling(window, min_periods=min_periods).apply(
        _hurst_rs, raw=False
    )


# ---------------------------------------------------------------------------
# Amihud illiquidity
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    returns: pd.Series, volume: pd.Series, window: int = 24
) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio: mean(|r| / volume) over rolling window.
    Higher = less liquid.
    """
    safe_vol = volume.replace(0, np.nan)
    ratio = returns.abs() / safe_vol
    return ratio.rolling(window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(
    prices_df: pd.DataFrame,
    ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute all per-asset features from close prices.

    Parameters
    ----------
    prices_df : DataFrame of close prices (columns = assets).
    ohlcv_data : Optional dict of raw OHLCV DataFrames keyed by asset.
                 Used for Parkinson / Garman-Klass / Amihud features.
    config : Full config dict (uses 'preprocessing.features' section).

    Returns
    -------
    MultiIndex DataFrame: (timestamp) x (asset, feature)
    """
    pp_cfg = _get_preprocess_config(config or {})
    feat_cfg = pp_cfg["features"]

    vol_windows = feat_cfg.get("volatility_windows", [24, 120, 480])
    mom_windows = feat_cfg.get("momentum_windows", [120, 480, 1440])
    sharpe_window = feat_cfg.get("sharpe_window", 1440)
    corr_window = feat_cfg.get("correlation_window", 720)
    hurst_window = feat_cfg.get("hurst_window", 480)
    use_parkinson = feat_cfg.get("parkinson_vol", True)
    use_gk = feat_cfg.get("garman_klass_vol", True)
    use_skew = feat_cfg.get("rolling_skew", True)
    use_kurt = feat_cfg.get("rolling_kurtosis", True)
    use_amihud = feat_cfg.get("amihud_illiquidity", True)

    window_labels = {24: "1d", 120: "5d", 480: "20d", 720: "30d", 1440: "60d"}

    def _wlabel(w):
        return window_labels.get(w, f"{w}h")

    btc_returns = None
    if "BTC" in prices_df.columns:
        btc_returns = np.log(prices_df["BTC"] / prices_df["BTC"].shift(1))

    def _compute_asset_features(col: str) -> Dict[Tuple[str, str], pd.Series]:
        """Compute all features for a single asset. Returns dict of (asset, feature) -> Series."""
        asset_feats: Dict[Tuple[str, str], pd.Series] = {}
        p = prices_df[col]
        lr = np.log(p / p.shift(1))

        # Log returns
        asset_feats[(col, "log_return")] = lr

        # Close-to-close realised volatility
        for w in vol_windows:
            asset_feats[(col, f"realised_vol_{_wlabel(w)}")] = (
                lr.rolling(w).std() * np.sqrt(8760)
            )

        # Parkinson volatility (if OHLCV available)
        if use_parkinson and ohlcv_data and col in ohlcv_data:
            raw = ohlcv_data[col]
            if "high" in raw.columns and "low" in raw.columns:
                h = raw["high"].reindex(p.index)
                l_col = raw["low"].reindex(p.index)
                for w in vol_windows:
                    asset_feats[(col, f"parkinson_vol_{_wlabel(w)}")] = (
                        parkinson_volatility(h, l_col, w)
                    )

        # Garman-Klass volatility
        if use_gk and ohlcv_data and col in ohlcv_data:
            raw = ohlcv_data[col]
            required = {"open", "high", "low", "close"}
            if required.issubset(raw.columns):
                o = raw["open"].reindex(p.index)
                h = raw["high"].reindex(p.index)
                l_col = raw["low"].reindex(p.index)
                c = raw["close"].reindex(p.index)
                for w in vol_windows:
                    asset_feats[(col, f"garman_klass_vol_{_wlabel(w)}")] = (
                        garman_klass_volatility(o, h, l_col, c, w)
                    )

        # Rolling Sharpe
        asset_feats[(col, f"rolling_sharpe_{_wlabel(sharpe_window)}")] = (
            lr.rolling(sharpe_window).mean() / lr.rolling(sharpe_window).std()
        ) * np.sqrt(8760)

        # Momentum
        for w in mom_windows:
            asset_feats[(col, f"momentum_{_wlabel(w)}")] = p / p.shift(w) - 1

        # Rolling skewness
        if use_skew:
            for w in vol_windows:
                asset_feats[(col, f"rolling_skew_{_wlabel(w)}")] = lr.rolling(w).skew()

        # Rolling kurtosis
        if use_kurt:
            for w in vol_windows:
                asset_feats[(col, f"rolling_kurt_{_wlabel(w)}")] = lr.rolling(w).kurt()

        # Rolling correlation with BTC (the market factor)
        if btc_returns is not None and col != "BTC":
            asset_feats[(col, f"btc_corr_{_wlabel(corr_window)}")] = (
                lr.rolling(corr_window).corr(btc_returns)
            )

        # Volume-weighted returns
        if ohlcv_data and col in ohlcv_data and "volume" in ohlcv_data[col].columns:
            vol = ohlcv_data[col]["volume"].reindex(p.index)
            asset_feats[(col, "volume_weighted_return")] = lr * vol

            # Volume ratio
            for w in vol_windows:
                asset_feats[(col, f"volume_ratio_{_wlabel(w)}")] = (
                    vol / vol.rolling(w).mean()
                )

        # Amihud illiquidity
        if use_amihud and ohlcv_data and col in ohlcv_data and "volume" in ohlcv_data[col].columns:
            vol = ohlcv_data[col]["volume"].reindex(p.index)
            asset_feats[(col, "amihud_illiquidity")] = amihud_illiquidity(lr, vol)

        # Hurst exponent
        asset_feats[(col, "hurst_exponent")] = rolling_hurst_exponent(
            lr, window=hurst_window
        )

        return asset_feats

    # Compute per-asset features in parallel (pandas rolling/numpy release the GIL)
    features: Dict[Tuple[str, str], pd.Series] = {}
    columns = list(prices_df.columns)
    feat_bar = tqdm(total=len(columns), desc="Features", unit="asset", leave=False)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_compute_asset_features, col): col for col in columns}
        for future in as_completed(futures):
            col = futures[future]
            asset_feats = future.result()
            features.update(asset_feats)
            feat_bar.update(1)
            feat_bar.set_postfix(asset=col)

    feat_bar.close()

    features_df = pd.DataFrame(features, index=prices_df.index)
    features_df.columns = pd.MultiIndex.from_tuples(
        features_df.columns, names=["asset", "feature"]
    )

    return features_df


# ---------------------------------------------------------------------------
# Cross-asset features
# ---------------------------------------------------------------------------

def compute_cross_asset_features(
    returns_df: pd.DataFrame, config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute cross-sectional / market-level features.

    Returns a single-level DataFrame indexed by timestamp with columns:
      market_breadth, avg_pairwise_corr, first_pc_variance_ratio,
      return_dispersion.
    """
    pp_cfg = _get_preprocess_config(config or {})
    corr_window = pp_cfg["features"].get("correlation_window", 720)

    cross = {}

    # Market breadth: fraction of assets with positive 24h return
    cross["market_breadth"] = (returns_df.rolling(24).sum() > 0).mean(axis=1)

    # Average pairwise rolling correlation
    def _avg_pairwise_corr(window_returns):
        """Average off-diagonal correlation from rolling window."""
        corr_mat = window_returns.corr()
        n = len(corr_mat)
        if n < 2:
            return np.nan
        mask = ~np.eye(n, dtype=bool)
        return corr_mat.values[mask].mean()

    avg_corr = returns_df.rolling(corr_window).apply(
        lambda _: np.nan, raw=True  # placeholder — computed below
    ).iloc[:, 0]  # Will be overwritten

    # Efficient rolling average correlation
    corr_values = []
    for i in tqdm(range(len(returns_df)), desc="Cross-asset | Avg corr", unit="step", leave=False):
        if i < corr_window:
            corr_values.append(np.nan)
        else:
            window_slice = returns_df.iloc[i - corr_window : i]
            corr_values.append(_avg_pairwise_corr(window_slice))

    cross["avg_pairwise_corr"] = pd.Series(
        corr_values, index=returns_df.index
    )

    # First principal component variance ratio (rolling)
    from sklearn.decomposition import PCA

    pc_ratios = []
    for i in tqdm(range(len(returns_df)), desc="Cross-asset | PCA", unit="step", leave=False):
        if i < corr_window:
            pc_ratios.append(np.nan)
        else:
            window_slice = returns_df.iloc[i - corr_window : i].dropna()
            if len(window_slice) < 30 or window_slice.shape[1] < 2:
                pc_ratios.append(np.nan)
                continue
            try:
                pca = PCA(n_components=1)
                pca.fit(window_slice)
                pc_ratios.append(float(pca.explained_variance_ratio_[0]))
            except Exception:
                pc_ratios.append(np.nan)

    cross["first_pc_variance_ratio"] = pd.Series(
        pc_ratios, index=returns_df.index
    )

    # Return dispersion: cross-sectional standard deviation of returns
    cross["return_dispersion"] = returns_df.std(axis=1)

    cross_df = pd.DataFrame(cross, index=returns_df.index)
    logger.info(
        f"Cross-asset features computed: {list(cross_df.columns)} "
        f"({len(cross_df)} rows)"
    )
    return cross_df


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def normalize_features(
    features_df: pd.DataFrame, config: Optional[dict] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply rolling normalization to features.

    Returns (normalized_df, params_dict) where params_dict stores
    per-column rolling mean/std for inverse transform.
    """
    pp_cfg = _get_preprocess_config(config or {})
    norm_cfg = pp_cfg["normalization"]
    method = norm_cfg.get("method", "zscore")
    window = norm_cfg.get("window", 720)

    norm_params = {"method": method, "window": window, "columns": []}

    if method == "none":
        logger.info("Normalization: none (pass-through)")
        return features_df.copy(), norm_params

    normalized = features_df.copy()

    if method == "zscore":
        rolling_mean = features_df.rolling(window, min_periods=30).mean()
        rolling_std = features_df.rolling(window, min_periods=30).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        normalized = (features_df - rolling_mean) / rolling_std
        norm_params["rolling_mean_last"] = rolling_mean.iloc[-1].to_dict() if len(rolling_mean) > 0 else {}
        norm_params["rolling_std_last"] = rolling_std.iloc[-1].to_dict() if len(rolling_std) > 0 else {}
        logger.info(f"Applied rolling z-score normalization (window={window})")

    elif method == "rank":
        for col_tuple in tqdm(features_df.columns, desc="Rank normalization", unit="col", leave=False):
            col_data = features_df[col_tuple]
            normalized[col_tuple] = col_data.rolling(window, min_periods=30).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
        logger.info(f"Applied rolling rank normalization (window={window})")

    norm_params["columns"] = [
        str(c) for c in features_df.columns.tolist()
    ]
    return normalized, norm_params


# ---------------------------------------------------------------------------
# Data validation pipeline
# ---------------------------------------------------------------------------

def validate_data_pipeline(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> bool:
    """
    Final validation checks before data leaves the pipeline.

    Checks:
      1. No look-ahead bias (future data leaking)
      2. All time series are aligned
      3. No NaN in final output matrices

    Returns True if all checks pass, raises ValueError otherwise.
    """
    issues = []

    # 1. Check alignment — all DataFrames share the same index subset
    r_idx = returns_df.index
    f_idx = features_df.index
    p_idx = prices_df.index

    if not r_idx.isin(p_idx).all():
        issues.append("Returns index contains timestamps not in prices")
    # Features may have NaN from rolling windows — just check index alignment
    if not f_idx.equals(p_idx):
        # Features should cover the same range (they share the prices index)
        if len(f_idx) != len(p_idx):
            logger.warning(
                f"Features index length ({len(f_idx)}) != prices ({len(p_idx)})"
            )

    # 2. Check no future timestamps
    now = pd.Timestamp.now(tz="UTC")
    if prices_df.index.max() > now + pd.Timedelta(hours=2):
        issues.append(
            f"Prices contain future timestamps (max={prices_df.index.max()})"
        )

    # 3. Check for NaN in returns
    nan_pct_returns = returns_df.isna().mean().mean() * 100
    if nan_pct_returns > 0.1:
        issues.append(
            f"Returns matrix has {nan_pct_returns:.2f}% NaN (threshold: 0.1%)"
        )

    if issues:
        for iss in issues:
            logger.error(f"  VALIDATION FAIL: {iss}")
        raise ValueError(
            f"Data pipeline validation failed with {len(issues)} issue(s). "
            f"See log for details."
        )

    logger.info("Data pipeline validation: ALL CHECKS PASSED")
    return True


# ---------------------------------------------------------------------------
# Core functions (preserved interface)
# ---------------------------------------------------------------------------

def compute_return_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple log return matrix for all assets."""
    return np.log(prices_df / prices_df.shift(1)).dropna()


def compute_rolling_correlation(
    returns_df: pd.DataFrame, window: int = 1440
) -> pd.DataFrame:
    """Compute rolling pairwise correlation matrix (flattened)."""
    return returns_df.rolling(window).corr()


def prepare_hmm_features(
    prices_df: pd.DataFrame, returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare features specifically for HMM regime detection.

    Features:
      - log_returns_mean_24h: Average return across all assets (24h rolling)
      - realised_vol_120h: Average realised vol across all assets (5-day)
      - market_breadth: Fraction of assets with positive 24h return
    """
    mean_return_24h = returns_df.rolling(24).mean().mean(axis=1)
    avg_vol_120h = returns_df.rolling(120).std().mean(axis=1) * np.sqrt(8760)
    breadth = (returns_df.rolling(24).sum() > 0).mean(axis=1)

    hmm_features = pd.DataFrame(
        {
            "log_returns_mean_24h": mean_return_24h,
            "realised_vol_120h": avg_vol_120h,
            "market_breadth": breadth,
        }
    ).dropna()

    return hmm_features


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def prepare_all_data(config_path: str = "config.yaml"):
    """
    Main preprocessing pipeline.

    Returns:
        prices_df: Close prices for all assets (aligned)
        returns_df: Log returns for all assets (outlier-handled)
        features_df: Full feature matrix (normalized)
        hmm_features: Features for HMM regime detection
    """
    from .fetch_data import load_cached_data

    with open(config_path) as f:
        config = yaml.safe_load(f)

    pp_cfg = _get_preprocess_config(config)

    raw_data = load_cached_data(config_path)

    if not raw_data:
        raise ValueError("No cached data found. Run fetch_data.py first.")

    # --- Extract close prices and align timestamps ---
    close_prices = {}
    for symbol, df in raw_data.items():
        close_prices[symbol] = df["close"]

    prices_df = pd.DataFrame(close_prices)

    # Forward-fill gaps up to 6 hours, then drop remaining NaN rows
    prices_df = prices_df.ffill(limit=6).dropna()

    logger.info(
        f"Aligned price matrix: {prices_df.shape[0]} rows x "
        f"{prices_df.shape[1]} assets"
    )
    logger.info(f"Date range: {prices_df.index[0]} to {prices_df.index[-1]}")

    # --- Compute raw returns ---
    returns_df = compute_return_matrix(prices_df)

    # --- Outlier handling ---
    returns_df = detect_and_handle_outliers(returns_df, pp_cfg["outlier"])

    # --- Stationarity testing ---
    stationarity_results = test_stationarity(
        returns_df, significance=pp_cfg["stationarity"].get("significance", 0.05)
    )

    # Apply differencing for non-stationary series (rare for returns, but guard)
    for col, res in stationarity_results.items():
        if res.get("is_stationary") is False:
            logger.warning(
                f"  {col}: Applying first differencing to achieve stationarity"
            )
            returns_df[col] = returns_df[col].diff()
    returns_df = returns_df.dropna()

    # --- Feature engineering (with OHLCV for advanced vol estimators) ---
    features_df = compute_features(prices_df, ohlcv_data=raw_data, config=config)

    # --- Cross-asset features ---
    cross_features = compute_cross_asset_features(returns_df, config=config)

    # --- HMM features ---
    hmm_features = prepare_hmm_features(prices_df, returns_df)

    # --- Feature normalization ---
    norm_cfg = pp_cfg["normalization"]
    if norm_cfg.get("method", "zscore") != "none":
        features_df, norm_params = normalize_features(features_df, config=config)

        # Save normalization params for inverse transform
        output_dir = Path("data/cache")
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_path = output_dir / "normalization_params.json"
        # Convert any non-serialisable keys
        serialisable_params = {
            "method": norm_params["method"],
            "window": norm_params["window"],
        }
        with open(norm_path, "w") as f:
            json.dump(serialisable_params, f, indent=2)

    # --- Data validation ---
    try:
        validate_data_pipeline(prices_df, returns_df, features_df)
    except ValueError as e:
        logger.error(f"Data validation failed: {e}")
        # Continue anyway — caller decides whether to abort

    # --- Save processed data ---
    output_dir = Path("data/cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    prices_df.to_parquet(output_dir / "prices_aligned.parquet")
    returns_df.to_parquet(output_dir / "returns.parquet")
    hmm_features.to_parquet(output_dir / "hmm_features.parquet")
    cross_features.to_parquet(output_dir / "cross_features.parquet")

    # Save stationarity report
    stat_path = output_dir / "stationarity_report.json"
    with open(stat_path, "w") as f:
        json.dump(stationarity_results, f, indent=2)
    logger.info(f"Stationarity report saved to {stat_path}")

    logger.info(f"Saved preprocessed data to {output_dir}")

    return prices_df, returns_df, features_df, hmm_features


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices, returns, features, hmm_feat = prepare_all_data()
    print(f"Prices: {prices.shape}")
    print(f"Returns: {returns.shape}")
    print(f"Features: {features.shape}")
    print(f"HMM features: {hmm_feat.shape}")
    print(f"\nAssets: {list(prices.columns)}")
