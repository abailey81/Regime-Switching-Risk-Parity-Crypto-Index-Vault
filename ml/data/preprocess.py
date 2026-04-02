"""
Feature engineering pipeline for the ML ensemble.
Computes: log returns, realised vol, rolling Sharpe, momentum, volume ratio,
rolling correlations — all aligned on a common hourly timestamp index.
"""
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from a DataFrame of close prices (columns = assets).

    Returns a MultiIndex DataFrame: (timestamp) x (asset, feature)
    """
    features = {}

    for col in prices_df.columns:
        p = prices_df[col]

        # Log returns
        features[(col, "log_return")] = np.log(p / p.shift(1))

        # Realised volatility (multiple windows)
        lr = np.log(p / p.shift(1))
        for window, label in [(24, "1d"), (120, "5d"), (480, "20d")]:
            features[(col, f"realised_vol_{label}")] = lr.rolling(window).std() * np.sqrt(8760)

        # Rolling Sharpe (60-day = 1440 hours)
        features[(col, "rolling_sharpe_60d")] = (
            lr.rolling(1440).mean() / lr.rolling(1440).std()
        ) * np.sqrt(8760)

        # Momentum
        for window, label in [(120, "5d"), (480, "20d"), (1440, "60d")]:
            features[(col, f"momentum_{label}")] = p / p.shift(window) - 1

        # Volume ratio (if volume available)
        # Will be added if volume data is present

    features_df = pd.DataFrame(features, index=prices_df.index)
    features_df.columns = pd.MultiIndex.from_tuples(features_df.columns, names=["asset", "feature"])

    return features_df


def compute_return_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple log return matrix for all assets."""
    return np.log(prices_df / prices_df.shift(1)).dropna()


def compute_rolling_correlation(returns_df: pd.DataFrame, window: int = 1440) -> pd.DataFrame:
    """Compute rolling pairwise correlation matrix (flattened)."""
    return returns_df.rolling(window).corr()


def prepare_hmm_features(prices_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features specifically for HMM regime detection.

    Features:
      - log_returns_mean_24h: Average return across all assets (24h rolling)
      - realised_vol_120h: Average realised vol across all assets (5-day)
      - volume_ratio_480h: Placeholder (not available for all assets)
    """
    mean_return_24h = returns_df.rolling(24).mean().mean(axis=1)
    avg_vol_120h = returns_df.rolling(120).std().mean(axis=1) * np.sqrt(8760)

    # Market breadth: fraction of assets with positive 24h return
    breadth = (returns_df.rolling(24).sum() > 0).mean(axis=1)

    hmm_features = pd.DataFrame({
        "log_returns_mean_24h": mean_return_24h,
        "realised_vol_120h": avg_vol_120h,
        "market_breadth": breadth,
    }).dropna()

    return hmm_features


def prepare_all_data(config_path: str = "config.yaml"):
    """
    Main preprocessing pipeline.

    Returns:
        prices_df: Close prices for all assets (aligned)
        returns_df: Log returns for all assets
        features_df: Full feature matrix
        hmm_features: Features for HMM regime detection
    """
    from .fetch_data import load_cached_data

    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_data = load_cached_data(config_path)

    if not raw_data:
        raise ValueError("No cached data found. Run fetch_data.py first.")

    # Extract close prices and align timestamps
    close_prices = {}
    for symbol, df in raw_data.items():
        close_prices[symbol] = df["close"]

    prices_df = pd.DataFrame(close_prices)

    # Forward-fill gaps up to 6 hours, then drop remaining NaN rows
    prices_df = prices_df.ffill(limit=6).dropna()

    logger.info(f"Aligned price matrix: {prices_df.shape[0]} rows x {prices_df.shape[1]} assets")
    logger.info(f"Date range: {prices_df.index[0]} to {prices_df.index[-1]}")

    # Compute features
    returns_df = compute_return_matrix(prices_df)
    features_df = compute_features(prices_df)
    hmm_features = prepare_hmm_features(prices_df, returns_df)

    # Save processed data
    output_dir = Path("data/cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    prices_df.to_parquet(output_dir / "prices_aligned.parquet")
    returns_df.to_parquet(output_dir / "returns.parquet")
    hmm_features.to_parquet(output_dir / "hmm_features.parquet")

    logger.info(f"Saved preprocessed data to {output_dir}")

    return prices_df, returns_df, features_df, hmm_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices, returns, features, hmm_feat = prepare_all_data()
    print(f"Prices: {prices.shape}")
    print(f"Returns: {returns.shape}")
    print(f"HMM features: {hmm_feat.shape}")
    print(f"\nAssets: {list(prices.columns)}")
