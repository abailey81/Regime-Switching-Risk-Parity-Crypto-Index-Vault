"""
Correlation Regime Analysis — Adapted from Crypto-Statistical-Arbitrage.

Detects when portfolio diversification breaks down by monitoring
cross-asset correlations. During crises, crypto correlations approach 1.0,
eliminating diversification benefits precisely when needed most.

Features:
  - Rolling pairwise correlations
  - EWMA (Exponentially Weighted) correlations
  - Correlation regime classification (low/normal/high/crisis)
  - Average correlation as HMM feature
  - Correlation breakdown detection (early warning)

Adapted from: Crypto-Statistical-Arbitrage/portfolio/correlation_analysis.py
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Correlation regime classifications from stat-arb."""
    LOW = "low"           # Avg corr < 0.3 — good diversification
    NORMAL = "normal"     # 0.3 - 0.6 — standard conditions
    HIGH = "high"         # 0.6 - 0.8 — diversification weakening
    CRISIS = "crisis"     # > 0.8 — diversification breakdown

    @classmethod
    def from_value(cls, avg_corr: float):
        if avg_corr < 0.3: return cls.LOW
        elif avg_corr < 0.6: return cls.NORMAL
        elif avg_corr < 0.8: return cls.HIGH
        else: return cls.CRISIS


class CorrelationAnalyzer:
    """
    Portfolio correlation analysis for diversification monitoring.

    Provides rolling and EWMA correlations, regime classification,
    and average correlation as a feature for HMM regime detection.
    """

    def __init__(self, returns_df: pd.DataFrame):
        self.returns = returns_df
        self.n_assets = returns_df.shape[1]
        self.asset_names = list(returns_df.columns)

    def rolling_correlation_matrix(self, window: int = 1440) -> pd.DataFrame:
        """Rolling pairwise correlation matrix (default: 60-day window)."""
        return self.returns.rolling(window).corr()

    def ewma_correlation_matrix(self, halflife: int = 720) -> np.ndarray:
        """
        EWMA correlation matrix — gives more weight to recent observations.
        Useful for detecting correlation regime changes faster than rolling.
        From stat-arb portfolio/correlation_analysis.py.
        """
        # Compute EWMA covariance
        ewma_cov = self.returns.ewm(halflife=halflife).cov()

        # Extract the most recent full matrix
        last_idx = ewma_cov.index.get_level_values(0)[-1]
        cov_matrix = ewma_cov.loc[last_idx].values

        # Convert to correlation
        vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        corr = cov_matrix / np.outer(vols, vols)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1, 1)

    def average_correlation(self, window: int = 1440) -> pd.Series:
        """
        Average off-diagonal correlation over time.
        Useful as a feature for HMM (high avg corr = crisis regime).
        """
        avg_corr = []
        for end in range(window, len(self.returns)):
            start = end - window
            corr = self.returns.iloc[start:end].corr().values
            mask = ~np.eye(corr.shape[0], dtype=bool)
            avg_corr.append(np.mean(np.abs(corr[mask])))

        result = pd.Series(avg_corr, index=self.returns.index[window:], name="avg_correlation")
        return result

    def current_regime(self, window: int = 1440) -> Dict:
        """Classify current correlation regime."""
        if len(self.returns) < window:
            return {"regime": CorrelationRegime.NORMAL, "avg_corr": 0.5}

        corr = self.returns.iloc[-window:].corr().values
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg = float(np.mean(np.abs(corr[mask])))
        regime = CorrelationRegime.from_value(avg)

        return {
            "regime": regime,
            "avg_correlation": avg,
            "max_pairwise": float(np.max(corr[mask])),
            "min_pairwise": float(np.min(corr[mask])),
        }

    def detect_correlation_breakdown(self, short_window: int = 168,
                                      long_window: int = 1440,
                                      spike_threshold: float = 0.15) -> bool:
        """
        Detect if correlations have spiked recently (early warning).
        Compares short-window avg correlation to long-window baseline.
        A spike > threshold indicates diversification breakdown.
        From stat-arb crisis_analyzer.py correlation spike detection.
        """
        if len(self.returns) < long_window:
            return False

        short_corr = self.returns.iloc[-short_window:].corr().values
        long_corr = self.returns.iloc[-long_window:].corr().values

        mask = ~np.eye(short_corr.shape[0], dtype=bool)
        short_avg = np.mean(np.abs(short_corr[mask]))
        long_avg = np.mean(np.abs(long_corr[mask]))

        spike = short_avg - long_avg
        return spike > spike_threshold

    def get_hmm_features(self, window: int = 1440) -> pd.DataFrame:
        """
        Extract correlation-based features for HMM regime detection.
        Returns avg_correlation and correlation_regime_score.
        """
        avg_corr = self.average_correlation(window)

        # Regime score: 0 (low corr, good) to 1 (crisis corr, bad)
        regime_score = avg_corr.clip(0, 1)

        return pd.DataFrame({
            "avg_correlation": avg_corr,
            "corr_regime_score": regime_score,
        })
