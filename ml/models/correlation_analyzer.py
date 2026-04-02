"""
Correlation Regime Analysis — Adapted from Crypto-Statistical-Arbitrage.

Detects when portfolio diversification breaks down by monitoring
cross-asset correlations. During crises, crypto correlations approach 1.0,
eliminating diversification benefits precisely when needed most.

Features:
  - Rolling pairwise correlations
  - EWMA (Exponentially Weighted) correlations
  - DCC-based dynamic correlations (uses GARCH residuals if available)
  - Correlation regime classification with hysteresis
  - Average correlation as HMM feature
  - Correlation breakdown detection (early warning)
  - Eigenvalue analysis (Marchenko-Pastur edge, concentration)
  - Minimum spanning tree of correlations
  - Rolling absorption ratio

Adapted from: Crypto-Statistical-Arbitrage/portfolio/correlation_analysis.py
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Correlation regime classifications from stat-arb."""
    LOW = "low"           # Avg corr < 0.3 — good diversification
    NORMAL = "normal"     # 0.3 - 0.6 — standard conditions
    HIGH = "high"         # 0.6 - 0.8 — diversification weakening
    CRISIS = "crisis"     # > 0.8 — diversification breakdown

    @classmethod
    def from_value(cls, avg_corr: float) -> "CorrelationRegime":
        if avg_corr < 0.3:
            return cls.LOW
        elif avg_corr < 0.6:
            return cls.NORMAL
        elif avg_corr < 0.8:
            return cls.HIGH
        else:
            return cls.CRISIS


class CorrelationAnalyzer:
    """
    Portfolio correlation analysis for diversification monitoring.

    Provides rolling and EWMA correlations, regime classification,
    and average correlation as a feature for HMM regime detection.
    """

    def __init__(self, returns_df: pd.DataFrame, config: Optional[Dict] = None):
        self.returns = returns_df
        self.n_assets = returns_df.shape[1]
        self.asset_names = list(returns_df.columns)

        cfg = config or {}
        # Hysteresis parameters
        self.hysteresis_hours = cfg.get("hysteresis_hours", 24)
        self.regime_thresholds = cfg.get("regime_thresholds", {
            "low": 0.3, "normal": 0.6, "high": 0.8,
        })

        # Absorption ratio parameters
        self.absorption_k = cfg.get("absorption_k_eigenvectors", 3)

        # Internal hysteresis state
        self._hysteresis_regime: Optional[CorrelationRegime] = None
        self._hysteresis_counter: int = 0

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

    # ─── DCC-BASED DYNAMIC CORRELATIONS ───

    def dcc_dynamic_correlations(
        self,
        garch_residuals: Optional[np.ndarray] = None,
        halflife: int = 120,
    ) -> np.ndarray:
        """
        DCC-inspired dynamic correlation matrix using GARCH residuals.

        If GARCH residuals are provided, standardises them and computes
        EWMA correlation on the standardised residuals (proxy for the
        DCC correlation dynamics).  Falls back to EWMA on raw returns
        if residuals are not available.

        Args:
            garch_residuals: (T x n_assets) array of GARCH model residuals.
            halflife: EWMA halflife in hours for correlation dynamics.

        Returns:
            (n_assets x n_assets) dynamic correlation matrix at the last timestamp.
        """
        if garch_residuals is not None and garch_residuals.shape[1] == self.n_assets:
            logger.info("Computing DCC correlations from GARCH residuals (halflife=%d)", halflife)
            # Standardise residuals column-wise
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                std_resid = garch_residuals / (np.std(garch_residuals, axis=0, keepdims=True) + 1e-12)

            resid_df = pd.DataFrame(std_resid, columns=self.asset_names)
        else:
            logger.info("No GARCH residuals; computing DCC proxy from raw returns (halflife=%d)", halflife)
            resid_df = self.returns

        # EWMA covariance on (standardised) residuals
        ewma_cov = resid_df.ewm(halflife=halflife).cov()
        last_idx = ewma_cov.index.get_level_values(0)[-1]
        cov_matrix = ewma_cov.loc[last_idx].values

        vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        corr = cov_matrix / np.outer(vols, vols)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1, 1)

    # ─── AVERAGE CORRELATION ───

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

    # ─── REGIME WITH HYSTERESIS ───

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

    def current_regime_with_hysteresis(self, window: int = 1440) -> Dict:
        """
        Classify correlation regime with hysteresis to prevent whipsaw.

        Only switches regime if the new classification persists for
        ``hysteresis_hours`` consecutive observations.

        Args:
            window: Correlation estimation window in hours.

        Returns:
            Dict with 'regime', 'avg_correlation', 'hysteresis_counter',
            'candidate_regime', and pairwise stats.
        """
        if len(self.returns) < window:
            return {"regime": CorrelationRegime.NORMAL, "avg_correlation": 0.5,
                    "hysteresis_counter": 0, "candidate_regime": CorrelationRegime.NORMAL}

        corr = self.returns.iloc[-window:].corr().values
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg = float(np.mean(np.abs(corr[mask])))
        candidate = CorrelationRegime.from_value(avg)

        # Initialize hysteresis state if needed
        if self._hysteresis_regime is None:
            self._hysteresis_regime = candidate
            self._hysteresis_counter = 0

        if candidate != self._hysteresis_regime:
            self._hysteresis_counter += 1
            if self._hysteresis_counter >= self.hysteresis_hours:
                logger.info(
                    "Correlation regime change confirmed: %s → %s after %d hours",
                    self._hysteresis_regime.value, candidate.value,
                    self._hysteresis_counter,
                )
                self._hysteresis_regime = candidate
                self._hysteresis_counter = 0
            else:
                logger.debug(
                    "Correlation regime candidate %s (%d/%d hours)",
                    candidate.value, self._hysteresis_counter, self.hysteresis_hours,
                )
        else:
            self._hysteresis_counter = 0

        return {
            "regime": self._hysteresis_regime,
            "avg_correlation": avg,
            "hysteresis_counter": self._hysteresis_counter,
            "candidate_regime": candidate,
            "max_pairwise": float(np.max(corr[mask])),
            "min_pairwise": float(np.min(corr[mask])),
        }

    # ─── EIGENVALUE ANALYSIS ───

    def eigenvalue_analysis(self, window: int = 1440) -> Dict:
        """
        Analyse eigenvalues of the correlation matrix.

        Tracks:
          - Largest eigenvalue vs. Marchenko-Pastur upper edge
          - Eigenvalue concentration (fraction of variance in top eigenvalue)
          - Effective rank (entropy-based)

        A largest eigenvalue well above the MP edge indicates a strong
        common factor; eigenvalue concentration approaching 1.0 signals
        complete diversification breakdown.

        Args:
            window: Correlation estimation window.

        Returns:
            Dict with 'eigenvalues', 'largest', 'mp_upper_edge',
            'concentration', 'effective_rank', 'above_mp'.
        """
        if len(self.returns) < window:
            logger.warning("Insufficient data for eigenvalue analysis")
            return {"eigenvalues": [], "largest": 0.0, "mp_upper_edge": 0.0,
                    "concentration": 0.0, "effective_rank": 0.0, "above_mp": False}

        corr = self.returns.iloc[-window:].corr().values
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.sort(eigvals)[::-1]  # Descending

        n = self.n_assets
        T = window
        q = n / T  # Aspect ratio

        # Marchenko-Pastur upper edge: sigma^2 * (1 + sqrt(q))^2
        # For correlation matrix, sigma^2 = 1
        mp_upper = (1 + np.sqrt(q)) ** 2

        largest = float(eigvals[0])
        concentration = largest / float(eigvals.sum()) if eigvals.sum() > 0 else 0.0

        # Effective rank: exp(entropy of normalised eigenvalues)
        normed = eigvals / eigvals.sum()
        normed = normed[normed > 1e-12]  # Avoid log(0)
        entropy = -np.sum(normed * np.log(normed))
        effective_rank = float(np.exp(entropy))

        above_mp = largest > mp_upper

        logger.info(
            "Eigenvalue analysis: largest=%.3f  MP_edge=%.3f  concentration=%.3f  "
            "eff_rank=%.1f  above_MP=%s",
            largest, mp_upper, concentration, effective_rank, above_mp,
        )

        return {
            "eigenvalues": eigvals.tolist(),
            "largest": largest,
            "mp_upper_edge": float(mp_upper),
            "concentration": concentration,
            "effective_rank": effective_rank,
            "above_mp": above_mp,
        }

    # ─── MINIMUM SPANNING TREE ───

    def minimum_spanning_tree(self, window: int = 1440) -> Dict:
        """
        Compute the minimum spanning tree (MST) of the correlation matrix.

        Uses correlation distance d(i,j) = sqrt(0.5 * (1 - rho)) and
        Kruskal's algorithm to find the MST. Identifies clusters and
        key linking assets.

        Args:
            window: Correlation estimation window.

        Returns:
            Dict with 'edges' (list of (i_name, j_name, distance)),
            'hub_asset' (most connected node), 'total_distance'.
        """
        if len(self.returns) < window:
            logger.warning("Insufficient data for MST")
            return {"edges": [], "hub_asset": "", "total_distance": 0.0}

        corr = self.returns.iloc[-window:].corr().values
        n = self.n_assets

        # Correlation distance
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0)

        # Kruskal's algorithm
        edges_all = []
        for i in range(n):
            for j in range(i + 1, n):
                edges_all.append((dist[i, j], i, j))
        edges_all.sort()

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        mst_edges: List[Tuple[str, str, float]] = []
        degree = np.zeros(n, dtype=int)
        total_dist = 0.0

        for d_val, i, j in edges_all:
            if union(i, j):
                mst_edges.append((self.asset_names[i], self.asset_names[j], float(d_val)))
                degree[i] += 1
                degree[j] += 1
                total_dist += d_val
                if len(mst_edges) == n - 1:
                    break

        hub_idx = int(np.argmax(degree))
        hub_asset = self.asset_names[hub_idx]

        logger.info(
            "MST: %d edges, total_distance=%.4f, hub=%s (degree=%d)",
            len(mst_edges), total_dist, hub_asset, degree[hub_idx],
        )

        return {
            "edges": mst_edges,
            "hub_asset": hub_asset,
            "total_distance": float(total_dist),
            "degrees": {self.asset_names[i]: int(degree[i]) for i in range(n)},
        }

    # ─── ROLLING ABSORPTION RATIO ───

    def rolling_absorption_ratio(self, window: int = 1440,
                                  k: Optional[int] = None) -> pd.Series:
        """
        Rolling absorption ratio: fraction of total variance explained
        by the first K eigenvectors of the correlation matrix.

        A high absorption ratio (approaching 1.0) indicates that a small
        number of factors drive all asset returns — i.e., the market is
        moving as a single block and diversification is ineffective.

        Kritzman et al. (2011): "Principal Components as a Measure of
        Systemic Risk."

        Args:
            window: Rolling window in hours.
            k: Number of top eigenvectors (default from config).

        Returns:
            pd.Series of absorption ratios indexed by time.
        """
        n_eigenvectors = k if k is not None else self.absorption_k
        n_eigenvectors = min(n_eigenvectors, self.n_assets)

        if len(self.returns) < window:
            logger.warning("Insufficient data for absorption ratio")
            return pd.Series(dtype=float, name="absorption_ratio")

        ratios = []
        indices = []

        for end in range(window, len(self.returns)):
            start = end - window
            corr = self.returns.iloc[start:end].corr().values
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = np.sort(eigvals)[::-1]

            total_var = eigvals.sum()
            top_k_var = eigvals[:n_eigenvectors].sum()
            ratio = top_k_var / total_var if total_var > 0 else 0.0

            ratios.append(ratio)
            indices.append(self.returns.index[end])

        result = pd.Series(ratios, index=indices, name="absorption_ratio")
        logger.info(
            "Absorption ratio (k=%d): latest=%.4f  mean=%.4f  max=%.4f",
            n_eigenvectors,
            result.iloc[-1] if len(result) > 0 else 0.0,
            result.mean() if len(result) > 0 else 0.0,
            result.max() if len(result) > 0 else 0.0,
        )
        return result

    # ─── CORRELATION BREAKDOWN DETECTION ───

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
        if spike > spike_threshold:
            logger.warning(
                "Correlation breakdown detected: short=%.3f long=%.3f spike=%.3f > %.3f",
                short_avg, long_avg, spike, spike_threshold,
            )
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
