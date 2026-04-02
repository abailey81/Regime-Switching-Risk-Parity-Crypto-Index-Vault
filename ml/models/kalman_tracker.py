"""
Kalman Filter Pair Tracker — Dynamic hedge ratio estimation.

Tracks time-varying relationships between asset pairs using a Kalman
filter with state = [alpha, beta].  The observation model is:

    y_t = alpha_t + beta_t * x_t + epsilon_t

where alpha and beta evolve as a random walk:

    [alpha_t]   [alpha_{t-1}]
    [beta_t ] = [beta_{t-1} ] + w_t     w_t ~ N(0, Q)

Key use case for the vault: prove that stETH tracks ETH and rETH
tracks ETH with near-unit beta, validating their inclusion as
separate portfolio constituents (staking yield layer) rather than
redundant crypto exposure.

Adapted from: Crypto-Statistical-Arbitrage/models/kalman_filter.py
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KalmanPairTracker:
    """
    Kalman Filter for dynamic hedge ratio estimation between asset pairs.

    Tracks time-varying alpha (intercept) and beta (hedge ratio) in:
        y_t = alpha_t + beta_t * x_t + epsilon_t

    Key use case: Prove stETH/ETH and rETH/ETH maintain stable tracking
    relationships, validating their inclusion as separate portfolio constituents.

    Adapted from Crypto-Statistical-Arbitrage (kalman_filter.py).

    Parameters
    ----------
    delta : float
        Controls the rate of state evolution.  Higher = more adaptive.
        The process noise covariance Q = delta / (1 - delta) * I.
        Default 1e-4 gives smooth, slowly-varying hedge ratios.
    obs_noise_init : float
        Initial observation noise variance (Ve).  Updated online via
        the Kalman innovation sequence.
    """

    def __init__(self, delta: float = 1e-4, obs_noise_init: float = 1e-3):
        self.delta = delta
        self.obs_noise_init = obs_noise_init

        # State dimension: [alpha, beta]
        self._n_state = 2

        # Computed by fit()
        self._alphas: Optional[np.ndarray] = None
        self._betas: Optional[np.ndarray] = None
        self._spreads: Optional[np.ndarray] = None
        self._spread_zscores: Optional[np.ndarray] = None
        self._Ve: Optional[np.ndarray] = None  # observation noise over time
        self._dates: Optional[pd.DatetimeIndex] = None
        self._fitted = False

    # ─── CORE KALMAN FILTER ──────────────────────────────────────────────

    def fit(self, y: pd.Series, x: pd.Series) -> "KalmanPairTracker":
        """
        Run the Kalman filter to estimate time-varying alpha and beta.

        Parameters
        ----------
        y : pd.Series
            Dependent asset prices (e.g., stETH).
        x : pd.Series
            Independent asset prices (e.g., ETH).

        Returns
        -------
        self
            For method chaining.
        """
        if len(y) != len(x):
            raise ValueError(
                f"y and x must have the same length (got {len(y)} vs {len(x)})"
            )
        if len(y) < 10:
            raise ValueError(f"Need at least 10 observations (got {len(y)})")

        T = len(y)
        y_vals = np.asarray(y, dtype=np.float64)
        x_vals = np.asarray(x, dtype=np.float64)

        # Process noise covariance
        Wt = self.delta / (1.0 - self.delta) * np.eye(self._n_state)

        # Storage
        alphas = np.zeros(T)
        betas = np.zeros(T)
        Ve_series = np.zeros(T)

        # Initial state: OLS on first min(50, T//5) observations as warm-up
        n_init = min(50, max(10, T // 5))
        X_init = np.column_stack([np.ones(n_init), x_vals[:n_init]])
        try:
            theta_init = np.linalg.lstsq(X_init, y_vals[:n_init], rcond=None)[0]
        except np.linalg.LinAlgError:
            theta_init = np.array([0.0, 1.0])

        # State vector and covariance
        theta = theta_init.copy()
        R = np.eye(self._n_state) * 1.0  # initial state covariance
        Ve = self.obs_noise_init  # observation noise variance

        for t in range(T):
            # Design vector: [1, x_t]
            F = np.array([1.0, x_vals[t]])

            # Prediction step
            # theta_{t|t-1} = theta_{t-1|t-1}  (random walk)
            R_pred = R + Wt

            # Observation prediction
            y_hat = F @ theta
            innovation = y_vals[t] - y_hat

            # Innovation variance
            S = F @ R_pred @ F + Ve

            # Kalman gain
            if abs(S) < 1e-16:
                K = np.zeros(self._n_state)
            else:
                K = R_pred @ F / S

            # Update step
            theta = theta + K * innovation
            R = R_pred - np.outer(K, K) * S

            # Update observation noise estimate (exponential smoothing)
            Ve = 0.99 * Ve + 0.01 * innovation ** 2

            alphas[t] = theta[0]
            betas[t] = theta[1]
            Ve_series[t] = Ve

        # Compute spreads: y - (alpha + beta * x)
        spreads = y_vals - (alphas + betas * x_vals)

        # Spread z-scores using expanding window (min 20 obs)
        spread_zscores = np.zeros(T)
        for t in range(20, T):
            window = spreads[:t + 1]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if sigma > 1e-12:
                spread_zscores[t] = (spreads[t] - mu) / sigma

        # Store results
        self._alphas = alphas
        self._betas = betas
        self._spreads = spreads
        self._spread_zscores = spread_zscores
        self._Ve = Ve_series
        self._dates = y.index if hasattr(y, "index") else pd.RangeIndex(T)
        self._fitted = True

        logger.info(
            "Kalman filter fitted: T=%d  mean_beta=%.4f  beta_std=%.4f  "
            "mean_spread=%.6f  spread_vol=%.6f",
            T, np.mean(betas), np.std(betas),
            np.mean(spreads), np.std(spreads),
        )
        return self

    # ─── TRACKING QUALITY ────────────────────────────────────────────────

    def get_tracking_quality(self) -> Dict[str, float]:
        """
        Assess how well y tracks x via the Kalman-estimated relationship.

        Returns
        -------
        dict
            mean_beta : Average hedge ratio (close to 1.0 = good tracking).
            beta_std : Std of hedge ratio (low = stable relationship).
            tracking_error : Annualised std of spread (lower = tighter peg).
            r_squared : Fraction of y variance explained by alpha + beta*x.
            mean_spread : Average pricing spread.
            spread_vol : Std of spread.
        """
        self._check_fitted()

        # R-squared from residual variance
        y_var = np.var(self._spreads + self._alphas + self._betas)
        spread_var = np.var(self._spreads)
        r_sq = 1.0 - spread_var / y_var if y_var > 1e-16 else 0.0

        # Annualised tracking error (hourly data)
        te_annual = float(np.std(self._spreads, ddof=1) * np.sqrt(8760))

        result = {
            "mean_beta": float(np.mean(self._betas)),
            "beta_std": float(np.std(self._betas, ddof=1)),
            "tracking_error": te_annual,
            "r_squared": float(np.clip(r_sq, 0.0, 1.0)),
            "mean_spread": float(np.mean(self._spreads)),
            "spread_vol": float(np.std(self._spreads, ddof=1)),
        }

        logger.info(
            "Tracking quality: beta=%.4f+/-%.4f  R2=%.4f  TE=%.4f",
            result["mean_beta"], result["beta_std"],
            result["r_squared"], result["tracking_error"],
        )
        return result

    # ─── HEDGE RATIOS ────────────────────────────────────────────────────

    def get_hedge_ratios(self) -> pd.DataFrame:
        """
        Return the full time series of Kalman-estimated parameters.

        Returns
        -------
        pd.DataFrame
            Columns: date, alpha, beta, spread, spread_zscore.
        """
        self._check_fitted()

        df = pd.DataFrame({
            "date": self._dates,
            "alpha": self._alphas,
            "beta": self._betas,
            "spread": self._spreads,
            "spread_zscore": self._spread_zscores,
        })
        return df

    # ─── DEPEG DETECTION ─────────────────────────────────────────────────

    def detect_depegs(
        self, threshold_sigma: float = 3.0
    ) -> List[Dict[str, object]]:
        """
        Detect depeg events where the spread exceeds a z-score threshold.

        A depeg event is defined as a contiguous block of observations where
        |spread_zscore| > threshold_sigma.  Events are merged if separated
        by fewer than 6 observations (to avoid fragmenting a single event).

        Parameters
        ----------
        threshold_sigma : float
            Z-score threshold for flagging a depeg.

        Returns
        -------
        list of dict
            Each dict: {start_date, end_date, peak_date, peak_spread,
                        peak_zscore, duration_hours, severity}.
            severity = integral of |zscore| above threshold over the event.
        """
        self._check_fitted()

        T = len(self._spread_zscores)
        is_depeg = np.abs(self._spread_zscores) > threshold_sigma

        # Find contiguous depeg blocks
        events: List[Dict[str, object]] = []
        in_event = False
        start = 0

        for t in range(T):
            if is_depeg[t] and not in_event:
                in_event = True
                start = t
            elif not is_depeg[t] and in_event:
                events.append(self._build_depeg_event(start, t - 1, threshold_sigma))
                in_event = False

        if in_event:
            events.append(self._build_depeg_event(start, T - 1, threshold_sigma))

        # Merge events separated by fewer than 6 hours
        merged: List[Dict[str, object]] = []
        for evt in events:
            if merged and (evt["_start_idx"] - merged[-1]["_end_idx"]) < 6:
                # Extend the previous event
                prev = merged[-1]
                new_end = evt["_end_idx"]
                merged[-1] = self._build_depeg_event(
                    prev["_start_idx"], new_end, threshold_sigma
                )
            else:
                merged.append(evt)

        # Remove internal index fields
        for evt in merged:
            evt.pop("_start_idx", None)
            evt.pop("_end_idx", None)

        logger.info(
            "Depeg detection (threshold=%.1f sigma): %d events found",
            threshold_sigma, len(merged),
        )
        return merged

    def _build_depeg_event(
        self, start: int, end: int, threshold: float
    ) -> Dict[str, object]:
        """Build a depeg event dict from index range."""
        zscores = self._spread_zscores[start:end + 1]
        spreads = self._spreads[start:end + 1]

        peak_idx_local = int(np.argmax(np.abs(zscores)))
        peak_idx_global = start + peak_idx_local

        # Severity: integral of excess z-score above threshold
        severity = float(np.sum(np.maximum(np.abs(zscores) - threshold, 0.0)))

        return {
            "start_date": self._dates[start],
            "end_date": self._dates[end],
            "peak_date": self._dates[peak_idx_global],
            "peak_spread": float(spreads[peak_idx_local]),
            "peak_zscore": float(zscores[peak_idx_local]),
            "duration_hours": int(end - start + 1),
            "severity": severity,
            "_start_idx": start,
            "_end_idx": end,
        }

    # ─── MULTI-PAIR ANALYSIS ─────────────────────────────────────────────

    def run_pair_analysis(
        self,
        pairs: List[Tuple[str, str]],
        prices_df: pd.DataFrame,
        config: Optional[Dict] = None,
    ) -> Dict[str, Dict]:
        """
        Run Kalman pair analysis on multiple asset pairs.

        Parameters
        ----------
        pairs : list of (y_name, x_name) tuples
            e.g., [("stETH", "ETH"), ("rETH", "ETH")]
        prices_df : pd.DataFrame
            Price DataFrame with asset columns.
        config : dict, optional
            Override delta and obs_noise_init via keys
            'kalman_delta' and 'kalman_obs_noise'.

        Returns
        -------
        dict
            Keys = "y_vs_x" strings, values = dict with
            'tracking_quality', 'hedge_ratios', 'depeg_events'.
        """
        cfg = config or {}
        delta = cfg.get("kalman_delta", self.delta)
        obs_noise = cfg.get("kalman_obs_noise", self.obs_noise_init)

        results: Dict[str, Dict] = {}

        for y_name, x_name in pairs:
            pair_key = f"{y_name}_vs_{x_name}"

            if y_name not in prices_df.columns:
                logger.warning("Asset %s not found in prices_df — skipping pair", y_name)
                continue
            if x_name not in prices_df.columns:
                logger.warning("Asset %s not found in prices_df — skipping pair", x_name)
                continue

            y = prices_df[y_name].dropna()
            x = prices_df[x_name].dropna()

            # Align indices
            common_idx = y.index.intersection(x.index)
            if len(common_idx) < 10:
                logger.warning(
                    "Pair %s: only %d common observations — skipping",
                    pair_key, len(common_idx),
                )
                continue

            y_aligned = y.loc[common_idx]
            x_aligned = x.loc[common_idx]

            # Create a fresh tracker for each pair
            tracker = KalmanPairTracker(delta=delta, obs_noise_init=obs_noise)
            try:
                tracker.fit(y_aligned, x_aligned)
            except Exception as e:
                logger.error("Pair %s: fit failed — %s", pair_key, e)
                continue

            results[pair_key] = {
                "tracking_quality": tracker.get_tracking_quality(),
                "hedge_ratios": tracker.get_hedge_ratios(),
                "depeg_events": tracker.detect_depegs(),
            }

            tq = results[pair_key]["tracking_quality"]
            logger.info(
                "Pair %s: beta=%.4f  R2=%.4f  TE=%.4f  depegs=%d",
                pair_key, tq["mean_beta"], tq["r_squared"],
                tq["tracking_error"], len(results[pair_key]["depeg_events"]),
            )

        return results

    # ─── HELPERS ──────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """Raise if fit() has not been called."""
        if not self._fitted:
            raise RuntimeError(
                "KalmanPairTracker has not been fitted. Call fit(y, x) first."
            )
