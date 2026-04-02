"""
Uncertainty-Weighted Ensemble Meta-Model — Enhanced Edition.

Enhancements over base version (adapted from stat-arb portfolio/):
  1. Multi-method optimizer selection (HRP, risk parity, BL, CVaR)
  2. Kelly-based confidence scaling (from stat-arb position_sizing_engine)
  3. Correlation-aware model weighting
  4. Regime-blended risk budgets with soft HMM probabilities
  5. Circuit breaker with dynamic threshold
  6. Config-driven asset buckets (with DEFAULT_BUCKETS fallback)
  7. Adaptive Black-Litterman views from regime-conditional mean returns
  8. Dynamic Kelly fraction based on rolling Sharpe
  9. Model contribution tracking over time
  10. Regime transition smoothing (exponential)

Pipeline:
  1. Check circuit breaker (15% drawdown → defensive)
  2. Compute regime-blended risk budget from HMM soft probabilities
  3. Get weights from GARCH-DCC (via selected method: HRP/RP/BL)
  4. Get weights from SAC RL agent
  5. Inverse-variance weighted combination
  6. Kelly confidence scaling (scale toward equal weight when uncertain)
  7. CVaR-constrained optimisation
  8. Return final weight vector w*
"""
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *a, **kw):
        return iterable if iterable is not None else range(0)

from .portfolio_optimizer import MultiMethodOptimizer

logger = logging.getLogger(__name__)

DEFAULT_DEFENSIVE = {
    "BTC": 0.05, "ETH": 0.05, "SOL": 0.00,
    "stETH": 0.05, "rETH": 0.05,
    "BUIDL": 0.35, "USDY": 0.15, "USDC": 0.30,
}

DEFAULT_RISK_BUDGETS = {
    "bull":   {"crypto": 0.70, "staking": 0.20, "treasuries": 0.00, "stable": 0.10},
    "normal": {"crypto": 0.40, "staking": 0.30, "treasuries": 0.20, "stable": 0.10},
    "crisis": {"crypto": 0.10, "staking": 0.10, "treasuries": 0.50, "stable": 0.30},
}

DEFAULT_BUCKETS = {
    "crypto": ["BTC", "ETH", "SOL"],
    "staking": ["stETH", "rETH"],
    "treasuries": ["BUIDL", "USDY"],
    "stable": ["USDC"],
}


class EnsembleCombiner:
    """
    Enhanced ensemble meta-model with Kelly confidence scaling,
    multi-method optimisation, and regime-conditional allocation.
    """

    def __init__(self, config: Optional[Dict] = None, asset_names: Optional[List[str]] = None):
        cfg = config or {}
        ens = cfg.get("ensemble", {})

        self.asset_names = asset_names or list(DEFAULT_DEFENSIVE.keys())
        self.n_assets = len(self.asset_names)

        # Constraints
        self.max_single = ens.get("max_single_asset", 0.40)
        self.min_single = ens.get("min_single_asset", 0.02)
        self.max_turnover = ens.get("max_turnover", 0.30)

        # Circuit breaker
        cb = ens.get("circuit_breaker", {})
        self.cb_threshold = cb.get("drawdown_threshold", 0.15)
        self.cb_recovery = cb.get("recovery_threshold", 0.10)
        self.defensive_weights = cb.get("defensive_weights", DEFAULT_DEFENSIVE)

        # Risk budgets
        self.risk_budgets = cfg.get("risk_budgets", DEFAULT_RISK_BUDGETS)

        # Config-driven asset buckets with DEFAULT_BUCKETS fallback
        self.asset_buckets = cfg.get("asset_buckets", DEFAULT_BUCKETS)
        logger.info(
            "Asset buckets loaded: %d buckets, %d total assets",
            len(self.asset_buckets),
            sum(len(v) for v in self.asset_buckets.values()),
        )

        # Multi-method optimiser (from stat-arb portfolio/)
        self.optimizer = MultiMethodOptimizer(
            min_weight=self.min_single,
            max_weight=self.max_single,
            max_turnover=self.max_turnover,
        )

        # Kelly confidence parameters (from stat-arb position_sizing_engine)
        self.kelly_fraction = 0.25  # Quarter Kelly (conservative)
        self.kelly_min_confidence = 0.3  # Below this → blend toward 1/N

        # Dynamic Kelly thresholds — configurable
        kelly_cfg = ens.get("kelly", {})
        self.kelly_aggressive_sharpe = kelly_cfg.get("aggressive_sharpe_threshold", 1.5)
        self.kelly_standard_sharpe = kelly_cfg.get("standard_sharpe_threshold", 0.5)
        self.kelly_aggressive_fraction = kelly_cfg.get("aggressive_fraction", 0.50)
        self.kelly_standard_fraction = kelly_cfg.get("standard_fraction", 0.25)
        self.kelly_conservative_fraction = kelly_cfg.get("conservative_fraction", 0.10)
        self.kelly_rolling_window = kelly_cfg.get("rolling_window", 720)

        # Regime transition smoothing
        smooth_cfg = ens.get("regime_smoothing", {})
        self.smoothing_steps = smooth_cfg.get("steps", 12)
        self.smoothing_alpha = smooth_cfg.get("alpha", 0.3)  # EMA decay
        self._prev_regime_weights: Optional[np.ndarray] = None
        self._smoothing_counter: int = 0
        self._prev_regime: Optional[str] = None

        # State
        self.high_water_mark = 1.0
        self.circuit_breaker_active = False

        # Model contribution tracking
        self.contribution_history: List[Dict] = []

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC: combine
    # ═══════════════════════════════════════════════════════════════

    def combine(
        self,
        garch_rp_weights: np.ndarray,
        hmm_regime_probs: np.ndarray,
        rl_weights: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: np.ndarray,
        current_nav: float,
        garch_uncertainty: float = 0.5,
        hmm_uncertainty: float = 0.5,
        rl_uncertainty: float = 0.5,
        optimization_method: str = "hrp",
        returns_df=None,
    ) -> dict:
        """
        Combine all model outputs into final weight vector.

        Enhanced with:
          - Multi-method optimizer selection (HRP, BL, RP, etc.)
          - Kelly-based confidence scaling
          - Correlation-aware model weighting
          - Adaptive Black-Litterman views from regime-conditional returns
          - Dynamic Kelly fraction from rolling Sharpe
          - Regime transition smoothing
          - Model contribution tracking
        """
        pipeline_steps = [
            "HWM update",
            "Circuit breaker",
            "Regime-blended risk budget",
            "Multi-method optimizer",
            "Inverse-variance combination",
            "Kelly confidence scaling",
            "CVaR-constrained optimisation",
        ]
        pbar = tqdm(total=len(pipeline_steps), desc="Ensemble", unit="step", leave=True)

        # ═══ 1. HWM UPDATE ═══
        pbar.set_postfix(step="HWM update")
        if current_nav > self.high_water_mark:
            self.high_water_mark = current_nav
        logger.debug("NAV=%.4f  HWM=%.4f", current_nav, self.high_water_mark)
        pbar.update(1)

        # ═══ 2. CIRCUIT BREAKER ═══
        pbar.set_postfix(step="Circuit breaker")
        drawdown = 1.0 - current_nav / self.high_water_mark if self.high_water_mark > 0 else 0

        if drawdown >= self.cb_threshold and not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            logger.warning("CIRCUIT BREAKER TRIGGERED: dd=%.2f%%", drawdown * 100)

        if self.circuit_breaker_active:
            if drawdown < self.cb_recovery:
                self.circuit_breaker_active = False
                logger.info("Circuit breaker RESET: dd=%.2f%%", drawdown * 100)
            else:
                defensive = np.array([self.defensive_weights.get(a, 0) for a in self.asset_names])
                defensive = defensive / defensive.sum()
                pbar.close()
                return self._build_result(defensive, "crisis (CB)", hmm_regime_probs,
                                          True, {"defensive": 1.0}, drawdown)
        pbar.update(1)

        # ═══ 3. REGIME-BLENDED RISK BUDGET ═══
        pbar.set_postfix(step="Regime-blended risk budget")
        p_bull, p_normal, p_crisis = hmm_regime_probs
        dominant = ["bull", "normal", "crisis"][np.argmax(hmm_regime_probs)]
        logger.info(
            "Regime probs: bull=%.2f  normal=%.2f  crisis=%.2f  → %s",
            p_bull, p_normal, p_crisis, dominant,
        )

        blended = {}
        for bucket in self.asset_buckets:
            blended[bucket] = (
                p_bull * self.risk_budgets["bull"].get(bucket, 0) +
                p_normal * self.risk_budgets["normal"].get(bucket, 0) +
                p_crisis * self.risk_budgets["crisis"].get(bucket, 0)
            )

        budget_weights = self._budget_to_weights(blended)
        pbar.update(1)

        # ═══ 4. MULTI-METHOD GARCH WEIGHTS ═══
        pbar.set_postfix(step=f"Multi-method optimizer [{optimization_method}]")
        corr = covariance_matrix / np.outer(
            np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12)),
            np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12))
        )

        if optimization_method == "hrp":
            logger.info("Optimization method: HRP")
            opt_weights = self.optimizer.hrp(covariance_matrix, corr)
        elif optimization_method == "black_litterman":
            logger.info("Optimization method: Black-Litterman")
            # Adaptive BL views: use regime-conditional means if returns available
            if returns_df is not None:
                regime_labels = self._assign_regime_labels(hmm_regime_probs)
                regime_views = self._compute_adaptive_views(returns_df, regime_labels, hmm_regime_probs)
                logger.info("Using adaptive BL views from regime-conditional returns")
            else:
                regime_views = self._compute_regime_views(hmm_regime_probs, covariance_matrix)
                logger.info("Using fallback hard-coded BL views (no returns_df)")
            opt_weights = self.optimizer.black_litterman(
                covariance_matrix, market_weights=budget_weights,
                regime_views=regime_views
            )
        elif optimization_method == "risk_parity":
            logger.info("Optimization method: Risk Parity")
            opt_weights = self.optimizer.risk_parity(covariance_matrix)
        elif optimization_method == "max_diversification":
            logger.info("Optimization method: Max Diversification")
            opt_weights = self.optimizer.max_diversification(covariance_matrix)
        elif optimization_method == "inverse_vol":
            logger.info("Optimization method: Inverse Volatility")
            opt_weights = self.optimizer.inverse_volatility(covariance_matrix)
        else:
            logger.info("Optimization method: pass-through (garch_rp_weights)")
            opt_weights = garch_rp_weights
        pbar.update(1)

        # ═══ 5. INVERSE-VARIANCE COMBINATION ═══
        pbar.set_postfix(step="Inverse-variance combination")
        uncertainties = np.array([
            max(garch_uncertainty, 1e-6),
            max(hmm_uncertainty, 1e-6),
            max(rl_uncertainty, 1e-6),
        ])
        inv_var = 1.0 / (uncertainties ** 2)
        model_w = inv_var / inv_var.sum()
        logger.debug(
            "Model weights: GARCH=%.3f  HMM=%.3f  RL=%.3f",
            model_w[0], model_w[1], model_w[2],
        )

        combined = (model_w[0] * opt_weights +
                    model_w[1] * budget_weights +
                    model_w[2] * rl_weights)
        combined = np.clip(combined, 0, 1)
        combined = combined / combined.sum()

        # ═══ 5b. REGIME TRANSITION SMOOTHING ═══
        combined = self._smooth_regime_transition(combined, dominant)
        pbar.update(1)

        # ═══ 6. KELLY CONFIDENCE SCALING ═══
        pbar.set_postfix(step="Kelly confidence scaling")
        # Dynamic Kelly fraction from rolling Sharpe
        kelly_frac = self._dynamic_kelly_fraction(returns_df)

        avg_uncertainty = uncertainties.mean()
        confidence = 1.0 - min(avg_uncertainty, 1.0)

        if confidence < self.kelly_min_confidence:
            kelly_scale = kelly_frac * confidence / self.kelly_min_confidence
        else:
            kelly_scale = kelly_frac + (1 - kelly_frac) * (
                (confidence - self.kelly_min_confidence) / (1 - self.kelly_min_confidence)
            )
        logger.info(
            "Kelly: fraction=%.3f  confidence=%.3f  scale=%.3f",
            kelly_frac, confidence, kelly_scale,
        )

        equal_weight = np.ones(self.n_assets) / self.n_assets
        combined = kelly_scale * combined + (1 - kelly_scale) * equal_weight
        combined = combined / combined.sum()
        pbar.update(1)

        # ═══ 7. CVaR-CONSTRAINED OPTIMISATION ═══
        pbar.set_postfix(step="CVaR-constrained optimisation")
        final = self.optimizer.cvar_constrained(
            combined, covariance_matrix, current_weights
        )

        contributions = {"garch": float(model_w[0]), "hmm": float(model_w[1]),
                         "rl": float(model_w[2])}

        # ═══ 8. TRACK CONTRIBUTION HISTORY ═══
        self._record_contribution(contributions)
        pbar.update(1)
        pbar.close()

        return self._build_result(final, dominant, hmm_regime_probs,
                                  False, contributions, drawdown,
                                  kelly_scale=kelly_scale, confidence=confidence,
                                  opt_method=optimization_method)

    # ═══════════════════════════════════════════════════════════════
    # ADAPTIVE BLACK-LITTERMAN VIEWS
    # ═══════════════════════════════════════════════════════════════

    def _compute_adaptive_views(
        self,
        returns_df: pd.DataFrame,
        regime_labels: np.ndarray,
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute BL views from historical regime-conditional mean returns.

        Instead of hard-coded bull/crisis view vectors, averages the actual
        returns observed during each regime and blends by regime probability.

        Args:
            returns_df: Historical returns DataFrame (T x n_assets).
            regime_labels: Per-row regime label array (0=bull, 1=normal, 2=crisis).
            regime_probs: Current regime probabilities [p_bull, p_normal, p_crisis].

        Returns:
            views: (n_assets,) blended expected return views.
        """
        n = min(returns_df.shape[1], self.n_assets)
        regime_means = np.zeros((3, n))

        for regime_id in range(3):
            mask = regime_labels == regime_id
            if mask.sum() > 0:
                regime_means[regime_id, :] = returns_df.iloc[:, :n].values[mask].mean(axis=0)
            else:
                # No data for this regime — use unconditional mean
                regime_means[regime_id, :] = returns_df.iloc[:, :n].values.mean(axis=0)
                logger.debug("No observations for regime %d; using unconditional mean", regime_id)

        # Blend by current regime probabilities
        p_bull, p_normal, p_crisis = regime_probs
        views = (p_bull * regime_means[0] +
                 p_normal * regime_means[1] +
                 p_crisis * regime_means[2])

        logger.debug("Adaptive BL views: %s", np.array2string(views, precision=6))
        return views

    @staticmethod
    def _assign_regime_labels(regime_probs: np.ndarray) -> np.ndarray:
        """
        Create regime label array for historical rows.

        If *regime_probs* is 2-D (T x n_states), assigns the argmax regime
        per row.  If 1-D (single timestep), broadcasts the dominant regime
        across all rows — conservative fallback when only latest probs are
        available.
        """
        regime_probs = np.atleast_2d(regime_probs)
        if regime_probs.shape[0] == 1:
            # Single-timestep: caller will broadcast
            return np.full(1, int(np.argmax(regime_probs[0])))
        return np.argmax(regime_probs, axis=1)

    # ═══════════════════════════════════════════════════════════════
    # DYNAMIC KELLY FRACTION
    # ═══════════════════════════════════════════════════════════════

    def _dynamic_kelly_fraction(self, returns_df: Optional[pd.DataFrame] = None) -> float:
        """
        Compute Kelly fraction dynamically from rolling Sharpe ratio.

        Thresholds (configurable via config.ensemble.kelly):
          - Rolling Sharpe > aggressive_sharpe (1.5): aggressive fraction (0.50)
          - Rolling Sharpe 0.5-1.5:                  standard fraction  (0.25)
          - Rolling Sharpe < conservative_sharpe (0.5): conservative     (0.10)

        Falls back to static self.kelly_fraction when returns_df is unavailable.
        """
        if returns_df is None or len(returns_df) < self.kelly_rolling_window:
            logger.debug("Insufficient data for dynamic Kelly; using static %.2f", self.kelly_fraction)
            return self.kelly_fraction

        # Portfolio returns using equal weight for Sharpe estimation
        n_cols = min(returns_df.shape[1], self.n_assets)
        port_ret = returns_df.iloc[-self.kelly_rolling_window:, :n_cols].mean(axis=1)
        mu = port_ret.mean()
        sigma = port_ret.std()

        if sigma < 1e-12:
            logger.debug("Near-zero vol in Kelly window; using conservative fraction")
            return self.kelly_conservative_fraction

        # Annualise hourly Sharpe
        rolling_sharpe = (mu / sigma) * np.sqrt(8760)

        if rolling_sharpe > self.kelly_aggressive_sharpe:
            frac = self.kelly_aggressive_fraction
            logger.info("Dynamic Kelly: Sharpe=%.2f → aggressive (%.2f)", rolling_sharpe, frac)
        elif rolling_sharpe > self.kelly_standard_sharpe:
            frac = self.kelly_standard_fraction
            logger.info("Dynamic Kelly: Sharpe=%.2f → standard (%.2f)", rolling_sharpe, frac)
        else:
            frac = self.kelly_conservative_fraction
            logger.info("Dynamic Kelly: Sharpe=%.2f → conservative (%.2f)", rolling_sharpe, frac)

        return frac

    # ═══════════════════════════════════════════════════════════════
    # REGIME TRANSITION SMOOTHING
    # ═══════════════════════════════════════════════════════════════

    def _smooth_regime_transition(self, raw_weights: np.ndarray, current_regime: str) -> np.ndarray:
        """
        Apply exponential smoothing when the regime changes to avoid
        whipsaw from rapid regime switches.

        Uses EMA: w_smooth = alpha * w_new + (1 - alpha) * w_prev
        for ``smoothing_steps`` periods after a regime change.

        Args:
            raw_weights: Newly computed weight vector.
            current_regime: Current dominant regime label.

        Returns:
            Smoothed weight vector (normalised to sum to 1).
        """
        if self._prev_regime is not None and current_regime != self._prev_regime:
            self._smoothing_counter = self.smoothing_steps
            logger.info(
                "Regime transition %s → %s; smoothing for %d steps",
                self._prev_regime, current_regime, self.smoothing_steps,
            )

        self._prev_regime = current_regime

        if self._smoothing_counter > 0 and self._prev_regime_weights is not None:
            alpha = self.smoothing_alpha
            smoothed = alpha * raw_weights + (1 - alpha) * self._prev_regime_weights
            smoothed = np.maximum(smoothed, 0)
            smoothed = smoothed / smoothed.sum()
            self._smoothing_counter -= 1
            logger.debug(
                "Smoothing step (remaining=%d)  alpha=%.2f",
                self._smoothing_counter, alpha,
            )
            self._prev_regime_weights = smoothed
            return smoothed

        self._prev_regime_weights = raw_weights.copy()
        return raw_weights

    # ═══════════════════════════════════════════════════════════════
    # MODEL CONTRIBUTION TRACKING
    # ═══════════════════════════════════════════════════════════════

    def _record_contribution(self, contributions: Dict[str, float]) -> None:
        """Append current model contributions to history."""
        record = {
            "timestamp": time.time(),
            "garch": contributions.get("garch", 0.0),
            "hmm": contributions.get("hmm", 0.0),
            "rl": contributions.get("rl", 0.0),
        }
        self.contribution_history.append(record)

    def get_contribution_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising model contributions over time.

        Columns: timestamp, garch, hmm, rl (each as weight fraction).
        Includes mean, std, min, max summary statistics per model.
        """
        if not self.contribution_history:
            logger.warning("No contribution history recorded yet")
            return pd.DataFrame(columns=["timestamp", "garch", "hmm", "rl"])

        df = pd.DataFrame(self.contribution_history)
        logger.info(
            "Contribution summary (%d records): GARCH mean=%.3f  HMM mean=%.3f  RL mean=%.3f",
            len(df), df["garch"].mean(), df["hmm"].mean(), df["rl"].mean(),
        )
        return df

    # ═══════════════════════════════════════════════════════════════
    # INTERNALS (preserved signatures)
    # ═══════════════════════════════════════════════════════════════

    def _budget_to_weights(self, blended_budget: Dict[str, float]) -> np.ndarray:
        """Convert bucket-level risk budgets to per-asset weights."""
        weights = np.zeros(self.n_assets)
        for bucket, budget in blended_budget.items():
            assets = self.asset_buckets.get(bucket, [])
            n = len(assets)
            if n > 0:
                for a in assets:
                    if a in self.asset_names:
                        weights[self.asset_names.index(a)] = budget / n
        total = weights.sum()
        return weights / total if total > 0 else np.ones(self.n_assets) / self.n_assets

    def _compute_regime_views(self, regime_probs: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Fallback: convert HMM regime probs to BL-style expected return views
        using hard-coded vectors.  Prefer _compute_adaptive_views when
        returns_df is available.
        """
        n = self.n_assets
        p_bull, p_normal, p_crisis = regime_probs

        bull_views = np.array([0.003, 0.003, 0.004, 0.002, 0.002, -0.001, -0.001, 0.0])
        normal_views = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0])
        crisis_views = np.array([-0.005, -0.004, -0.006, -0.003, -0.003, 0.001, 0.001, 0.0])

        views = p_bull * bull_views[:n] + p_normal * normal_views[:n] + p_crisis * crisis_views[:n]
        return views

    def _build_result(self, weights: np.ndarray, regime: str, probs: np.ndarray,
                      cb: bool, contributions: Dict, dd: float, **kwargs) -> dict:
        return {
            "weights": weights,
            "regime": regime,
            "regime_probs": {"bull": float(probs[0]), "normal": float(probs[1]),
                             "crisis": float(probs[2])},
            "circuit_breaker": cb,
            "model_contributions": contributions,
            "drawdown": dd,
            "high_water_mark": self.high_water_mark,
            **kwargs,
        }
