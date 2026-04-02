"""
Uncertainty-Weighted Ensemble Meta-Model — Enhanced Edition.

Enhancements over base version (adapted from stat-arb portfolio/):
  1. Multi-method optimizer selection (HRP, risk parity, BL, CVaR)
  2. Kelly-based confidence scaling (from stat-arb position_sizing_engine)
  3. Correlation-aware model weighting
  4. Regime-blended risk budgets with soft HMM probabilities
  5. Circuit breaker with dynamic threshold

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
import numpy as np
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

    def __init__(self, config=None, asset_names=None):
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
        self.asset_buckets = cfg.get("asset_buckets", DEFAULT_BUCKETS)

        # Multi-method optimiser (from stat-arb portfolio/)
        self.optimizer = MultiMethodOptimizer(
            min_weight=self.min_single,
            max_weight=self.max_single,
            max_turnover=self.max_turnover,
        )

        # Kelly confidence parameters (from stat-arb position_sizing_engine)
        self.kelly_fraction = 0.25  # Quarter Kelly (conservative)
        self.kelly_min_confidence = 0.3  # Below this → blend toward 1/N

        # State
        self.high_water_mark = 1.0
        self.circuit_breaker_active = False

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
        """
        # ═══ 1. HWM UPDATE ═══
        if current_nav > self.high_water_mark:
            self.high_water_mark = current_nav

        # ═══ 2. CIRCUIT BREAKER ═══
        drawdown = 1.0 - current_nav / self.high_water_mark if self.high_water_mark > 0 else 0

        if drawdown >= self.cb_threshold and not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            logger.warning(f"CIRCUIT BREAKER TRIGGERED: dd={drawdown:.2%}")

        if self.circuit_breaker_active:
            if drawdown < self.cb_recovery:
                self.circuit_breaker_active = False
                logger.info(f"Circuit breaker RESET: dd={drawdown:.2%}")
            else:
                defensive = np.array([self.defensive_weights.get(a, 0) for a in self.asset_names])
                defensive = defensive / defensive.sum()
                return self._build_result(defensive, "crisis (CB)", hmm_regime_probs,
                                          True, {"defensive": 1.0}, drawdown)

        # ═══ 3. REGIME-BLENDED RISK BUDGET ═══
        p_bull, p_normal, p_crisis = hmm_regime_probs
        dominant = ["bull", "normal", "crisis"][np.argmax(hmm_regime_probs)]

        blended = {}
        for bucket in self.asset_buckets:
            blended[bucket] = (
                p_bull * self.risk_budgets["bull"].get(bucket, 0) +
                p_normal * self.risk_budgets["normal"].get(bucket, 0) +
                p_crisis * self.risk_budgets["crisis"].get(bucket, 0)
            )

        budget_weights = self._budget_to_weights(blended)

        # ═══ 4. MULTI-METHOD GARCH WEIGHTS ═══
        corr = covariance_matrix / np.outer(
            np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12)),
            np.sqrt(np.maximum(np.diag(covariance_matrix), 1e-12))
        )

        if optimization_method == "hrp":
            opt_weights = self.optimizer.hrp(covariance_matrix, corr)
        elif optimization_method == "black_litterman":
            # HMM regime views as BL views
            regime_views = self._compute_regime_views(hmm_regime_probs, covariance_matrix)
            opt_weights = self.optimizer.black_litterman(
                covariance_matrix, market_weights=budget_weights,
                regime_views=regime_views
            )
        elif optimization_method == "risk_parity":
            opt_weights = self.optimizer.risk_parity(covariance_matrix)
        elif optimization_method == "max_diversification":
            opt_weights = self.optimizer.max_diversification(covariance_matrix)
        elif optimization_method == "inverse_vol":
            opt_weights = self.optimizer.inverse_volatility(covariance_matrix)
        else:
            opt_weights = garch_rp_weights

        # ═══ 5. INVERSE-VARIANCE COMBINATION ═══
        uncertainties = np.array([
            max(garch_uncertainty, 1e-6),
            max(hmm_uncertainty, 1e-6),
            max(rl_uncertainty, 1e-6),
        ])
        inv_var = 1.0 / (uncertainties ** 2)
        model_w = inv_var / inv_var.sum()

        combined = (model_w[0] * opt_weights +
                    model_w[1] * budget_weights +
                    model_w[2] * rl_weights)
        combined = np.clip(combined, 0, 1)
        combined = combined / combined.sum()

        # ═══ 6. KELLY CONFIDENCE SCALING ═══
        # From stat-arb position_sizing_engine: when confidence is low,
        # blend toward 1/N equal weight to reduce active risk
        avg_uncertainty = uncertainties.mean()
        confidence = 1.0 - min(avg_uncertainty, 1.0)

        if confidence < self.kelly_min_confidence:
            # Very uncertain: mostly equal weight
            kelly_scale = self.kelly_fraction * confidence / self.kelly_min_confidence
        else:
            kelly_scale = self.kelly_fraction + (1 - self.kelly_fraction) * (
                (confidence - self.kelly_min_confidence) / (1 - self.kelly_min_confidence)
            )

        equal_weight = np.ones(self.n_assets) / self.n_assets
        combined = kelly_scale * combined + (1 - kelly_scale) * equal_weight
        combined = combined / combined.sum()

        # ═══ 7. CVaR-CONSTRAINED OPTIMISATION ═══
        final = self.optimizer.cvar_constrained(
            combined, covariance_matrix, current_weights
        )

        contributions = {"garch": float(model_w[0]), "hmm": float(model_w[1]),
                         "rl": float(model_w[2])}

        return self._build_result(final, dominant, hmm_regime_probs,
                                  False, contributions, drawdown,
                                  kelly_scale=kelly_scale, confidence=confidence,
                                  opt_method=optimization_method)

    def _budget_to_weights(self, blended_budget):
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

    def _compute_regime_views(self, regime_probs, cov):
        """Convert HMM regime probs to BL-style expected return views."""
        n = self.n_assets
        views = np.zeros(n)
        p_bull, p_normal, p_crisis = regime_probs

        # Bull: positive views on crypto, negative on Treasuries
        bull_views = np.array([0.003, 0.003, 0.004, 0.002, 0.002, -0.001, -0.001, 0.0])
        normal_views = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0])
        crisis_views = np.array([-0.005, -0.004, -0.006, -0.003, -0.003, 0.001, 0.001, 0.0])

        views = p_bull * bull_views[:n] + p_normal * normal_views[:n] + p_crisis * crisis_views[:n]
        return views

    def _build_result(self, weights, regime, probs, cb, contributions, dd, **kwargs):
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
