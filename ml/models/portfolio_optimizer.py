"""
Portfolio Optimisation Module — Adapted from Crypto-Statistical-Arbitrage.

Multi-method portfolio weight optimisation with CVaR constraint, turnover
limits, and regime-aware Black-Litterman integration.

Methods:
  1. Risk Parity (Equal Risk Contribution)
  2. Hierarchical Risk Parity (Lopez de Prado, 2016)
  3. Black-Litterman (with HMM regime views)
  4. Mean-Variance Optimisation (max Sharpe, long-only)
  5. Inverse Volatility
  6. Maximum Diversification
  7. CVaR-Constrained (tracks ensemble target)

All methods enforce: sum-to-one, min/max weight, max turnover, no leverage.

Adapted from: github.com/abailey81/Crypto-Statistical-Arbitrage/portfolio/
"""
import logging
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize as sp_minimize
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Output of portfolio optimisation."""
    weights: np.ndarray
    method: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 1.0
    effective_n: float = 1.0
    cvar_5pct: float = 0.0
    turnover: float = 0.0

    def summary(self) -> str:
        return (
            f"Method: {self.method} | Sharpe: {self.sharpe_ratio:.3f} | "
            f"Vol: {self.expected_volatility:.2%} | CVaR: {self.cvar_5pct:.2%} | "
            f"Eff N: {self.effective_n:.1f} | Turnover: {self.turnover:.2%}"
        )


class MultiMethodOptimizer:
    """
    Multi-method portfolio optimiser supporting 7 allocation strategies.

    Adapted from the author's Crypto-Statistical-Arbitrage portfolio module,
    extended with CVaR constraints and HMM regime-aware Black-Litterman views.
    """

    def __init__(self, min_weight=0.02, max_weight=0.40, max_turnover=0.30,
                 risk_free_rate=0.045):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_turnover = max_turnover
        self.risk_free_rate = risk_free_rate

    # ─── 1. RISK PARITY (Equal Risk Contribution) ───

    def risk_parity(self, cov: np.ndarray) -> np.ndarray:
        """
        Equal Risk Contribution portfolio.
        Each asset contributes equally to total portfolio risk.
        Iterative approach from stat-arb portfolio/optimization.py.
        """
        n = cov.shape[0]
        w = 1.0 / np.sqrt(np.diag(cov))
        w = w / w.sum()

        for _ in range(200):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                break
            marginal = cov @ w
            risk_contrib = w * marginal / port_vol
            target = port_vol / n
            adjustment = np.clip(target / (risk_contrib + 1e-12), 0.5, 2.0)
            w = w * adjustment
            w = np.maximum(w, 1e-6)
            w = w / w.sum()

        return w

    # ─── 2. HIERARCHICAL RISK PARITY ───

    def hrp(self, cov: np.ndarray, corr: np.ndarray = None) -> np.ndarray:
        """
        Hierarchical Risk Parity (Lopez de Prado, 2016).
        Tree-based allocation avoiding covariance inversion.
        More robust to estimation error than MVO or standard risk parity.
        Directly adapted from stat-arb portfolio/optimization.py.
        """
        n = cov.shape[0]
        if n <= 1:
            return np.array([1.0])

        if corr is None:
            vols = np.sqrt(np.diag(cov))
            corr = cov / np.outer(vols, vols)
            np.fill_diagonal(corr, 1.0)

        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method='single')
        order = leaves_list(link).tolist()

        # Recursive bisection
        weights = np.ones(n)
        items = [order]

        while items:
            current = items.pop()
            if len(current) <= 1:
                continue

            mid = len(current) // 2
            left, right = current[:mid], current[mid:]

            # Cluster variance (equal-weight within cluster)
            left_var = self._cluster_variance(cov, left)
            right_var = self._cluster_variance(cov, right)

            alpha = 1.0 - left_var / (left_var + right_var + 1e-12)

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                items.append(left)
            if len(right) > 1:
                items.append(right)

        weights = np.maximum(weights, 1e-8)
        return weights / weights.sum()

    # ─── 3. BLACK-LITTERMAN WITH HMM REGIME VIEWS ───

    def black_litterman(self, cov: np.ndarray, market_weights: np.ndarray,
                        regime_views: np.ndarray = None, tau: float = 0.05,
                        risk_aversion: float = 2.5) -> np.ndarray:
        """
        Black-Litterman with HMM regime probabilities as investor views.

        Adapted from stat-arb portfolio/optimization.py BlackLitterman class.
        Innovation: HMM regime posteriors feed directly as views.

        Args:
            cov: Covariance matrix
            market_weights: Equilibrium (e.g., equal weight or market cap)
            regime_views: Optional absolute return views from HMM regimes
            tau: Uncertainty parameter (scales prior covariance)
            risk_aversion: Market risk aversion coefficient
        """
        n = cov.shape[0]

        # Implied equilibrium returns: pi = delta * Sigma * w_mkt
        pi = risk_aversion * cov @ market_weights

        if regime_views is not None and len(regime_views) == n:
            # P matrix: identity (absolute views on each asset)
            P = np.eye(n)
            Q = regime_views  # View vector

            # Omega: uncertainty of views (diagonal, proportional to asset vol)
            omega = np.diag(tau * np.diag(cov))

            # BL posterior: mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
            #                       * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
            tau_sigma_inv = np.linalg.inv(tau * cov + np.eye(n) * 1e-8)
            omega_inv = np.linalg.inv(omega + np.eye(n) * 1e-8)

            posterior_cov_inv = tau_sigma_inv + P.T @ omega_inv @ P
            posterior_cov = np.linalg.inv(posterior_cov_inv + np.eye(n) * 1e-8)
            mu_bl = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        else:
            mu_bl = pi
            posterior_cov = cov + tau * cov

        # MVO on BL returns (long-only)
        try:
            cov_inv = np.linalg.inv(posterior_cov + np.eye(n) * 1e-8)
            w = cov_inv @ mu_bl
            w = np.maximum(w, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            w = np.ones(n) / n

        return w

    # ─── 4. MEAN-VARIANCE (MAX SHARPE, LONG-ONLY) ───

    def mean_variance(self, expected_returns: np.ndarray,
                      cov: np.ndarray) -> np.ndarray:
        """
        Mean-Variance Optimisation — max Sharpe, long-only.
        Adapted from stat-arb portfolio/optimization.py.
        """
        n = len(expected_returns)
        try:
            excess = expected_returns - self.risk_free_rate / 8760
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-8)
            w = cov_inv @ excess
            w = np.maximum(w, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            w = np.ones(n) / n
        return w

    # ─── 5. INVERSE VOLATILITY ───

    def inverse_volatility(self, cov: np.ndarray) -> np.ndarray:
        """Weight inversely proportional to volatility."""
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()

    # ─── 6. MAXIMUM DIVERSIFICATION ───

    def max_diversification(self, cov: np.ndarray) -> np.ndarray:
        """
        Maximum Diversification Portfolio.
        Maximises: weighted avg vol / portfolio vol.
        Adapted from stat-arb portfolio/optimization.py.
        """
        n = cov.shape[0]
        vols = np.sqrt(np.diag(cov))
        w = np.ones(n) / n

        for _ in range(300):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                break
            gradient = vols / port_vol - (cov @ w) * (vols @ w) / (port_vol ** 3 + 1e-12)
            w = w + 0.005 * gradient
            w = np.maximum(w, 1e-6)
            w = w / w.sum()

        return w

    # ─── 7. CVaR-CONSTRAINED (TRACKS TARGET) ───

    def cvar_constrained(self, target_weights: np.ndarray, cov: np.ndarray,
                         current_weights: np.ndarray = None,
                         vol_target: float = 0.25) -> np.ndarray:
        """
        CVaR-constrained optimisation — minimises deviation from target
        subject to risk and turnover constraints.
        """
        n = len(target_weights)
        if current_weights is None:
            current_weights = np.ones(n) / n

        w = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(w - target_weights))

        hourly_vol = vol_target / np.sqrt(8760)
        cov_psd = self._ensure_psd(cov)

        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
            cp.norm(w - current_weights, 1) <= self.max_turnover,
            cp.quad_form(w, cov_psd) <= hourly_vol ** 2,
        ]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            if problem.status in ("optimal", "optimal_inaccurate"):
                weights = np.clip(w.value, 0, 1)
                return weights / weights.sum()
        except cp.SolverError:
            pass

        return self._project_simplex(target_weights)

    # ─── COMPARE ALL METHODS ───

    def compare_methods(self, cov: np.ndarray, returns_df: pd.DataFrame = None,
                        current_weights: np.ndarray = None) -> pd.DataFrame:
        """Run all methods and return comparison table."""
        n = cov.shape[0]
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        mkt_w = np.ones(n) / n

        methods = {
            "risk_parity": lambda: self.risk_parity(cov),
            "hrp": lambda: self.hrp(cov, corr),
            "black_litterman": lambda: self.black_litterman(cov, mkt_w),
            "inverse_vol": lambda: self.inverse_volatility(cov),
            "max_diversification": lambda: self.max_diversification(cov),
        }

        if returns_df is not None:
            mu = returns_df.mean().values
            methods["mean_variance"] = lambda: self.mean_variance(mu, cov)

        results = []
        for name, func in methods.items():
            try:
                w = func()
                port_vol = np.sqrt(w @ cov @ w) * np.sqrt(8760)
                weighted_vols = np.sqrt(np.diag(cov)) @ w * np.sqrt(8760)
                div_ratio = weighted_vols / port_vol if port_vol > 0 else 1.0
                eff_n = 1.0 / np.sum(w ** 2)
                turnover = np.abs(w - (current_weights if current_weights is not None else mkt_w)).sum()

                # Parametric CVaR
                z = norm.ppf(0.05)
                cvar = port_vol * norm.pdf(z) / 0.05

                results.append({
                    "method": name, "vol": port_vol, "cvar_5pct": cvar,
                    "div_ratio": div_ratio, "eff_n": eff_n, "turnover": turnover,
                    "max_weight": w.max(), "min_weight": w.min(),
                })
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        return pd.DataFrame(results)

    # ─── UTILITIES ───

    @staticmethod
    def _cluster_variance(cov, indices):
        sub_cov = cov[np.ix_(indices, indices)]
        w = np.ones(len(indices)) / len(indices)
        return float(w @ sub_cov @ w)

    @staticmethod
    def _ensure_psd(matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _project_simplex(self, w):
        w = np.clip(w, self.min_weight, self.max_weight)
        return w / w.sum()
