"""
Risk Analysis Module — Adapted from Crypto-Statistical-Arbitrage.

Comprehensive risk analytics for the vault portfolio:
  - Parametric / Historical / Monte Carlo VaR
  - Conditional VaR (Expected Shortfall)
  - Cornish-Fisher VaR (skewness/kurtosis adjusted)
  - Conditional Drawdown at Risk (CDaR)
  - Drawdown analysis with top-N events and recovery tracking
  - Risk decomposition by asset and asset class
  - BTC correlation and beta
  - Tail dependence (copula-based estimates)
  - Risk attribution over time (rolling marginal risk contribution)
  - Stress testing (user-defined scenarios)
  - Portfolio efficiency metrics (Diversification ratio, effective N, HHI)
  - Risk limit monitoring with breach detection

Directly adapted from: github.com/abailey81/Crypto-Statistical-Arbitrage/portfolio/risk_analysis.py
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    confidence: float
    horizon: str
    method: str
    var_pct: float
    cvar_pct: float

    def __repr__(self):
        return f"VaR({self.method}, {self.confidence:.0%}): {self.var_pct:.2%} | CVaR: {self.cvar_pct:.2%}"


@dataclass
class DrawdownEvent:
    """Single drawdown event with timing."""
    drawdown: float
    peak_date: str
    trough_date: str
    recovery_date: Optional[str]
    duration_days: int
    recovery_days: int


@dataclass
class RiskDecomposition:
    """Risk contribution by asset."""
    total_vol_annual: float
    asset_contributions: Dict[str, float]
    marginal_risk: Dict[str, float]
    diversification_ratio: float


class VaultRiskAnalyzer:
    """
    Comprehensive risk analysis for the vault portfolio.

    Adapted from the author's Crypto-Statistical-Arbitrage RiskAnalyzer,
    retargeted for hourly crypto portfolio risk assessment.
    """

    def __init__(self, returns_df: pd.DataFrame, weights: np.ndarray = None,
                 asset_names: list = None, config: Optional[Dict] = None):
        self.returns_df = returns_df
        self.asset_names = asset_names or list(returns_df.columns)
        n = returns_df.shape[1]
        self.weights = weights if weights is not None else np.ones(n) / n
        self.portfolio_returns = (returns_df.values * self.weights).sum(axis=1)

        cfg = config or {}
        self.cdar_quantile = cfg.get("cdar_quantile", 0.05)
        self.tail_dep_threshold = cfg.get("tail_dependence_threshold", 0.05)
        self.rolling_risk_window = cfg.get("rolling_risk_window", 720)

    # ─── VaR / CVaR ───

    def calculate_var(self, confidence: float = 0.95, horizon_hours: int = 24,
                      method: str = "historical") -> VaRResult:
        """Calculate VaR using parametric, historical, or Monte Carlo method."""
        if method == "parametric":
            var, cvar = self._parametric_var(confidence, horizon_hours)
        elif method == "historical":
            var, cvar = self._historical_var(confidence, horizon_hours)
        elif method == "monte_carlo":
            var, cvar = self._monte_carlo_var(confidence, horizon_hours)
        elif method == "cornish_fisher":
            var, cvar = self._cornish_fisher_var(confidence, horizon_hours)
        else:
            raise ValueError(f"Unknown method: {method}")

        return VaRResult(confidence=confidence, horizon=f"{horizon_hours}h",
                         method=method, var_pct=var, cvar_pct=cvar)

    def _parametric_var(self, confidence: float, horizon: int) -> Tuple[float, float]:
        mu = self.portfolio_returns.mean() * horizon
        sigma = self.portfolio_returns.std() * np.sqrt(horizon)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma)
        cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
        return float(var), float(cvar)

    def _historical_var(self, confidence: float, horizon: int) -> Tuple[float, float]:
        if horizon > 1:
            rolling = pd.Series(self.portfolio_returns).rolling(horizon).sum().dropna().values
        else:
            rolling = self.portfolio_returns
        var = -np.percentile(rolling, (1 - confidence) * 100)
        tail = rolling[rolling <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var
        return float(var), float(cvar)

    def _monte_carlo_var(self, confidence: float, horizon: int,
                         n_sims: int = 10000) -> Tuple[float, float]:
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        sims = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
        var = -np.percentile(sims, (1 - confidence) * 100)
        tail = sims[sims <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var
        return float(var), float(cvar)

    # ─── CORNISH-FISHER VaR ───

    def _cornish_fisher_var(self, confidence: float, horizon: int) -> Tuple[float, float]:
        """
        Cornish-Fisher VaR — adjusts the Gaussian quantile for
        skewness and excess kurtosis of the return distribution.

        z_cf = z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36

        where S = skewness, K = excess kurtosis, z = Gaussian quantile.
        """
        mu = self.portfolio_returns.mean() * horizon
        sigma = self.portfolio_returns.std() * np.sqrt(horizon)
        z = stats.norm.ppf(1 - confidence)

        skew = float(stats.skew(self.portfolio_returns))
        kurt = float(stats.kurtosis(self.portfolio_returns))  # excess kurtosis

        z_cf = (z
                + (z ** 2 - 1) * skew / 6.0
                + (z ** 3 - 3 * z) * kurt / 24.0
                - (2 * z ** 3 - 5 * z) * skew ** 2 / 36.0)

        var = -(mu + z_cf * sigma)

        # Approximate CVaR: use parametric CVaR with CF-adjusted z
        cvar = -(mu - sigma * stats.norm.pdf(z_cf) / (1 - confidence))
        logger.info(
            "Cornish-Fisher VaR: skew=%.3f kurtosis=%.3f z_cf=%.4f var=%.4f",
            skew, kurt, z_cf, var,
        )
        return float(var), float(cvar)

    def cornish_fisher_var(self, confidence: float = 0.95,
                           horizon_hours: int = 24) -> VaRResult:
        """Public Cornish-Fisher VaR with skewness/kurtosis adjustment."""
        var, cvar = self._cornish_fisher_var(confidence, horizon_hours)
        return VaRResult(confidence=confidence, horizon=f"{horizon_hours}h",
                         method="cornish_fisher", var_pct=var, cvar_pct=cvar)

    def var_comparison(self, confidence: float = 0.95, horizon: int = 24) -> pd.DataFrame:
        """Compare all VaR methods including Cornish-Fisher."""
        results = []
        for method in ["parametric", "historical", "monte_carlo", "cornish_fisher"]:
            r = self.calculate_var(confidence, horizon, method)
            results.append({"method": method, "VaR": r.var_pct, "CVaR": r.cvar_pct})
        return pd.DataFrame(results)

    # ─── CONDITIONAL DRAWDOWN AT RISK (CDaR) ───

    def conditional_drawdown_at_risk(self, quantile: Optional[float] = None) -> Dict[str, float]:
        """
        Conditional Drawdown at Risk (CDaR).

        Average of the worst *quantile* fraction of drawdowns.
        Analogous to CVaR but for drawdowns instead of returns.

        Args:
            quantile: Fraction of worst drawdowns to average (default from config, 0.05).

        Returns:
            Dict with 'cdar', 'max_drawdown', 'n_drawdown_obs', 'threshold_drawdown'.
        """
        q = quantile if quantile is not None else self.cdar_quantile

        cum = np.exp(np.cumsum(self.portfolio_returns))
        peak = np.maximum.accumulate(cum)
        dd = cum / peak - 1.0  # negative values

        # CDaR: average of the worst q% drawdowns
        threshold = np.percentile(dd, q * 100)
        tail_dd = dd[dd <= threshold]
        cdar = float(tail_dd.mean()) if len(tail_dd) > 0 else float(dd.min())

        logger.info(
            "CDaR(%.0f%%): %.4f  (max_dd=%.4f, n_tail=%d)",
            q * 100, cdar, dd.min(), len(tail_dd),
        )
        return {
            "cdar": cdar,
            "max_drawdown": float(dd.min()),
            "n_drawdown_obs": int(len(tail_dd)),
            "threshold_drawdown": float(threshold),
        }

    # ─── DRAWDOWN ANALYSIS ───

    def analyze_drawdowns(self, top_n: int = 5) -> Tuple[float, List[DrawdownEvent]]:
        """
        Full drawdown analysis with top-N events.
        Adapted from stat-arb portfolio/risk_analysis.py.
        """
        cum = np.exp(np.cumsum(self.portfolio_returns))
        peak = np.maximum.accumulate(cum)
        dd = cum / peak - 1.0

        max_dd = dd.min()

        # Find top N drawdown events
        events = []
        dd_series = pd.Series(dd, index=self.returns_df.index if hasattr(self.returns_df, 'index') else range(len(dd)))
        dd_copy = dd_series.copy()

        for _ in range(min(top_n, len(dd_copy))):
            if dd_copy.min() >= -0.01:
                break
            trough_idx = dd_copy.idxmin()
            trough_loc = dd_copy.index.get_loc(trough_idx)

            # Find peak before trough
            peak_loc = np.argmax(cum[:trough_loc + 1])

            # Find recovery after trough
            post_trough = cum[trough_loc:]
            peak_val = cum[peak_loc]
            recovered = np.where(post_trough >= peak_val)[0]
            recovery_loc = trough_loc + recovered[0] if len(recovered) > 0 else None

            events.append(DrawdownEvent(
                drawdown=float(dd_copy[trough_idx]),
                peak_date=str(dd_copy.index[peak_loc]),
                trough_date=str(trough_idx),
                recovery_date=str(dd_copy.index[recovery_loc]) if recovery_loc else None,
                duration_days=(trough_loc - peak_loc) // 24,
                recovery_days=(recovery_loc - trough_loc) // 24 if recovery_loc else -1,
            ))

            # Mask nearby values
            mask_start = max(0, trough_loc - 480)
            mask_end = min(len(dd_copy), trough_loc + 480)
            dd_copy.iloc[mask_start:mask_end] = 0

        return float(max_dd), events

    # ─── RISK DECOMPOSITION ───

    def decompose_risk(self) -> RiskDecomposition:
        """Decompose portfolio risk by asset contribution."""
        cov = self.returns_df.cov().values
        w = self.weights

        port_var = w @ cov @ w
        port_vol = np.sqrt(port_var) * np.sqrt(8760)

        marginal = cov @ w
        risk_contrib = w * marginal
        rc_pct = risk_contrib / port_var if port_var > 0 else np.zeros_like(w)

        # Diversification ratio
        weighted_vols = np.sqrt(np.diag(cov)) @ w * np.sqrt(8760)
        div_ratio = weighted_vols / port_vol if port_vol > 0 else 1.0

        return RiskDecomposition(
            total_vol_annual=port_vol,
            asset_contributions={self.asset_names[i]: float(rc_pct[i]) for i in range(len(w))},
            marginal_risk={self.asset_names[i]: float(marginal[i] * np.sqrt(8760)) for i in range(len(w))},
            diversification_ratio=float(div_ratio),
        )

    # ─── TAIL DEPENDENCE ───

    def tail_dependence(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Estimate upper and lower tail dependence between asset pairs
        using empirical copula-based estimates.

        For each pair (i, j), compute:
          - Lower tail: P(U_j <= q | U_i <= q)  where U = rank / (T+1)
          - Upper tail: P(U_j >= 1-q | U_i >= 1-q)

        Args:
            threshold: Quantile threshold for tail (default from config, 0.05).

        Returns:
            Dict with 'avg_lower_tail_dep', 'avg_upper_tail_dep',
            'max_lower_pair', 'max_upper_pair', 'lower_matrix', 'upper_matrix'.
        """
        q = threshold if threshold is not None else self.tail_dep_threshold
        data = self.returns_df.values
        T, n = data.shape

        if T < 50:
            logger.warning("Insufficient data (%d rows) for tail dependence; returning zeros", T)
            return {"avg_lower_tail_dep": 0.0, "avg_upper_tail_dep": 0.0,
                    "max_lower_pair": "", "max_upper_pair": "",
                    "lower_matrix": np.zeros((n, n)), "upper_matrix": np.zeros((n, n))}

        # Convert to pseudo-observations (empirical copula)
        ranks = np.zeros_like(data)
        for col in range(n):
            ranks[:, col] = stats.rankdata(data[:, col]) / (T + 1)

        lower_dep = np.zeros((n, n))
        upper_dep = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Lower tail
                mask_i_low = ranks[:, i] <= q
                if mask_i_low.sum() > 0:
                    lower_dep[i, j] = np.mean(ranks[mask_i_low, j] <= q)
                    lower_dep[j, i] = lower_dep[i, j]

                # Upper tail
                mask_i_high = ranks[:, i] >= (1 - q)
                if mask_i_high.sum() > 0:
                    upper_dep[i, j] = np.mean(ranks[mask_i_high, j] >= (1 - q))
                    upper_dep[j, i] = upper_dep[i, j]

        # Find max pairs
        np.fill_diagonal(lower_dep, 0)
        np.fill_diagonal(upper_dep, 0)

        max_lower_idx = np.unravel_index(np.argmax(lower_dep), lower_dep.shape)
        max_upper_idx = np.unravel_index(np.argmax(upper_dep), upper_dep.shape)

        max_lower_pair = f"{self.asset_names[max_lower_idx[0]]}-{self.asset_names[max_lower_idx[1]]}"
        max_upper_pair = f"{self.asset_names[max_upper_idx[0]]}-{self.asset_names[max_upper_idx[1]]}"

        # Off-diagonal averages
        mask = ~np.eye(n, dtype=bool)
        avg_lower = float(np.mean(lower_dep[mask])) if n > 1 else 0.0
        avg_upper = float(np.mean(upper_dep[mask])) if n > 1 else 0.0

        logger.info(
            "Tail dependence (q=%.2f): avg_lower=%.4f avg_upper=%.4f  "
            "max_lower=%s(%.4f) max_upper=%s(%.4f)",
            q, avg_lower, avg_upper,
            max_lower_pair, lower_dep[max_lower_idx],
            max_upper_pair, upper_dep[max_upper_idx],
        )

        return {
            "avg_lower_tail_dep": avg_lower,
            "avg_upper_tail_dep": avg_upper,
            "max_lower_pair": max_lower_pair,
            "max_upper_pair": max_upper_pair,
            "lower_matrix": lower_dep,
            "upper_matrix": upper_dep,
        }

    # ─── RISK ATTRIBUTION OVER TIME ───

    def rolling_risk_attribution(self, window: Optional[int] = None) -> pd.DataFrame:
        """
        Rolling marginal risk contribution per asset over time.

        Computes the percentage risk contribution of each asset using
        a rolling covariance window.

        Args:
            window: Rolling window in hours (default from config, 720).

        Returns:
            DataFrame with index = timestamp, columns = asset names,
            values = fractional risk contribution.
        """
        win = window if window is not None else self.rolling_risk_window
        T, n = self.returns_df.shape
        w = self.weights

        if T < win:
            logger.warning("Insufficient data (%d < %d) for rolling risk attribution", T, win)
            return pd.DataFrame(columns=self.asset_names)

        results = []
        indices = []

        for end in range(win, T):
            start = end - win
            sub = self.returns_df.values[start:end]
            cov = np.cov(sub, rowvar=False)

            port_var = w @ cov @ w
            if port_var > 1e-16:
                marginal = cov @ w
                rc = w * marginal / port_var
            else:
                rc = np.ones(n) / n

            results.append(rc)
            indices.append(self.returns_df.index[end] if hasattr(self.returns_df, 'index') else end)

        df = pd.DataFrame(results, index=indices, columns=self.asset_names[:n])
        logger.info("Rolling risk attribution computed: %d windows of %d hours", len(df), win)
        return df

    # ─── STRESS TESTING ───

    def stress_test(self, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Apply user-defined stress scenarios to the portfolio.

        Each scenario is a dict mapping asset names to return shocks.
        Example:
            {
                "BTC crash": {"BTC": -0.30, "ETH": -0.25, "SOL": -0.40},
                "Yields +200bps": {"BUIDL": -0.02, "USDY": -0.01},
            }

        Args:
            scenarios: Dict of scenario_name -> {asset_name: return_shock}.

        Returns:
            DataFrame with scenario name, portfolio impact, and per-asset impacts.
        """
        results = []

        for scenario_name, shocks in scenarios.items():
            asset_impacts = np.zeros(len(self.asset_names))
            for asset, shock in shocks.items():
                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    asset_impacts[idx] = shock

            portfolio_impact = float(np.sum(self.weights * asset_impacts))

            result = {"scenario": scenario_name, "portfolio_impact": portfolio_impact}
            for i, name in enumerate(self.asset_names):
                result[f"{name}_impact"] = float(self.weights[i] * asset_impacts[i])

            results.append(result)
            logger.info(
                "Stress test '%s': portfolio impact = %.4f (%.2f%%)",
                scenario_name, portfolio_impact, portfolio_impact * 100,
            )

        return pd.DataFrame(results)

    # ─── PORTFOLIO EFFICIENCY METRICS ───

    def portfolio_efficiency(self) -> Dict[str, float]:
        """
        Compute portfolio efficiency metrics:
          - Diversification ratio: weighted avg vol / portfolio vol
          - Effective N: 1 / HHI (how many equal-weight assets equivalent)
          - HHI: Herfindahl-Hirschman Index (weight concentration)
          - Max weight, min weight
        """
        cov = self.returns_df.cov().values
        w = self.weights
        n = len(w)

        port_var = w @ cov @ w
        port_vol = np.sqrt(port_var)

        asset_vols = np.sqrt(np.diag(cov))
        weighted_vols = asset_vols @ w
        div_ratio = weighted_vols / port_vol if port_vol > 0 else 1.0

        hhi = float(np.sum(w ** 2))
        eff_n = 1.0 / hhi if hhi > 0 else float(n)

        result = {
            "diversification_ratio": float(div_ratio),
            "effective_n": float(eff_n),
            "hhi": hhi,
            "max_weight": float(np.max(w)),
            "min_weight": float(np.min(w)),
            "n_assets": n,
        }

        logger.info(
            "Portfolio efficiency: DivR=%.3f  EffN=%.1f  HHI=%.4f  MaxW=%.3f",
            result["diversification_ratio"], result["effective_n"],
            result["hhi"], result["max_weight"],
        )
        return result

    # ─── BTC CORRELATION ───

    def btc_correlation(self, btc_returns: np.ndarray) -> Dict[str, float]:
        """Portfolio and per-asset correlation to BTC."""
        min_len = min(len(self.portfolio_returns), len(btc_returns))
        port_corr = float(np.corrcoef(self.portfolio_returns[:min_len], btc_returns[:min_len])[0, 1])

        asset_corrs = {}
        for i, name in enumerate(self.asset_names):
            asset_ret = self.returns_df.values[:min_len, i]
            asset_corrs[name] = float(np.corrcoef(asset_ret, btc_returns[:min_len])[0, 1])

        # BTC beta
        cov_with_btc = np.cov(self.portfolio_returns[:min_len], btc_returns[:min_len])
        beta = cov_with_btc[0, 1] / cov_with_btc[1, 1] if cov_with_btc[1, 1] > 0 else 0

        return {
            "portfolio_btc_correlation": port_corr,
            "portfolio_btc_beta": float(beta),
            "asset_btc_correlations": asset_corrs,
        }

    # ─── RISK LIMIT CHECK ───

    def check_risk_limits(self, var_limit: float = 0.03, dd_limit: float = 0.20,
                          corr_limit: float = 0.30,
                          btc_returns: np.ndarray = None) -> Dict:
        """Check all risk limits. Adapted from stat-arb RiskMonitor."""
        var = self.calculate_var(0.95, 24, "historical")
        max_dd, _ = self.analyze_drawdowns(1)

        breaches = []
        if var.var_pct > var_limit:
            breaches.append(f"VaR {var.var_pct:.2%} > {var_limit:.2%}")
        if abs(max_dd) > dd_limit:
            breaches.append(f"MaxDD {max_dd:.2%} > {dd_limit:.2%}")

        if btc_returns is not None:
            btc = self.btc_correlation(btc_returns)
            if abs(btc["portfolio_btc_correlation"]) > corr_limit:
                breaches.append(f"BTC corr {btc['portfolio_btc_correlation']:.2f} > {corr_limit:.2f}")

        return {"compliant": len(breaches) == 0, "breaches": breaches,
                "var_1d_95": var.var_pct, "max_drawdown": max_dd}
