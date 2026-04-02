"""
Risk Analysis Module — Adapted from Crypto-Statistical-Arbitrage.

Comprehensive risk analytics for the vault portfolio:
  - Parametric / Historical / Monte Carlo VaR
  - Conditional VaR (Expected Shortfall)
  - Drawdown analysis with top-N events and recovery tracking
  - Risk decomposition by asset and asset class
  - BTC correlation and beta
  - Risk limit monitoring with breach detection

Directly adapted from: github.com/abailey81/Crypto-Statistical-Arbitrage/portfolio/risk_analysis.py
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
                 asset_names: list = None):
        self.returns_df = returns_df
        self.asset_names = asset_names or list(returns_df.columns)
        n = returns_df.shape[1]
        self.weights = weights if weights is not None else np.ones(n) / n
        self.portfolio_returns = (returns_df.values * self.weights).sum(axis=1)

    # ─── VaR / CVaR ───

    def calculate_var(self, confidence=0.95, horizon_hours=24,
                      method="historical") -> VaRResult:
        """Calculate VaR using parametric, historical, or Monte Carlo method."""
        if method == "parametric":
            var, cvar = self._parametric_var(confidence, horizon_hours)
        elif method == "historical":
            var, cvar = self._historical_var(confidence, horizon_hours)
        elif method == "monte_carlo":
            var, cvar = self._monte_carlo_var(confidence, horizon_hours)
        else:
            raise ValueError(f"Unknown method: {method}")

        return VaRResult(confidence=confidence, horizon=f"{horizon_hours}h",
                         method=method, var_pct=var, cvar_pct=cvar)

    def _parametric_var(self, confidence, horizon):
        mu = self.portfolio_returns.mean() * horizon
        sigma = self.portfolio_returns.std() * np.sqrt(horizon)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma)
        cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
        return float(var), float(cvar)

    def _historical_var(self, confidence, horizon):
        if horizon > 1:
            rolling = pd.Series(self.portfolio_returns).rolling(horizon).sum().dropna().values
        else:
            rolling = self.portfolio_returns
        var = -np.percentile(rolling, (1 - confidence) * 100)
        tail = rolling[rolling <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var
        return float(var), float(cvar)

    def _monte_carlo_var(self, confidence, horizon, n_sims=10000):
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        sims = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
        var = -np.percentile(sims, (1 - confidence) * 100)
        tail = sims[sims <= -var]
        cvar = -tail.mean() if len(tail) > 0 else var
        return float(var), float(cvar)

    def var_comparison(self, confidence=0.95, horizon=24) -> pd.DataFrame:
        """Compare all three VaR methods."""
        results = []
        for method in ["parametric", "historical", "monte_carlo"]:
            r = self.calculate_var(confidence, horizon, method)
            results.append({"method": method, "VaR": r.var_pct, "CVaR": r.cvar_pct})
        return pd.DataFrame(results)

    # ─── DRAWDOWN ANALYSIS ───

    def analyze_drawdowns(self, top_n=5) -> Tuple[float, List[DrawdownEvent]]:
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

    def check_risk_limits(self, var_limit=0.03, dd_limit=0.20,
                          corr_limit=0.30, btc_returns=None) -> Dict:
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
