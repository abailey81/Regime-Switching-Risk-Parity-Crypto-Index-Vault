"""
Portfolio Analysis Module — Statistical tests and decomposition for the 8-asset vault.

Provides a unified ``PortfolioAnalyzer`` class that runs:
  1.  Mean-Variance Spanning Test (Huberman & Kandel, 1987)
  2.  Diversification Benefit Decomposition by asset class
  3.  Optimal-N Analysis (greedy vol-minimisation)
  4.  Regime-Conditional Correlation Matrices
  5.  Copula Tail Dependence Analysis (empirical t-copula)
  6.  Eigenvalue Decomposition / Random Matrix Theory
  7.  Granger Causality Network
  8.  Johansen Cointegration Testing
  9.  Ledoit-Wolf Shrinkage Comparison
  10. Rolling Absorption Ratio (Kritzman et al., 2011)
  11. Correlation Network (MST + centrality)
  12. Full Analysis Runner

All methods return structured dicts / DataFrames suitable for programmatic
use and downstream chart generation.

Dependencies: numpy, pandas, scipy, sklearn, statsmodels, warnings, logging.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

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

# ── Default asset-class mapping ──────────────────────────────────────────────
_DEFAULT_ASSET_CLASSES: Dict[str, List[str]] = {
    "crypto": ["BTC", "ETH", "SOL"],
    "staking": ["stETH", "rETH"],
    "treasuries": ["BUIDL", "USDY"],
    "stable": ["USDC"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PortfolioAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioAnalyzer:
    """Comprehensive portfolio analysis for the 8-asset vault.

    Parameters
    ----------
    returns_df : pd.DataFrame
        (T, N) DataFrame of log returns indexed by datetime.
    prices_df : pd.DataFrame, optional
        (T, N) DataFrame of prices (needed for cointegration tests).
    config : dict, optional
        Full config dict (from ``config.yaml``).
    asset_classes : dict, optional
        Mapping ``class_name -> [asset_names]``.  Falls back to the
        default 4-class structure when omitted.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        prices_df: pd.DataFrame = None,
        config: dict = None,
        asset_classes: Dict[str, List[str]] = None,
    ):
        self.returns = returns_df.copy()
        self.prices = prices_df.copy() if prices_df is not None else None
        self.config = config or {}
        self.asset_classes = asset_classes or _DEFAULT_ASSET_CLASSES
        self.asset_names: List[str] = list(returns_df.columns)
        self.n_assets: int = len(self.asset_names)
        self.T: int = len(returns_df)

        logger.info(
            "PortfolioAnalyzer initialised: %d assets x %d observations",
            self.n_assets, self.T,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  1. Mean-Variance Spanning Test
    # ──────────────────────────────────────────────────────────────────────────

    def spanning_test(
        self,
        base_assets: List[str],
        test_assets: List[str],
        significance: float = 0.05,
    ) -> dict:
        """Huberman & Kandel (1987) mean-variance spanning test.

        H0: the efficient frontier of ``base_assets`` is identical to the
        frontier of ``base_assets + test_assets`` (i.e., the new assets are
        redundant).

        Uses the Wald-type F-test:  regress each test-asset return on
        base-asset returns, then test jointly that all intercepts = 0 and
        all beta row-sums = 1.

        Parameters
        ----------
        base_assets : list of str
        test_assets : list of str
        significance : float
            p-value threshold for rejection.

        Returns
        -------
        dict with test_statistic, p_value, is_significant, interpretation.
        """
        missing_b = [a for a in base_assets if a not in self.asset_names]
        missing_t = [a for a in test_assets if a not in self.asset_names]
        if missing_b or missing_t:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
                "interpretation": (
                    f"Missing assets — base: {missing_b}, test: {missing_t}"
                ),
            }

        R_base = self.returns[base_assets].values  # (T, K)
        R_test = self.returns[test_assets].values   # (T, L)
        T, K = R_base.shape
        L = R_test.shape[1]

        if T < K + L + 5:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
                "interpretation": "Insufficient observations for spanning test.",
            }

        # --- OLS: R_test = alpha + R_base @ Beta + epsilon ---
        X = np.column_stack([np.ones(T), R_base])  # (T, K+1)
        try:
            # Multivariate OLS:  B_hat = (X'X)^{-1} X' Y
            XtX_inv = np.linalg.inv(X.T @ X)
            B_hat = XtX_inv @ X.T @ R_test  # (K+1, L)
            E = R_test - X @ B_hat           # (T, L) residuals
            Sigma_e = (E.T @ E) / (T - K - 1)

            # H0: alpha = 0  AND  sum(betas per test asset) = 1
            # Constraint matrix C (2L restrictions):
            #   row 0..L-1:  alpha_l = 0
            #   row L..2L-1: sum_k beta_{k,l} = 1  for each l
            n_params = K + 1  # intercept + K slopes
            C = np.zeros((2 * L, n_params * L))
            d = np.zeros(2 * L)

            for l in range(L):
                # alpha_l = 0  →  select B_hat[0, l]
                C[l, l * n_params] = 1.0
                # sum_k beta_{k,l} = 1
                for k in range(1, n_params):
                    C[L + l, l * n_params + k] = 1.0
                d[L + l] = 1.0

            b_vec = B_hat.T.ravel()  # vectorise column-major per test asset
            V_b = np.kron(Sigma_e, XtX_inv)  # variance of vec(B_hat')

            diff = C @ b_vec - d
            CV = C @ V_b @ C.T
            CV += np.eye(CV.shape[0]) * 1e-12  # regularise
            W = float(diff @ np.linalg.inv(CV) @ diff)

            q = 2 * L  # number of restrictions
            # Approximate F-distribution
            F_stat = W / q
            df1 = q
            df2 = T - K - 1 - L + 1
            if df2 < 1:
                df2 = 1
            p_value = float(1.0 - stats.f.cdf(F_stat, df1, df2))

        except np.linalg.LinAlgError:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "is_significant": False,
                "interpretation": "Singular matrix — spanning test could not be computed.",
            }

        is_sig = p_value < significance
        if is_sig:
            interp = (
                f"Reject H0 (p={p_value:.4f}): adding {test_assets} to "
                f"{base_assets} significantly improves the efficient frontier."
            )
        else:
            interp = (
                f"Fail to reject H0 (p={p_value:.4f}): {test_assets} do NOT "
                f"significantly expand the frontier spanned by {base_assets}."
            )

        logger.info("Spanning test: F=%.4f  p=%.4f  sig=%s", F_stat, p_value, is_sig)
        return {
            "test_statistic": float(F_stat),
            "p_value": p_value,
            "is_significant": is_sig,
            "interpretation": interp,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  2. Diversification Benefit Decomposition
    # ──────────────────────────────────────────────────────────────────────────

    def diversification_benefit(self) -> pd.DataFrame:
        """Decompose diversification gains by asset class.

        For each asset class compute:
          * marginal_vol_reduction_pct — pct vol reduction from adding this class
          * cvar_reduction_pct — pct CVaR(5%) improvement
          * diversification_ratio_contribution

        Returns
        -------
        pd.DataFrame with columns:
            asset_class, marginal_vol_reduction_pct, cvar_reduction_pct,
            diversification_ratio_contribution
        """
        all_assets = [a for a in self.asset_names]
        cov_full = self.returns[all_assets].cov().values
        w_eq = np.ones(len(all_assets)) / len(all_assets)
        vol_full = float(np.sqrt(w_eq @ cov_full @ w_eq))

        # Full-portfolio CVaR (parametric, 5%)
        port_ret_full = self.returns[all_assets].values @ w_eq
        cvar_full = self._parametric_cvar(port_ret_full)

        # Full diversification ratio
        asset_vols = np.sqrt(np.maximum(np.diag(cov_full), 1e-12))
        dr_full = float((asset_vols @ w_eq) / vol_full) if vol_full > 0 else 1.0

        rows = []
        for cls_name, cls_assets in self.asset_classes.items():
            present = [a for a in cls_assets if a in self.asset_names]
            if not present:
                rows.append({
                    "asset_class": cls_name,
                    "marginal_vol_reduction_pct": 0.0,
                    "cvar_reduction_pct": 0.0,
                    "diversification_ratio_contribution": 0.0,
                })
                continue

            # Portfolio WITHOUT this class
            remaining = [a for a in all_assets if a not in present]
            if len(remaining) < 1:
                rows.append({
                    "asset_class": cls_name,
                    "marginal_vol_reduction_pct": 0.0,
                    "cvar_reduction_pct": 0.0,
                    "diversification_ratio_contribution": 0.0,
                })
                continue

            cov_sub = self.returns[remaining].cov().values
            w_sub = np.ones(len(remaining)) / len(remaining)
            vol_sub = float(np.sqrt(w_sub @ cov_sub @ w_sub))

            port_ret_sub = self.returns[remaining].values @ w_sub
            cvar_sub = self._parametric_cvar(port_ret_sub)

            marginal_vol = (vol_sub - vol_full) / vol_sub * 100.0 if vol_sub > 0 else 0.0
            marginal_cvar = (cvar_sub - cvar_full) / cvar_sub * 100.0 if cvar_sub != 0 else 0.0

            # Diversification ratio contribution
            sub_vols = np.sqrt(np.maximum(np.diag(cov_sub), 1e-12))
            dr_sub = float((sub_vols @ w_sub) / vol_sub) if vol_sub > 0 else 1.0
            dr_contrib = dr_full - dr_sub

            rows.append({
                "asset_class": cls_name,
                "marginal_vol_reduction_pct": round(marginal_vol, 4),
                "cvar_reduction_pct": round(marginal_cvar, 4),
                "diversification_ratio_contribution": round(dr_contrib, 4),
            })

        df = pd.DataFrame(rows)
        logger.info("Diversification benefit:\n%s", df.to_string(index=False))
        return df

    # ──────────────────────────────────────────────────────────────────────────
    #  3. Optimal-N Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def optimal_n_analysis(self) -> pd.DataFrame:
        """Greedy asset addition — add the asset that reduces vol the most.

        Starts with BTC (or the first asset) and iteratively adds the asset
        that minimises equal-weight portfolio volatility.

        Returns
        -------
        pd.DataFrame with columns:
            n_assets, assets, portfolio_vol, portfolio_sharpe,
            portfolio_cvar, marginal_improvement
        """
        available = list(self.asset_names)
        if "BTC" in available:
            selected = ["BTC"]
            available.remove("BTC")
        else:
            selected = [available.pop(0)]

        rows = []

        # First entry
        vol_prev = self._eq_weight_vol(selected)
        sharpe_prev = self._eq_weight_sharpe(selected)
        cvar_prev = self._eq_weight_cvar(selected)
        rows.append({
            "n_assets": 1,
            "assets": ",".join(selected),
            "portfolio_vol": round(vol_prev, 6),
            "portfolio_sharpe": round(sharpe_prev, 4),
            "portfolio_cvar": round(cvar_prev, 6),
            "marginal_improvement": 0.0,
        })

        opt_n_bar = tqdm(total=len(available), desc="Optimal N", unit="asset", leave=False)
        while available:
            best_asset = None
            best_vol = np.inf
            for candidate in available:
                trial = selected + [candidate]
                v = self._eq_weight_vol(trial)
                if v < best_vol:
                    best_vol = v
                    best_asset = candidate

            selected.append(best_asset)
            available.remove(best_asset)

            sharpe_now = self._eq_weight_sharpe(selected)
            cvar_now = self._eq_weight_cvar(selected)
            improvement = (vol_prev - best_vol) / vol_prev * 100.0 if vol_prev > 0 else 0.0

            opt_n_bar.update(1)
            opt_n_bar.set_description(
                f"Optimal N | Adding asset {len(selected)}/{self.n_assets}"
            )
            opt_n_bar.set_postfix(asset=best_asset, vol=f"{best_vol:.4f}")

            rows.append({
                "n_assets": len(selected),
                "assets": ",".join(selected),
                "portfolio_vol": round(best_vol, 6),
                "portfolio_sharpe": round(sharpe_now, 4),
                "portfolio_cvar": round(cvar_now, 6),
                "marginal_improvement": round(improvement, 4),
            })
            vol_prev = best_vol
        opt_n_bar.close()

        df = pd.DataFrame(rows)
        logger.info("Optimal-N analysis:\n%s", df[["n_assets", "portfolio_vol", "marginal_improvement"]].to_string(index=False))
        return df

    # ──────────────────────────────────────────────────────────────────────────
    #  4. Regime-Conditional Correlation Matrices
    # ──────────────────────────────────────────────────────────────────────────

    def regime_conditional_correlations(
        self, regime_labels: np.ndarray
    ) -> dict:
        """Compute separate correlation matrices per regime.

        Parameters
        ----------
        regime_labels : np.ndarray of int
            Same length as ``self.returns``.
            Convention: 0 = bull, 1 = normal, 2 = crisis.

        Returns
        -------
        dict with keys: bull, normal, crisis, crisis_minus_bull,
        avg_crypto_corr_by_regime.
        """
        labels = np.asarray(regime_labels).ravel()
        if len(labels) != self.T:
            return {
                "error": (
                    f"regime_labels length ({len(labels)}) != "
                    f"returns length ({self.T})"
                )
            }

        regime_map = {0: "bull", 1: "normal", 2: "crisis"}
        corr_matrices: Dict[str, pd.DataFrame] = {}

        for code, name in regime_map.items():
            mask = labels == code
            n_obs = int(mask.sum())
            if n_obs < 30:
                logger.warning(
                    "Regime '%s' has only %d observations — correlation may be unreliable",
                    name, n_obs,
                )
            if n_obs < 2:
                corr_matrices[name] = pd.DataFrame(
                    np.nan, index=self.asset_names, columns=self.asset_names,
                )
                continue
            corr_matrices[name] = self.returns.iloc[mask].corr()

        # Difference matrix
        crisis_minus_bull = corr_matrices.get("crisis", pd.DataFrame())
        bull_mat = corr_matrices.get("bull", pd.DataFrame())
        if not crisis_minus_bull.empty and not bull_mat.empty:
            diff = crisis_minus_bull - bull_mat
        else:
            diff = pd.DataFrame()

        # Average crypto-only correlation per regime
        crypto_assets = [a for a in self.asset_classes.get("crypto", [])
                         if a in self.asset_names]
        avg_crypto: Dict[str, float] = {}
        for name, cmat in corr_matrices.items():
            if cmat.empty or len(crypto_assets) < 2:
                avg_crypto[name] = np.nan
                continue
            sub = cmat.loc[crypto_assets, crypto_assets].values
            mask_off = ~np.eye(len(crypto_assets), dtype=bool)
            avg_crypto[name] = float(np.nanmean(sub[mask_off]))

        logger.info("Regime correlations — avg crypto corr: %s", avg_crypto)
        return {
            "bull": corr_matrices.get("bull", pd.DataFrame()),
            "normal": corr_matrices.get("normal", pd.DataFrame()),
            "crisis": corr_matrices.get("crisis", pd.DataFrame()),
            "crisis_minus_bull": diff,
            "avg_crypto_corr_by_regime": avg_crypto,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  5. Copula Tail Dependence Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def tail_dependence_analysis(self, quantile: float = 0.05) -> dict:
        """Empirical copula-based tail dependence estimation.

        Estimates lower (crash) and upper (rally) tail dependence for each
        pair using pseudo-observations (rank-based empirical copula) and
        fits a Student-t copula to extract parametric tail-dependence
        coefficients.

        Returns
        -------
        dict with lower_tail, upper_tail, asymmetry matrices,
        most_tail_dependent_pairs, interpretation.
        """
        data = self.returns.values
        T, n = data.shape

        if T < 100:
            logger.warning("Only %d observations — tail dependence estimates unreliable", T)

        # Pseudo-observations (empirical copula)
        ranks = np.zeros_like(data)
        for col in range(n):
            ranks[:, col] = stats.rankdata(data[:, col]) / (T + 1)

        lower_dep = np.zeros((n, n))
        upper_dep = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Lower tail: P(U_j <= q | U_i <= q)
                mask_low = ranks[:, i] <= quantile
                if mask_low.sum() > 0:
                    lower_dep[i, j] = np.mean(ranks[mask_low, j] <= quantile)
                lower_dep[j, i] = lower_dep[i, j]

                # Upper tail: P(U_j >= 1-q | U_i >= 1-q)
                mask_high = ranks[:, i] >= (1 - quantile)
                if mask_high.sum() > 0:
                    upper_dep[i, j] = np.mean(ranks[mask_high, j] >= (1 - quantile))
                upper_dep[j, i] = upper_dep[i, j]

        # Parametric t-copula tail dependence (approximate)
        # For a bivariate t-copula with rho and nu dof:
        #   lambda_L = lambda_U = 2 * t_{nu+1}( -sqrt((nu+1)*(1-rho)/(1+rho)) )
        # We estimate rho from Kendall's tau and fit nu via MLE on the margins.
        t_lower = np.zeros((n, n))
        t_upper = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    tau, _ = stats.kendalltau(data[:, i], data[:, j])
                    rho_hat = np.sin(np.pi * tau / 2.0)  # Kendall->Pearson for elliptical
                    rho_hat = np.clip(rho_hat, -0.999, 0.999)

                    # Estimate degrees of freedom from marginal kurtosis
                    kurt_i = float(stats.kurtosis(data[:, i], fisher=True))
                    kurt_j = float(stats.kurtosis(data[:, j], fisher=True))
                    avg_kurt = (kurt_i + kurt_j) / 2.0
                    # For Student-t: excess kurtosis = 6/(nu-4) for nu>4
                    if avg_kurt > 0:
                        nu_hat = max(4.01, 6.0 / avg_kurt + 4.0)
                    else:
                        nu_hat = 30.0  # near-Gaussian
                    nu_hat = min(nu_hat, 100.0)

                    arg = -np.sqrt((nu_hat + 1) * (1 - rho_hat) / (1 + rho_hat + 1e-12))
                    lam = 2.0 * stats.t.cdf(arg, df=nu_hat + 1)
                    t_lower[i, j] = t_lower[j, i] = lam
                    t_upper[i, j] = t_upper[j, i] = lam  # symmetric for t-copula

                except Exception:
                    t_lower[i, j] = t_lower[j, i] = np.nan
                    t_upper[i, j] = t_upper[j, i] = np.nan

        asymmetry = lower_dep - upper_dep  # positive => more crash dependence

        # Identify most tail-dependent pairs
        np.fill_diagonal(lower_dep, 0)
        np.fill_diagonal(upper_dep, 0)
        flat_lower = []
        for i in range(n):
            for j in range(i + 1, n):
                flat_lower.append((
                    self.asset_names[i], self.asset_names[j],
                    float(lower_dep[i, j]),
                    float(upper_dep[i, j]),
                ))
        flat_lower.sort(key=lambda x: x[2], reverse=True)
        top_pairs = flat_lower[:5]

        # Build interpretation
        crypto = self.asset_classes.get("crypto", [])
        crypto_idx = [self.asset_names.index(a) for a in crypto if a in self.asset_names]
        avg_crypto_lower = np.nan
        avg_crypto_upper = np.nan
        if len(crypto_idx) >= 2:
            pairs_lower = []
            pairs_upper = []
            for i in crypto_idx:
                for j in crypto_idx:
                    if i < j:
                        pairs_lower.append(lower_dep[i, j])
                        pairs_upper.append(upper_dep[i, j])
            avg_crypto_lower = float(np.mean(pairs_lower)) if pairs_lower else np.nan
            avg_crypto_upper = float(np.mean(pairs_upper)) if pairs_upper else np.nan

        interp = (
            f"Crypto pairs avg lower (crash) tail dep: {avg_crypto_lower:.4f}, "
            f"avg upper (rally) tail dep: {avg_crypto_upper:.4f}.  "
        )
        if avg_crypto_lower > avg_crypto_upper + 0.02:
            interp += (
                "Asymmetric: crypto assets are MORE dependent in crashes than rallies, "
                "undermining diversification precisely when it is needed most."
            )
        else:
            interp += "Tail dependence is roughly symmetric for crypto pairs."

        lower_df = pd.DataFrame(lower_dep, index=self.asset_names, columns=self.asset_names)
        upper_df = pd.DataFrame(upper_dep, index=self.asset_names, columns=self.asset_names)
        asym_df = pd.DataFrame(asymmetry, index=self.asset_names, columns=self.asset_names)
        t_lower_df = pd.DataFrame(t_lower, index=self.asset_names, columns=self.asset_names)
        t_upper_df = pd.DataFrame(t_upper, index=self.asset_names, columns=self.asset_names)

        logger.info("Tail dependence — %s", interp[:120])
        return {
            "lower_tail": lower_df,
            "upper_tail": upper_df,
            "asymmetry": asym_df,
            "t_copula_lower": t_lower_df,
            "t_copula_upper": t_upper_df,
            "most_tail_dependent_pairs": top_pairs,
            "interpretation": interp,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  6. Eigenvalue Decomposition / Random Matrix Theory
    # ──────────────────────────────────────────────────────────────────────────

    def eigenvalue_analysis(self) -> dict:
        """Eigenvalue decomposition with Marchenko-Pastur noise threshold.

        Computes eigenvalues of the full-sample correlation matrix and
        compares against the Marchenko-Pastur upper bound to separate
        signal from noise eigenvalues.

        Returns
        -------
        dict with eigenvalues, mp_upper_bound, n_signal_eigenvalues,
        variance_explained, interpretation.
        """
        corr = self.returns.corr().values
        eigvals_raw = np.linalg.eigvalsh(corr)
        eigvals = np.sort(eigvals_raw)[::-1]  # descending

        N = self.n_assets
        T = self.T
        q = N / T  # aspect ratio

        # Marchenko-Pastur upper edge: lambda_+ = (1 + sqrt(N/T))^2
        mp_upper = (1.0 + np.sqrt(q)) ** 2

        total_var = eigvals.sum()
        variance_explained = eigvals / total_var if total_var > 0 else eigvals
        n_signal = int(np.sum(eigvals > mp_upper))

        # Effective rank (Shannon entropy)
        normed = eigvals / total_var
        normed_pos = normed[normed > 1e-12]
        entropy = -np.sum(normed_pos * np.log(normed_pos))
        effective_rank = float(np.exp(entropy))

        interp = (
            f"{n_signal} eigenvalue(s) above MP noise edge ({mp_upper:.4f}).  "
            f"Largest eigenvalue {eigvals[0]:.4f} explains "
            f"{variance_explained[0]*100:.1f}% of variance "
            f"(market factor).  Effective rank = {effective_rank:.1f}/{N}."
        )

        logger.info("Eigenvalue analysis: %s", interp)
        return {
            "eigenvalues": eigvals.tolist(),
            "mp_upper_bound": float(mp_upper),
            "n_signal_eigenvalues": n_signal,
            "variance_explained": variance_explained.tolist(),
            "effective_rank": effective_rank,
            "interpretation": interp,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  7. Granger Causality Network
    # ──────────────────────────────────────────────────────────────────────────

    def granger_causality_network(self, max_lag: int = 24) -> dict:
        """Pairwise Granger causality tests and directed network.

        Parameters
        ----------
        max_lag : int
            Maximum lag order to test (hours).

        Returns
        -------
        dict with edges, hub_asset (most outgoing), reactive_asset
        (most incoming), adjacency_matrix.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        edges: List[Tuple[str, str, int, float]] = []
        adj = pd.DataFrame(
            np.zeros((self.n_assets, self.n_assets)),
            index=self.asset_names,
            columns=self.asset_names,
        )

        n_pairs = self.n_assets * (self.n_assets - 1)
        granger_bar = tqdm(total=n_pairs, desc="Granger Causality", unit="pair", leave=False)

        for i, src in enumerate(self.asset_names):
            for j, tgt in enumerate(self.asset_names):
                if i == j:
                    continue
                granger_bar.update(1)
                granger_bar.set_postfix(pair=f"{src}->{tgt}")
                pair_data = self.returns[[tgt, src]].dropna()
                if len(pair_data) < max_lag + 30:
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = grangercausalitytests(
                            pair_data.values, maxlag=max_lag, verbose=False,
                        )

                    # Find the lag with the smallest p-value
                    best_lag = 1
                    best_p = 1.0
                    for lag in range(1, max_lag + 1):
                        if lag not in results:
                            continue
                        p_val = results[lag][0]["ssr_ftest"][1]
                        if p_val < best_p:
                            best_p = p_val
                            best_lag = lag

                    if best_p < 0.05:
                        edges.append((src, tgt, best_lag, float(best_p)))
                        adj.loc[src, tgt] = 1.0

                except Exception as exc:
                    logger.debug("Granger %s->%s failed: %s", src, tgt, exc)

        granger_bar.close()

        # Hub = most outgoing edges; reactive = most incoming
        outgoing = adj.sum(axis=1)
        incoming = adj.sum(axis=0)
        hub_asset = str(outgoing.idxmax()) if outgoing.max() > 0 else ""
        reactive_asset = str(incoming.idxmax()) if incoming.max() > 0 else ""

        logger.info(
            "Granger network: %d edges, hub=%s (out=%d), reactive=%s (in=%d)",
            len(edges), hub_asset, int(outgoing.max()),
            reactive_asset, int(incoming.max()),
        )
        return {
            "edges": edges,
            "hub_asset": hub_asset,
            "reactive_asset": reactive_asset,
            "adjacency_matrix": adj,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  8. Johansen Cointegration Testing
    # ──────────────────────────────────────────────────────────────────────────

    def cointegration_analysis(self) -> dict:
        """Johansen cointegration test on key pairs.

        Default pairs: (stETH, ETH), (rETH, ETH), (BUIDL, USDY), (BTC, ETH).

        Returns
        -------
        dict with 'pairs': list of dicts per pair.
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        default_pairs = [
            ("stETH", "ETH"),
            ("rETH", "ETH"),
            ("BUIDL", "USDY"),
            ("BTC", "ETH"),
        ]

        source = self.prices if self.prices is not None else self.returns
        pair_results = []

        for a, b in tqdm(default_pairs, desc="Cointegration tests", unit="pair", leave=False):
            if a not in source.columns or b not in source.columns:
                pair_results.append({
                    "pair": f"{a}-{b}",
                    "trace_stat": np.nan,
                    "critical_value_5pct": np.nan,
                    "is_cointegrated": False,
                    "rank": 0,
                    "interpretation": f"Asset(s) missing: {a} or {b}",
                })
                continue

            pair_data = source[[a, b]].dropna()
            if len(pair_data) < 100:
                pair_results.append({
                    "pair": f"{a}-{b}",
                    "trace_stat": np.nan,
                    "critical_value_5pct": np.nan,
                    "is_cointegrated": False,
                    "rank": 0,
                    "interpretation": f"Insufficient data ({len(pair_data)} rows).",
                })
                continue

            try:
                # det_order=-1: no constant in cointegrating relation
                # k_ar_diff=2: 2 lags of first differences
                result = coint_johansen(pair_data.values, det_order=0, k_ar_diff=2)

                trace_stat = float(result.lr1[0])  # first (r=0) trace statistic
                crit_5 = float(result.cvt[0, 1])   # 5% critical value

                # Cointegration rank: count rejections
                rank = 0
                for r in range(len(result.lr1)):
                    if result.lr1[r] > result.cvt[r, 1]:
                        rank += 1
                    else:
                        break

                is_coint = trace_stat > crit_5

                if a in ("stETH", "rETH") and b == "ETH":
                    if is_coint:
                        interp = (
                            f"{a}-{b} ARE cointegrated (trace={trace_stat:.2f} > "
                            f"cv={crit_5:.2f}). Expected: same underlying + yield."
                        )
                    else:
                        interp = (
                            f"{a}-{b} are NOT cointegrated (trace={trace_stat:.2f} < "
                            f"cv={crit_5:.2f}). Implies meaningful basis risk."
                        )
                else:
                    status = "cointegrated" if is_coint else "not cointegrated"
                    interp = (
                        f"{a}-{b}: {status} (trace={trace_stat:.2f}, "
                        f"cv_5%={crit_5:.2f}, rank={rank})."
                    )

                pair_results.append({
                    "pair": f"{a}-{b}",
                    "trace_stat": round(trace_stat, 4),
                    "critical_value_5pct": round(crit_5, 4),
                    "is_cointegrated": is_coint,
                    "rank": rank,
                    "interpretation": interp,
                })
                logger.info("Cointegration %s-%s: %s", a, b, interp)

            except Exception as exc:
                pair_results.append({
                    "pair": f"{a}-{b}",
                    "trace_stat": np.nan,
                    "critical_value_5pct": np.nan,
                    "is_cointegrated": False,
                    "rank": 0,
                    "interpretation": f"Johansen test failed: {exc}",
                })
                logger.warning("Cointegration %s-%s failed: %s", a, b, exc)

        return {"pairs": pair_results}

    # ──────────────────────────────────────────────────────────────────────────
    #  9. Ledoit-Wolf Shrinkage Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def shrinkage_comparison(self) -> dict:
        """Compare covariance estimators: sample, Ledoit-Wolf, OAS.

        For each estimator compute: condition number, equal-weight portfolio
        vol, risk-parity portfolio vol, and effective N.

        Returns
        -------
        dict with keys 'sample', 'ledoit_wolf', 'oracle', 'best'.
        """
        from sklearn.covariance import LedoitWolf, OAS

        data = self.returns.values
        n = self.n_assets

        estimators = {}

        # 1. Sample covariance
        cov_sample = np.cov(data, rowvar=False)
        estimators["sample"] = cov_sample

        # 2. Ledoit-Wolf
        try:
            lw = LedoitWolf().fit(data)
            cov_lw = lw.covariance_
            estimators["ledoit_wolf"] = cov_lw
        except Exception as exc:
            logger.warning("Ledoit-Wolf fit failed: %s", exc)
            cov_lw = cov_sample
            estimators["ledoit_wolf"] = cov_lw

        # 3. Oracle Approximating Shrinkage
        try:
            oas = OAS().fit(data)
            cov_oas = oas.covariance_
            estimators["oracle"] = cov_oas
        except Exception as exc:
            logger.warning("OAS fit failed: %s", exc)
            cov_oas = cov_sample
            estimators["oracle"] = cov_oas

        results = {}
        for name, cov in estimators.items():
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-16)
            cond = float(eigvals.max() / eigvals.min())

            # Equal-weight vol
            w_eq = np.ones(n) / n
            vol_eq = float(np.sqrt(w_eq @ cov @ w_eq) * np.sqrt(8760))

            # Risk-parity vol (simple iterative)
            w_rp = self._simple_risk_parity(cov)
            vol_rp = float(np.sqrt(w_rp @ cov @ w_rp) * np.sqrt(8760))

            eff_n = float(1.0 / np.sum(w_rp ** 2))

            results[name] = {
                "condition_number": round(cond, 2),
                "eq_weight_vol_annual": round(vol_eq, 6),
                "risk_parity_vol_annual": round(vol_rp, 6),
                "effective_n": round(eff_n, 2),
            }

        # Determine best by condition number (lower = better conditioned)
        best = min(results, key=lambda k: results[k]["condition_number"])
        results["best"] = best

        logger.info(
            "Shrinkage comparison — condition numbers: sample=%.1f  LW=%.1f  OAS=%.1f  best=%s",
            results["sample"]["condition_number"],
            results["ledoit_wolf"]["condition_number"],
            results["oracle"]["condition_number"],
            best,
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  10. Rolling Absorption Ratio
    # ──────────────────────────────────────────────────────────────────────────

    def rolling_absorption_ratio(
        self, window: int = 720, n_components: int = 3
    ) -> pd.Series:
        """Rolling absorption ratio (Kritzman et al., 2011).

        Fraction of total variance explained by the first ``n_components``
        eigenvectors, computed on a rolling window.  Spikes signal rising
        systemic risk and diversification breakdown.

        Parameters
        ----------
        window : int
            Rolling window in hours (default 720 = 30 days).
        n_components : int
            Number of top eigenvectors (default 3).

        Returns
        -------
        pd.Series indexed by datetime.
        """
        k = min(n_components, self.n_assets)
        if self.T < window:
            logger.warning(
                "Insufficient data (%d < %d) for rolling absorption ratio",
                self.T, window,
            )
            return pd.Series(dtype=float, name="absorption_ratio")

        ratios = []
        indices = []

        for end in tqdm(range(window, self.T), desc="Absorption ratio", unit="window", leave=False):
            start = end - window
            sub = self.returns.values[start:end]
            corr = np.corrcoef(sub, rowvar=False)
            # Guard against NaN
            if np.isnan(corr).any():
                ratios.append(np.nan)
            else:
                eigvals = np.linalg.eigvalsh(corr)
                eigvals = np.sort(eigvals)[::-1]
                total = eigvals.sum()
                ar = eigvals[:k].sum() / total if total > 0 else 0.0
                ratios.append(float(ar))
            indices.append(self.returns.index[end])

        result = pd.Series(ratios, index=indices, name="absorption_ratio")
        if len(result) > 0:
            logger.info(
                "Absorption ratio (k=%d, w=%d): latest=%.4f  mean=%.4f  max=%.4f",
                k, window,
                result.iloc[-1] if not np.isnan(result.iloc[-1]) else 0.0,
                result.mean(), result.max(),
            )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    #  11. Correlation Network (MST + Centrality)
    # ──────────────────────────────────────────────────────────────────────────

    def correlation_network(self) -> dict:
        """Build minimum spanning tree from correlation distances and compute
        centrality measures.

        Distance: d(i,j) = sqrt(0.5 * (1 - rho_{i,j})).
        MST via Kruskal's algorithm.
        Centrality: degree, betweenness, eigenvector.

        Returns
        -------
        dict with mst_edges, centrality (DataFrame), hub, peripherals.
        """
        corr = self.returns.corr().values
        n = self.n_assets

        # Correlation distance
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0)

        # Kruskal's MST
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((dist[i, j], i, j))
        edge_list.sort()

        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> bool:
            rx, ry = _find(x), _find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        mst_edges: List[Tuple[str, str, float]] = []
        adj_list: Dict[int, List[int]] = {i: [] for i in range(n)}

        for d_val, i, j in edge_list:
            if _union(i, j):
                mst_edges.append((self.asset_names[i], self.asset_names[j], float(d_val)))
                adj_list[i].append(j)
                adj_list[j].append(i)
                if len(mst_edges) == n - 1:
                    break

        # Degree centrality
        degree = {self.asset_names[i]: len(adj_list[i]) for i in range(n)}

        # Betweenness centrality (BFS-based for tree — every path is unique)
        betweenness = {self.asset_names[i]: 0.0 for i in range(n)}
        for src in range(n):
            # BFS from src to find all shortest paths (tree => unique)
            visited = {src}
            queue = [src]
            parent_map: Dict[int, int] = {}
            order = []
            while queue:
                node = queue.pop(0)
                order.append(node)
                for nb in adj_list[node]:
                    if nb not in visited:
                        visited.add(nb)
                        parent_map[nb] = node
                        queue.append(nb)

            # Every node on the path between src and any other node gets credit
            for tgt in range(n):
                if tgt == src:
                    continue
                path = []
                cur = tgt
                while cur != src:
                    if cur not in parent_map:
                        break
                    path.append(cur)
                    cur = parent_map[cur]
                # Intermediate nodes (exclude src and tgt)
                for node in path[1:]:
                    betweenness[self.asset_names[node]] += 1.0

        # Normalise betweenness
        norm_factor = max(1, (n - 1) * (n - 2))
        betweenness = {k: v / norm_factor for k, v in betweenness.items()}

        # Eigenvector centrality from adjacency matrix of MST
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in adj_list[i]:
                adj_matrix[i, j] = 1.0

        eigvals_adj, eigvecs_adj = np.linalg.eigh(adj_matrix)
        max_idx = np.argmax(eigvals_adj)
        eig_cent_raw = np.abs(eigvecs_adj[:, max_idx])
        eig_cent_raw = eig_cent_raw / (eig_cent_raw.sum() + 1e-12)
        eigenvector_cent = {self.asset_names[i]: float(eig_cent_raw[i]) for i in range(n)}

        centrality_df = pd.DataFrame({
            "asset": self.asset_names,
            "degree": [degree[a] for a in self.asset_names],
            "betweenness": [betweenness[a] for a in self.asset_names],
            "eigenvector": [eigenvector_cent[a] for a in self.asset_names],
        }).set_index("asset")

        hub = centrality_df["degree"].idxmax()
        peripherals = centrality_df[centrality_df["degree"] == centrality_df["degree"].min()].index.tolist()

        logger.info(
            "Correlation network: MST %d edges, hub=%s, peripherals=%s",
            len(mst_edges), hub, peripherals,
        )
        return {
            "mst_edges": mst_edges,
            "centrality": centrality_df,
            "hub": hub,
            "peripherals": peripherals,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  12. Full Analysis Runner
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_analysis(
        self, regime_labels: np.ndarray = None
    ) -> dict:
        """Run ALL portfolio analyses and return a combined results dict.

        Each analysis is wrapped in a try/except so that a single failure
        does not abort the remaining tests.

        Parameters
        ----------
        regime_labels : np.ndarray, optional
            Regime labels for conditional correlation analysis (0=bull,
            1=normal, 2=crisis).

        Returns
        -------
        dict mapping analysis name to its result.
        """
        results: Dict[str, object] = {}

        analysis_steps = [
            ("spanning_tests", "Mean-Variance Spanning Tests", None),
            ("diversification_benefit", "Diversification Benefit Decomposition", None),
            ("optimal_n", "Optimal-N Analysis", None),
            ("regime_correlations", "Regime-Conditional Correlations", None),
            ("tail_dependence", "Copula Tail Dependence", None),
            ("eigenvalue_analysis", "Eigenvalue Decomposition / RMT", None),
            ("granger_causality", "Granger Causality Network", None),
            ("cointegration", "Johansen Cointegration", None),
            ("shrinkage_comparison", "Covariance Shrinkage Comparison", None),
            ("absorption_ratio", "Rolling Absorption Ratio", None),
            ("correlation_network", "Correlation Network (MST)", None),
        ]

        analysis_bar = tqdm(analysis_steps, desc="Portfolio Analysis", unit="test", leave=True)

        for step_key, step_name, _ in analysis_bar:
            step_idx = analysis_steps.index((step_key, step_name, _)) + 1
            analysis_bar.set_description(
                f"Portfolio Analysis | {step_idx}/11 [{step_name[:25]}]"
            )

            logger.info("=" * 60)
            logger.info("[%d/11] %s", step_idx, step_name)
            logger.info("=" * 60)

            try:
                if step_key == "spanning_tests":
                    base = ["BTC", "ETH", "SOL"]
                    test_candidates = ["stETH", "rETH", "BUIDL", "USDY", "USDC"]
                    spanning = {}
                    for asset in test_candidates:
                        if asset in self.asset_names:
                            spanning[asset] = self.spanning_test(base, [asset])
                    full_test = [a for a in test_candidates if a in self.asset_names]
                    if full_test:
                        spanning["full_3v8"] = self.spanning_test(base, full_test)
                    results[step_key] = spanning

                elif step_key == "diversification_benefit":
                    results[step_key] = self.diversification_benefit()

                elif step_key == "optimal_n":
                    results[step_key] = self.optimal_n_analysis()

                elif step_key == "regime_correlations":
                    if regime_labels is not None:
                        results[step_key] = self.regime_conditional_correlations(
                            regime_labels
                        )
                    else:
                        logger.info("No regime labels provided — skipping regime correlations")
                        results[step_key] = {"skipped": "no regime_labels"}

                elif step_key == "tail_dependence":
                    results[step_key] = self.tail_dependence_analysis()

                elif step_key == "eigenvalue_analysis":
                    results[step_key] = self.eigenvalue_analysis()

                elif step_key == "granger_causality":
                    results[step_key] = self.granger_causality_network()

                elif step_key == "cointegration":
                    results[step_key] = self.cointegration_analysis()

                elif step_key == "shrinkage_comparison":
                    results[step_key] = self.shrinkage_comparison()

                elif step_key == "absorption_ratio":
                    results[step_key] = self.rolling_absorption_ratio()

                elif step_key == "correlation_network":
                    results[step_key] = self.correlation_network()

            except Exception as exc:
                logger.error("%s failed: %s", step_name, exc)
                results[step_key] = {"error": str(exc)}

        # ── Summary ──────────────────────────────────────────────────────
        n_ok = sum(1 for v in results.values()
                   if not (isinstance(v, dict) and ("error" in v or "skipped" in v)))
        n_err = sum(1 for v in results.values()
                    if isinstance(v, dict) and "error" in v)
        n_skip = sum(1 for v in results.values()
                     if isinstance(v, dict) and "skipped" in v)

        logger.info("=" * 60)
        logger.info(
            "Full analysis complete: %d OK, %d errors, %d skipped",
            n_ok, n_err, n_skip,
        )
        logger.info("=" * 60)

        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parametric_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Parametric CVaR (Expected Shortfall) at ``confidence`` level."""
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < 1e-12:
            return 0.0
        z = stats.norm.ppf(1 - confidence)
        cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
        return float(cvar)

    def _eq_weight_vol(self, assets: List[str]) -> float:
        """Equal-weight annualised portfolio volatility."""
        sub = self.returns[assets].values
        n = len(assets)
        w = np.ones(n) / n
        cov = np.cov(sub, rowvar=False)
        if cov.ndim == 0:
            return float(np.sqrt(cov) * np.sqrt(8760))
        return float(np.sqrt(w @ cov @ w) * np.sqrt(8760))

    def _eq_weight_sharpe(self, assets: List[str]) -> float:
        """Equal-weight annualised Sharpe ratio."""
        sub = self.returns[assets].values
        n = len(assets)
        w = np.ones(n) / n
        port_ret = sub @ w
        mu = float(np.mean(port_ret)) * 8760
        sigma = float(np.std(port_ret, ddof=1)) * np.sqrt(8760)
        rf = self.config.get("risk_free_rate", 0.045)
        return (mu - rf) / sigma if sigma > 1e-12 else 0.0

    def _eq_weight_cvar(self, assets: List[str]) -> float:
        """Equal-weight parametric CVaR."""
        sub = self.returns[assets].values
        n = len(assets)
        w = np.ones(n) / n
        port_ret = sub @ w
        return self._parametric_cvar(port_ret)

    @staticmethod
    def _simple_risk_parity(cov: np.ndarray, n_iter: int = 200) -> np.ndarray:
        """Lightweight risk-parity solver for shrinkage comparison."""
        n = cov.shape[0]
        w = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
        w = w / w.sum()

        for _ in range(n_iter):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                break
            marginal = cov @ w
            rc = w * marginal / port_vol
            target = port_vol / n
            adj = np.clip(target / (rc + 1e-12), 0.5, 2.0)
            w = w * adj
            w = np.maximum(w, 1e-8)
            w = w / w.sum()

        return w
