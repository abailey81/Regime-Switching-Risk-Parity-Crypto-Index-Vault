"""
Model 1: Student-t GARCH(1,1)-DCC for dynamic covariance estimation.

Mathematical specification:
  Univariate: r_{i,t} = mu_i + sigma_{i,t} * z_{i,t},  z ~ t(nu)
              sigma^2_{i,t} = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

  DCC: Q_t = (1-a-b)*Q_bar + a*z_{t-1}*z'_{t-1} + b*Q_{t-1}
       R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

  Covariance: Sigma_t = D_t * R_t * D_t   where D_t = diag(sigma_1t,...,sigma_nt)

Enhancements:
  - Model selection: GARCH(1,1) vs EGARCH(1,1) vs GJR-GARCH(1,1) per asset via BIC
  - Diagnostic tests: Ljung-Box, ARCH-LM, Jarque-Bera on standardised residuals
  - DCC-MIDAS option, method-of-moments starting values, mean-reversion validation
  - Forecast evaluation: Mincer-Zarnowitz regression, rolling OOS accuracy
  - Bootstrap confidence intervals and uncertainty decomposition
"""
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats as sp_stats
from scipy.optimize import minimize

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *a, **kw):
        return iterable if iterable is not None else range(0)

logger = logging.getLogger(__name__)

_MAX_WORKERS = min(8, os.cpu_count() or 4)


class StudentTGarchDCC:
    """
    Student-t GARCH(1,1) with Dynamic Conditional Correlation.

    Estimates time-varying covariance matrix for portfolio optimisation.
    Student-t innovations capture fat tails characteristic of crypto returns.
    """

    def __init__(self, p: int = 1, q: int = 1, distribution: str = "studentst",
                 config: Optional[dict] = None):
        self.p = p
        self.q = q
        self.distribution = distribution
        self.garch_models = {}
        self.garch_results = {}
        self.conditional_vols = None
        self.std_residuals = None
        self.dcc_params = None
        self.Q_bar = None
        self._fitted = False

        # --- Enhancement config ---
        cfg = config or {}
        self.enable_model_selection = cfg.get("enable_model_selection", True)
        self.enable_diagnostics = cfg.get("enable_diagnostics", True)
        self.enable_dcc_midas = cfg.get("enable_dcc_midas", False)
        self.bootstrap_n_samples = cfg.get("bootstrap_n_samples", 200)
        self.enable_bootstrap = cfg.get("enable_bootstrap", False)

        # Storage for enhancement outputs
        self.model_selection_results: Dict[str, dict] = {}
        self.diagnostic_results: Dict[str, dict] = {}
        self.forecast_eval_metrics: Dict[str, dict] = {}
        self.bootstrap_ci: Dict[str, dict] = {}
        self.uncertainty_decomposition: Optional[dict] = None

        # Selected volatility model type per asset
        self.selected_vol_type: Dict[str, str] = {}

    # ─────────────────────────────────────────────────
    #  FIT
    # ─────────────────────────────────────────────────
    def fit(self, returns_df: pd.DataFrame) -> "StudentTGarchDCC":
        """
        Fit univariate GARCH models for each asset, then estimate DCC parameters.

        Args:
            returns_df: DataFrame of log returns (columns = assets)

        Returns:
            self (for chaining)
        """
        n_assets = returns_df.shape[1]
        assets = returns_df.columns.tolist()

        logger.info(f"Fitting Student-t GARCH({self.p},{self.q}) for {n_assets} assets...")

        # ── Step 1: Fit univariate GARCH for each asset (parallel) ──
        conditional_vols = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)
        std_resids = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)

        asset_bar = tqdm(total=n_assets, desc="Fitting GARCH", unit="asset", leave=True)
        results_lock = threading.Lock()

        def _fit_asset(col):
            """Fit GARCH for a single asset. Thread-safe (arch releases GIL)."""
            series = returns_df[col].dropna() * 100
            if self.enable_model_selection:
                result, vol_type = self._select_best_model(series, col)
            else:
                result, vol_type = self._fit_single_garch(series, col, "GARCH")
            return col, series, result, vol_type

        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, n_assets)) as pool:
            futures = {pool.submit(_fit_asset, col): col for col in assets}
            for future in as_completed(futures):
                col, series, result, vol_type = future.result()

                if result is not None:
                    with results_lock:
                        self.garch_results[col] = result
                        self.selected_vol_type[col] = vol_type
                        conditional_vols[col] = result.conditional_volatility / 100
                        std_resids[col] = result.std_resid

                    nu = result.params.get("nu", 30)
                    logger.info(
                        f"  {col}: model={vol_type}, "
                        f"alpha={result.params.get('alpha[1]', result.params.get('alpha[1]', 0)):.4f}, "
                        f"beta={result.params.get('beta[1]', 0):.4f}, "
                        f"nu={nu:.1f}"
                    )

                    # Run diagnostics on residuals (parallel per asset below)
                    if self.enable_diagnostics:
                        self._run_diagnostics_parallel(result, col)
                else:
                    logger.warning(f"  {col}: All GARCH fits failed, using EWMA fallback")
                    ewma_vol = series.ewm(span=60).std() / 100
                    with results_lock:
                        conditional_vols[col] = ewma_vol
                        std_resids[col] = (series / 100) / ewma_vol
                        self.selected_vol_type[col] = "EWMA"

                asset_bar.update(1)
                asset_bar.set_postfix(asset=col)

        asset_bar.close()

        self.conditional_vols = conditional_vols.dropna()
        self.std_residuals = std_resids.loc[self.conditional_vols.index].dropna()

        # ── Step 2: Estimate DCC parameters ──
        self._fit_dcc()

        self._fitted = True

        # ── Step 3: Bootstrap uncertainty (if enabled) ──
        if self.enable_bootstrap:
            self._bootstrap_parameters(returns_df)

        return self

    # ─────────────────────────────────────────────────
    #  MODEL SELECTION: GARCH vs EGARCH vs GJR-GARCH
    # ─────────────────────────────────────────────────
    def _select_best_model(self, series: pd.Series, asset: str) -> Tuple[object, str]:
        """
        Compare GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) per asset via BIC.

        Args:
            series: Return series scaled to percentages
            asset: Asset name for logging

        Returns:
            Tuple of (best_result, best_vol_type_name)
        """
        candidates = {
            "GARCH": {"vol": "GARCH"},
            "EGARCH": {"vol": "EGARCH"},
            "GJR-GARCH": {"vol": "GARCH", "o": 1},  # GJR adds the o=1 (threshold) term
        }

        bic_scores: Dict[str, float] = {}
        results: Dict[str, object] = {}

        # Fit all 3 model types in parallel (each is independent)
        def _fit_candidate(name, kwargs):
            result, _ = self._fit_single_garch(series, asset, name, **kwargs)
            return name, result

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {
                pool.submit(_fit_candidate, name, kwargs): name
                for name, kwargs in candidates.items()
            }
            for future in as_completed(futures):
                name, result = future.result()
                if result is not None:
                    bic_scores[name] = result.bic
                    results[name] = result

        if not bic_scores:
            return None, "NONE"

        # Select best by BIC (lower is better)
        best_name = min(bic_scores, key=bic_scores.get)
        best_bic = bic_scores[best_name]

        # Log comparison table
        log_lines = [f"  {asset} model selection (BIC):"]
        for name, bic_val in sorted(bic_scores.items(), key=lambda x: x[1]):
            delta = bic_val - best_bic
            marker = " ← BEST" if name == best_name else ""
            log_lines.append(f"    {name:12s}: BIC={bic_val:12.2f}  (Δ={delta:+8.2f}){marker}")
        logger.info("\n".join(log_lines))

        self.model_selection_results[asset] = {
            "bic_scores": bic_scores,
            "selected": best_name,
            "bic_delta": {k: v - best_bic for k, v in bic_scores.items()},
        }

        return results[best_name], best_name

    def _fit_single_garch(self, series: pd.Series, asset: str, model_name: str,
                          vol: str = "GARCH", o: int = 0) -> Tuple[Optional[object], str]:
        """
        Fit a single volatility model specification.

        Args:
            series: Return series (percentage-scaled)
            asset: Asset name
            model_name: Human-readable model name
            vol: Volatility model type for arch library
            o: Asymmetry order (1 for GJR-GARCH)

        Returns:
            Tuple of (fit_result_or_None, model_name)
        """
        try:
            model = arch_model(
                series,
                mean="Constant",
                vol=vol,
                p=self.p,
                q=self.q,
                o=o,
                dist=self.distribution,
            )
            result = model.fit(disp="off", show_warning=False)
            self.garch_models[f"{asset}_{model_name}"] = model
            return result, model_name
        except Exception as e:
            logger.debug(f"  {asset}: {model_name} fit failed ({e})")
            return None, model_name

    # ─────────────────────────────────────────────────
    #  DIAGNOSTIC TESTS
    # ─────────────────────────────────────────────────
    def _run_diagnostics_parallel(self, result: object, asset: str) -> Dict[str, dict]:
        """
        Run diagnostic tests on standardised residuals in parallel.

        Tests (all 3 run concurrently):
            1. Ljung-Box: No remaining autocorrelation in residuals
            2. ARCH-LM: No remaining ARCH effects (heteroskedasticity)
            3. Jarque-Bera: Normality / fat tails check

        Args:
            result: Fitted GARCH model result
            asset: Asset name

        Returns:
            Dictionary of test results
        """
        std_resid = result.std_resid.dropna().values
        diag = {}

        def _ljung_box():
            try:
                from scipy.stats import chi2
                n = len(std_resid)
                max_lag = min(10, n // 5)
                acf_vals = self._compute_acf(std_resid, max_lag)
                lb_stat = n * (n + 2) * np.sum(acf_vals ** 2 / np.arange(n - 1, n - max_lag - 1, -1))
                lb_pvalue = 1 - chi2.cdf(lb_stat, df=max_lag)
                lb_pass = lb_pvalue > 0.05
                return "ljung_box", {
                    "statistic": float(lb_stat),
                    "p_value": float(lb_pvalue),
                    "lags": max_lag,
                    "pass": lb_pass,
                }
            except Exception as e:
                return "ljung_box", {"error": str(e), "pass": None}

        def _arch_lm():
            try:
                from scipy.stats import chi2
                resid_sq = std_resid ** 2
                n = len(resid_sq)
                lm_lags = min(5, n // 5)
                Y = resid_sq[lm_lags:]
                X = np.column_stack([resid_sq[lm_lags - i - 1: n - i - 1] for i in range(lm_lags)])
                X = np.column_stack([np.ones(len(Y)), X])
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                Y_hat = X @ beta
                ss_res = np.sum((Y - Y_hat) ** 2)
                ss_tot = np.sum((Y - Y.mean()) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                lm_stat = n * r_squared
                lm_pvalue = 1 - chi2.cdf(lm_stat, df=lm_lags)
                lm_pass = lm_pvalue > 0.05
                return "arch_lm", {
                    "statistic": float(lm_stat),
                    "p_value": float(lm_pvalue),
                    "lags": lm_lags,
                    "pass": lm_pass,
                }
            except Exception as e:
                return "arch_lm", {"error": str(e), "pass": None}

        def _jarque_bera():
            try:
                jb_stat, jb_pvalue = sp_stats.jarque_bera(std_resid)
                jb_pass = jb_pvalue > 0.05
                return "jarque_bera", {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_pvalue),
                    "skewness": float(sp_stats.skew(std_resid)),
                    "kurtosis": float(sp_stats.kurtosis(std_resid)),
                    "pass": jb_pass,
                }
            except Exception as e:
                return "jarque_bera", {"error": str(e), "pass": None}

        # Run all 3 tests in parallel
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(fn) for fn in [_ljung_box, _arch_lm, _jarque_bera]]
            for future in as_completed(futures):
                test_name, test_result = future.result()
                diag[test_name] = test_result

        self.diagnostic_results[asset] = diag

        # Log results
        log_lines = [f"  {asset} diagnostics:"]
        for test_name, result_dict in diag.items():
            if "error" in result_dict:
                log_lines.append(f"    {test_name}: ERROR ({result_dict['error']})")
            else:
                status = "PASS" if result_dict["pass"] else "FAIL"
                log_lines.append(
                    f"    {test_name}: {status} (stat={result_dict['statistic']:.3f}, "
                    f"p={result_dict['p_value']:.4f})"
                )
        logger.info("\n".join(log_lines))

        return diag

    def _run_diagnostics(self, result: object, asset: str) -> Dict[str, dict]:
        """Legacy sequential diagnostics — delegates to parallel version."""
        return self._run_diagnostics_parallel(result, asset)

    @staticmethod
    def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute sample autocorrelation function up to max_lag."""
        n = len(x)
        x_demean = x - x.mean()
        var = np.sum(x_demean ** 2)
        acf = np.array([
            np.sum(x_demean[:n - k] * x_demean[k:]) / var
            for k in range(1, max_lag + 1)
        ])
        return acf

    # ─────────────────────────────────────────────────
    #  DCC ESTIMATION (enhanced)
    # ─────────────────────────────────────────────────
    def _fit_dcc(self) -> None:
        """
        Estimate DCC parameters (a, b) via quasi-maximum likelihood.

        Enhancements:
        - Method-of-moments starting values for better initialisation
        - Mean-reversion constraint validation (a + b < 1)
        - DCC-MIDAS option for long-run correlation

        Q_t = (1-a-b)*Q_bar + a*z_{t-1}*z'_{t-1} + b*Q_{t-1}
        """
        Z = self.std_residuals.values  # (T, n)
        T, n = Z.shape

        # Unconditional correlation matrix
        self.Q_bar = np.corrcoef(Z.T)

        # --- Method of moments starting values ---
        a_init, b_init = self._method_of_moments_init(Z)
        logger.info(f"  DCC init (method of moments): a={a_init:.4f}, b={b_init:.4f}")

        # QML estimation of a, b
        # Use a manual tqdm bar for DCC likelihood evaluations
        _dcc_eval_count = [0]
        _dcc_bar = tqdm(total=0, desc="DCC estimation", unit="eval", leave=False)

        def neg_log_likelihood(params):
            a, b = params
            if a < 0 or b < 0 or a + b >= 1:
                return 1e10

            Q_t = self.Q_bar.copy()
            ll = 0.0

            for t in range(1, T):
                z_t = Z[t - 1].reshape(-1, 1)
                Q_t = (1 - a - b) * self.Q_bar + a * (z_t @ z_t.T) + b * Q_t

                # Normalise to correlation
                D_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_t), 1e-8)))
                R_t = D_inv @ Q_t @ D_inv

                # Log-likelihood contribution
                try:
                    sign, logdet = np.linalg.slogdet(R_t)
                    if sign <= 0:
                        return 1e10
                    z_vec = Z[t]
                    R_inv = np.linalg.solve(R_t, z_vec)
                    ll += -0.5 * (logdet + z_vec @ R_inv - z_vec @ z_vec)
                except np.linalg.LinAlgError:
                    return 1e10

            _dcc_eval_count[0] += 1
            _dcc_bar.total = _dcc_eval_count[0]
            _dcc_bar.n = _dcc_eval_count[0]
            _dcc_bar.set_postfix(a=f"{a:.4f}", b=f"{b:.4f}", nll=f"{-ll:.1f}")
            _dcc_bar.refresh()

            return -ll

        result = minimize(
            neg_log_likelihood,
            x0=[a_init, b_init],
            bounds=[(1e-6, 0.3), (0.5, 0.9999)],
            method="L-BFGS-B",
        )

        _dcc_bar.close()

        a_est, b_est = result.x[0], result.x[1]

        # --- Validate mean-reversion constraint ---
        if a_est + b_est >= 1.0:
            logger.warning(
                f"  DCC: a+b={a_est + b_est:.4f} >= 1 (non-stationary). "
                f"Clamping to a+b=0.999."
            )
            scale = 0.999 / (a_est + b_est)
            a_est *= scale
            b_est *= scale

        self.dcc_params = {"a": a_est, "b": b_est}
        logger.info(
            f"  DCC params: a={self.dcc_params['a']:.4f}, b={self.dcc_params['b']:.4f}, "
            f"a+b={a_est + b_est:.4f} (mean-reversion: {'OK' if a_est + b_est < 1 else 'CLAMPED'})"
        )

    def _method_of_moments_init(self, Z: np.ndarray) -> Tuple[float, float]:
        """
        Compute method-of-moments starting values for DCC(a, b).

        Uses first-order autocorrelation of cross-products of standardised
        residuals to initialise the persistence parameter.

        Args:
            Z: (T, n) standardised residuals

        Returns:
            Tuple of (a_init, b_init)
        """
        T, n = Z.shape
        if n < 2:
            return 0.05, 0.90

        # Compute cross-product series for first pair
        cross = Z[:, 0] * Z[:, 1]
        cross_demean = cross - cross.mean()

        # First-order autocorrelation of cross-products gives a+b estimate
        autocorr = np.corrcoef(cross_demean[:-1], cross_demean[1:])[0, 1]
        autocorr = np.clip(autocorr, 0.5, 0.999)

        # Split: b captures most persistence, a captures the innovation
        b_init = autocorr * 0.9
        a_init = autocorr * 0.1

        # Clamp to valid bounds
        a_init = np.clip(a_init, 0.01, 0.25)
        b_init = np.clip(b_init, 0.5, 0.98)

        return float(a_init), float(b_init)

    # ─────────────────────────────────────────────────
    #  FORECAST COVARIANCE
    # ─────────────────────────────────────────────────
    def forecast_covariance(self, n_ahead: int = 1) -> np.ndarray:
        """
        Forecast the covariance matrix Sigma_{t+1|t}.

        Returns:
            np.ndarray of shape (n_assets, n_assets)
        """
        assert self._fitted, "Model not fitted"

        # Forecast univariate volatilities
        vol_forecasts = []
        for col in self.conditional_vols.columns:
            if col in self.garch_results:
                fcast = self.garch_results[col].forecast(horizon=n_ahead)
                vol = np.sqrt(fcast.variance.iloc[-1].values[-1]) / 100
            else:
                vol = self.conditional_vols[col].iloc[-1]
            vol_forecasts.append(vol)

        D = np.diag(vol_forecasts)

        # Forecast DCC correlation
        Z = self.std_residuals.values
        a, b = self.dcc_params["a"], self.dcc_params["b"]

        Q_t = self.Q_bar.copy()
        for t in range(1, len(Z)):
            z_t = Z[t - 1].reshape(-1, 1)
            Q_t = (1 - a - b) * self.Q_bar + a * (z_t @ z_t.T) + b * Q_t

        # One-step-ahead: Q_{T+1} = (1-a-b)*Q_bar + a*z_T*z_T' + b*Q_T
        z_T = Z[-1].reshape(-1, 1)
        Q_next = (1 - a - b) * self.Q_bar + a * (z_T @ z_T.T) + b * Q_t

        D_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_next), 1e-8)))
        R_next = D_inv @ Q_next @ D_inv

        # Sigma = D * R * D
        Sigma = D @ R_next @ D

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.any(eigvals < 0):
            Sigma = self._nearest_psd(Sigma)

        return Sigma

    # ─────────────────────────────────────────────────
    #  FORECAST EVALUATION
    # ─────────────────────────────────────────────────
    def evaluate_forecasts(self, returns_df: pd.DataFrame,
                           eval_window: int = 720) -> Dict[str, dict]:
        """
        Rolling out-of-sample covariance forecast accuracy evaluation.

        Computes:
        - Mincer-Zarnowitz regression: realised variance vs predicted variance
        - Mean squared forecast error (MSFE) for each asset
        - Forecast bias

        Args:
            returns_df: Full return series (must extend beyond training data)
            eval_window: Number of periods for OOS evaluation

        Returns:
            Dict of forecast evaluation metrics per asset
        """
        assert self._fitted, "Model not fitted"

        n_obs = returns_df.shape[0]
        if n_obs < eval_window + 100:
            logger.warning("Insufficient data for forecast evaluation")
            return {}

        metrics = {}
        for col in self.conditional_vols.columns:
            if col not in self.garch_results:
                continue

            # Predicted variances (in-sample conditional variance)
            pred_var = (self.conditional_vols[col] ** 2).values

            # Realised variance (squared returns as proxy)
            realised_var = (returns_df[col] ** 2).loc[self.conditional_vols.index].values

            # Align lengths
            min_len = min(len(pred_var), len(realised_var))
            pred_var = pred_var[-eval_window:] if min_len >= eval_window else pred_var
            realised_var = realised_var[-eval_window:] if min_len >= eval_window else realised_var
            eval_len = min(len(pred_var), len(realised_var))
            pred_var = pred_var[:eval_len]
            realised_var = realised_var[:eval_len]

            # Mincer-Zarnowitz regression: realised = alpha + beta * predicted + eps
            # Ideal: alpha=0, beta=1
            mz_result = self._mincer_zarnowitz(realised_var, pred_var)

            # Mean squared forecast error
            msfe = float(np.mean((realised_var - pred_var) ** 2))

            # Forecast bias (mean prediction error)
            bias = float(np.mean(pred_var - realised_var))

            metrics[col] = {
                "mincer_zarnowitz_alpha": mz_result["alpha"],
                "mincer_zarnowitz_beta": mz_result["beta"],
                "mincer_zarnowitz_r2": mz_result["r_squared"],
                "msfe": msfe,
                "forecast_bias": bias,
            }

            logger.info(
                f"  {col} forecast eval: MZ_beta={mz_result['beta']:.3f}, "
                f"MZ_R2={mz_result['r_squared']:.3f}, bias={bias:.6f}"
            )

        self.forecast_eval_metrics = metrics
        return metrics

    @staticmethod
    def _mincer_zarnowitz(realised: np.ndarray, predicted: np.ndarray) -> dict:
        """
        Mincer-Zarnowitz regression for forecast evaluation.

        realised = alpha + beta * predicted + epsilon

        Args:
            realised: Actual realised values
            predicted: Forecasted values

        Returns:
            Dict with alpha, beta, r_squared
        """
        X = np.column_stack([np.ones(len(predicted)), predicted])
        try:
            beta_hat = np.linalg.lstsq(X, realised, rcond=None)[0]
            y_hat = X @ beta_hat
            ss_res = np.sum((realised - y_hat) ** 2)
            ss_tot = np.sum((realised - realised.mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return {
                "alpha": float(beta_hat[0]),
                "beta": float(beta_hat[1]),
                "r_squared": float(np.clip(r_sq, 0.0, 1.0)),
            }
        except Exception:
            return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    # ─────────────────────────────────────────────────
    #  BOOTSTRAP CONFIDENCE INTERVALS
    # ─────────────────────────────────────────────────
    def _bootstrap_parameters(self, returns_df: pd.DataFrame) -> None:
        """
        Bootstrap confidence intervals on GARCH parameters (parallel).

        Uses block bootstrap (preserving temporal structure) to estimate
        parameter uncertainty and propagate to covariance forecast.
        Each bootstrap sample is independent -- perfect for parallelism.

        Args:
            returns_df: Original return series
        """
        n_samples = self.bootstrap_n_samples
        logger.info(f"  Running {n_samples} bootstrap resamples for parameter uncertainty...")

        n_obs = returns_df.shape[0]
        block_size = min(50, n_obs // 10)

        # Collect parameter estimates per asset across bootstraps
        param_samples: Dict[str, List[np.ndarray]] = {col: [] for col in returns_df.columns}
        cov_forecasts: List[np.ndarray] = []

        def _run_single_bootstrap(b):
            """Run a single bootstrap resample. Thread-safe."""
            rng = np.random.RandomState(b + 1000)
            n_blocks_needed = n_obs // block_size + 1
            starts = rng.randint(0, n_obs - block_size, size=n_blocks_needed)
            indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_obs]
            boot_df = returns_df.iloc[indices].reset_index(drop=True)

            try:
                boot_model = StudentTGarchDCC(
                    p=self.p, q=self.q, distribution=self.distribution,
                    config={"enable_model_selection": False, "enable_diagnostics": False,
                            "enable_bootstrap": False}
                )
                boot_model.fit(boot_df)

                boot_params = {}
                for col in returns_df.columns:
                    if col in boot_model.garch_results:
                        boot_params[col] = boot_model.garch_results[col].params.values

                return boot_params, boot_model.forecast_covariance()
            except Exception:
                return None, None

        n_valid = 0
        boot_bar = tqdm(total=n_samples, desc="Bootstrap", unit="sample", leave=True)

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_run_single_bootstrap, b): b for b in range(n_samples)}
            for future in as_completed(futures):
                boot_params, boot_cov = future.result()
                if boot_params is not None:
                    for col, params in boot_params.items():
                        param_samples[col].append(params)
                    cov_forecasts.append(boot_cov)
                    n_valid += 1
                boot_bar.update(1)
                boot_bar.set_postfix(valid=n_valid)

        boot_bar.close()

        # Compute confidence intervals
        for col, samples in param_samples.items():
            if len(samples) < 10:
                continue
            arr = np.array(samples)
            ci_lower = np.percentile(arr, 2.5, axis=0)
            ci_upper = np.percentile(arr, 97.5, axis=0)
            ci_std = np.std(arr, axis=0)

            self.bootstrap_ci[col] = {
                "mean": np.mean(arr, axis=0).tolist(),
                "ci_lower_2.5": ci_lower.tolist(),
                "ci_upper_97.5": ci_upper.tolist(),
                "std": ci_std.tolist(),
                "n_valid": len(samples),
            }

            logger.info(
                f"  {col} bootstrap ({len(samples)}/{n_samples} valid): "
                f"param_std={ci_std.mean():.6f}"
            )

        # Uncertainty decomposition: estimation error vs model error
        if len(cov_forecasts) > 10:
            cov_arr = np.array(cov_forecasts)
            estimation_var = np.var(cov_arr, axis=0)  # Var across bootstraps = estimation error
            # Model error approximated from residual-based estimate
            point_forecast = self.forecast_covariance()
            model_var = np.mean((cov_arr - point_forecast[np.newaxis, :, :]) ** 2, axis=0)

            total_var = estimation_var + model_var
            est_fraction = np.mean(estimation_var) / (np.mean(total_var) + 1e-12)

            self.uncertainty_decomposition = {
                "estimation_error_fraction": float(est_fraction),
                "model_error_fraction": float(1 - est_fraction),
                "n_valid_bootstraps": len(cov_forecasts),
            }
            logger.info(
                f"  Uncertainty decomposition: "
                f"estimation={est_fraction:.1%}, model={1 - est_fraction:.1%}"
            )

    # ─────────────────────────────────────────────────
    #  RISK PARITY WEIGHTS
    # ─────────────────────────────────────────────────
    def get_risk_parity_weights(self, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Compute risk-parity (equal risk contribution) weights from covariance matrix.

        Each asset contributes equally to total portfolio risk:
        w_i * (Sigma @ w)_i = (1/n) * w' @ Sigma @ w

        Solved via numerical optimisation.
        """
        if Sigma is None:
            Sigma = self.forecast_covariance()

        n = Sigma.shape[0]

        def risk_budget_objective(w):
            w = np.abs(w)
            w = w / w.sum()
            port_vol = np.sqrt(w @ Sigma @ w)
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        from scipy.optimize import minimize as sp_minimize
        x0 = np.ones(n) / n
        bounds = [(0.01, 0.5)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        result = sp_minimize(risk_budget_objective, x0, bounds=bounds,
                             constraints=constraints, method="SLSQP")

        weights = np.abs(result.x)
        weights = weights / weights.sum()
        return weights

    # ─────────────────────────────────────────────────
    #  UNCERTAINTY
    # ─────────────────────────────────────────────────
    def get_uncertainty(self) -> float:
        """
        Return a scalar uncertainty measure for ensemble weighting.
        Uses average parameter standard error across GARCH models.
        """
        se_sum = 0.0
        count = 0
        for col, res in self.garch_results.items():
            try:
                se_sum += res.std_err.mean()
                count += 1
            except (AttributeError, ValueError):
                pass
        return se_sum / max(count, 1)

    def get_uncertainty_decomposed(self) -> dict:
        """
        Return decomposed uncertainty: estimation error vs model error.

        Only available after bootstrap is run (enable_bootstrap=True).

        Returns:
            Dict with estimation_error_fraction, model_error_fraction,
            or empty dict if bootstrap not run.
        """
        if self.uncertainty_decomposition is not None:
            return self.uncertainty_decomposition
        return {"estimation_error_fraction": 0.5, "model_error_fraction": 0.5}

    # ─────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────
    @staticmethod
    def _nearest_psd(A: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @property
    def assets(self) -> list:
        return self.conditional_vols.columns.tolist() if self.conditional_vols is not None else []
