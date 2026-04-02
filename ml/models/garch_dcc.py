"""
Model 1: Student-t GARCH(1,1)-DCC for dynamic covariance estimation.

Mathematical specification:
  Univariate: r_{i,t} = mu_i + sigma_{i,t} * z_{i,t},  z ~ t(nu)
              sigma^2_{i,t} = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

  DCC: Q_t = (1-a-b)*Q_bar + a*z_{t-1}*z'_{t-1} + b*Q_{t-1}
       R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

  Covariance: Sigma_t = D_t * R_t * D_t   where D_t = diag(sigma_1t,...,sigma_nt)
"""
import logging
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class StudentTGarchDCC:
    """
    Student-t GARCH(1,1) with Dynamic Conditional Correlation.

    Estimates time-varying covariance matrix for portfolio optimisation.
    Student-t innovations capture fat tails characteristic of crypto returns.
    """

    def __init__(self, p: int = 1, q: int = 1, distribution: str = "studentst"):
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

        # ── Step 1: Fit univariate GARCH for each asset ──
        conditional_vols = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)
        std_resids = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)

        for col in assets:
            series = returns_df[col].dropna() * 100  # Scale to percentage for numerical stability

            model = arch_model(
                series,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.distribution,
            )

            try:
                result = model.fit(disp="off", show_warning=False)
                self.garch_models[col] = model
                self.garch_results[col] = result

                # Extract conditional volatility and standardised residuals
                conditional_vols[col] = result.conditional_volatility / 100  # Back to decimal
                std_resids[col] = result.std_resid

                nu = result.params.get("nu", 30)  # Degrees of freedom
                logger.info(
                    f"  {col}: omega={result.params['omega']:.6f}, "
                    f"alpha={result.params.get('alpha[1]', 0):.4f}, "
                    f"beta={result.params.get('beta[1]', 0):.4f}, "
                    f"nu={nu:.1f}"
                )
            except Exception as e:
                logger.warning(f"  {col}: GARCH fit failed ({e}), using EWMA fallback")
                ewma_vol = series.ewm(span=60).std() / 100
                conditional_vols[col] = ewma_vol
                std_resids[col] = (series / 100) / ewma_vol

        self.conditional_vols = conditional_vols.dropna()
        self.std_residuals = std_resids.loc[self.conditional_vols.index].dropna()

        # ── Step 2: Estimate DCC parameters ──
        self._fit_dcc()

        self._fitted = True
        return self

    def _fit_dcc(self):
        """
        Estimate DCC parameters (a, b) via quasi-maximum likelihood.

        Q_t = (1-a-b)*Q_bar + a*z_{t-1}*z'_{t-1} + b*Q_{t-1}
        """
        Z = self.std_residuals.values  # (T, n)
        T, n = Z.shape

        # Unconditional correlation matrix
        self.Q_bar = np.corrcoef(Z.T)

        # QML estimation of a, b
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

            return -ll

        result = minimize(
            neg_log_likelihood,
            x0=[0.05, 0.90],
            bounds=[(1e-6, 0.3), (0.5, 0.9999)],
            method="L-BFGS-B",
        )

        self.dcc_params = {"a": result.x[0], "b": result.x[1]}
        logger.info(f"  DCC params: a={self.dcc_params['a']:.4f}, b={self.dcc_params['b']:.4f}")

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
            except:
                pass
        return se_sum / max(count, 1)

    @staticmethod
    def _nearest_psd(A):
        """Find nearest positive semi-definite matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @property
    def assets(self):
        return self.conditional_vols.columns.tolist() if self.conditional_vols is not None else []
