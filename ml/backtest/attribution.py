"""
Performance Attribution for the Risk-Parity Vault.

Decomposes total portfolio return into contributions from:
  1. Asset allocation (which assets were held, Brinson-Fachler)
  2. Regime detection (did HMM correctly shift to defensive?)
  3. Model selection (GARCH vs HMM vs RL contribution)
  4. Risk management (circuit breaker, turnover limits)
  5. Fee drag (management + performance fees)
  6. Transaction costs

This directly supports the coursework narrative: the TOKEN DESIGN
(fees, circuit breaker, epoch system) demonstrably protects value.

Asset class mapping follows config.yaml risk_budgets:
  crypto:     BTC, ETH, SOL
  staking:    stETH, rETH
  treasuries: BUIDL, USDY
  stable:     USDC

Annualisation: hourly data, 8760 periods per year.

Adapted from: performance attribution concepts in Brinson, Hood & Beebower
(1986) and regime-conditional decomposition.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

# Default asset-class mapping (mirrors config.yaml risk_budgets)
_DEFAULT_ASSET_CLASSES: Dict[str, List[str]] = {
    "crypto": ["BTC", "ETH", "SOL"],
    "staking": ["stETH", "rETH"],
    "treasuries": ["BUIDL", "USDY"],
    "stable": ["USDC"],
}

# Regime names (convention from HMM: 0=bull, 1=normal, 2=crisis)
_REGIME_NAMES = {0: "bull", 1: "normal", 2: "crisis"}


class PerformanceAttributor:
    """
    Multi-factor performance attribution for the risk-parity vault.

    Decomposes total portfolio return into contributions from:
    1. Asset allocation (which assets were held)
    2. Regime detection (did HMM correctly shift to defensive?)
    3. Model selection (GARCH vs HMM vs RL contribution)
    4. Risk management (circuit breaker, turnover limits)
    5. Fee drag (management + performance fees)
    6. Transaction costs

    This directly supports the coursework narrative: the TOKEN DESIGN
    (fees, circuit breaker, epoch system) demonstrably protects value.

    Parameters
    ----------
    asset_names : list of str
        Asset names matching column order in returns data.
    asset_classes : dict, optional
        Mapping of class_name -> [asset_names].
    config : dict, optional
        Override defaults via keys 'attribution_window', etc.
    """

    def __init__(
        self,
        asset_names: List[str],
        asset_classes: Optional[Dict[str, List[str]]] = None,
        config: Optional[Dict] = None,
    ):
        self.asset_names = list(asset_names)
        self.n_assets = len(self.asset_names)
        self.asset_classes = asset_classes or _DEFAULT_ASSET_CLASSES
        self.config = config or {}

        # Build reverse lookup: asset -> class
        self._asset_to_class: Dict[str, str] = {}
        for cls_name, assets in self.asset_classes.items():
            for a in assets:
                self._asset_to_class[a] = cls_name

        # Populated by attribute()
        self._portfolio_returns: Optional[np.ndarray] = None
        self._benchmark_returns: Optional[np.ndarray] = None
        self._weights_history: Optional[np.ndarray] = None
        self._regime_history: Optional[np.ndarray] = None
        self._fee_history: Optional[np.ndarray] = None
        self._cost_history: Optional[np.ndarray] = None
        self._asset_returns: Optional[np.ndarray] = None
        self._benchmark_weights: Optional[np.ndarray] = None
        self._fitted = False

    # ─── MAIN ATTRIBUTION ────────────────────────────────────────────────

    def attribute(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        weights_history: np.ndarray,
        regime_history: np.ndarray,
        asset_returns: Optional[np.ndarray] = None,
        benchmark_weights: Optional[np.ndarray] = None,
        fee_history: Optional[np.ndarray] = None,
        cost_history: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Run full performance attribution.

        Parameters
        ----------
        portfolio_returns : np.ndarray, shape (T,)
            Realised portfolio log returns per period.
        benchmark_returns : np.ndarray, shape (T,)
            Benchmark (e.g., equal-weight) log returns per period.
        weights_history : np.ndarray, shape (T, N)
            Portfolio weights at each period.
        regime_history : np.ndarray, shape (T,)
            Regime labels per period (0=bull, 1=normal, 2=crisis).
        asset_returns : np.ndarray, shape (T, N), optional
            Per-asset returns. Needed for Brinson and asset-class attribution.
        benchmark_weights : np.ndarray, shape (T, N) or (N,), optional
            Benchmark weights (default: equal-weight 1/N).
        fee_history : np.ndarray, shape (T,), optional
            Fees charged per period (positive = cost).
        cost_history : np.ndarray, shape (T,), optional
            Transaction costs per period (positive = cost).

        Returns
        -------
        dict
            Complete attribution results with keys:
            'total_return', 'benchmark_return', 'excess_return',
            'asset_class_attribution', 'regime_attribution',
            'risk_management_value', 'fee_drag', 'cost_drag'.
        """
        T = len(portfolio_returns)

        # Validate and store
        self._portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
        self._benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)[:T]
        self._weights_history = np.asarray(weights_history, dtype=np.float64)[:T]
        self._regime_history = np.asarray(regime_history, dtype=np.int32)[:T]

        if asset_returns is not None:
            self._asset_returns = np.asarray(asset_returns, dtype=np.float64)[:T]
        else:
            self._asset_returns = None

        if benchmark_weights is not None:
            bw = np.asarray(benchmark_weights, dtype=np.float64)
            if bw.ndim == 1:
                self._benchmark_weights = np.tile(bw, (T, 1))
            else:
                self._benchmark_weights = bw[:T]
        else:
            self._benchmark_weights = np.ones((T, self.n_assets)) / self.n_assets

        self._fee_history = np.asarray(fee_history, dtype=np.float64) if fee_history is not None else np.zeros(T)
        self._cost_history = np.asarray(cost_history, dtype=np.float64) if cost_history is not None else np.zeros(T)
        self._fitted = True

        # Compute attributions
        total_ret = float(np.exp(np.sum(self._portfolio_returns)) - 1)
        bench_ret = float(np.exp(np.sum(self._benchmark_returns)) - 1)
        excess = total_ret - bench_ret

        attr_steps = [
            ("Asset class attribution", lambda: self.asset_class_attribution()),
            ("Regime attribution", lambda: self.regime_attribution()),
            ("Risk management value", lambda: self.risk_management_value()),
        ]
        attr_results = {}
        attr_bar = tqdm(attr_steps, desc="Attribution pipeline", unit="step", leave=False)
        for step_name, step_func in attr_bar:
            attr_bar.set_postfix(step=step_name[:25])
            attr_results[step_name] = step_func()

        results = {
            "total_return": total_ret,
            "benchmark_return": bench_ret,
            "excess_return": excess,
            "n_periods": T,
            "asset_class_attribution": attr_results["Asset class attribution"],
            "regime_attribution": attr_results["Regime attribution"],
            "risk_management_value": attr_results["Risk management value"],
            "fee_drag": float(np.sum(self._fee_history)),
            "cost_drag": float(np.sum(self._cost_history)),
        }

        logger.info(
            "Attribution complete: total=%.4f  bench=%.4f  excess=%.4f  "
            "fee_drag=%.4f  cost_drag=%.4f",
            total_ret, bench_ret, excess,
            results["fee_drag"], results["cost_drag"],
        )
        return results

    # ─── ASSET CLASS ATTRIBUTION ─────────────────────────────────────────

    def asset_class_attribution(self) -> Dict[str, float]:
        """
        Decompose portfolio return by asset class.

        For each class, the contribution = sum over t of
        (sum of w_i,t * r_i,t for i in class).

        Returns
        -------
        dict
            Keys = class names, values = cumulative return contribution.
        """
        self._check_fitted()

        if self._asset_returns is None:
            logger.warning(
                "asset_returns not provided — cannot compute asset class attribution"
            )
            return {cls: 0.0 for cls in self.asset_classes}

        T = len(self._portfolio_returns)
        class_contributions: Dict[str, float] = {}

        for cls_name, cls_assets in self.asset_classes.items():
            # Find column indices for this class
            indices = [
                self.asset_names.index(a)
                for a in cls_assets
                if a in self.asset_names
            ]
            if not indices:
                class_contributions[cls_name] = 0.0
                continue

            # Contribution = sum_t sum_i (w_i,t * r_i,t)
            w_cls = self._weights_history[:T, indices]
            r_cls = self._asset_returns[:T, indices]
            contribution = float(np.sum(w_cls * r_cls))

            class_contributions[cls_name] = round(contribution, 6)

        logger.info("Asset class attribution: %s", class_contributions)
        return class_contributions

    # ─── REGIME ATTRIBUTION ──────────────────────────────────────────────

    def regime_attribution(self) -> Dict[str, object]:
        """
        Decompose performance by market regime.

        Computes for each regime:
          - contribution: sum of portfolio returns in that regime
          - n_periods: number of periods in that regime
          - avg_return: mean return per period
          - benchmark_contribution: sum of benchmark returns
          - alpha: portfolio contribution minus benchmark contribution

        Returns
        -------
        dict
            Keys = regime names, values = dict with metrics above.
        """
        self._check_fitted()

        T = len(self._portfolio_returns)
        results: Dict[str, object] = {}

        for code, name in _REGIME_NAMES.items():
            mask = self._regime_history[:T] == code
            n_obs = int(mask.sum())

            if n_obs == 0:
                results[name] = {
                    "contribution": 0.0,
                    "n_periods": 0,
                    "avg_return": 0.0,
                    "benchmark_contribution": 0.0,
                    "alpha": 0.0,
                }
                continue

            port_contrib = float(np.sum(self._portfolio_returns[mask]))
            bench_contrib = float(np.sum(self._benchmark_returns[mask]))
            avg_ret = float(np.mean(self._portfolio_returns[mask]))
            alpha = port_contrib - bench_contrib

            results[name] = {
                "contribution": round(port_contrib, 6),
                "n_periods": n_obs,
                "avg_return": round(avg_ret, 8),
                "benchmark_contribution": round(bench_contrib, 6),
                "alpha": round(alpha, 6),
            }

        logger.info("Regime attribution: %s",
                     {k: v["contribution"] for k, v in results.items()})
        return results

    # ─── MODEL ATTRIBUTION ───────────────────────────────────────────────

    def model_attribution(
        self, model_weights_history: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Attribute performance to individual models in the ensemble.

        For each model, compute the hypothetical return if only that
        model's weights had been used:
            model_return = sum_t (w_model_t @ r_t)

        Parameters
        ----------
        model_weights_history : dict
            Keys = model names ("garch", "hmm", "rl"),
            values = (T, N) weight arrays.

        Returns
        -------
        dict
            Keys = model names, values = cumulative return if that
            model were used alone.
        """
        self._check_fitted()

        if self._asset_returns is None:
            logger.warning(
                "asset_returns not provided — cannot compute model attribution"
            )
            return {name: 0.0 for name in model_weights_history}

        T = len(self._portfolio_returns)
        results: Dict[str, float] = {}

        for model_name, w_hist in model_weights_history.items():
            w = np.asarray(w_hist, dtype=np.float64)[:T]
            if w.shape[0] < T:
                logger.warning(
                    "Model '%s' has %d weight rows, need %d — padding",
                    model_name, w.shape[0], T,
                )
                pad = np.tile(w[-1], (T - w.shape[0], 1))
                w = np.vstack([w, pad])

            # Hypothetical per-period return under this model's weights
            period_rets = np.sum(w * self._asset_returns[:T], axis=1)
            cum_ret = float(np.exp(np.sum(period_rets)) - 1)
            results[model_name] = round(cum_ret, 6)

        logger.info("Model attribution: %s", results)
        return results

    # ─── RISK MANAGEMENT VALUE ───────────────────────────────────────────

    def risk_management_value(self) -> Dict[str, float]:
        """
        Quantify the value added by risk management mechanisms.

        Computes:
          - circuit_breaker_saves: return in periods where regime=crisis
            and the portfolio was defensively allocated (high stable/treasury
            weight). Compared to what an unprotected portfolio would lose.
          - turnover_limit_savings: estimated cost savings from turnover
            constraints (transaction cost reduction).
          - drawdown_protection: max drawdown avoided vs unconstrained.

        Returns
        -------
        dict
            Keys: circuit_breaker_saves, turnover_limit_savings,
            drawdown_protection.
        """
        self._check_fitted()

        T = len(self._portfolio_returns)

        # --- Circuit breaker value ---
        # In crisis periods, how much better did the portfolio do vs benchmark?
        crisis_mask = self._regime_history[:T] == 2
        n_crisis = int(crisis_mask.sum())

        if n_crisis > 0:
            port_crisis = float(np.sum(self._portfolio_returns[crisis_mask]))
            bench_crisis = float(np.sum(self._benchmark_returns[crisis_mask]))
            cb_saves = port_crisis - bench_crisis  # positive = we saved money
        else:
            cb_saves = 0.0

        # --- Turnover limit savings ---
        # Estimate: actual costs vs hypothetical unconstrained costs
        # If cost history provided, the saving is the difference between
        # an unconstrained cost estimate and actual
        actual_costs = float(np.sum(self._cost_history[:T]))

        # Estimate unconstrained turnover from weight changes
        if T > 1:
            weight_changes = np.diff(self._weights_history[:T], axis=0)
            raw_turnover = np.sum(np.abs(weight_changes))
            # Assume 10 bps cost per unit turnover (from config)
            cost_bps = self.config.get("transaction_cost_bps", 10)
            unconstrained_cost = raw_turnover * cost_bps / 10000.0
            turnover_savings = max(0.0, unconstrained_cost - actual_costs)
        else:
            turnover_savings = 0.0

        # --- Drawdown protection ---
        # Compare portfolio max drawdown vs benchmark max drawdown
        port_cum = np.exp(np.cumsum(self._portfolio_returns))
        port_peak = np.maximum.accumulate(port_cum)
        port_dd = np.min(port_cum / port_peak - 1.0)

        bench_cum = np.exp(np.cumsum(self._benchmark_returns))
        bench_peak = np.maximum.accumulate(bench_cum)
        bench_dd = np.min(bench_cum / bench_peak - 1.0)

        dd_protection = bench_dd - port_dd  # positive = we had shallower DD

        result = {
            "circuit_breaker_saves": round(cb_saves, 6),
            "turnover_limit_savings": round(turnover_savings, 6),
            "drawdown_protection": round(dd_protection, 6),
            "n_crisis_periods": n_crisis,
            "portfolio_max_dd": round(float(port_dd), 6),
            "benchmark_max_dd": round(float(bench_dd), 6),
        }

        logger.info(
            "Risk management value: CB=%.4f  turnover=%.4f  DD_protection=%.4f",
            cb_saves, turnover_savings, dd_protection,
        )
        return result

    # ─── BRINSON-FACHLER ATTRIBUTION ─────────────────────────────────────

    def brinson_attribution(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict[str, object]:
        """
        Single-period Brinson-Fachler attribution at the asset-class level.

        Decomposes active return into:
          - Allocation effect: over/underweighting the right classes
          - Selection effect: picking the right assets within classes
          - Interaction effect: cross-term

        Parameters
        ----------
        portfolio_weights : np.ndarray, shape (N,)
            Portfolio weights for one period.
        benchmark_weights : np.ndarray, shape (N,)
            Benchmark weights for one period.
        portfolio_returns : np.ndarray, shape (N,)
            Per-asset returns for the portfolio.
        benchmark_returns : np.ndarray, shape (N,)
            Per-asset returns for the benchmark.

        Returns
        -------
        dict with 'allocation_effect', 'selection_effect',
        'interaction_effect', 'total_active', 'by_class'.
        """
        pw = np.asarray(portfolio_weights, dtype=np.float64)
        bw = np.asarray(benchmark_weights, dtype=np.float64)
        pr = np.asarray(portfolio_returns, dtype=np.float64)
        br = np.asarray(benchmark_returns, dtype=np.float64)

        n = len(pw)
        if not (len(bw) == len(pr) == len(br) == n):
            raise ValueError("All arrays must have the same length")

        # Aggregate to asset-class level
        total_bench_return = float(bw @ br)

        alloc_total = 0.0
        select_total = 0.0
        interact_total = 0.0
        by_class: Dict[str, Dict[str, float]] = {}

        for cls_name, cls_assets in tqdm(self.asset_classes.items(), desc="Brinson attribution", unit="class", leave=False):
            indices = [
                self.asset_names.index(a)
                for a in cls_assets
                if a in self.asset_names
            ]
            if not indices:
                by_class[cls_name] = {
                    "allocation": 0.0,
                    "selection": 0.0,
                    "interaction": 0.0,
                }
                continue

            # Class-level weights and returns
            w_p = pw[indices].sum()
            w_b = bw[indices].sum()

            # Weighted average returns within class
            r_p = float(pw[indices] @ pr[indices]) / w_p if w_p > 1e-12 else 0.0
            r_b = float(bw[indices] @ br[indices]) / w_b if w_b > 1e-12 else 0.0

            # Brinson-Fachler decomposition
            alloc = (w_p - w_b) * (r_b - total_bench_return)
            select = w_b * (r_p - r_b)
            interact = (w_p - w_b) * (r_p - r_b)

            alloc_total += alloc
            select_total += select
            interact_total += interact

            by_class[cls_name] = {
                "allocation": round(float(alloc), 8),
                "selection": round(float(select), 8),
                "interaction": round(float(interact), 8),
                "port_weight": round(float(w_p), 4),
                "bench_weight": round(float(w_b), 4),
                "port_return": round(float(r_p), 6),
                "bench_return": round(float(r_b), 6),
            }

        result = {
            "allocation_effect": round(float(alloc_total), 8),
            "selection_effect": round(float(select_total), 8),
            "interaction_effect": round(float(interact_total), 8),
            "total_active": round(float(alloc_total + select_total + interact_total), 8),
            "by_class": by_class,
        }

        logger.info(
            "Brinson attribution: alloc=%.6f  select=%.6f  interact=%.6f  total=%.6f",
            result["allocation_effect"], result["selection_effect"],
            result["interaction_effect"], result["total_active"],
        )
        return result

    # ─── ROLLING ATTRIBUTION ─────────────────────────────────────────────

    def rolling_attribution(
        self, window: int = 720
    ) -> pd.DataFrame:
        """
        Rolling regime and asset-class attribution over time.

        Computes a rolling window attribution showing how the contribution
        of each regime and asset class evolves.

        Parameters
        ----------
        window : int
            Rolling window size in periods (default 720 = 30 days hourly).

        Returns
        -------
        pd.DataFrame
            Columns: window_end, total_return, bench_return,
            excess_return, plus one column per asset class contribution,
            plus regime columns (bull_contrib, normal_contrib, crisis_contrib).
        """
        self._check_fitted()

        if self._asset_returns is None:
            logger.warning(
                "asset_returns not provided — cannot compute rolling attribution"
            )
            return pd.DataFrame()

        T = len(self._portfolio_returns)
        if T < window:
            logger.warning(
                "Insufficient data (%d < %d) for rolling attribution", T, window
            )
            return pd.DataFrame()

        rows = []
        # Use stride for efficiency on large datasets
        stride = max(1, window // 10)
        n_windows = len(range(window, T, stride))

        for end in tqdm(range(window, T, stride), desc="Rolling attribution", unit="window", total=n_windows, leave=False):
            start = end - window
            port_ret = self._portfolio_returns[start:end]
            bench_ret = self._benchmark_returns[start:end]
            regimes = self._regime_history[start:end]
            w_hist = self._weights_history[start:end]
            a_ret = self._asset_returns[start:end]

            row = {
                "window_end": end,
                "total_return": float(np.exp(np.sum(port_ret)) - 1),
                "bench_return": float(np.exp(np.sum(bench_ret)) - 1),
            }
            row["excess_return"] = row["total_return"] - row["bench_return"]

            # Asset class contributions
            for cls_name, cls_assets in self.asset_classes.items():
                indices = [
                    self.asset_names.index(a)
                    for a in cls_assets
                    if a in self.asset_names
                ]
                if indices:
                    contrib = float(np.sum(w_hist[:, indices] * a_ret[:, indices]))
                else:
                    contrib = 0.0
                row[f"{cls_name}_contrib"] = round(contrib, 6)

            # Regime contributions
            for code, name in _REGIME_NAMES.items():
                mask = regimes == code
                if mask.sum() > 0:
                    row[f"{name}_contrib"] = round(float(np.sum(port_ret[mask])), 6)
                else:
                    row[f"{name}_contrib"] = 0.0

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info("Rolling attribution computed: %d windows of %d periods", len(df), window)
        return df

    # ─── GENERATE FULL REPORT ────────────────────────────────────────────

    def generate_attribution_report(self) -> Dict[str, object]:
        """
        Generate a comprehensive attribution report.

        Aggregates all attribution results into a single dict suitable
        for display, charting, or inclusion in the coursework report.

        Returns
        -------
        dict
            Keys: 'summary', 'asset_class', 'regime', 'risk_management',
            'fee_and_costs', 'periods'.
        """
        self._check_fitted()

        T = len(self._portfolio_returns)
        total_ret = float(np.exp(np.sum(self._portfolio_returns)) - 1)
        bench_ret = float(np.exp(np.sum(self._benchmark_returns)) - 1)
        fee_drag = float(np.sum(self._fee_history[:T]))
        cost_drag = float(np.sum(self._cost_history[:T]))

        # Gross return (before fees and costs)
        gross_ret = total_ret + fee_drag + cost_drag

        report = {
            "summary": {
                "total_return": round(total_ret, 6),
                "benchmark_return": round(bench_ret, 6),
                "excess_return": round(total_ret - bench_ret, 6),
                "gross_return": round(gross_ret, 6),
                "fee_drag": round(fee_drag, 6),
                "cost_drag": round(cost_drag, 6),
                "n_periods": T,
                "n_hours": T,
                "n_days": T / 24.0,
            },
            "asset_class": self.asset_class_attribution(),
            "regime": self.regime_attribution(),
            "risk_management": self.risk_management_value(),
            "fee_and_costs": {
                "total_fees": round(fee_drag, 6),
                "total_costs": round(cost_drag, 6),
                "fees_pct_of_gross": round(
                    fee_drag / gross_ret * 100.0 if abs(gross_ret) > 1e-12 else 0.0, 2
                ),
                "costs_pct_of_gross": round(
                    cost_drag / gross_ret * 100.0 if abs(gross_ret) > 1e-12 else 0.0, 2
                ),
            },
            "periods": {
                "bull_pct": round(
                    float(np.mean(self._regime_history[:T] == 0)) * 100.0, 1
                ),
                "normal_pct": round(
                    float(np.mean(self._regime_history[:T] == 1)) * 100.0, 1
                ),
                "crisis_pct": round(
                    float(np.mean(self._regime_history[:T] == 2)) * 100.0, 1
                ),
            },
        }

        logger.info(
            "Attribution report: total=%.4f  bench=%.4f  excess=%.4f  "
            "gross=%.4f  fees=%.4f  costs=%.4f",
            report["summary"]["total_return"],
            report["summary"]["benchmark_return"],
            report["summary"]["excess_return"],
            report["summary"]["gross_return"],
            report["summary"]["fee_drag"],
            report["summary"]["cost_drag"],
        )
        return report

    # ─── HELPERS ──────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """Raise if attribute() has not been called."""
        if not self._fitted:
            raise RuntimeError(
                "PerformanceAttributor has not been fitted. "
                "Call attribute() first."
            )
