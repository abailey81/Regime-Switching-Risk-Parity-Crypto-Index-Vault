"""
Benchmark Portfolio Strategies.

Implements six comparison strategies for evaluating the ensemble:
  1. Equal Weight (1/N) -- monthly rebalance
  2. BTC Only -- 100% buy-and-hold
  3. 60/40 BTC/USDC -- monthly rebalance
  4. Market-Cap Weighted -- monthly rebalance (approximated)
  5. Risk Parity (static) -- equal risk contribution, monthly rebalance
  6. Minimum Variance -- global minimum variance, monthly rebalance
"""
import os
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

_MAX_WORKERS = min(8, os.cpu_count() or 4)

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


class BenchmarkStrategy:
    """Base class for benchmark strategies."""

    def __init__(self, name: str, asset_names: list):
        self.name = name
        self.asset_names = asset_names
        self.n_assets = len(asset_names)

    def get_weights(self, t: int, returns_df: pd.DataFrame,
                    current_weights: np.ndarray) -> np.ndarray:
        """Return target weights at time t."""
        raise NotImplementedError


class EqualWeight(BenchmarkStrategy):
    """1/N equal-weight portfolio, rebalanced monthly."""

    def __init__(self, asset_names: list, rebalance_hours: int = 720):
        super().__init__("Equal Weight (1/N)", asset_names)
        self.rebalance_hours = rebalance_hours
        self._step = 0

    def get_weights(self, t, returns_df, current_weights):
        self._step += 1
        if self._step % self.rebalance_hours == 0 or current_weights is None:
            return np.ones(self.n_assets) / self.n_assets
        return current_weights


class BTCOnly(BenchmarkStrategy):
    """100% BTC buy-and-hold."""

    def __init__(self, asset_names: list):
        super().__init__("BTC Only", asset_names)
        self.btc_idx = asset_names.index("BTC") if "BTC" in asset_names else 0

    def get_weights(self, t, returns_df, current_weights):
        w = np.zeros(self.n_assets)
        w[self.btc_idx] = 1.0
        return w


class SixtyForty(BenchmarkStrategy):
    """60% BTC / 40% USDC, rebalanced monthly."""

    def __init__(self, asset_names: list, rebalance_hours: int = 720):
        super().__init__("60/40 BTC/USDC", asset_names)
        self.rebalance_hours = rebalance_hours
        self.btc_idx = asset_names.index("BTC") if "BTC" in asset_names else 0
        self.usdc_idx = asset_names.index("USDC") if "USDC" in asset_names else -1
        self._step = 0

    def get_weights(self, t, returns_df, current_weights):
        self._step += 1
        if self._step % self.rebalance_hours == 0 or current_weights is None:
            w = np.zeros(self.n_assets)
            w[self.btc_idx] = 0.60
            if self.usdc_idx >= 0:
                w[self.usdc_idx] = 0.40
            else:
                w[self.btc_idx] = 1.0
            return w
        return current_weights


class MarketCapWeighted(BenchmarkStrategy):
    """
    Market-cap weighted portfolio, rebalanced monthly.
    Approximates market cap weights using price x constant supply factors.
    """

    def __init__(self, asset_names: list, rebalance_hours: int = 720):
        super().__init__("Market-Cap Weighted", asset_names)
        self.rebalance_hours = rebalance_hours
        self._step = 0
        # Approximate supply factors (relative market cap proxies)
        self.supply_factors = {
            "BTC": 19.5e6, "ETH": 120e6, "SOL": 440e6,
            "stETH": 10e6, "rETH": 0.5e6,
            "BUIDL": 2.2e9, "USDY": 0.5e9, "USDC": 30e9,
        }

    def get_weights(self, t, returns_df, current_weights):
        self._step += 1
        if self._step % self.rebalance_hours == 0 or current_weights is None:
            # Use latest prices to compute approximate market caps
            if t < len(returns_df):
                prices = np.exp(returns_df.iloc[:t+1].sum())  # Cumulative price from returns
            else:
                prices = np.ones(self.n_assets)

            caps = np.array([
                prices[i] * self.supply_factors.get(self.asset_names[i], 1e6)
                for i in range(self.n_assets)
            ])
            w = caps / caps.sum()
            return w
        return current_weights


class RiskParityStatic(BenchmarkStrategy):
    """
    Equal Risk Contribution (Risk Parity) portfolio, rebalanced monthly.

    Each asset contributes equally to total portfolio risk. Uses a rolling
    covariance estimate to compute risk contributions and then solves for
    weights such that w_i * (Sigma w)_i / w^T Sigma w = 1/N for all i.

    This is a static (non-ML) risk parity benchmark -- shows what pure RP
    does without the ensemble's dynamic regime adjustment.
    """

    def __init__(self, asset_names: list, rebalance_hours: int = 720,
                 lookback: int = 720):
        super().__init__("Risk Parity (Static)", asset_names)
        self.rebalance_hours = rebalance_hours
        self.lookback = lookback
        self._step = 0
        self._cached_weights: Optional[np.ndarray] = None

    def get_weights(self, t, returns_df, current_weights):
        self._step += 1
        if self._step % self.rebalance_hours == 0 or current_weights is None:
            self._cached_weights = self._compute_rp_weights(t, returns_df)
            return self._cached_weights
        return current_weights if current_weights is not None else np.ones(self.n_assets) / self.n_assets

    def _compute_rp_weights(self, t: int, returns_df: pd.DataFrame) -> np.ndarray:
        """Solve for equal risk contribution weights via iterative bisection."""
        start = max(0, t - self.lookback)
        window = returns_df.iloc[start:t + 1]

        if len(window) < 30:
            return np.ones(self.n_assets) / self.n_assets

        cov = window.cov().values
        # Regularise: add small ridge to avoid singularity
        cov += np.eye(self.n_assets) * 1e-8

        # Newton-style iterative risk parity
        w = np.ones(self.n_assets) / self.n_assets
        target_rc = 1.0 / self.n_assets

        for _ in range(50):
            sigma_w = cov @ w
            port_vol = np.sqrt(w @ sigma_w)
            if port_vol < 1e-12:
                break
            # Marginal risk contribution
            mrc = sigma_w / port_vol
            # Risk contribution
            rc = w * mrc
            rc_share = rc / rc.sum()

            # Gradient step: adjust weights toward equal RC
            adjustment = target_rc - rc_share
            w = w * np.exp(0.5 * adjustment)
            w = np.maximum(w, 1e-6)
            w = w / w.sum()

        return w


class MinimumVariance(BenchmarkStrategy):
    """
    Global Minimum Variance portfolio, rebalanced monthly.

    Minimises portfolio variance: min w^T Sigma w s.t. sum(w)=1, w>=0.
    Uses analytical solution with long-only constraint via projected gradient.
    """

    def __init__(self, asset_names: list, rebalance_hours: int = 720,
                 lookback: int = 720):
        super().__init__("Minimum Variance", asset_names)
        self.rebalance_hours = rebalance_hours
        self.lookback = lookback
        self._step = 0
        self._cached_weights: Optional[np.ndarray] = None

    def get_weights(self, t, returns_df, current_weights):
        self._step += 1
        if self._step % self.rebalance_hours == 0 or current_weights is None:
            self._cached_weights = self._compute_minvar_weights(t, returns_df)
            return self._cached_weights
        return current_weights if current_weights is not None else np.ones(self.n_assets) / self.n_assets

    def _compute_minvar_weights(self, t: int, returns_df: pd.DataFrame) -> np.ndarray:
        """Compute global minimum variance weights with long-only constraint."""
        start = max(0, t - self.lookback)
        window = returns_df.iloc[start:t + 1]

        if len(window) < 30:
            return np.ones(self.n_assets) / self.n_assets

        cov = window.cov().values
        cov += np.eye(self.n_assets) * 1e-8

        # Analytical unconstrained solution: w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return np.ones(self.n_assets) / self.n_assets

        ones = np.ones(self.n_assets)
        w = cov_inv @ ones
        w = w / w.sum()

        # Project to long-only: clip negatives, renormalise
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total < 1e-10:
            return np.ones(self.n_assets) / self.n_assets
        w = w / total

        return w


def simulate_benchmark(strategy: BenchmarkStrategy, returns_df: pd.DataFrame,
                       tc_bps: float = 10) -> dict:
    """
    Simulate a benchmark strategy over the full return series.

    Args:
        strategy: Benchmark strategy instance
        returns_df: DataFrame of log returns (columns = assets)
        tc_bps: Transaction cost in basis points per unit of turnover

    Returns:
        dict with 'returns', 'equity_curve', 'weights_history', 'weight_changes'
    """
    tc_rate = tc_bps / 10000
    n_steps = len(returns_df)
    returns_arr = returns_df.values
    n_assets = returns_arr.shape[1]

    portfolio_returns = []
    weights_history = []
    weight_changes = []
    current_weights = None

    for t in range(n_steps):
        new_weights = strategy.get_weights(t, returns_df, current_weights)

        # Transaction cost
        if current_weights is not None:
            turnover = np.abs(new_weights - current_weights).sum()
            tc = turnover * tc_rate
        else:
            tc = 0.0

        # Portfolio return
        period_return = np.dot(new_weights, returns_arr[t]) - tc
        portfolio_returns.append(period_return)

        if current_weights is not None:
            weight_changes.append(new_weights - current_weights)

        # Drift weights by asset returns (before next rebalance)
        drifted = new_weights * np.exp(returns_arr[t])
        current_weights = drifted / drifted.sum()

        weights_history.append(new_weights.copy())

    portfolio_returns = np.array(portfolio_returns)
    equity_curve = np.exp(np.cumsum(portfolio_returns))

    return {
        "returns": portfolio_returns,
        "equity_curve": equity_curve,
        "weights_history": np.array(weights_history),
        "weight_changes": weight_changes,
        "name": strategy.name,
    }


def run_all_benchmarks(returns_df: pd.DataFrame, asset_names: list,
                       tc_bps: float = 10) -> dict:
    """Run all six benchmark strategies in parallel and return results."""
    benchmarks = {
        "equal_weight": EqualWeight(asset_names),
        "btc_only": BTCOnly(asset_names),
        "sixty_forty": SixtyForty(asset_names),
        "market_cap": MarketCapWeighted(asset_names),
        "risk_parity_static": RiskParityStatic(asset_names),
        "min_variance": MinimumVariance(asset_names),
    }

    def _run_single(key: str, strategy: BenchmarkStrategy) -> tuple:
        logger.info(f"  Simulating {strategy.name}...")
        return key, simulate_benchmark(strategy, returns_df, tc_bps)

    results = {}
    bench_bar = tqdm(total=len(benchmarks), desc="Benchmarks", unit="strategy", leave=True)

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(benchmarks))) as pool:
        futures = {
            pool.submit(_run_single, key, strategy): key
            for key, strategy in benchmarks.items()
        }
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
            bench_bar.update(1)
            bench_bar.set_postfix(strategy=result["name"][:25])

    bench_bar.close()
    return results
