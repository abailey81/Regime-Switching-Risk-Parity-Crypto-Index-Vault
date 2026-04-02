"""
Walk-Forward Backtesting Engine.

Protocol:
  - Train window: 180 days (4320 hours), Test window: 30 days (720 hours)
  - Roll forward by 30 days, refit GARCH-DCC and HMM each fold
  - SAC RL uses pre-trained model (inference only, no retraining)
  - Transaction costs: venue-specific via RebalancingCostModel
  - Compare against 6 benchmarks

Outputs:
  - Performance metrics (Sharpe, CVaR, MDD, etc.) with bootstrap CIs
  - Equity curves, drawdown plots, regime timeline, weight evolution
  - Monthly returns heatmap, 4-panel risk dashboard
  - Rolling Sharpe comparison
  - Statistical significance tests vs benchmarks
"""
import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

from ..models.garch_dcc import StudentTGarchDCC
from ..models.bayesian_hmm import BayesianRegimeHMM
from ..models.ensemble import EnsembleCombiner
from ..data.preprocess import prepare_hmm_features
from .benchmarks import run_all_benchmarks, simulate_benchmark
from .benchmarks import (
    EqualWeight, BTCOnly, SixtyForty, MarketCapWeighted,
    RiskParityStatic, MinimumVariance,
)
from .metrics import (
    compute_all_metrics, format_metrics_table,
    bootstrap_metric, paired_bootstrap_test,
    sharpe_ratio, maximum_drawdown, cvar, drawdown_series,
)
from .transaction_costs import RebalancingCostModel

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#              PUBLICATION-QUALITY STYLE
# ═══════════════════════════════════════════════════

def _set_plot_style():
    """Configure matplotlib/seaborn for publication-quality charts."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

_set_plot_style()

COLOURS = {
    "ensemble": "#1f77b4",
    "equal_weight": "#ff7f0e",
    "btc_only": "#2ca02c",
    "sixty_forty": "#d62728",
    "market_cap": "#9467bd",
    "risk_parity_static": "#8c564b",
    "min_variance": "#e377c2",
}

REGIME_COLOURS = {"bull": "#27ae60", "normal": "#f39c12", "crisis": "#e74c3c"}


# ═══════════════════════════════════════════════════
#              SAC OBSERVATION BUILDER
# ═══════════════════════════════════════════════════

def _build_sac_observation(
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    hmm_features: pd.DataFrame,
    regime_probs: np.ndarray,
    t: int,
    n_assets: int,
    cumulative_return: float,
    drawdown: float,
    steps_since_rebalance: int,
    return_history: List[float],
    config: dict,
) -> np.ndarray:
    """
    Build a proper observation vector for the SAC agent from real data.

    Matches the PortfolioEnv state space (dim=48):
      [0..23]   Per-asset features: returns_5d, returns_20d, vol_20d  (n_assets * 3)
      [24..31]  GARCH vol forecasts (approx from rolling vol)         (n_assets)
      [32..39]  Rolling correlation with BTC                          (n_assets)
      [40..42]  HMM regime probabilities                              (3)
      [43]      Regime transition probability                         (1)
      [44]      Cumulative return                                     (1)
      [45]      Current drawdown                                      (1)
      [46]      Days since rebalance                                  (1)
      [47]      Current Kelly fraction                                (1)

    Args:
        returns_df: Full returns DataFrame
        features_df: Full feature DataFrame (may be MultiIndex)
        hmm_features: HMM feature DataFrame
        regime_probs: Current HMM regime probabilities (3,)
        t: Current global time index into returns_df
        n_assets: Number of assets
        cumulative_return: Portfolio cumulative return
        drawdown: Current drawdown from high-water mark
        steps_since_rebalance: Steps since last rebalance
        return_history: List of recent portfolio returns
        config: Full config dict

    Returns:
        obs: np.ndarray of shape (48,) or (state_dim,)
    """
    rl_cfg = config.get("rl", {})
    state_dim = rl_cfg.get("state_dim", 48)
    btc_corr_lookback = rl_cfg.get("btc_corr_lookback", 168)

    asset_names = list(returns_df.columns)

    # ── Per-asset features: returns_5d, returns_20d, vol_20d ──
    asset_features = np.zeros(n_assets * 3)
    if t >= 480:
        for i, col in enumerate(asset_names):
            # 5d returns (120h)
            r5d = returns_df.iloc[max(0, t-120):t][col].sum()
            # 20d returns (480h)
            r20d = returns_df.iloc[max(0, t-480):t][col].sum()
            # 20d volatility
            v20d = returns_df.iloc[max(0, t-480):t][col].std() * np.sqrt(8760)

            asset_features[i] = r5d
            asset_features[n_assets + i] = r20d
            asset_features[2 * n_assets + i] = v20d

    # ── GARCH vol proxies (use rolling 20d vol as approximation) ──
    garch_feat = np.zeros(n_assets)
    if t >= 480:
        for i, col in enumerate(asset_names):
            garch_feat[i] = returns_df.iloc[max(0, t-480):t][col].std() * np.sqrt(8760)

    # ── Rolling BTC correlation ──
    btc_corr = np.zeros(n_assets)
    if t >= btc_corr_lookback and "BTC" in asset_names:
        btc_idx = asset_names.index("BTC")
        window = returns_df.iloc[t - btc_corr_lookback:t]
        btc_rets = window.iloc[:, btc_idx].values
        btc_std = btc_rets.std()
        if btc_std > 1e-10:
            for i in range(n_assets):
                asset_rets = window.iloc[:, i].values
                if asset_rets.std() > 1e-10:
                    btc_corr[i] = np.corrcoef(btc_rets, asset_rets)[0, 1]

    # ── Regime transition probability (entropy-based) ──
    probs = np.clip(regime_probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    transition_prob = entropy / max_entropy if max_entropy > 0 else 0.0

    # ── Kelly fraction estimate ──
    kelly_frac = 0.5
    if len(return_history) >= 50:
        recent = np.array(return_history[-480:])
        mu = recent.mean()
        var = recent.var()
        if var > 1e-12:
            kelly_frac = float(np.clip(mu / var, 0.0, 1.0))

    # ── Portfolio state ──
    portfolio_state = np.array([
        cumulative_return,
        drawdown,
        steps_since_rebalance / 24.0,  # Normalise to days
        kelly_frac,
    ])

    obs = np.concatenate([
        asset_features,       # n_assets * 3
        garch_feat,           # n_assets
        btc_corr,             # n_assets
        regime_probs,         # 3
        [transition_prob],    # 1
        portfolio_state,      # 4
    ]).astype(np.float32)

    # Pad or truncate to expected dimension
    if len(obs) < state_dim:
        obs = np.pad(obs, (0, state_dim - len(obs)))
    elif len(obs) > state_dim:
        obs = obs[:state_dim]

    # Replace NaN/inf
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    return obs


# ═══════════════════════════════════════════════════
#              VISUALISATION HELPERS
# ═══════════════════════════════════════════════════

def plot_equity_curves(ensemble_equity: np.ndarray, benchmark_results: dict,
                       timestamps: pd.DatetimeIndex, output_dir: Path) -> None:
    """Plot overlaid equity curves for all strategies (log-scale)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(timestamps, ensemble_equity, label="Ensemble (Ours)",
            color=COLOURS["ensemble"], linewidth=2)

    for key, result in benchmark_results.items():
        n = min(len(result["equity_curve"]), len(timestamps))
        ax.plot(timestamps[:n], result["equity_curve"][:n],
                label=result["name"], color=COLOURS.get(key, "gray"),
                linewidth=1, alpha=0.8)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($1 invested)", fontsize=12)
    ax.set_title("Walk-Forward Backtest: Equity Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved equity_curves.png")


def plot_drawdowns(ensemble_returns: np.ndarray, benchmark_results: dict,
                   timestamps: pd.DatetimeIndex, output_dir: Path) -> None:
    """Plot drawdown time series."""
    fig, ax = plt.subplots(figsize=(14, 5))

    dd = drawdown_series(ensemble_returns)
    n = min(len(dd), len(timestamps))
    ax.fill_between(timestamps[:n], dd[:n], 0, alpha=0.4, color=COLOURS["ensemble"], label="Ensemble")

    for key, result in benchmark_results.items():
        dd_b = drawdown_series(result["returns"])
        n_b = min(len(dd_b), len(timestamps))
        ax.plot(timestamps[:n_b], dd_b[:n_b], label=result["name"],
                color=COLOURS.get(key, "gray"), linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown", fontsize=12)
    ax.set_title("Drawdown Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved drawdown_plot.png")


def plot_regime_timeline(regimes: list, timestamps: pd.DatetimeIndex,
                         output_dir: Path) -> None:
    """Plot regime classification over time (color-coded)."""
    fig, ax = plt.subplots(figsize=(14, 2.5))

    for i, (regime, colour) in enumerate(REGIME_COLOURS.items()):
        mask = np.array(regimes) == regime
        if mask.any():
            n = min(len(mask), len(timestamps))
            for j in range(n):
                if mask[j]:
                    ax.axvspan(timestamps[j], timestamps[min(j+1, n-1)],
                              alpha=0.6, color=colour, linewidth=0)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=r.capitalize())
                      for r, c in REGIME_COLOURS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("HMM Regime Classification", fontsize=14, fontweight="bold")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "regime_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved regime_timeline.png")


def plot_weight_evolution(weights_history: list, asset_names: list,
                          timestamps: pd.DatetimeIndex, output_dir: Path) -> None:
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n = min(len(weights_history), len(timestamps))
    weights_arr = np.array(weights_history[:n])

    ax.stackplot(timestamps[:n], weights_arr.T, labels=asset_names, alpha=0.8)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Weight", fontsize=12)
    ax.set_title("Portfolio Weight Evolution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=4)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "weight_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved weight_evolution.png")


def plot_rolling_sharpe(ensemble_returns: np.ndarray, benchmark_results: dict,
                        timestamps: pd.DatetimeIndex, output_dir: Path,
                        window: int = 2160) -> None:
    """Rolling 90-day Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(14, 5))

    def rolling_sr(rets, w):
        sr = pd.Series(rets).rolling(w).apply(
            lambda x: x.mean() / x.std() * np.sqrt(8760) if x.std() > 0 else 0, raw=True)
        return sr.values

    n = min(len(ensemble_returns), len(timestamps))
    sr = rolling_sr(ensemble_returns[:n], window)
    ax.plot(timestamps[:n], sr, label="Ensemble", color=COLOURS["ensemble"], linewidth=1.5)

    for key, result in benchmark_results.items():
        n_b = min(len(result["returns"]), len(timestamps))
        sr_b = rolling_sr(result["returns"][:n_b], window)
        ax.plot(timestamps[:n_b], sr_b, label=result["name"],
                color=COLOURS.get(key, "gray"), linewidth=0.8, alpha=0.7)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rolling 90-day Sharpe Ratio", fontsize=12)
    ax.set_title("Rolling Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_sharpe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved rolling_sharpe.png")


def plot_monthly_returns_heatmap(returns: np.ndarray, timestamps: pd.DatetimeIndex,
                                  output_dir: Path) -> None:
    """
    Monthly returns heatmap -- rows=years, columns=months.
    """
    import calendar
    returns_series = pd.Series(returns, index=timestamps)

    # Resample to monthly returns
    monthly = returns_series.resample("M").sum()
    monthly_pct = (np.exp(monthly) - 1) * 100  # Convert to percentage

    # Pivot to year x month
    df = pd.DataFrame({
        "year": monthly_pct.index.year,
        "month": monthly_pct.index.month,
        "return": monthly_pct.values,
    })
    pivot = df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Monthly Return (%)"})
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Year")
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_returns_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved monthly_returns_heatmap.png")


def plot_risk_metrics_dashboard(returns: np.ndarray, timestamps: pd.DatetimeIndex,
                                 output_dir: Path) -> None:
    """
    4-panel risk dashboard: rolling vol, rolling CVaR, drawdown, VaR histogram.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n = min(len(returns), len(timestamps))
    returns_s = pd.Series(returns[:n], index=timestamps[:n])

    # 1. Rolling 30-day volatility
    rolling_vol = returns_s.rolling(720).std() * np.sqrt(8760) * 100
    axes[0, 0].plot(timestamps[:n], rolling_vol, color="#e74c3c", linewidth=1)
    axes[0, 0].set_title("Rolling 30-Day Volatility (%)", fontweight="bold")
    axes[0, 0].set_ylabel("Annualised Vol (%)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Rolling 30-day CVaR
    cvar_values = []
    for i in range(720, n):
        from .metrics import cvar as compute_cvar
        cv = compute_cvar(returns[i-720:i], 0.05) * 100
        cvar_values.append(cv)
    if cvar_values:
        axes[0, 1].plot(timestamps[720:n], cvar_values, color="#8e44ad", linewidth=1)
    axes[0, 1].set_title("Rolling 30-Day CVaR 5% (%)", fontweight="bold")
    axes[0, 1].set_ylabel("CVaR (%)")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Drawdown
    dd = drawdown_series(returns[:n])
    axes[1, 0].fill_between(timestamps[:n], dd * 100, 0, alpha=0.5, color="#e74c3c")
    axes[1, 0].set_title("Drawdown (%)", fontweight="bold")
    axes[1, 0].set_ylabel("Drawdown (%)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Return distribution with VaR
    axes[1, 1].hist(returns[:n] * 100, bins=100, color="#3498db", alpha=0.7, edgecolor="white")
    var_5 = np.percentile(returns[:n], 5) * 100
    axes[1, 1].axvline(var_5, color="red", linewidth=2, linestyle="--", label=f"VaR 5%: {var_5:.2f}%")
    axes[1, 1].set_title("Return Distribution with VaR", fontweight="bold")
    axes[1, 1].set_xlabel("Hourly Return (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Risk Metrics Dashboard", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved risk_dashboard.png")


# ═══════════════════════════════════════════════════
#          BOOTSTRAP & SIGNIFICANCE TESTING
# ═══════════════════════════════════════════════════

def _compute_bootstrap_cis(returns: np.ndarray, n_resamples: int = 1000) -> dict:
    """Compute bootstrap 95% CIs for key metrics (parallel across metrics)."""
    metrics_to_bootstrap = [
        ("sharpe_ci", sharpe_ratio, "Sharpe"),
        ("max_drawdown_ci", maximum_drawdown, "MaxDD"),
        ("cvar_5pct_ci", lambda r: cvar(r, 0.05), "CVaR"),
    ]

    def _bootstrap_single(key, func, label):
        return key, bootstrap_metric(returns, func, n_resamples=n_resamples)

    results = {}
    with ThreadPoolExecutor(max_workers=len(metrics_to_bootstrap)) as pool:
        futures = {
            pool.submit(_bootstrap_single, key, func, label): key
            for key, func, label in metrics_to_bootstrap
        }
        for future in as_completed(futures):
            key, ci_result = future.result()
            results[key] = ci_result

    return results


def _compute_significance_tests(
    ensemble_returns: np.ndarray,
    benchmark_results: dict,
    n_resamples: int = 1000,
) -> dict:
    """
    Paired bootstrap test: is ensemble Sharpe significantly > each benchmark?
    Returns p-values for each benchmark (parallel across benchmarks).
    """
    def _test_single(key, bm):
        test = paired_bootstrap_test(
            ensemble_returns, bm["returns"],
            metric_func=sharpe_ratio,
            n_resamples=n_resamples,
        )
        return bm["name"], test

    results = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(benchmark_results))) as pool:
        futures = {
            pool.submit(_test_single, key, bm): key
            for key, bm in benchmark_results.items()
        }
        for future in as_completed(futures):
            name, test = future.result()
            results[name] = test

    return results


# ═══════════════════════════════════════════════════
#              MAIN WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════

def run_walk_forward(
    config_path: str = "config.yaml",
    use_rl: bool = False,
    rl_model_path: str = "models/saved/sac_best",
) -> dict:
    """
    Execute walk-forward backtest of the full ensemble pipeline.

    Protocol:
      For each fold:
        1. Train GARCH-DCC on [t, t+train_window]
        2. Train HMM on [t, t+train_window]
        3. Generate weights for [t+train_window, t+train_window+test_window]
        4. Simulate portfolio returns with venue-specific transaction costs
        5. Roll forward by test_window

    Returns:
        Dictionary with all results, metrics, and file paths of generated charts
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    bt_cfg = config.get("backtest", {})
    train_window = bt_cfg.get("train_window", 4320)
    test_window = bt_cfg.get("test_window", 720)
    step_size = bt_cfg.get("step_size", 720)
    tc_bps = bt_cfg.get("transaction_cost_bps", 10)
    bootstrap_n = bt_cfg.get("bootstrap_resamples", 1000)

    # Venue-specific cost model
    cost_model = RebalancingCostModel()
    portfolio_value_usd = bt_cfg.get("portfolio_value_usd", 1_000_000)

    # Load data
    from ..data.preprocess import prepare_all_data
    logger.info("Loading data...")
    prices_df, returns_df, features_df, hmm_features = prepare_all_data(config_path)
    asset_names = list(returns_df.columns)
    n_assets = len(asset_names)
    n_total = len(returns_df)

    logger.info(f"Data: {n_total} observations, {n_assets} assets")
    logger.info(f"Walk-forward: train={train_window}h, test={test_window}h, step={step_size}h")

    # Prepare SAC agent if requested
    sac_agent = None
    if use_rl:
        from ..models.sac_agent import SACAllocator
        sac_agent = SACAllocator(config)
        sac_agent.load(rl_model_path)
        logger.info("SAC agent loaded for inference")

    # ── Walk-Forward Loop ──
    ensemble = EnsembleCombiner(config=config, asset_names=asset_names)

    all_returns: List[float] = []
    all_weights: List[np.ndarray] = []
    all_regimes: List[str] = []
    all_weight_changes: List[np.ndarray] = []
    fold_metrics: List[dict] = []
    timestamps: List = []

    t = 0
    fold = 0
    current_weights = np.ones(n_assets) / n_assets
    cumulative_return = 0.0
    high_water_mark = 1.0

    # Pre-compute total folds for progress bar
    total_folds = 0
    _t = 0
    while _t + train_window + test_window <= n_total:
        total_folds += 1
        _t += step_size
    fold_bar = tqdm(total=total_folds, desc="Walk-Forward", unit="fold", leave=True)

    while t + train_window + test_window <= n_total:
        fold += 1
        train_start = t
        train_end = t + train_window
        test_start = train_end
        test_end = min(train_end + test_window, n_total)

        # Update fold progress bar with date range
        train_date = str(returns_df.index[train_start].date()) if hasattr(returns_df.index[train_start], 'date') else str(train_start)
        test_date = str(returns_df.index[min(test_end - 1, n_total - 1)].date()) if hasattr(returns_df.index[min(test_end - 1, n_total - 1)], 'date') else str(test_end)
        fold_bar.set_description(f"Walk-Forward | Fold {fold}/{total_folds}")
        fold_bar.set_postfix(train=train_date, test=test_date)

        logger.info(f"\n  Fold {fold}: train [{train_start}:{train_end}], "
                    f"test [{test_start}:{test_end}]")

        # ── 1. Fit GARCH-DCC on training window ──
        train_returns = returns_df.iloc[train_start:train_end]
        garch = StudentTGarchDCC(config=config.get("garch", {}))
        try:
            garch.fit(train_returns)
            sigma = garch.forecast_covariance()
            garch_rp_weights = garch.get_risk_parity_weights(sigma)
            garch_uncertainty = garch.get_uncertainty()
        except Exception as e:
            logger.warning(f"  GARCH failed: {e}, using equal weights")
            sigma = np.eye(n_assets) * 0.01
            garch_rp_weights = np.ones(n_assets) / n_assets
            garch_uncertainty = 1.0

        # ── 2. Fit HMM on training window ──
        train_hmm_feat = hmm_features.iloc[train_start:train_end]
        hmm_model = BayesianRegimeHMM(config=config.get("hmm", {}))
        try:
            hmm_model.fit(train_hmm_feat)
            hmm_uncertainty = hmm_model.get_uncertainty()
        except Exception as e:
            logger.warning(f"  HMM failed: {e}, using uniform probs")
            hmm_uncertainty = 1.0

        # ── 3. Simulate test window ──
        test_returns = returns_df.iloc[test_start:test_end].values
        test_hmm_feat = hmm_features.iloc[test_start:test_end]
        test_timestamps = returns_df.index[test_start:test_end]

        for i in range(len(test_returns)):
            global_t = test_start + i

            # Get HMM regime probs
            try:
                if i < len(test_hmm_feat):
                    regime_probs = hmm_model.predict_proba(
                        test_hmm_feat.iloc[i:i+1]
                    ).values[0]
                else:
                    regime_probs = np.array([0.33, 0.34, 0.33])
            except Exception:
                regime_probs = np.array([0.33, 0.34, 0.33])

            # Get RL weights (if available) — FIXED: build real observations
            if sac_agent:
                obs = _build_sac_observation(
                    returns_df=returns_df,
                    features_df=features_df,
                    hmm_features=hmm_features,
                    regime_probs=regime_probs,
                    t=global_t,
                    n_assets=n_assets,
                    cumulative_return=cumulative_return,
                    drawdown=1.0 - np.exp(sum(all_returns)) / high_water_mark if all_returns else 0.0,
                    steps_since_rebalance=i,
                    return_history=all_returns[-480:] if all_returns else [],
                    config=config,
                )
                rl_weights = sac_agent.predict(obs)
                rl_uncertainty = sac_agent.get_uncertainty()
            else:
                rl_weights = np.ones(n_assets) / n_assets
                rl_uncertainty = 1.0

            # Ensemble combine
            nav = np.exp(sum(all_returns)) if all_returns else 1.0
            result = ensemble.combine(
                garch_rp_weights=garch_rp_weights,
                hmm_regime_probs=regime_probs,
                rl_weights=rl_weights,
                covariance_matrix=sigma,
                current_weights=current_weights,
                current_nav=nav,
                garch_uncertainty=garch_uncertainty,
                hmm_uncertainty=hmm_uncertainty,
                rl_uncertainty=rl_uncertainty,
            )

            new_weights = result["weights"]
            regime = result["regime"]

            # ── Venue-specific transaction costs ──
            # Map regime to volatility regime label for cost model
            vol_regime_map = {"bull": "calm", "normal": "normal", "crisis": "volatile"}
            vol_regime = vol_regime_map.get(regime, "normal")
            tc_cost_bps = cost_model.compute_rebalance_cost(
                old_weights=current_weights,
                new_weights=new_weights,
                asset_names=asset_names,
                portfolio_value_usd=portfolio_value_usd * nav,
                volatility_regime=vol_regime,
            )
            tc = tc_cost_bps / 10000  # Convert bps to decimal

            # Portfolio return
            period_return = np.dot(new_weights, test_returns[i]) - tc

            # Update tracking
            cumulative_return += period_return
            current_nav = np.exp(cumulative_return)
            if current_nav > high_water_mark:
                high_water_mark = current_nav

            # Record
            all_returns.append(period_return)
            all_weights.append(new_weights.copy())
            all_regimes.append(regime)
            all_weight_changes.append(new_weights - current_weights)
            timestamps.append(test_timestamps[i])

            current_weights = new_weights

        t += step_size
        fold_bar.update(1)

    fold_bar.close()

    # ── Convert to arrays ──
    all_returns = np.array(all_returns)
    timestamps = pd.DatetimeIndex(timestamps)
    equity_curve = np.exp(np.cumsum(all_returns))

    logger.info(f"\nWalk-forward complete: {fold} folds, {len(all_returns)} test observations")

    # ── Run Benchmarks (parallel -- each benchmark is independent) ──
    logger.info("\nRunning benchmarks (parallel)...")
    test_start_global = train_window
    test_returns_df = returns_df.iloc[test_start_global:test_start_global + len(all_returns)]

    _benchmark_specs = {
        "equal_weight": EqualWeight(asset_names),
        "btc_only": BTCOnly(asset_names),
        "sixty_forty": SixtyForty(asset_names),
        "market_cap": MarketCapWeighted(asset_names),
        "risk_parity_static": RiskParityStatic(asset_names),
        "min_variance": MinimumVariance(asset_names),
    }

    def _run_single_benchmark(key, strategy):
        return key, simulate_benchmark(strategy, test_returns_df, tc_bps)

    benchmark_results = {}
    bench_bar = tqdm(total=len(_benchmark_specs), desc="Benchmarks", unit="strategy", leave=True)
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(_benchmark_specs))) as pool:
        futures = {
            pool.submit(_run_single_benchmark, key, strat): key
            for key, strat in _benchmark_specs.items()
        }
        for future in as_completed(futures):
            key, result = future.result()
            benchmark_results[key] = result
            bench_bar.update(1)
            bench_bar.set_postfix(strategy=result["name"][:25])
    bench_bar.close()

    # ── Compute Metrics ──
    logger.info("\nComputing metrics...")
    bench_eq_returns = benchmark_results["equal_weight"]["returns"]

    ensemble_metrics = compute_all_metrics(
        all_returns, benchmark_returns=bench_eq_returns,
        weight_changes=all_weight_changes
    )

    all_metrics = {"Ensemble (Ours)": ensemble_metrics}
    for key, result in benchmark_results.items():
        bm = compute_all_metrics(result["returns"])
        all_metrics[result["name"]] = bm

    # ── Bootstrap Confidence Intervals ──
    logger.info("\nComputing bootstrap confidence intervals...")
    bootstrap_cis = _compute_bootstrap_cis(all_returns, n_resamples=bootstrap_n)

    # Add CIs to ensemble metrics
    for metric_key, ci_data in bootstrap_cis.items():
        base_key = metric_key.replace("_ci", "")
        ensemble_metrics[f"{base_key}_ci_lower"] = ci_data["ci_lower"]
        ensemble_metrics[f"{base_key}_ci_upper"] = ci_data["ci_upper"]

    # ── Statistical Significance Tests ──
    logger.info("Computing significance tests (paired bootstrap)...")
    significance_results = _compute_significance_tests(
        all_returns, benchmark_results, n_resamples=bootstrap_n,
    )

    # ── Display Metrics ──
    metrics_df = pd.DataFrame(all_metrics).T
    logger.info(f"\n{'='*60}")
    logger.info(f"  PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{metrics_df.round(4).to_string()}")

    # Display bootstrap CIs
    logger.info(f"\n{'='*60}")
    logger.info(f"  BOOTSTRAP 95% CONFIDENCE INTERVALS")
    logger.info(f"{'='*60}")
    for metric_key, ci_data in bootstrap_cis.items():
        logger.info(f"  {metric_key}: {ci_data['point_estimate']:.4f} "
                    f"[{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]")

    # Display significance tests
    logger.info(f"\n{'='*60}")
    logger.info(f"  SIGNIFICANCE TESTS (Ensemble Sharpe > Benchmark)")
    logger.info(f"{'='*60}")
    for name, test_res in significance_results.items():
        sig_marker = "*" if test_res["significant_5pct"] else ""
        logger.info(f"  vs {name}: diff={test_res['diff']:.4f}, "
                    f"p={test_res['p_value']:.4f}{sig_marker}")

    # ── Generate All 9 Charts ──
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating visualisations (parallel)...")
    chart_tasks = [
        ("equity_curves", lambda: plot_equity_curves(equity_curve, benchmark_results, timestamps, output_dir)),
        ("drawdowns", lambda: plot_drawdowns(all_returns, benchmark_results, timestamps, output_dir)),
        ("regime_timeline", lambda: plot_regime_timeline(all_regimes, timestamps, output_dir)),
        ("weight_evolution", lambda: plot_weight_evolution(all_weights, asset_names, timestamps, output_dir)),
        ("rolling_sharpe", lambda: plot_rolling_sharpe(all_returns, benchmark_results, timestamps, output_dir)),
        ("monthly_heatmap", lambda: plot_monthly_returns_heatmap(all_returns, timestamps, output_dir)),
        ("risk_dashboard", lambda: plot_risk_metrics_dashboard(all_returns, timestamps, output_dir)),
    ]

    # matplotlib is not fully thread-safe, so we use ThreadPoolExecutor with
    # max_workers=1 for safety, but can increase if backend is Agg (non-interactive)
    # Agg backend is thread-safe for independent figure objects
    chart_bar = tqdm(total=len(chart_tasks), desc="Generating charts", unit="chart", leave=True)

    def _run_chart(chart_name, chart_func):
        chart_func()
        return chart_name

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(chart_tasks))) as pool:
        futures = {
            pool.submit(_run_chart, name, func): name
            for name, func in chart_tasks
        }
        for future in as_completed(futures):
            chart_name = future.result()
            chart_bar.update(1)
            chart_bar.set_postfix(chart=chart_name)

    chart_bar.close()
    # monte_carlo_fan_6m.png and monte_carlo_fan_12m.png are generated
    # by monte_carlo.py when called separately

    # ── Save Metrics CSV with CIs ──
    # Add significance column
    sig_df = pd.DataFrame(significance_results).T
    if not sig_df.empty:
        sig_df.to_csv(output_dir / "significance_tests.csv")
        logger.info("  Saved significance_tests.csv")

    # Save bootstrap CIs
    ci_df = pd.DataFrame(bootstrap_cis).T
    ci_df.to_csv(output_dir / "bootstrap_confidence_intervals.csv")
    logger.info("  Saved bootstrap_confidence_intervals.csv")

    metrics_df.to_csv(output_dir / "performance_summary.csv")
    logger.info("  Saved performance_summary.csv")

    # ── Save Full Results ──
    results = {
        "ensemble_returns": all_returns,
        "equity_curve": equity_curve,
        "timestamps": timestamps,
        "regimes": all_regimes,
        "weights_history": all_weights,
        "metrics": all_metrics,
        "benchmark_results": benchmark_results,
        "bootstrap_cis": bootstrap_cis,
        "significance_tests": significance_results,
        "n_folds": fold,
        "config": config,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKTEST COMPLETE")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--rl-model", default="models/saved/sac_best")
    args = parser.parse_args()

    results = run_walk_forward(args.config, use_rl=args.rl, rl_model_path=args.rl_model)
