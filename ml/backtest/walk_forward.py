"""
Walk-Forward Backtesting Engine.

Protocol:
  - Train window: 180 days (4320 hours), Test window: 30 days (720 hours)
  - Roll forward by 30 days, refit GARCH-DCC and HMM each fold
  - SAC RL uses pre-trained model (inference only, no retraining)
  - Transaction costs: 10 bps per unit of turnover
  - Compare against 4 benchmarks

Outputs:
  - Performance metrics (Sharpe, CVaR, MDD, etc.)
  - Equity curves, drawdown plots, regime timeline, weight evolution
"""
import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from ..models.garch_dcc import StudentTGarchDCC
from ..models.bayesian_hmm import BayesianRegimeHMM
from ..models.ensemble import EnsembleCombiner
from ..data.preprocess import prepare_hmm_features
from .benchmarks import run_all_benchmarks
from .metrics import compute_all_metrics, format_metrics_table

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#              VISUALISATION HELPERS
# ═══════════════════════════════════════════════════

COLOURS = {
    "ensemble": "#1f77b4",
    "equal_weight": "#ff7f0e",
    "btc_only": "#2ca02c",
    "sixty_forty": "#d62728",
    "market_cap": "#9467bd",
}

REGIME_COLOURS = {"bull": "#27ae60", "normal": "#f39c12", "crisis": "#e74c3c"}


def plot_equity_curves(ensemble_equity, benchmark_results, timestamps, output_dir):
    """Plot overlaid equity curves for all strategies."""
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
    ax.legend(loc="upper left", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved equity_curves.png")


def plot_drawdowns(ensemble_returns, benchmark_results, timestamps, output_dir):
    """Plot drawdown time series."""
    from .metrics import drawdown_series

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
    logger.info(f"  Saved drawdown_plot.png")


def plot_regime_timeline(regimes, timestamps, output_dir):
    """Plot regime classification over time."""
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
    logger.info(f"  Saved regime_timeline.png")


def plot_weight_evolution(weights_history, asset_names, timestamps, output_dir):
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
    logger.info(f"  Saved weight_evolution.png")


def plot_rolling_sharpe(ensemble_returns, benchmark_results, timestamps,
                        output_dir, window=2160):
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
    logger.info(f"  Saved rolling_sharpe.png")


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
        4. Simulate portfolio returns with transaction costs
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
    tc_rate = tc_bps / 10000

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

    all_returns = []
    all_weights = []
    all_regimes = []
    all_weight_changes = []
    fold_metrics = []
    timestamps = []

    t = 0
    fold = 0
    current_weights = np.ones(n_assets) / n_assets

    while t + train_window + test_window <= n_total:
        fold += 1
        train_start = t
        train_end = t + train_window
        test_start = train_end
        test_end = min(train_end + test_window, n_total)

        logger.info(f"\n  Fold {fold}: train [{train_start}:{train_end}], "
                    f"test [{test_start}:{test_end}]")

        # ── 1. Fit GARCH-DCC on training window ──
        train_returns = returns_df.iloc[train_start:train_end]
        garch = StudentTGarchDCC()
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
        hmm_model = BayesianRegimeHMM()
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
            # Get HMM regime probs
            try:
                if i < len(test_hmm_feat):
                    regime_probs = hmm_model.predict_proba(
                        test_hmm_feat.iloc[i:i+1]
                    ).values[0]
                else:
                    regime_probs = np.array([0.33, 0.34, 0.33])
            except:
                regime_probs = np.array([0.33, 0.34, 0.33])

            # Get RL weights (if available)
            if sac_agent:
                obs = np.zeros(38, dtype=np.float32)
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

            # Transaction cost
            turnover = np.abs(new_weights - current_weights).sum()
            tc = turnover * tc_rate

            # Portfolio return
            period_return = np.dot(new_weights, test_returns[i]) - tc

            # Record
            all_returns.append(period_return)
            all_weights.append(new_weights.copy())
            all_regimes.append(regime)
            all_weight_changes.append(new_weights - current_weights)
            timestamps.append(test_timestamps[i])

            current_weights = new_weights

        t += step_size

    # ── Convert to arrays ──
    all_returns = np.array(all_returns)
    timestamps = pd.DatetimeIndex(timestamps)
    equity_curve = np.exp(np.cumsum(all_returns))

    logger.info(f"\nWalk-forward complete: {fold} folds, {len(all_returns)} test observations")

    # ── Run Benchmarks ──
    logger.info("\nRunning benchmarks...")
    # Use same test period as ensemble
    test_start_global = train_window
    test_returns_df = returns_df.iloc[test_start_global:test_start_global + len(all_returns)]
    benchmark_results = run_all_benchmarks(test_returns_df, asset_names, tc_bps)

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

    # ── Display Metrics ──
    metrics_df = pd.DataFrame(all_metrics).T
    logger.info(f"\n{'='*60}")
    logger.info(f"  PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{metrics_df.round(4).to_string()}")

    # ── Generate Charts ──
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating visualisations...")
    plot_equity_curves(equity_curve, benchmark_results, timestamps, output_dir)
    plot_drawdowns(all_returns, benchmark_results, timestamps, output_dir)
    plot_regime_timeline(all_regimes, timestamps, output_dir)
    plot_weight_evolution(all_weights, asset_names, timestamps, output_dir)
    plot_rolling_sharpe(all_returns, benchmark_results, timestamps, output_dir)

    # ── Save Metrics ──
    metrics_df.to_csv(output_dir / "performance_summary.csv")
    logger.info(f"  Saved performance_summary.csv")

    # ── Save Full Results ──
    results = {
        "ensemble_returns": all_returns,
        "equity_curve": equity_curve,
        "timestamps": timestamps,
        "regimes": all_regimes,
        "weights_history": all_weights,
        "metrics": all_metrics,
        "benchmark_results": benchmark_results,
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


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL VISUALISATIONS (from stat-arb backtesting/visualization.py)
# ═══════════════════════════════════════════════════════════════

def plot_monthly_returns_heatmap(returns, timestamps, output_dir):
    """
    Monthly returns heatmap — rows=years, columns=months.
    Adapted from stat-arb backtesting/visualization.py plot_monthly_returns_heatmap.
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
    logger.info(f"  Saved monthly_returns_heatmap.png")


def plot_risk_metrics_dashboard(returns, timestamps, output_dir):
    """
    Risk metrics dashboard — 4 subplots: rolling vol, rolling CVaR, drawdown, VaR histogram.
    Adapted from stat-arb backtesting/visualization.py plot_risk_metrics_dashboard.
    """
    from .metrics import drawdown_series, cvar as compute_cvar

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
        cv = compute_cvar(returns[i-720:i], 0.05) * 100
        cvar_values.append(cv)
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
    logger.info(f"  Saved risk_dashboard.png")
