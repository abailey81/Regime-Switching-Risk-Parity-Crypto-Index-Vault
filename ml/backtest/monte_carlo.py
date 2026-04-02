"""
Monte Carlo Stress Testing.

Protocol:
  - 10,000 simulated paths via regime-conditioned block bootstrap
  - Block size: 1 week (168 hours)
  - Horizons: 6-month (4320h), 12-month (8640h)
  - Reports: fan chart, terminal NAV distribution, drawdown distribution,
    probability of loss, probability of Sharpe > 1

Uses the HMM transition matrix to sample regime sequences, then
bootstraps return blocks from historical data conditioned on regime.

Also provides an alternative t-copula simulation path for comparison.
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
from typing import Dict, Optional

from .metrics import maximum_drawdown, sharpe_ratio, cvar, annualised_return

logger = logging.getLogger(__name__)


def regime_conditioned_bootstrap(
    returns_df: pd.DataFrame,
    regimes: np.ndarray,
    transition_matrix: np.ndarray,
    n_simulations: int = 10000,
    horizon_hours: int = 4320,
    block_size: int = 168,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Monte Carlo paths via regime-conditioned block bootstrap.

    For each simulation:
      1. Sample regime sequence from HMM transition matrix
      2. For each block, draw a random block of returns from historical
         data matching that regime
      3. Concatenate blocks to form a full simulated path

    Args:
        returns_df: Historical return matrix (T x n_assets)
        regimes: Regime labels for each historical observation (T,)
        transition_matrix: HMM transition matrix (3x3)
        n_simulations: Number of MC paths
        horizon_hours: Length of each simulated path
        block_size: Block size for bootstrap (hours)
        seed: Random seed

    Returns:
        simulated_returns: (n_simulations, horizon_hours, n_assets)
    """
    rng = np.random.RandomState(seed)
    returns = returns_df.values
    T, n_assets = returns.shape
    n_blocks = (horizon_hours + block_size - 1) // block_size  # ceiling division

    # Index returns by regime
    regime_labels = ["bull", "normal", "crisis"]
    regime_indices: Dict[int, np.ndarray] = {}
    for i, label in enumerate(regime_labels):
        mask = regimes == label
        valid_starts = np.where(mask)[0]
        # Filter starts where a full block is available
        valid_starts = valid_starts[valid_starts + block_size <= T]
        regime_indices[i] = valid_starts

    # Handle empty regimes -- fallback to all valid starts
    all_valid = np.arange(max(0, T - block_size))
    if len(all_valid) == 0:
        all_valid = np.array([0])
    for i in range(3):
        if len(regime_indices[i]) == 0:
            logger.warning(f"  No blocks for regime {regime_labels[i]}, using all data as fallback")
            regime_indices[i] = all_valid

    simulated = np.zeros((n_simulations, horizon_hours, n_assets))

    for sim in range(n_simulations):
        # Sample starting regime from stationary distribution (first row)
        current_regime = rng.choice(3, p=transition_matrix[0])

        path_returns = []
        for block in range(n_blocks):
            # Sample a block from the current regime
            valid = regime_indices[current_regime]
            start_idx = valid[rng.randint(len(valid))]
            end_idx = min(start_idx + block_size, T)
            block_data = returns[start_idx:end_idx]
            path_returns.append(block_data)

            # Transition to next regime
            current_regime = rng.choice(3, p=transition_matrix[current_regime])

        path = np.concatenate(path_returns, axis=0)[:horizon_hours]

        # Pad if necessary (edge case: not enough data)
        if len(path) < horizon_hours:
            pad_len = horizon_hours - len(path)
            pad_start = rng.randint(0, max(1, T - pad_len))
            pad = returns[pad_start:pad_start + pad_len]
            if len(pad) < pad_len:
                pad = np.tile(returns.mean(axis=0), (pad_len, 1))
            path = np.concatenate([path, pad[:pad_len]], axis=0)

        simulated[sim, :len(path)] = path[:horizon_hours]

    return simulated


def t_copula_simulation(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10000,
    horizon_hours: int = 4320,
    seed: int = 42,
    df: int = 5,
) -> np.ndarray:
    """
    Generate Monte Carlo portfolio return paths using a t-copula model.

    Fits marginal distributions and a t-copula to the historical return
    data, then generates simulated paths from the fitted model.

    Args:
        returns_df: Historical returns (T x n_assets)
        weights: Portfolio weights (n_assets,)
        n_simulations: Number of MC paths
        horizon_hours: Length of each path
        seed: Random seed
        df: Degrees of freedom for the t-copula

    Returns:
        portfolio_returns: (n_simulations, horizon_hours) portfolio return paths
    """
    from scipy.stats import t as t_dist, norm as norm_dist
    from scipy.linalg import cholesky

    rng = np.random.RandomState(seed)
    returns = returns_df.values
    T, n_assets = returns.shape

    # Fit marginal parameters (mean, std) per asset
    means = returns.mean(axis=0)
    stds = returns.std(axis=0)
    stds = np.maximum(stds, 1e-10)

    # Compute correlation matrix from standardised returns
    standardised = (returns - means) / stds
    corr_matrix = np.corrcoef(standardised.T)
    # Regularise for positive definiteness
    eigvals = np.linalg.eigvalsh(corr_matrix)
    if eigvals.min() < 1e-6:
        corr_matrix += np.eye(n_assets) * (1e-6 - eigvals.min())

    try:
        L = cholesky(corr_matrix, lower=True)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal (independent assets)
        L = np.eye(n_assets)

    # Generate t-copula samples
    portfolio_returns = np.zeros((n_simulations, horizon_hours))

    for sim in range(n_simulations):
        # Draw from multivariate t via: Z = sqrt(df/chi2) * L @ N(0,I)
        chi2_samples = rng.chisquare(df, size=horizon_hours)
        normal_samples = rng.randn(horizon_hours, n_assets)

        for h in range(horizon_hours):
            z = normal_samples[h]
            correlated = L @ z
            # Scale by chi-squared for t-distribution
            t_factor = np.sqrt(df / chi2_samples[h])
            t_samples = correlated * t_factor

            # Transform through marginals: t-CDF -> uniform -> asset return
            # Use t-distribution CDF then inverse-normal to get correlated normals
            # Then scale back to asset returns
            asset_returns = means + stds * t_samples

            portfolio_returns[sim, h] = np.dot(weights, asset_returns)

    return portfolio_returns


def run_monte_carlo(
    returns_df: pd.DataFrame,
    regimes: np.ndarray,
    transition_matrix: np.ndarray,
    weights: np.ndarray,
    config: dict = None,
) -> dict:
    """
    Run full Monte Carlo stress test and generate visualisations.

    Includes both regime-conditioned block bootstrap and t-copula simulation.

    Args:
        returns_df: Historical returns
        regimes: Regime labels
        transition_matrix: 3x3 HMM transition matrix
        weights: Current/target portfolio weights (for computing portfolio returns)
        config: Configuration dictionary

    Returns:
        dict with simulation results and statistics
    """
    cfg = config or {}
    bt_cfg = cfg.get("backtest", {})
    n_sims = bt_cfg.get("monte_carlo_sims", 10000)
    block_size = bt_cfg.get("block_size", 168)
    horizons = [4320, 8640]  # 6-month, 12-month

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Publication-quality plot settings
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    all_results = {}

    for horizon in horizons:
        label = f"{horizon // (24 * 30)}m"  # "6m" or "12m"
        logger.info(f"\nMonte Carlo: {n_sims} paths x {horizon}h ({label})...")

        # ── Bootstrap simulation ──
        sim_returns = regime_conditioned_bootstrap(
            returns_df, regimes, transition_matrix,
            n_simulations=n_sims, horizon_hours=horizon,
            block_size=block_size,
        )

        # Compute portfolio returns for each path
        # sim_returns shape: (n_sims, horizon, n_assets)
        # weights shape: (n_assets,)
        portfolio_returns = np.einsum("ijk,k->ij", sim_returns, weights)

        # Terminal values
        terminal_values = np.exp(portfolio_returns.sum(axis=1))

        # Max drawdowns per path
        max_drawdowns = np.array([
            maximum_drawdown(portfolio_returns[i]) for i in range(n_sims)
        ])

        # Sharpe ratios per path
        sharpe_ratios = np.array([
            sharpe_ratio(portfolio_returns[i]) for i in range(n_sims)
        ])

        # Conditional Tail Expectation (CTE) at 5%
        sorted_terminal = np.sort(terminal_values)
        n_tail = max(1, int(0.05 * len(sorted_terminal)))
        cte_5 = float(np.mean(sorted_terminal[:n_tail]))

        # ── t-Copula simulation (comparison) ──
        logger.info(f"  Running t-copula simulation for comparison...")
        copula_returns = t_copula_simulation(
            returns_df, weights,
            n_simulations=min(n_sims, 5000),  # Fewer sims for speed
            horizon_hours=horizon,
        )
        copula_terminal = np.exp(copula_returns.sum(axis=1))

        # Statistics
        stats = {
            "horizon": label,
            "n_simulations": n_sims,
            "terminal_value": {
                "median": float(np.median(terminal_values)),
                "p5": float(np.percentile(terminal_values, 5)),
                "p10": float(np.percentile(terminal_values, 10)),
                "p25": float(np.percentile(terminal_values, 25)),
                "p75": float(np.percentile(terminal_values, 75)),
                "p90": float(np.percentile(terminal_values, 90)),
                "p95": float(np.percentile(terminal_values, 95)),
                "mean": float(np.mean(terminal_values)),
            },
            "max_drawdown": {
                "median": float(np.median(max_drawdowns)),
                "p5": float(np.percentile(max_drawdowns, 5)),
                "p95": float(np.percentile(max_drawdowns, 95)),
            },
            "probabilities": {
                "P(loss)": float((terminal_values < 1.0).mean()),
                "P(loss > 20%)": float((terminal_values < 0.8).mean()),
                "P(gain > 50%)": float((terminal_values > 1.5).mean()),
                "P(MDD > 30%)": float((max_drawdowns < -0.30).mean()),
                "P(Sharpe > 1)": float((sharpe_ratios > 1.0).mean()),
            },
            "conditional_tail_expectation_5pct": cte_5,
            "copula_comparison": {
                "copula_median": float(np.median(copula_terminal)),
                "copula_p5": float(np.percentile(copula_terminal, 5)),
                "copula_p95": float(np.percentile(copula_terminal, 95)),
                "copula_P_loss": float((copula_terminal < 1.0).mean()),
            },
        }

        all_results[label] = stats

        logger.info(f"  Terminal value: median={stats['terminal_value']['median']:.3f}, "
                    f"5th={stats['terminal_value']['p5']:.3f}, "
                    f"95th={stats['terminal_value']['p95']:.3f}")
        logger.info(f"  CTE(5%)={cte_5:.3f}")
        logger.info(f"  P(loss)={stats['probabilities']['P(loss)']:.1%}, "
                    f"P(Sharpe>1)={stats['probabilities']['P(Sharpe > 1)']:.1%}")
        logger.info(f"  Copula comparison: median={stats['copula_comparison']['copula_median']:.3f}, "
                    f"P(loss)={stats['copula_comparison']['copula_P_loss']:.1%}")

        # ── Fan Chart ──
        fig, ax = plt.subplots(figsize=(14, 7))

        cum_returns = np.exp(np.cumsum(portfolio_returns, axis=1))
        percentiles = [5, 25, 50, 75, 95]
        curves = {p: np.percentile(cum_returns, p, axis=0) for p in percentiles}

        x = np.arange(horizon) / 24  # Convert to days
        ax.fill_between(x, curves[5], curves[95], alpha=0.15, color="#1f77b4", label="5th-95th")
        ax.fill_between(x, curves[25], curves[75], alpha=0.3, color="#1f77b4", label="25th-75th")
        ax.plot(x, curves[50], color="#1f77b4", linewidth=2, label="Median")
        ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

        ax.set_xlabel("Days", fontsize=12)
        ax.set_ylabel("Portfolio Value ($1 invested)", fontsize=12)
        ax.set_title(f"Monte Carlo Fan Chart ({label} horizon, {n_sims:,} paths)",
                    fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"monte_carlo_fan_{label}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved monte_carlo_fan_{label}.png")

        # ── Terminal Value Histogram ──
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(terminal_values, bins=100, color="#1f77b4", alpha=0.7, edgecolor="white",
                label="Bootstrap")
        ax.hist(copula_terminal, bins=100, color="#ff7f0e", alpha=0.4, edgecolor="white",
                label="t-Copula")
        ax.axvline(x=1.0, color="red", linewidth=1.5, linestyle="--", label="Break-even")
        ax.axvline(x=np.median(terminal_values), color="green", linewidth=1.5,
                  linestyle="--", label=f"Median: ${np.median(terminal_values):.2f}")
        ax.axvline(x=cte_5, color="purple", linewidth=1.5, linestyle=":",
                  label=f"CTE(5%): ${cte_5:.2f}")
        ax.set_xlabel("Terminal Portfolio Value ($1 invested)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Terminal Value Distribution ({label})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"terminal_dist_{label}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved terminal_dist_{label}.png")

    # Save statistics to JSON
    with open(output_dir / "monte_carlo_stats.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n  All Monte Carlo results saved to {output_dir}")
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Run via walk_forward.py or as a standalone module")
    logger.info("Requires: returns_df, regimes, transition_matrix, weights")
