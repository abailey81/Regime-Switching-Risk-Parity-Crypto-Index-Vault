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
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
    n_blocks = horizon_hours // block_size

    # Index returns by regime
    regime_labels = ["bull", "normal", "crisis"]
    regime_indices = {}
    for i, label in enumerate(regime_labels):
        mask = regimes == label
        valid_starts = np.where(mask)[0]
        # Filter starts where a full block is available
        valid_starts = valid_starts[valid_starts + block_size <= T]
        regime_indices[i] = valid_starts

    # Handle empty regimes
    all_valid = np.arange(T - block_size)
    for i in range(3):
        if len(regime_indices[i]) == 0:
            regime_indices[i] = all_valid

    simulated = np.zeros((n_simulations, horizon_hours, n_assets))

    for sim in range(n_simulations):
        # Sample starting regime
        current_regime = rng.choice(3, p=transition_matrix[0])

        path_returns = []
        for block in range(n_blocks):
            # Sample a block from the current regime
            valid = regime_indices[current_regime]
            if len(valid) == 0:
                valid = all_valid
            start_idx = valid[rng.randint(len(valid))]
            block_data = returns[start_idx:start_idx + block_size]
            path_returns.append(block_data)

            # Transition to next regime
            current_regime = rng.choice(3, p=transition_matrix[current_regime])

        path = np.concatenate(path_returns, axis=0)[:horizon_hours]

        # Pad if necessary
        if len(path) < horizon_hours:
            pad_len = horizon_hours - len(path)
            pad = returns[rng.randint(0, T - pad_len):rng.randint(0, T - pad_len) + pad_len]
            if len(pad) >= pad_len:
                path = np.concatenate([path, pad[:pad_len]], axis=0)

        simulated[sim, :len(path)] = path[:horizon_hours]

    return simulated


def run_monte_carlo(
    returns_df: pd.DataFrame,
    regimes: np.ndarray,
    transition_matrix: np.ndarray,
    weights: np.ndarray,
    config: dict = None,
) -> dict:
    """
    Run full Monte Carlo stress test and generate visualisations.

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

    all_results = {}

    for horizon in horizons:
        label = f"{horizon // (24 * 30)}m"  # "6m" or "12m"
        logger.info(f"\nMonte Carlo: {n_sims} paths × {horizon}h ({label})...")

        # Generate simulated return paths
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

        # Statistics
        stats = {
            "horizon": label,
            "n_simulations": n_sims,
            "terminal_value": {
                "median": float(np.median(terminal_values)),
                "p5": float(np.percentile(terminal_values, 5)),
                "p25": float(np.percentile(terminal_values, 25)),
                "p75": float(np.percentile(terminal_values, 75)),
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
        }

        all_results[label] = stats

        logger.info(f"  Terminal value: median={stats['terminal_value']['median']:.3f}, "
                    f"5th={stats['terminal_value']['p5']:.3f}, "
                    f"95th={stats['terminal_value']['p95']:.3f}")
        logger.info(f"  P(loss)={stats['probabilities']['P(loss)']:.1%}, "
                    f"P(Sharpe>1)={stats['probabilities']['P(Sharpe > 1)']:.1%}")

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
        ax.hist(terminal_values, bins=100, color="#1f77b4", alpha=0.7, edgecolor="white")
        ax.axvline(x=1.0, color="red", linewidth=1.5, linestyle="--", label="Break-even")
        ax.axvline(x=np.median(terminal_values), color="green", linewidth=1.5,
                  linestyle="--", label=f"Median: ${np.median(terminal_values):.2f}")
        ax.set_xlabel("Terminal Portfolio Value ($1 invested)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Terminal Value Distribution ({label})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"terminal_dist_{label}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved terminal_dist_{label}.png")

    # Save statistics
    import json
    with open(output_dir / "monte_carlo_stats.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n  All Monte Carlo results saved to {output_dir}")
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Run via walk_forward.py or as a standalone module")
    logger.info("Requires: returns_df, regimes, transition_matrix, weights")
