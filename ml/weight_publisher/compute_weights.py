"""
Run the full ensemble pipeline: data -> models -> weights -> Merkle root.

This is the main entry point for the keeper bot.
Called periodically (e.g., daily) to compute new portfolio weights
and prepare them for on-chain commitment.

Enhancements:
  - Per-step error handling with graceful fallbacks
  - Intermediate result saving (GARCH fit, HMM fit) for debugging
  - --dry-run flag that computes weights but does not publish
  - Proper SAC observation construction
"""
import logging
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..data.preprocess import prepare_all_data
from ..models.garch_dcc import StudentTGarchDCC
from ..models.bayesian_hmm import BayesianRegimeHMM
from ..models.ensemble import EnsembleCombiner
from .merkle import compute_merkle_tree

logger = logging.getLogger(__name__)


def _save_intermediate(name: str, data: dict, output_dir: Path) -> None:
    """Save intermediate model results for debugging."""
    path = output_dir / f"intermediate_{name}.json"
    try:
        serialisable = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                serialisable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serialisable[k] = float(v)
            else:
                serialisable[k] = v
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        logger.info(f"  Intermediate saved: {path}")
    except Exception as e:
        logger.warning(f"  Failed to save intermediate {name}: {e}")


def compute_weights(
    config_path: str = "config.yaml",
    use_rl: bool = False,
    rl_model_path: str = "models/saved/sac_best",
    dry_run: bool = False,
) -> dict:
    """
    Full pipeline: preprocess -> GARCH-DCC -> HMM -> (optional RL) -> Ensemble -> Merkle.

    Args:
        config_path: Path to config.yaml
        use_rl: Whether to include SAC RL agent (requires trained model)
        rl_model_path: Path to pre-trained SAC model
        dry_run: If True, compute weights but do not generate Merkle root or save

    Returns:
        dict with keys:
          - weights: dict mapping asset -> weight (0-1)
          - weights_bps: dict mapping asset -> weight in basis points
          - merkle_root: hex string (None if dry_run)
          - regime: current regime classification
          - metadata: model diagnostics
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    asset_names = config["data"]["assets"]
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  WEIGHT COMPUTATION PIPELINE")
    if dry_run:
        logger.info("  MODE: DRY RUN (no publishing)")
    logger.info("=" * 60)

    # ── 1. Load preprocessed data ──
    logger.info("\n1. Loading preprocessed data...")
    try:
        prices_df, returns_df, features_df, hmm_features = prepare_all_data(config_path)
    except Exception as e:
        logger.error(f"  FATAL: Data loading failed: {e}")
        raise RuntimeError(f"Data pipeline failed: {e}") from e

    _save_intermediate("data_summary", {
        "n_rows": len(returns_df),
        "n_assets": len(returns_df.columns),
        "date_start": str(returns_df.index[0]),
        "date_end": str(returns_df.index[-1]),
        "assets": list(returns_df.columns),
    }, output_dir)

    # ── 2. Fit GARCH-DCC ──
    logger.info("\n2. Fitting Student-t GARCH-DCC...")
    try:
        garch = StudentTGarchDCC(
            p=config.get("garch", {}).get("p", 1),
            q=config.get("garch", {}).get("q", 1),
            distribution=config.get("garch", {}).get("distribution", "studentst"),
        )
        garch.fit(returns_df)
        sigma = garch.forecast_covariance()
        garch_rp_weights = garch.get_risk_parity_weights(sigma)
        garch_uncertainty = garch.get_uncertainty()

        _save_intermediate("garch_fit", {
            "rp_weights": dict(zip(asset_names, garch_rp_weights.round(6).tolist())),
            "uncertainty": float(garch_uncertainty),
            "covariance_diagonal": np.diag(sigma).tolist(),
        }, output_dir)

        logger.info(f"  GARCH risk-parity weights: {dict(zip(asset_names, garch_rp_weights.round(4)))}")
    except Exception as e:
        logger.error(f"  GARCH failed: {e}, falling back to equal weights")
        sigma = np.eye(len(asset_names)) * 0.01
        garch_rp_weights = np.ones(len(asset_names)) / len(asset_names)
        garch_uncertainty = 1.0

    # ── 3. Fit HMM ──
    logger.info("\n3. Fitting Bayesian HMM...")
    try:
        hmm_model = BayesianRegimeHMM(
            n_states=config.get("hmm", {}).get("n_states", 3),
        )
        hmm_model.fit(hmm_features)
        regime_probs = hmm_model.predict_proba(hmm_features)
        latest_probs = regime_probs.iloc[-1].values
        hmm_uncertainty = hmm_model.get_uncertainty()

        _save_intermediate("hmm_fit", {
            "latest_probs": {"bull": float(latest_probs[0]),
                            "normal": float(latest_probs[1]),
                            "crisis": float(latest_probs[2])},
            "uncertainty": float(hmm_uncertainty),
            "n_states": config.get("hmm", {}).get("n_states", 3),
        }, output_dir)

        logger.info(f"  Latest regime probs: bull={latest_probs[0]:.3f}, "
                    f"normal={latest_probs[1]:.3f}, crisis={latest_probs[2]:.3f}")
    except Exception as e:
        logger.error(f"  HMM failed: {e}, falling back to uniform probs")
        latest_probs = np.array([0.33, 0.34, 0.33])
        hmm_uncertainty = 1.0

    # ── 4. (Optional) SAC RL Agent ──
    if use_rl:
        logger.info("\n4. Loading SAC RL agent...")
        try:
            from ..models.sac_agent import SACAllocator
            sac = SACAllocator(config)
            sac.load(rl_model_path)

            # Build proper observation from latest data
            from ..backtest.walk_forward import _build_sac_observation
            obs = _build_sac_observation(
                returns_df=returns_df,
                features_df=features_df,
                hmm_features=hmm_features,
                regime_probs=latest_probs,
                t=len(returns_df) - 1,
                n_assets=len(asset_names),
                cumulative_return=0.0,
                drawdown=0.0,
                steps_since_rebalance=0,
                return_history=[],
                config=config,
            )

            rl_weights = sac.predict(obs)
            rl_uncertainty = sac.get_uncertainty()

            _save_intermediate("sac_inference", {
                "weights": dict(zip(asset_names, rl_weights.round(6).tolist())),
                "uncertainty": float(rl_uncertainty),
            }, output_dir)

            logger.info(f"  SAC weights: {dict(zip(asset_names, rl_weights.round(4)))}")
        except Exception as e:
            logger.error(f"  SAC agent failed: {e}, using equal weights")
            rl_weights = np.ones(len(asset_names)) / len(asset_names)
            rl_uncertainty = 1.0
    else:
        logger.info("\n4. SAC RL agent: SKIPPED (use --rl flag to enable)")
        rl_weights = np.ones(len(asset_names)) / len(asset_names)
        rl_uncertainty = 1.0  # Max uncertainty -> minimal influence

    # ── 5. Ensemble Combiner ──
    logger.info("\n5. Running ensemble combiner...")
    try:
        ensemble = EnsembleCombiner(config=config, asset_names=asset_names)
        current_weights = np.ones(len(asset_names)) / len(asset_names)  # Initial equal weight

        result = ensemble.combine(
            garch_rp_weights=garch_rp_weights,
            hmm_regime_probs=latest_probs,
            rl_weights=rl_weights,
            covariance_matrix=sigma,
            current_weights=current_weights,
            current_nav=1.0,
            garch_uncertainty=garch_uncertainty,
            hmm_uncertainty=hmm_uncertainty,
            rl_uncertainty=rl_uncertainty,
        )

        final_weights = result["weights"]
        weights_dict = dict(zip(asset_names, final_weights.round(6)))
        weights_bps = {a: int(w * 10000) for a, w in weights_dict.items()}

        logger.info(f"\n  Final weights: {weights_dict}")
        logger.info(f"  Regime: {result['regime']}")
        logger.info(f"  Circuit breaker: {result['circuit_breaker']}")
        logger.info(f"  Model contributions: {result['model_contributions']}")
    except Exception as e:
        logger.error(f"  Ensemble combiner failed: {e}")
        raise RuntimeError(f"Ensemble combiner failed: {e}") from e

    # ── 6. Compute Merkle Root ──
    merkle_root = None
    merkle_proofs = None
    if not dry_run:
        logger.info("\n6. Computing Merkle tree...")
        try:
            placeholder_addresses = [f"0x{'0' * 39}{i}" for i in range(len(asset_names))]
            bps_list = [weights_bps[a] for a in asset_names]
            tree_result = compute_merkle_tree(placeholder_addresses, bps_list)
            merkle_root = tree_result["root"]
            merkle_proofs = tree_result["proofs"]
            logger.info(f"  Merkle root: {merkle_root[:18]}...")
        except Exception as e:
            logger.error(f"  Merkle tree computation failed: {e}")
            merkle_root = "0x" + "0" * 64  # Fallback null root
            merkle_proofs = []
    else:
        logger.info("\n6. Merkle tree: SKIPPED (dry run)")

    # ── 7. Save output ──
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "weights": weights_dict,
        "weights_bps": weights_bps,
        "merkle_root": merkle_root,
        "merkle_proofs": merkle_proofs,
        "regime": result["regime"],
        "regime_probs": result.get("regime_probs", {}),
        "circuit_breaker": result["circuit_breaker"],
        "model_contributions": result["model_contributions"],
        "metadata": {
            "garch_uncertainty": float(garch_uncertainty),
            "hmm_uncertainty": float(hmm_uncertainty),
            "rl_uncertainty": float(rl_uncertainty),
            "drawdown": float(result.get("drawdown", 0)),
        },
    }

    if not dry_run:
        output_path = output_dir / "latest_weights.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"\n  Saved to {output_path}")
    else:
        logger.info("\n  Dry run complete -- weights NOT saved to disk")

    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute portfolio weights")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rl", action="store_true", help="Include SAC RL agent")
    parser.add_argument("--rl-model", default="models/saved/sac_best")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute weights without publishing or saving")
    args = parser.parse_args()

    result = compute_weights(args.config, use_rl=args.rl,
                            rl_model_path=args.rl_model, dry_run=args.dry_run)
    print(f"\nFinal weights: {result['weights']}")
    print(f"Regime: {result['regime']}")
    if result.get("merkle_root"):
        print(f"Merkle root: {result['merkle_root']}")
