"""
Run the full ensemble pipeline: data → models → weights → Merkle root.

This is the main entry point for the keeper bot.
Called periodically (e.g., daily) to compute new portfolio weights
and prepare them for on-chain commitment.
"""
import logging
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from ..data.preprocess import prepare_all_data
from ..models.garch_dcc import StudentTGarchDCC
from ..models.bayesian_hmm import BayesianRegimeHMM
from ..models.ensemble import EnsembleCombiner
from .merkle import compute_merkle_root

logger = logging.getLogger(__name__)


def compute_weights(
    config_path: str = "config.yaml",
    use_rl: bool = False,
    rl_model_path: str = "models/saved/sac_best",
) -> dict:
    """
    Full pipeline: preprocess → GARCH-DCC → HMM → (optional RL) → Ensemble → Merkle.

    Args:
        config_path: Path to config.yaml
        use_rl: Whether to include SAC RL agent (requires trained model)
        rl_model_path: Path to pre-trained SAC model

    Returns:
        dict with keys:
          - weights: dict mapping asset -> weight (0-1)
          - weights_bps: dict mapping asset -> weight in basis points
          - merkle_root: hex string
          - regime: current regime classification
          - metadata: model diagnostics
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    asset_names = config["data"]["assets"]
    logger.info("=" * 60)
    logger.info("  WEIGHT COMPUTATION PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load preprocessed data ──
    logger.info("\n1. Loading preprocessed data...")
    prices_df, returns_df, features_df, hmm_features = prepare_all_data(config_path)

    # ── 2. Fit GARCH-DCC ──
    logger.info("\n2. Fitting Student-t GARCH-DCC...")
    garch = StudentTGarchDCC(
        p=config.get("garch", {}).get("p", 1),
        q=config.get("garch", {}).get("q", 1),
        distribution=config.get("garch", {}).get("distribution", "studentst"),
    )
    garch.fit(returns_df)
    sigma = garch.forecast_covariance()
    garch_rp_weights = garch.get_risk_parity_weights(sigma)
    garch_uncertainty = garch.get_uncertainty()

    logger.info(f"  GARCH risk-parity weights: {dict(zip(asset_names, garch_rp_weights.round(4)))}")

    # ── 3. Fit HMM ──
    logger.info("\n3. Fitting Bayesian HMM...")
    hmm_model = BayesianRegimeHMM(
        n_states=config.get("hmm", {}).get("n_states", 3),
    )
    hmm_model.fit(hmm_features)
    regime_probs = hmm_model.predict_proba(hmm_features)
    latest_probs = regime_probs.iloc[-1].values
    hmm_uncertainty = hmm_model.get_uncertainty()

    logger.info(f"  Latest regime probs: bull={latest_probs[0]:.3f}, "
                f"normal={latest_probs[1]:.3f}, crisis={latest_probs[2]:.3f}")

    # ── 4. (Optional) SAC RL Agent ──
    if use_rl:
        logger.info("\n4. Loading SAC RL agent...")
        from ..models.sac_agent import SACAllocator
        sac = SACAllocator(config)
        sac.load(rl_model_path)
        # Build observation for latest timestep
        obs = np.zeros(38, dtype=np.float32)  # Placeholder
        rl_weights = sac.predict(obs)
        rl_uncertainty = sac.get_uncertainty()
    else:
        logger.info("\n4. SAC RL agent: SKIPPED (use --rl flag to enable)")
        rl_weights = np.ones(len(asset_names)) / len(asset_names)
        rl_uncertainty = 1.0  # Max uncertainty → minimal influence

    # ── 5. Ensemble Combiner ──
    logger.info("\n5. Running ensemble combiner...")
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

    # ── 6. Compute Merkle Root ──
    # Note: In production, token addresses would be used.
    # For now, use placeholder addresses for demonstration.
    logger.info("\n6. Computing Merkle root...")
    placeholder_addresses = [f"0x{'0' * 39}{i}" for i in range(len(asset_names))]
    bps_list = [weights_bps[a] for a in asset_names]
    merkle_root = compute_merkle_root(placeholder_addresses, bps_list)
    logger.info(f"  Merkle root: {merkle_root[:18]}...")

    # ── 7. Save output ──
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "weights": weights_dict,
        "weights_bps": weights_bps,
        "merkle_root": merkle_root,
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

    output_path = Path("results") / "latest_weights.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Saved to {output_path}")
    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute portfolio weights")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rl", action="store_true", help="Include SAC RL agent")
    parser.add_argument("--rl-model", default="models/saved/sac_best")
    args = parser.parse_args()

    result = compute_weights(args.config, use_rl=args.rl, rl_model_path=args.rl_model)
    print(f"\nFinal weights: {result['weights']}")
    print(f"Regime: {result['regime']}")
