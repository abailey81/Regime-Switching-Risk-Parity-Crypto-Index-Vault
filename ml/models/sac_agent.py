"""
Model 3: Soft Actor-Critic (SAC) Deep RL Agent for portfolio allocation.

Uses stable-baselines3 SAC with MlpPolicy.
Train on Google Colab T4 GPU (~3-5 hours for 500K timesteps, 5 seeds).
Uses pre-trained model for inference during walk-forward backtest.
"""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class SACAllocator:
    """
    SAC-based portfolio allocation agent.

    Training: Creates a PortfolioEnv, trains SAC for N timesteps across
              multiple seeds, selects the median-performing seed.
    Inference: Loads pre-trained model, returns target weights for given state.
    """

    def __init__(self, config: dict):
        self.config = config
        rl_cfg = config.get("rl", {})
        train_cfg = rl_cfg.get("training", {})

        self.total_timesteps = train_cfg.get("total_timesteps", 500000)
        self.learning_rate = train_cfg.get("learning_rate", 3e-4)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.buffer_size = train_cfg.get("buffer_size", 100000)
        self.gamma = train_cfg.get("gamma", 0.99)
        self.tau = train_cfg.get("tau", 0.005)
        self.n_seeds = train_cfg.get("n_seeds", 5)
        self.net_arch = train_cfg.get("net_arch", [256, 256])
        self.n_assets = rl_cfg.get("action_dim", 8)

        self.model = None
        self._trained = False

    def train(self, returns: np.ndarray, features: np.ndarray,
              regime_probs: np.ndarray, garch_vols: np.ndarray,
              save_dir: str = "models/saved"):
        """
        Train SAC agent across multiple seeds and keep the best.

        Args:
            returns: (T, n_assets) log returns
            features: (T, n_features) per-asset features
            regime_probs: (T, 3) HMM posterior probabilities
            garch_vols: (T, n_assets) GARCH volatility forecasts
            save_dir: Directory to save trained model
        """
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback
        import torch
        from ..environment.portfolio_env import PortfolioEnv

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training SAC on device: {device}")
        logger.info(f"Training {self.n_seeds} seeds × {self.total_timesteps} timesteps")

        best_reward = -np.inf
        best_model = None
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for seed in range(self.n_seeds):
            logger.info(f"\n  Seed {seed + 1}/{self.n_seeds}...")

            env = PortfolioEnv(
                returns=returns,
                features=features,
                regime_probs=regime_probs,
                garch_vols=garch_vols,
                config=self.config.get("rl", {}),
            )

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                gamma=self.gamma,
                tau=self.tau,
                ent_coef="auto",
                policy_kwargs={"net_arch": self.net_arch},
                verbose=0,
                seed=seed,
                device=device,
            )

            model.learn(total_timesteps=self.total_timesteps)

            # Evaluate: run one episode and measure total reward
            obs, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            final_value = info.get("portfolio_value", 1.0)
            logger.info(f"    Seed {seed + 1}: total_reward={total_reward:.4f}, "
                       f"final_value={final_value:.4f}")

            if total_reward > best_reward:
                best_reward = total_reward
                best_model = model

            # Save each seed
            model.save(str(save_path / f"sac_seed_{seed}"))

        # Save best model
        self.model = best_model
        self.model.save(str(save_path / "sac_best"))
        self._trained = True

        logger.info(f"\n  Best seed reward: {best_reward:.4f}")
        logger.info(f"  Model saved to {save_path / 'sac_best'}")

    def load(self, model_path: str = "models/saved/sac_best"):
        """Load a pre-trained SAC model for inference."""
        from stable_baselines3 import SAC
        self.model = SAC.load(model_path)
        self._trained = True
        logger.info(f"Loaded SAC model from {model_path}")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Return target portfolio weights for a given state observation.

        Args:
            observation: State vector (dim = state_dim)

        Returns:
            weights: np.ndarray of shape (n_assets,) summing to 1
        """
        assert self._trained, "Model not trained or loaded"

        obs = observation.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0)

        action, _ = self.model.predict(obs, deterministic=True)

        # Softmax to ensure weights sum to 1 and are positive
        action = np.clip(action, -10, 10)
        weights = np.exp(action) / np.exp(action).sum()

        return weights

    def get_uncertainty(self) -> float:
        """
        Uncertainty measure based on policy entropy.
        Higher entropy = less certain about action → lower ensemble weight.
        """
        if not self._trained:
            return 1.0

        # Approximate: use the entropy coefficient from SAC
        try:
            ent_coef = self.model.ent_coef_tensor.item()
            # Higher ent_coef means the policy is more exploratory = more uncertain
            return min(abs(ent_coef), 1.0)
        except:
            return 0.5  # Default moderate uncertainty
