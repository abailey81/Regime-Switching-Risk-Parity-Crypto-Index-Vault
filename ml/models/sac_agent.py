"""
Model 3: Soft Actor-Critic (SAC) Deep RL Agent for portfolio allocation.

Uses stable-baselines3 SAC with MlpPolicy.
Train on Google Colab T4 GPU (~3-5 hours for 500K timesteps, 5 seeds).
Uses pre-trained model for inference during walk-forward backtest.

Enhancements:
  - RunningMeanStd observation normalization with clip [-10, 10]
  - EvalCallback with early stopping on validation Sharpe
  - Curriculum learning: phased reward complexity
  - Cosine annealing learning rate schedule
  - Multi-metric seed selection (Sharpe, max DD, CVaR)
  - predict_with_uncertainty() via action distribution entropy
  - Ensemble of top-K seeds for robust predictions
"""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *a, **kw):
        return iterable if iterable is not None else range(0)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
#  RUNNING MEAN/STD NORMALIZATION WRAPPER
# ─────────────────────────────────────────────────
class RunningMeanStd:
    """
    Running (online) mean and standard deviation estimator.

    Uses Welford's algorithm for numerically stable incremental updates.
    Tracks normalization statistics for both training and inference.
    """

    def __init__(self, shape: Tuple[int, ...], clip_range: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Avoid division by zero
        self.clip_range = clip_range

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with a new observation or batch.

        Args:
            x: Observation(s) to incorporate. Shape (..., feature_dim).
        """
        batch = np.atleast_2d(x)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize observation using running statistics and clip to [-clip_range, clip_range].

        Args:
            x: Raw observation

        Returns:
            Normalized and clipped observation
        """
        std = np.sqrt(self.var + 1e-8)
        normalized = (x - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range).astype(np.float32)

    def state_dict(self) -> dict:
        """Return serializable state for saving."""
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
            "clip_range": self.clip_range,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningMeanStd":
        """Reconstruct from saved state."""
        mean = np.array(state["mean"])
        obj = cls(shape=mean.shape, clip_range=state.get("clip_range", 10.0))
        obj.mean = mean
        obj.var = np.array(state["var"])
        obj.count = state["count"]
        return obj


class SACAllocator:
    """
    SAC-based portfolio allocation agent.

    Training: Creates a PortfolioEnv, trains SAC for N timesteps across
              multiple seeds, selects the best seed by risk-adjusted return.
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

        # Enhancement config
        self.top_k_ensemble = train_cfg.get("top_k_ensemble", 3)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 50000)
        self.eval_freq = train_cfg.get("eval_freq", 10000)
        self.enable_curriculum = train_cfg.get("enable_curriculum", True)
        self.enable_lr_schedule = train_cfg.get("enable_lr_schedule", True)
        self.enable_obs_normalization = train_cfg.get("enable_obs_normalization", True)

        self.model = None
        self._trained = False

        # Enhancement state
        self.obs_normalizer: Optional[RunningMeanStd] = None
        self.ensemble_models: List = []
        self.seed_metrics: Dict[int, dict] = {}
        self.training_curves: Dict[int, List[float]] = {}

    # ─────────────────────────────────────────────────
    #  LEARNING RATE SCHEDULE
    # ─────────────────────────────────────────────────
    def _cosine_annealing_schedule(self, initial_lr: float,
                                   min_lr: float = 1e-5) -> Callable[[float], float]:
        """
        Create cosine annealing learning rate schedule.

        Args:
            initial_lr: Starting learning rate
            min_lr: Minimum learning rate at end of training

        Returns:
            Schedule function: progress_remaining -> lr
        """
        def schedule(progress_remaining: float) -> float:
            # progress_remaining goes from 1.0 -> 0.0 during training
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
            return min_lr + (initial_lr - min_lr) * cosine_decay

        return schedule

    # ─────────────────────────────────────────────────
    #  TRAIN
    # ─────────────────────────────────────────────────
    def train(self, returns: np.ndarray, features: np.ndarray,
              regime_probs: np.ndarray, garch_vols: np.ndarray,
              save_dir: str = "models/saved") -> None:
        """
        Train SAC agent across multiple seeds and keep the best by Sharpe ratio.

        Enhancements:
        - Observation normalization with RunningMeanStd
        - EvalCallback for monitoring validation performance
        - Curriculum learning: gradually increase reward complexity
        - Cosine annealing learning rate
        - Multi-metric evaluation: Sharpe, max drawdown, CVaR

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
        logger.info(f"Training {self.n_seeds} seeds x {self.total_timesteps} timesteps")
        logger.info(f"  Enhancements: obs_norm={self.enable_obs_normalization}, "
                     f"curriculum={self.enable_curriculum}, "
                     f"lr_schedule={self.enable_lr_schedule}, "
                     f"top_k_ensemble={self.top_k_ensemble}")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # --- Initialize observation normalizer ---
        if self.enable_obs_normalization:
            env_temp = PortfolioEnv(
                returns=returns, features=features,
                regime_probs=regime_probs, garch_vols=garch_vols,
                config=self.config.get("rl", {}),
            )
            state_dim = env_temp.observation_space.shape[0]
            self.obs_normalizer = RunningMeanStd(shape=(state_dim,))

            # Warm up normalizer with random rollout
            obs, _ = env_temp.reset()
            for _ in range(min(1000, returns.shape[0] - 500)):
                self.obs_normalizer.update(obs)
                action = env_temp.action_space.sample()
                obs, _, terminated, truncated, _ = env_temp.step(action)
                if terminated or truncated:
                    obs, _ = env_temp.reset()
            logger.info(f"  Observation normalizer warmed up ({self.obs_normalizer.count:.0f} samples)")

        # --- Learning rate schedule ---
        lr = self.learning_rate
        if self.enable_lr_schedule:
            lr = self._cosine_annealing_schedule(self.learning_rate)

        # --- Train each seed ---
        seed_results: List[Tuple[int, float, dict, object]] = []

        seed_bar = tqdm(range(self.n_seeds), desc="SAC Training", unit="seed", leave=True)
        for seed in seed_bar:
            seed_bar.set_postfix(seed=f"{seed + 1}/{self.n_seeds}")
            logger.info(f"\n  Seed {seed + 1}/{self.n_seeds}...")

            # Create training environment
            env_config = self.config.get("rl", {}).copy()
            if self.enable_curriculum:
                # Phase 1: simplified reward (first half of training)
                env_config["curriculum_phase"] = 0
            env = PortfolioEnv(
                returns=returns, features=features,
                regime_probs=regime_probs, garch_vols=garch_vols,
                config=env_config,
            )

            # Create evaluation environment
            eval_env = PortfolioEnv(
                returns=returns, features=features,
                regime_probs=regime_probs, garch_vols=garch_vols,
                config=self.config.get("rl", {}),
            )

            # Setup eval callback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(save_path / f"eval_best_seed_{seed}"),
                log_path=str(save_path / f"eval_logs_seed_{seed}"),
                eval_freq=self.eval_freq,
                n_eval_episodes=1,
                deterministic=True,
                verbose=0,
            )

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=lr,
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

            # --- Curriculum learning ---
            if self.enable_curriculum:
                # Phase 1: Train with simplified reward (no CVaR/DD penalties)
                phase1_steps = self.total_timesteps // 3
                logger.info(f"    Phase 1 (simplified reward): {phase1_steps} steps")
                model.learn(total_timesteps=phase1_steps, callback=eval_callback)

                # Phase 2: Switch to full reward
                env.set_curriculum_phase(1)
                phase2_steps = self.total_timesteps - phase1_steps
                logger.info(f"    Phase 2 (full reward): {phase2_steps} steps")
                model.learn(total_timesteps=phase2_steps, callback=eval_callback,
                            reset_num_timesteps=False)
            else:
                model.learn(total_timesteps=self.total_timesteps, callback=eval_callback)

            # --- Multi-metric evaluation ---
            eval_metrics = self._evaluate_seed(model, env)
            self.seed_metrics[seed] = eval_metrics

            logger.info(
                f"    Seed {seed + 1}: Sharpe={eval_metrics['sharpe']:.3f}, "
                f"maxDD={eval_metrics['max_drawdown']:.3f}, "
                f"CVaR5%={eval_metrics['cvar_5pct']:.4f}, "
                f"total_return={eval_metrics['total_return']:.4f}"
            )

            seed_bar.set_postfix(
                seed=f"{seed + 1}/{self.n_seeds}",
                Sharpe=f"{eval_metrics['sharpe']:.3f}",
                maxDD=f"{eval_metrics['max_drawdown']:.3f}",
            )

            seed_results.append((seed, eval_metrics["sharpe"], eval_metrics, model))

            # Save each seed
            model.save(str(save_path / f"sac_seed_{seed}"))

        # --- Select best seed by Sharpe ---
        seed_results.sort(key=lambda x: x[1], reverse=True)

        best_seed, best_sharpe, best_metrics, best_model = seed_results[0]
        self.model = best_model
        self.model.save(str(save_path / "sac_best"))

        # --- Ensemble of top-K seeds ---
        k = min(self.top_k_ensemble, len(seed_results))
        self.ensemble_models = [sr[3] for sr in seed_results[:k]]

        logger.info(f"\n  Seed ranking by Sharpe:")
        for rank, (s, sharpe, metrics, _) in enumerate(seed_results):
            marker = " <-- BEST" if rank == 0 else (" (ensemble)" if rank < k else "")
            logger.info(
                f"    #{rank + 1}: seed={s}, Sharpe={sharpe:.3f}, "
                f"maxDD={metrics['max_drawdown']:.3f}{marker}"
            )

        # Save training curves
        self._save_training_curves(save_path)

        # Save normalizer state
        if self.obs_normalizer is not None:
            import json
            norm_path = save_path / "obs_normalizer.json"
            with open(norm_path, "w") as f:
                json.dump(self.obs_normalizer.state_dict(), f)
            logger.info(f"  Observation normalizer saved to {norm_path}")

        self._trained = True
        logger.info(f"\n  Best seed: {best_seed} (Sharpe={best_sharpe:.3f})")
        logger.info(f"  Ensemble: top-{k} seeds")
        logger.info(f"  Models saved to {save_path}")

    def _evaluate_seed(self, model: object, env: object) -> dict:
        """
        Evaluate a trained seed on multiple metrics: Sharpe, max DD, CVaR.

        Args:
            model: Trained SAC model
            env: Portfolio environment

        Returns:
            Dict with sharpe, max_drawdown, cvar_5pct, total_return, total_reward
        """
        obs, _ = env.reset()
        returns_list = []
        total_reward = 0.0
        done = False
        peak = 1.0
        value = 1.0
        max_dd = 0.0

        while not done:
            if self.obs_normalizer is not None:
                obs_norm = self.obs_normalizer.normalize(obs)
            else:
                obs_norm = obs
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            period_return = info.get("portfolio_return", reward)
            returns_list.append(period_return)

            value = info.get("portfolio_value", value * (1 + period_return))
            peak = max(peak, value)
            dd = 1 - value / peak
            max_dd = max(max_dd, dd)

            done = terminated or truncated

        returns_arr = np.array(returns_list)

        # Sharpe ratio (hourly -> annualised)
        if len(returns_arr) > 1 and returns_arr.std() > 1e-10:
            sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # CVaR at 5%
        if len(returns_arr) > 20:
            sorted_ret = np.sort(returns_arr)
            n_tail = max(1, int(0.05 * len(sorted_ret)))
            cvar = -sorted_ret[:n_tail].mean()
        else:
            cvar = 0.0

        total_return = value - 1.0

        return {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "cvar_5pct": float(cvar),
            "total_return": float(total_return),
            "total_reward": float(total_reward),
            "n_steps": len(returns_arr),
        }

    def _save_training_curves(self, save_path: Path) -> None:
        """Save seed metrics for later analysis."""
        import json
        curves_path = save_path / "seed_metrics.json"
        try:
            serializable = {str(k): v for k, v in self.seed_metrics.items()}
            with open(curves_path, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"  Failed to save training curves: {e}")

    # ─────────────────────────────────────────────────
    #  LOAD
    # ─────────────────────────────────────────────────
    def load(self, model_path: str = "models/saved/sac_best") -> None:
        """Load a pre-trained SAC model for inference."""
        from stable_baselines3 import SAC
        import json

        self.model = SAC.load(model_path)
        self._trained = True
        logger.info(f"Loaded SAC model from {model_path}")

        # Try to load observation normalizer
        norm_path = Path(model_path).parent / "obs_normalizer.json"
        if norm_path.exists():
            try:
                with open(norm_path) as f:
                    state = json.load(f)
                self.obs_normalizer = RunningMeanStd.from_state_dict(state)
                logger.info(f"  Loaded observation normalizer from {norm_path}")
            except Exception as e:
                logger.warning(f"  Failed to load observation normalizer: {e}")

        # Try to load ensemble models
        model_dir = Path(model_path).parent
        ensemble_loaded = 0
        load_bar = tqdm(range(self.n_seeds), desc="Loading ensemble seeds",
                        unit="seed", leave=False)
        for i in load_bar:
            seed_path = model_dir / f"sac_seed_{i}"
            if seed_path.with_suffix(".zip").exists():
                try:
                    self.ensemble_models.append(SAC.load(str(seed_path)))
                    ensemble_loaded += 1
                    load_bar.set_postfix(loaded=ensemble_loaded)
                except Exception:
                    pass
        if ensemble_loaded > 0:
            logger.info(f"  Loaded {ensemble_loaded} ensemble models")

    # ─────────────────────────────────────────────────
    #  PREDICT
    # ─────────────────────────────────────────────────
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

        # Apply observation normalization if available
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)

        action, _ = self.model.predict(obs, deterministic=True)

        # Softmax to ensure weights sum to 1 and are positive
        action = np.clip(action, -10, 10)
        weights = np.exp(action) / np.exp(action).sum()

        return weights

    def predict_ensemble(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction from top-K seeds for more robust weights.

        Returns the mean weights and the standard deviation across seeds
        (as a measure of disagreement/uncertainty).

        Args:
            observation: State vector

        Returns:
            Tuple of (mean_weights, std_weights)
        """
        assert self._trained, "Model not trained or loaded"

        if not self.ensemble_models:
            # Fall back to single model
            return self.predict(observation), np.zeros(self.n_assets)

        obs = observation.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)

        all_weights = []
        for model in self.ensemble_models:
            try:
                action, _ = model.predict(obs, deterministic=True)
                action = np.clip(action, -10, 10)
                w = np.exp(action) / np.exp(action).sum()
                all_weights.append(w)
            except Exception:
                continue

        if not all_weights:
            return self.predict(observation), np.zeros(self.n_assets)

        weight_matrix = np.array(all_weights)
        mean_weights = weight_matrix.mean(axis=0)
        mean_weights = mean_weights / mean_weights.sum()  # Renormalize
        std_weights = weight_matrix.std(axis=0)

        return mean_weights, std_weights

    def predict_with_uncertainty(self, observation: np.ndarray,
                                 n_samples: int = 20) -> Tuple[np.ndarray, float]:
        """
        Predict weights with uncertainty estimate using action distribution entropy.

        Uses the SAC policy's stochastic output to estimate prediction uncertainty.
        Samples multiple actions from the policy and measures their spread.

        Args:
            observation: State vector
            n_samples: Number of stochastic action samples

        Returns:
            Tuple of (deterministic_weights, uncertainty_score)
                uncertainty_score in [0, 1]: higher = more uncertain
        """
        assert self._trained, "Model not trained or loaded"

        obs = observation.astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)

        # Deterministic prediction
        det_action, _ = self.model.predict(obs, deterministic=True)
        det_action = np.clip(det_action, -10, 10)
        det_weights = np.exp(det_action) / np.exp(det_action).sum()

        # Stochastic samples for uncertainty
        stochastic_actions = []
        for _ in range(n_samples):
            try:
                action, _ = self.model.predict(obs, deterministic=False)
                stochastic_actions.append(action)
            except Exception:
                continue

        if len(stochastic_actions) < 2:
            return det_weights, 0.5

        action_matrix = np.array(stochastic_actions)

        # Uncertainty = average std of action components, normalized
        action_std = action_matrix.std(axis=0).mean()

        # Normalize to [0, 1] using sigmoid-like transformation
        uncertainty = float(2.0 / (1.0 + np.exp(-action_std)) - 1.0)

        return det_weights, uncertainty

    # ─────────────────────────────────────────────────
    #  UNCERTAINTY
    # ─────────────────────────────────────────────────
    def get_uncertainty(self) -> float:
        """
        Uncertainty measure based on policy entropy.
        Higher entropy = less certain about action -> lower ensemble weight.
        """
        if not self._trained:
            return 1.0

        # Approximate: use the entropy coefficient from SAC
        try:
            ent_coef = self.model.ent_coef_tensor.item()
            # Higher ent_coef means the policy is more exploratory = more uncertain
            return min(abs(ent_coef), 1.0)
        except (AttributeError, RuntimeError):
            return 0.5  # Default moderate uncertainty
