"""
Custom Gymnasium Environment for SAC Portfolio Allocation.

State space (dim=38):
  - Per-asset features: 5d returns, 20d returns, 20d vol, GARCH vol forecast (8 assets × 4 = 32)
  - HMM regime probabilities (3)
  - Portfolio: cumulative return, drawdown, days since rebalance (3)
  Total: 38

Action space (dim=8):
  - Continuous [0,1]^8 → softmax → target weights summing to 1

Reward:
  r_t = sharpe_coeff * portfolio_return
      - cvar_penalty * max(0, CVaR_threshold - rolling_CVaR)
      - drawdown_penalty * max(0, drawdown - threshold)
      - turnover_penalty * |w_new - w_old|_1
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio allocation environment.

    Simulates daily rebalancing with transaction costs, realistic constraints,
    and composite reward function for training the SAC agent.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,        # (T, n_assets) log returns
        features: np.ndarray,       # (T, n_features) per-asset features
        regime_probs: np.ndarray,   # (T, 3) HMM posterior probabilities
        garch_vols: np.ndarray,     # (T, n_assets) GARCH vol forecasts
        config: dict = None,
        rebalance_freq: int = 24,   # Rebalance every 24 hours
        lookback: int = 0,          # No lookback needed (features pre-computed)
    ):
        super().__init__()

        self.returns = returns
        self.features = features
        self.regime_probs = regime_probs
        self.garch_vols = garch_vols
        self.n_assets = returns.shape[1]
        self.n_steps = returns.shape[0]
        self.rebalance_freq = rebalance_freq

        # Config
        cfg = config or {}
        reward_cfg = cfg.get("reward", {})
        self.sharpe_coeff = reward_cfg.get("sharpe_coeff", 1.0)
        self.cvar_penalty = reward_cfg.get("cvar_penalty", 2.0)
        self.dd_penalty = reward_cfg.get("drawdown_penalty", 3.0)
        self.dd_threshold = reward_cfg.get("drawdown_threshold", 0.10)
        self.turnover_penalty = reward_cfg.get("turnover_penalty", 0.5)
        self.tc_bps = cfg.get("transaction_cost_bps", 10) / 10000

        # Spaces
        # State: per-asset features (n_assets * 4) + regime (3) + portfolio (3)
        state_dim = self.n_assets * 4 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Tracking
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = max(480, 1)  # Start after enough history for features
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.cumulative_return = 0.0
        self.return_history = []
        self.steps_since_rebalance = 0

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # Convert action to weights via softmax
        action = np.clip(action, 1e-6, None)
        new_weights = np.exp(action) / np.exp(action).sum()

        # Transaction costs from turnover
        turnover = np.abs(new_weights - self.weights).sum()
        tc = turnover * self.tc_bps

        # Portfolio return (using next period's returns)
        if self.step_idx >= self.n_steps:
            return self._get_observation(), 0.0, True, False, {}

        period_returns = self.returns[self.step_idx]
        portfolio_return = np.dot(new_weights, period_returns) - tc

        # Update portfolio state
        self.portfolio_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.cumulative_return = self.portfolio_value - 1.0
        drawdown = 1 - self.portfolio_value / self.peak_value

        self.return_history.append(portfolio_return)
        self.weights = new_weights
        self.step_idx += 1
        self.steps_since_rebalance += 1

        # ── Compute reward ──
        reward = self._compute_reward(portfolio_return, drawdown, turnover)

        # Check termination
        terminated = self.step_idx >= self.n_steps - 1
        truncated = False

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "drawdown": drawdown,
            "turnover": turnover,
            "weights": new_weights.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, ret, drawdown, turnover):
        """
        Composite reward function:
          + sharpe_coeff * return (encourages positive returns)
          - cvar_penalty * CVaR shortfall (penalises tail risk)
          - dd_penalty * excess drawdown (penalises drawdowns beyond threshold)
          - turnover_penalty * turnover (penalises excessive trading)
        """
        reward = self.sharpe_coeff * ret

        # CVaR penalty (rolling 480-hour = 20-day CVaR at 5%)
        if len(self.return_history) > 100:
            sorted_returns = np.sort(self.return_history[-480:])
            n_tail = max(1, int(0.05 * len(sorted_returns)))
            cvar = -sorted_returns[:n_tail].mean()
            reward -= self.cvar_penalty * max(0, cvar - 0.02)  # Penalise if CVaR > 2%

        # Drawdown penalty
        if drawdown > self.dd_threshold:
            reward -= self.dd_penalty * (drawdown - self.dd_threshold)

        # Turnover penalty
        reward -= self.turnover_penalty * turnover

        return float(reward)

    def _get_observation(self):
        """Construct state vector from current market data and portfolio state."""
        t = min(self.step_idx, self.n_steps - 1)

        # Per-asset features (pre-computed): returns_5d, returns_20d, vol_20d, garch_vol
        # Shape: (n_assets * 4,)
        if t < self.features.shape[0]:
            asset_features = self.features[t].flatten()[:self.n_assets * 3]
        else:
            asset_features = np.zeros(self.n_assets * 3)

        # GARCH vol forecasts
        if t < self.garch_vols.shape[0]:
            garch_feat = self.garch_vols[t]
        else:
            garch_feat = np.zeros(self.n_assets)

        # HMM regime probabilities
        if t < self.regime_probs.shape[0]:
            regime = self.regime_probs[t]
        else:
            regime = np.array([0.33, 0.34, 0.33])

        # Portfolio state
        drawdown = 1 - self.portfolio_value / self.peak_value if self.peak_value > 0 else 0
        portfolio_state = np.array([
            self.cumulative_return,
            drawdown,
            self.steps_since_rebalance / 24.0,  # Normalise to days
        ])

        obs = np.concatenate([
            asset_features,
            garch_feat,
            regime,
            portfolio_state,
        ]).astype(np.float32)

        # Pad or truncate to expected dimension
        expected_dim = self.observation_space.shape[0]
        if len(obs) < expected_dim:
            obs = np.pad(obs, (0, expected_dim - len(obs)))
        elif len(obs) > expected_dim:
            obs = obs[:expected_dim]

        # Replace NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs
