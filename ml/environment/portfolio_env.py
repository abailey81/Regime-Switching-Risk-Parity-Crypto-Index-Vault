"""
Custom Gymnasium Environment for SAC Portfolio Allocation.

State space (dim=48):
  Per-asset features:
    [0..7]   5d returns          (8 assets)
    [8..15]  20d returns         (8 assets)
    [16..23] 20d volatility      (8 assets)
    [24..31] GARCH vol forecast  (8 assets)
    [32..39] Rolling corr w/ BTC (8 assets)  [NEW]
  Regime & portfolio:
    [40..42] HMM regime probs    (3 values: P(bull), P(normal), P(crisis))
    [43]     Regime transition P  (1 value)  [NEW]
    [44]     Cumulative return   (1 value)
    [45]     Current drawdown    (1 value)
    [46]     Days since rebal    (1 value)
    [47]     Current Kelly frac  (1 value)  [NEW]
  Total: 48

Action space (dim=8):
  - Continuous [0,1]^8 -> softmax -> target weights summing to 1

Reward:
  r_t = sharpe_coeff * portfolio_return
      - cvar_penalty * max(0, CVaR_threshold - rolling_CVaR)
      - drawdown_penalty * max(0, drawdown - threshold)
      - turnover_penalty * |w_new - w_old|_1
      - risk_parity_penalty * risk_contribution_deviation    [NEW]
      - concentration_penalty * max(0, HHI - hhi_threshold)  [NEW]

Simulation:
  - Square-root market impact model                         [NEW]
  - Slippage proportional to position size and volatility   [NEW]
  - Configurable episode length                             [NEW]
  - Realized Sharpe tracking during episode                 [NEW]
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio allocation environment.

    Simulates daily rebalancing with transaction costs, market impact,
    slippage, and composite reward function for training the SAC agent.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,        # (T, n_assets) log returns
        features: np.ndarray,       # (T, n_features) per-asset features
        regime_probs: np.ndarray,   # (T, 3) HMM posterior probabilities
        garch_vols: np.ndarray,     # (T, n_assets) GARCH vol forecasts
        config: Optional[dict] = None,
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

        # --- NEW: enhanced reward penalties ---
        self.risk_parity_penalty = reward_cfg.get("risk_parity_penalty", 1.0)
        self.concentration_penalty_coeff = reward_cfg.get("concentration_penalty", 0.5)
        self.hhi_threshold = reward_cfg.get("hhi_threshold", 0.25)
        self.reward_clip = reward_cfg.get("reward_clip", 5.0)

        # --- NEW: market microstructure ---
        impact_cfg = cfg.get("market_impact", {})
        self.impact_coeff = impact_cfg.get("coefficient", 0.1)
        self.impact_exponent = impact_cfg.get("exponent", 0.5)  # Square-root model
        self.slippage_vol_mult = impact_cfg.get("slippage_vol_mult", 0.05)

        # --- NEW: episode management ---
        self.episode_length = cfg.get("episode_length", 0)  # 0 = use all data
        self.start_offset = max(cfg.get("start_offset", 480), 1)

        # --- NEW: curriculum learning phase ---
        self._curriculum_phase = cfg.get("curriculum_phase", 1)  # 0=simplified, 1=full

        # --- NEW: BTC correlation lookback ---
        self.btc_corr_lookback = cfg.get("btc_corr_lookback", 168)  # 7 days in hours

        # Spaces
        # State: per-asset features (n_assets * 3) + garch_vol (n_assets)
        #        + btc_corr (n_assets) + regime (3) + transition_prob (1)
        #        + portfolio (3) + kelly (1)
        # = n_assets * 5 + 3 + 1 + 3 + 1 = n_assets * 5 + 8
        state_dim = self.n_assets * 5 + 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Precompute BTC correlations
        self._btc_rolling_corr = self._precompute_btc_correlations()

        # Tracking
        self.reset()

    # ─────────────────────────────────────────────────
    #  PRECOMPUTATION
    # ─────────────────────────────────────────────────
    def _precompute_btc_correlations(self) -> np.ndarray:
        """
        Precompute rolling correlation of each asset with BTC (asset 0).

        Returns:
            (T, n_assets) array of rolling correlations
        """
        T = self.n_steps
        n = self.n_assets
        corr = np.zeros((T, n))
        lb = self.btc_corr_lookback

        btc_returns = self.returns[:, 0]  # Assume BTC is column 0

        for t in range(lb, T):
            window = self.returns[t - lb:t, :]
            btc_window = btc_returns[t - lb:t]
            btc_std = btc_window.std()
            if btc_std < 1e-10:
                corr[t, :] = 0.0
                continue
            for j in range(n):
                asset_std = window[:, j].std()
                if asset_std < 1e-10:
                    corr[t, j] = 0.0
                else:
                    corr[t, j] = np.corrcoef(btc_window, window[:, j])[0, 1]

        return corr

    # ─────────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = self.start_offset
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.cumulative_return = 0.0
        self.return_history: list = []
        self.steps_since_rebalance = 0

        # --- NEW: PnL and Sharpe tracking ---
        self.pnl_history: list = []
        self.realized_sharpe = 0.0

        # --- NEW: episode end ---
        if self.episode_length > 0:
            self._episode_end = min(self.step_idx + self.episode_length, self.n_steps - 1)
        else:
            self._episode_end = self.n_steps - 1

        obs = self._get_observation()
        return obs, {}

    # ─────────────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────────────
    def step(self, action):
        # Convert action to weights via softmax
        action = np.clip(action, 1e-6, None)
        new_weights = np.exp(action) / np.exp(action).sum()

        # --- Transaction costs from turnover ---
        turnover = np.abs(new_weights - self.weights).sum()
        flat_tc = turnover * self.tc_bps

        # --- NEW: Market impact (square-root model) ---
        # Impact = coefficient * |delta_w| ^ exponent * sigma
        # Approximation: total impact across assets
        market_impact = self._compute_market_impact(new_weights)

        # --- NEW: Slippage proportional to position size and volatility ---
        slippage = self._compute_slippage(new_weights)

        total_costs = flat_tc + market_impact + slippage

        # Portfolio return (using next period's returns)
        if self.step_idx >= self.n_steps:
            return self._get_observation(), 0.0, True, False, {}

        period_returns = self.returns[self.step_idx]
        portfolio_return = np.dot(new_weights, period_returns) - total_costs

        # Update portfolio state
        self.portfolio_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.cumulative_return = self.portfolio_value - 1.0
        drawdown = 1 - self.portfolio_value / self.peak_value

        self.return_history.append(portfolio_return)
        self.weights = new_weights
        self.step_idx += 1
        self.steps_since_rebalance += 1

        # --- NEW: Track PnL and realized Sharpe ---
        pnl = self.portfolio_value - (1.0 if len(self.pnl_history) == 0
                                      else self.pnl_history[-1])
        self.pnl_history.append(self.portfolio_value)
        self._update_realized_sharpe()

        # ── Compute reward ──
        reward = self._compute_reward(portfolio_return, drawdown, turnover, new_weights)

        # Check termination
        terminated = self.step_idx >= self._episode_end
        truncated = False

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "drawdown": drawdown,
            "turnover": turnover,
            "weights": new_weights.copy(),
            "market_impact": market_impact,
            "slippage": slippage,
            "total_costs": total_costs,
            "realized_sharpe": self.realized_sharpe,
        }

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────
    #  MARKET IMPACT (SQUARE-ROOT MODEL)
    # ─────────────────────────────────────────────────
    def _compute_market_impact(self, new_weights: np.ndarray) -> float:
        """
        Compute market impact using square-root model.

        Impact_i = coeff * |delta_w_i|^exponent * sigma_i
        Total impact = sum over assets

        Args:
            new_weights: Target portfolio weights

        Returns:
            Total market impact cost (fraction of portfolio value)
        """
        delta_w = np.abs(new_weights - self.weights)
        t = min(self.step_idx, self.garch_vols.shape[0] - 1)
        vols = self.garch_vols[t] if t < self.garch_vols.shape[0] else np.ones(self.n_assets) * 0.01

        impact_per_asset = self.impact_coeff * (delta_w ** self.impact_exponent) * vols
        return float(np.sum(impact_per_asset))

    def _compute_slippage(self, new_weights: np.ndarray) -> float:
        """
        Compute slippage proportional to position size and volatility.

        Slippage_i = slippage_vol_mult * w_i * sigma_i * |delta_w_i|

        Args:
            new_weights: Target portfolio weights

        Returns:
            Total slippage cost
        """
        t = min(self.step_idx, self.garch_vols.shape[0] - 1)
        vols = self.garch_vols[t] if t < self.garch_vols.shape[0] else np.ones(self.n_assets) * 0.01
        delta_w = np.abs(new_weights - self.weights)

        slippage_per_asset = self.slippage_vol_mult * new_weights * vols * delta_w
        return float(np.sum(slippage_per_asset))

    # ─────────────────────────────────────────────────
    #  REWARD FUNCTION (enhanced)
    # ─────────────────────────────────────────────────
    def _compute_reward(self, ret: float, drawdown: float, turnover: float,
                        weights: np.ndarray) -> float:
        """
        Composite reward function:
          + sharpe_coeff * return (encourages positive returns)
          - cvar_penalty * CVaR shortfall (penalises tail risk)
          - dd_penalty * excess drawdown (penalises drawdowns beyond threshold)
          - turnover_penalty * turnover (penalises excessive trading)
          - risk_parity_penalty * risk_contribution_deviation  [NEW]
          - concentration_penalty * max(0, HHI - threshold)    [NEW]

        Curriculum learning: Phase 0 uses only return + turnover.
        Phase 1 uses the full reward function.

        Args:
            ret: Period portfolio return
            drawdown: Current drawdown from peak
            turnover: Total absolute weight change
            weights: Current portfolio weights

        Returns:
            Clipped scalar reward
        """
        # --- Phase 0 (curriculum): simplified reward ---
        if self._curriculum_phase == 0:
            reward = self.sharpe_coeff * ret - self.turnover_penalty * turnover
            return float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # --- Phase 1: full reward ---
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

        # --- NEW: Risk-parity deviation penalty ---
        rp_deviation = self._compute_risk_parity_deviation(weights)
        reward -= self.risk_parity_penalty * rp_deviation

        # --- NEW: Concentration penalty (HHI) ---
        hhi = float(np.sum(weights ** 2))
        if hhi > self.hhi_threshold:
            reward -= self.concentration_penalty_coeff * (hhi - self.hhi_threshold)

        # --- Reward clipping for training stability ---
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        return reward

    def _compute_risk_parity_deviation(self, weights: np.ndarray) -> float:
        """
        Compute deviation from equal risk contribution.

        Measures how far each asset's marginal risk contribution is from
        the equal-weight target (1/n).

        Args:
            weights: Current portfolio weights

        Returns:
            Scalar deviation measure (0 = perfect risk parity)
        """
        t = min(self.step_idx, self.garch_vols.shape[0] - 1)
        vols = self.garch_vols[t] if t < self.garch_vols.shape[0] else np.ones(self.n_assets) * 0.01

        # Approximate risk contribution: w_i * sigma_i
        risk_contrib = weights * vols
        total_risk = risk_contrib.sum()

        if total_risk < 1e-10:
            return 0.0

        risk_shares = risk_contrib / total_risk
        target_share = 1.0 / self.n_assets
        deviation = np.sum((risk_shares - target_share) ** 2)

        return float(deviation)

    # ─────────────────────────────────────────────────
    #  OBSERVATION (enhanced state space: 48 dims)
    # ─────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        """
        Construct state vector from current market data and portfolio state.

        State dimensions (48 total):
          [0..23]   Per-asset features: returns_5d, returns_20d, vol_20d  (n_assets * 3)
          [24..31]  GARCH vol forecasts                                   (n_assets)
          [32..39]  Rolling correlation with BTC                          (n_assets)  [NEW]
          [40..42]  HMM regime probabilities                              (3)
          [43]      Regime transition probability                         (1)         [NEW]
          [44]      Cumulative return                                     (1)
          [45]      Current drawdown                                      (1)
          [46]      Days since rebalance                                  (1)
          [47]      Current Kelly fraction                                (1)         [NEW]
        """
        t = min(self.step_idx, self.n_steps - 1)

        # Per-asset features (pre-computed): returns_5d, returns_20d, vol_20d
        # Shape: (n_assets * 3,)
        if t < self.features.shape[0]:
            asset_features = self.features[t].flatten()[:self.n_assets * 3]
        else:
            asset_features = np.zeros(self.n_assets * 3)

        # GARCH vol forecasts (n_assets)
        if t < self.garch_vols.shape[0]:
            garch_feat = self.garch_vols[t]
        else:
            garch_feat = np.zeros(self.n_assets)

        # --- NEW: Rolling correlation with BTC (n_assets) ---
        if t < self._btc_rolling_corr.shape[0]:
            btc_corr = self._btc_rolling_corr[t]
        else:
            btc_corr = np.zeros(self.n_assets)

        # HMM regime probabilities (3)
        if t < self.regime_probs.shape[0]:
            regime = self.regime_probs[t]
        else:
            regime = np.array([0.33, 0.34, 0.33])

        # --- NEW: Regime transition probability (1) ---
        transition_prob = self._estimate_transition_probability(regime)

        # Portfolio state
        drawdown = 1 - self.portfolio_value / self.peak_value if self.peak_value > 0 else 0

        # --- NEW: Kelly fraction estimate ---
        kelly_frac = self._estimate_kelly_fraction()

        portfolio_state = np.array([
            self.cumulative_return,
            drawdown,
            self.steps_since_rebalance / 24.0,  # Normalise to days
            kelly_frac,
        ])

        obs = np.concatenate([
            asset_features,         # n_assets * 3
            garch_feat,             # n_assets
            btc_corr,               # n_assets  [NEW]
            regime,                 # 3
            [transition_prob],      # 1          [NEW]
            portfolio_state,        # 4 (was 3, +1 for kelly) [NEW]
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

    def _estimate_transition_probability(self, regime_probs: np.ndarray) -> float:
        """
        Estimate regime transition probability from current posterior.

        High entropy in regime probabilities indicates uncertain regime,
        which correlates with higher transition probability.

        Args:
            regime_probs: Current HMM posterior probabilities

        Returns:
            Estimated transition probability [0, 1]
        """
        probs = np.clip(regime_probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        # Higher entropy = more uncertain = higher expected transition
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _estimate_kelly_fraction(self) -> float:
        """
        Estimate Kelly criterion fraction from recent return history.

        Kelly f* = mu / sigma^2 (simplified for portfolio)
        Clipped to [0, 1] for practical use.

        Returns:
            Kelly fraction [0, 1]
        """
        if len(self.return_history) < 50:
            return 0.5  # Default moderate bet size

        recent = np.array(self.return_history[-480:])
        mu = recent.mean()
        var = recent.var()

        if var < 1e-12:
            return 0.5

        kelly = mu / var
        return float(np.clip(kelly, 0.0, 1.0))

    def _update_realized_sharpe(self) -> None:
        """
        Update realized Sharpe ratio from return history (for monitoring).

        Computes annualized Sharpe on the returns observed so far in the episode.
        """
        if len(self.return_history) < 50:
            self.realized_sharpe = 0.0
            return

        ret_arr = np.array(self.return_history)
        std = ret_arr.std()
        if std < 1e-10:
            self.realized_sharpe = 0.0
        else:
            self.realized_sharpe = float((ret_arr.mean() / std) * np.sqrt(8760))

    # ─────────────────────────────────────────────────
    #  CURRICULUM LEARNING SUPPORT
    # ─────────────────────────────────────────────────
    def set_curriculum_phase(self, phase: int) -> None:
        """
        Switch curriculum learning phase.

        Phase 0: Simplified reward (return + turnover only)
        Phase 1: Full reward function with all penalties

        Args:
            phase: Curriculum phase (0 or 1)
        """
        self._curriculum_phase = phase
        logger.info(f"  Portfolio env: curriculum phase set to {phase}")
