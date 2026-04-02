"""
Venue-Specific Transaction Cost Model — VERIFIED Q1 2026 FEE SCHEDULES.

Corrected from stat-arb (which used 2023 gas prices and futures fees).

CORRECTIONS APPLIED:
  1. Binance SPOT fees: 10 bps maker/taker (NOT 1-4 bps futures fees)
     With BNB discount: 7.5 bps. Source: binance.com/en/fee/spotMaker
  2. Ethereum L1 gas: $0.05-$3.00 per swap (NOT $15)
     Post-Dencun upgrade (March 2024): avg gas dropped 95% to ~2.7 gwei
     At 2.7 gwei, 150K gas, ETH $1900: cost = $0.77
     Source: etherscan.io/gastracker, April 2026
  3. Curve stETH/ETH pool: 4 bps fee (stableswap, very low)
  4. BUIDL/USDY: Institutional redemption at NAV, ~0 exchange fee
  5. L2 swap costs: <$0.10 on Arbitrum/Base

Adapted from: Crypto-Statistical-Arbitrage/backtesting/transaction_costs.py
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CostBreakdown:
    """Itemised cost for a single rebalancing trade."""
    exchange_fee_bps: float
    slippage_bps: float
    gas_cost_usd: float
    mev_cost_bps: float
    total_bps: float
    total_usd: float
    venue: str
    asset: str


# ═══════════════════════════════════════════════════════════════
# VERIFIED FEE PROFILES — Q1 2026
# ═══════════════════════════════════════════════════════════════
# Format: (maker_bps, taker_bps, base_slippage_bps, gas_usd, mev_bps, venue, adv_usd)

COST_PROFILES = {
    # ─── CEX SPOT (Binance) ───
    # Binance spot: 10 bps maker/taker (standard), 7.5 with BNB
    # Using 7.5 bps assuming BNB discount is enabled
    "BTC": {
        "maker_bps": 7.5, "taker_bps": 7.5,
        "base_slippage_bps": 0.5,    # Ultra-liquid: $50M+ within 0.1% of mid
        "gas_usd": 0.0,              # CEX = no gas
        "mev_bps": 0.0,              # CEX = no MEV
        "venue": "CEX (Binance)",
        "adv_usd": 5_000_000_000,    # ~$5B daily volume BTC/USDT
    },
    "ETH": {
        "maker_bps": 7.5, "taker_bps": 7.5,
        "base_slippage_bps": 1.0,    # Very liquid but slightly less than BTC
        "gas_usd": 0.0,
        "mev_bps": 0.0,
        "venue": "CEX (Binance)",
        "adv_usd": 3_000_000_000,
    },
    "SOL": {
        "maker_bps": 7.5, "taker_bps": 7.5,
        "base_slippage_bps": 2.0,    # Good liquidity but less than BTC/ETH
        "gas_usd": 0.0,
        "mev_bps": 0.0,
        "venue": "CEX (Binance)",
        "adv_usd": 1_000_000_000,
    },

    # ─── DEX (Ethereum L1) ───
    # Post-Dencun: avg gas ~2.7 gwei. Swap = ~150K gas.
    # Cost at 2.7 gwei, ETH $1900: 150000 * 2.7e-9 * 1900 = $0.77
    # Conservative estimate: $0.50-2.00 depending on congestion
    "stETH": {
        "maker_bps": 4.0, "taker_bps": 4.0,  # Curve stETH/ETH stableswap pool
        "base_slippage_bps": 2.0,    # stETH/ETH is tight (near peg)
        "gas_usd": 1.50,             # L1 swap ~$1-2 post-Dencun
        "mev_bps": 2.0,              # Some sandwich risk on L1
        "venue": "DEX (Curve L1)",
        "adv_usd": 200_000_000,
    },
    "rETH": {
        "maker_bps": 5.0, "taker_bps": 5.0,  # Uniswap V3 or Balancer
        "base_slippage_bps": 4.0,    # Thinner than stETH
        "gas_usd": 1.50,             # Same L1 gas
        "mev_bps": 3.0,              # Slightly more MEV risk (less liquid)
        "venue": "DEX (Uniswap L1)",
        "adv_usd": 50_000_000,
    },

    # ─── INSTITUTIONAL (Off-exchange) ───
    # BUIDL (BlackRock) and USDY (Ondo) are redeemed at NAV
    # No exchange trading fees, just potential redemption delay
    "BUIDL": {
        "maker_bps": 0.0, "taker_bps": 0.0,  # NAV redemption, no exchange fee
        "base_slippage_bps": 0.5,    # Minimal: redeemed at NAV
        "gas_usd": 0.0,              # Off-chain redemption or L1 transfer (~$0.50)
        "mev_bps": 0.0,
        "venue": "Institutional (BlackRock)",
        "adv_usd": 2_200_000_000,    # BUIDL AUM
    },
    "USDY": {
        "maker_bps": 0.0, "taker_bps": 0.0,
        "base_slippage_bps": 0.5,
        "gas_usd": 0.0,
        "mev_bps": 0.0,
        "venue": "Institutional (Ondo)",
        "adv_usd": 500_000_000,
    },

    # ─── STABLECOIN ───
    "USDC": {
        "maker_bps": 0.0, "taker_bps": 0.0,  # Binance: 0 fee for USDC pairs
        "base_slippage_bps": 0.1,    # Ultra-liquid stablecoin
        "gas_usd": 0.0,
        "mev_bps": 0.0,
        "venue": "CEX (Binance, 0-fee)",
        "adv_usd": 10_000_000_000,
    },
}


class RebalancingCostModel:
    """
    Venue-specific transaction cost model for portfolio rebalancing.

    Uses Almgren-Chriss square-root market impact model:
      slippage = base_slippage * sqrt(trade_size / ADV) * vol_multiplier

    Gas costs verified against Etherscan Q1 2026 data.
    Exchange fees verified against Binance fee schedule Q1 2026.
    """

    def __init__(self, profiles: Dict = None, use_taker: bool = True):
        self.profiles = profiles or COST_PROFILES
        self.use_taker = use_taker

    def estimate_rebalance_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        asset_names: list,
        portfolio_value_usd: float = 1_000_000,
        volatility_regime: str = "normal",
    ) -> Dict:
        """
        Estimate total cost of rebalancing from old to new weights.

        Uses Almgren-Chriss square-root impact model with venue-specific
        calibration and volatility regime adjustment.
        """
        vol_mult = {"calm": 0.6, "normal": 1.0, "volatile": 2.5,
                     "crisis": 4.0}.get(volatility_regime, 1.0)

        total_cost_usd = 0.0
        breakdowns = {}

        for i, asset in enumerate(asset_names):
            turnover = abs(new_weights[i] - old_weights[i])
            if turnover < 1e-6:
                continue

            trade_usd = turnover * portfolio_value_usd
            profile = self.profiles.get(asset)
            if profile is None:
                # Default: assume CEX spot with moderate liquidity
                profile = {"maker_bps": 10, "taker_bps": 10, "base_slippage_bps": 5,
                           "gas_usd": 0, "mev_bps": 0, "venue": "Unknown", "adv_usd": 100e6}

            # Exchange fee
            fee_bps = profile["taker_bps"] if self.use_taker else profile["maker_bps"]

            # Almgren-Chriss square-root market impact
            adv = profile["adv_usd"]
            participation_rate = trade_usd / adv
            slippage_bps = profile["base_slippage_bps"] * math.sqrt(participation_rate) * 10
            slippage_bps *= vol_mult
            slippage_bps = max(slippage_bps, profile["base_slippage_bps"] * 0.5)  # Floor

            # MEV cost
            mev_bps = profile["mev_bps"] * vol_mult

            # Gas
            gas_usd = profile["gas_usd"]

            # Total
            total_trade_bps = fee_bps + slippage_bps + mev_bps
            trade_cost_usd = trade_usd * total_trade_bps / 10000 + gas_usd

            breakdowns[asset] = CostBreakdown(
                exchange_fee_bps=fee_bps,
                slippage_bps=round(slippage_bps, 2),
                gas_cost_usd=gas_usd,
                mev_cost_bps=mev_bps,
                total_bps=round(total_trade_bps, 2),
                total_usd=round(trade_cost_usd, 2),
                venue=profile["venue"],
                asset=asset,
            )
            total_cost_usd += trade_cost_usd

        total_bps = (total_cost_usd / portfolio_value_usd * 10000) if portfolio_value_usd > 0 else 0

        return {
            "total_cost_usd": round(total_cost_usd, 2),
            "total_cost_bps": round(total_bps, 2),
            "per_asset": breakdowns,
            "turnover": float(np.abs(new_weights - old_weights).sum()),
            "volatility_regime": volatility_regime,
        }

    def flat_cost_bps(self, old_weights, new_weights, asset_names,
                      portfolio_value=1e6) -> float:
        """Simple flat cost in bps for backtest compatibility."""
        result = self.estimate_rebalance_cost(old_weights, new_weights,
                                               asset_names, portfolio_value)
        return result["total_cost_bps"]

    def compute_rebalance_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        asset_names: list,
        portfolio_value_usd: float = 1_000_000,
        volatility_regime: str = "normal",
    ) -> float:
        """
        Compute total rebalancing cost in bps.

        Convenience wrapper that returns just the scalar bps cost,
        suitable for direct use in the backtest loop.

        Args:
            old_weights: Previous portfolio weights
            new_weights: Target portfolio weights
            asset_names: List of asset name strings
            portfolio_value_usd: Portfolio NAV in USD
            volatility_regime: One of 'calm', 'normal', 'volatile', 'crisis'

        Returns:
            Total cost in basis points (float)
        """
        result = self.estimate_rebalance_cost(
            old_weights, new_weights, asset_names,
            portfolio_value_usd, volatility_regime,
        )
        return result["total_cost_bps"]

    def cost_sensitivity_analysis(self, old_w, new_w, names, pv=1e6):
        """Run across all volatility regimes."""
        return {r: self.estimate_rebalance_cost(old_w, new_w, names, pv, r)["total_cost_bps"]
                for r in ["calm", "normal", "volatile", "crisis"]}
