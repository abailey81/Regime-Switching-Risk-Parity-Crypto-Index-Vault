# Limitations and Future Work

This section provides a critical assessment of the design decisions, known limitations, and potential extensions of the rpCRYPTO vault system. Honest evaluation of these trade-offs is central to demonstrating that the design choices were deliberate rather than arbitrary.

---

## 1. Design Limitations

### 1.1 Model Limitations

**HMM State Space Assumption**: The Hidden Markov Model assumes three discrete market regimes (bull, normal, crisis). In practice, regime dynamics may be considerably more complex. The 2022-2023 period exhibited a prolonged "macro-driven crypto winter" that does not fit cleanly into a three-state taxonomy -- it displayed characteristics of both crisis (sustained drawdowns) and normal (low volatility consolidation) regimes simultaneously. A continuous latent-state model or a hierarchical HMM with sub-states would better capture this nuance, at the cost of increased estimation complexity and overfitting risk.

**SAC Policy Stationarity**: The SAC reinforcement learning agent is trained on data from 2022-2025. This period was dominated by specific market microstructure conditions: post-pandemic retail participation, centralised exchange collapses (FTX), and the early stages of institutional adoption. Structural changes -- most notably the approval and launch of spot Bitcoin ETFs in January 2024 -- have fundamentally altered crypto market dynamics. ETF-driven flows now account for a material fraction of BTC daily volume, introducing correlations with traditional equity markets that did not exist in the training data. The policy may therefore be partially invalidated by out-of-distribution market conditions, a known challenge in offline RL.

**GARCH-DCC Stationarity Assumption**: GARCH models assume that the data-generating process is stationary within the estimation window. During regime transitions -- precisely when accurate covariance estimation matters most -- this assumption breaks down. The 1,440-hour (60-day) rolling window in `config.yaml` represents a compromise: long enough for stable parameter estimation, short enough to adapt to regime changes, but inherently lagged during rapid transitions.

**Ensemble Weight Stability**: The inverse-variance weighting scheme for combining model outputs is itself a model. When all three models simultaneously report high uncertainty (as may occur during a black swan event), the ensemble defaults toward equal or defensive weighting. This is a reasonable fail-safe, but it means the system effectively "gives up" during the market conditions where intelligent allocation would be most valuable.

### 1.2 Smart Contract Limitations

**ERC-4626 Synchronous Redemption**: The ERC-4626 standard processes deposits and withdrawals synchronously within a single transaction. For a vault holding illiquid assets or processing large institutional redemptions, this creates a potential problem: the vault must have sufficient liquid reserves to honour redemptions immediately. While the 20% epoch redemption gate mitigates this, it is a blunt instrument. ERC-7540 (the asynchronous vault extension) provides a request-fulfil pattern that would allow the vault to liquidate positions over multiple blocks before settling redemptions, and would be the superior choice for production deployment with large AUM.

**Oracle Dependency and Heartbeat Risk**: The vault relies on Chainlink price feeds for NAV computation. Chainlink's ETH/USD feed has a 1-hour heartbeat (updates at minimum every hour) and a 0.5% deviation threshold. A flash crash that recovers within the heartbeat window could cause the vault to compute an inaccurate NAV, allowing arbitrageurs to deposit at a stale-low NAV and redeem after the oracle updates. The staleness check in the contract mitigates stale-high scenarios (reverting if the price is too old) but cannot detect stale-low scenarios within the heartbeat window.

**Single Keeper Point of Failure**: Weight updates depend on a single keeper bot with the `KEEPER_ROLE`. If the keeper goes offline, the vault continues operating with stale weights indefinitely. There is no on-chain mechanism to detect keeper liveness or trigger a fallback allocation after prolonged keeper absence. A timeout-based fallback to defensive weights (e.g., if no weight update occurs within 48 hours, automatically shift to crisis allocation) would improve resilience.

**Redemption Gate and the Liquidity Trilemma**: The 20% per-epoch redemption gate creates a fundamental tension. It protects the portfolio from forced liquidation during panic selling, but it also means that in a genuine crisis, investors who need liquidity most urgently are precisely those who cannot obtain it. This is the open-ended fund liquidity trilemma: instant liquidity, portfolio integrity, and run prevention cannot coexist simultaneously. The gate resolves this in favour of portfolio integrity at the expense of instant liquidity -- a defensible choice, but one that creates real investor harm during tail events.

### 1.3 Data and Infrastructure Limitations

**Binance Single-Source Risk**: Price data is sourced exclusively from Binance, with Kraken and Coinbase as fallbacks. If Binance experiences an outage, data manipulation, or geopolitical restriction, the ML pipeline cannot compute weights. A production system would require a multi-exchange aggregation layer with outlier detection.

**Synthetic RWA Data**: BUIDL (BlackRock tokenised Treasury) and USDY (Ondo yield-bearing stablecoin) have limited historical data relative to BTC and ETH. Where historical data is unavailable, the pipeline generates synthetic returns based on Treasury yield proxies. This introduces basis risk between the synthetic training data and the actual token price dynamics, particularly during periods of stablecoin depegging stress (e.g., USDC's brief depeg in March 2023).

---

## 2. Production Upgrades

The following enhancements would be necessary to transition rpCRYPTO from a coursework prototype to a production-grade system.

### 2.1 Governance and Security

- **Multi-signature administration**: Replace the single admin key with a 5-of-9 multi-sig (e.g., Gnosis Safe). Critical operations -- fee changes, circuit breaker reset, constituent management, contract upgrades -- should require supermajority approval to prevent single-key compromise.
- **Formal smart contract audit**: Engage a tier-1 security auditor (Trail of Bits, ConsenSys Diligence, or OpenZeppelin) for a comprehensive audit before mainnet deployment. The cost (typically $50,000-$200,000) is justified by the custodial nature of the vault.
- **Tiered rollout with deposit caps**: Deploy with a $10,000 total deposit cap in month one, $100,000 in month two, and uncapped thereafter. This limits exposure during the critical early period when undiscovered bugs are most likely.
- **Bug bounty programme**: Establish an Immunefi bug bounty with payouts up to 10% of AUM for critical vulnerabilities.

### 2.2 Oracle Infrastructure

- **Multi-oracle aggregation**: Implement a price feed aggregator that queries Chainlink, Pyth Network, and RedStone, taking the median price. This protects against single-oracle manipulation or staleness.
- **TWAP circuit breaker**: In addition to the existing HWM-based circuit breaker, add a time-weighted average price (TWAP) deviation check. If the spot oracle price deviates from a 1-hour TWAP by more than 5%, flag the price as potentially manipulated and pause rebalancing.
- **Chainlink Functions for on-chain ML**: Replace the keeper pattern with Chainlink Functions, which allow verifiable off-chain computation within a decentralised oracle network. This would eliminate the single-keeper trust assumption.

### 2.3 Portfolio Management

- **Automated rebalancing execution**: Currently, the vault commits and executes weight targets but does not execute the underlying swaps. A production system would integrate with DEX aggregators (1inch, Paraswap) or use Chainlink Automation to execute rebalancing trades atomically.
- **Dynamic fee calibration**: Management and performance fees are currently fixed at deployment. A production system could adjust fees based on AUM (fee breakpoints), market regime (lower fees during drawdowns to retain investors), or competitive benchmarking.
- **Cross-chain deployment**: Deploy vault instances on multiple L2s (Arbitrum, Base, Optimism) with cross-chain share bridging via LayerZero or Wormhole. This would reduce gas costs and broaden accessibility.

---

## 3. Academic Extensions

The following extensions connect the coursework to broader research questions in quantitative finance and decentralised systems.

### 3.1 Tail Risk Hedging with Options

The current defensive allocation (shifting to stablecoins and Treasuries during crisis) is effective but reactive -- it triggers only after a 15% drawdown has already occurred. A proactive tail-hedging strategy using on-chain options (via Lyra, Premia, or Hegic) could purchase protective puts on BTC and ETH, capping downside losses while preserving upside participation. The cost of the option premium would be offset against the portfolio's yield from staking and Treasury positions. This extension would require integrating options pricing models (Black-Scholes adapted for crypto vol surfaces) into the ensemble.

### 3.2 rpCRYPTO as DeFi Collateral

A yield-bearing, risk-managed vault token is well-suited as collateral in DeFi lending protocols. Listing rpCRYPTO as collateral on Aave or Compound would allow holders to borrow against their vault position without redeeming, creating a capital-efficient loop. This requires the vault token to have sufficient liquidity, oracle support, and a track record of NAV stability -- each of which is addressed by the architecture but would need empirical validation.

### 3.3 Formal Verification

The financial invariants of the vault (shares outstanding equals sum of minted minus burned, NAV per share is monotonically non-decreasing after fee deduction, redemption gate never exceeds 20% within an epoch) can be expressed as formal properties and verified using tools such as Certora Prover or Halmos. Formal verification provides mathematical guarantees that the contract implementation satisfies its specification, complementing but not replacing traditional testing and auditing.

### 3.4 LLM-Driven Reward Engineering

The SAC agent's reward function is hand-designed with five penalty terms (Sharpe coefficient, CVaR penalty, drawdown penalty, turnover penalty, risk-parity penalty) and their associated hyperparameters. An emerging research direction uses large language models to automatically generate and iteratively refine reward functions (the "Eureka" approach). Applying this to the vault's RL agent would create a direct comparison between human-designed and LLM-designed reward functions evaluated on identical market data, contributing to the growing literature on AI-assisted financial engineering.
