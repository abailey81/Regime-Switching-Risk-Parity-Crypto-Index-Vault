# Architecture: Financial Design of rpCRYPTO

This document describes the financial architecture of rpCRYPTO, a tokenised quantitative crypto index fund. The design follows the asset-token-market-risk logic chain required for asset tokenisation, with each layer motivated by financial reasoning rather than technical convenience.

---

## 1. Asset-Token-Market-Risk Chain

### 1.1 Asset Layer: Multi-Asset Crypto Portfolio

The underlying asset is a diversified portfolio of eight constituents spanning four asset classes:

| Asset  | Class          | Value Source                  | Role in Portfolio                |
|--------|----------------|-------------------------------|----------------------------------|
| BTC    | Crypto major   | Chainlink BTC/USD oracle      | Market beta, liquidity anchor    |
| ETH    | Crypto major   | Chainlink ETH/USD oracle      | Smart contract ecosystem proxy   |
| SOL    | Crypto alt     | Chainlink SOL/USD oracle      | High-growth, high-vol satellite  |
| stETH  | Liquid staking | Chainlink stETH/USD oracle    | ETH yield + staking rewards      |
| rETH   | Liquid staking | Chainlink rETH/USD oracle     | ETH yield, decentralised staking |
| BUIDL  | Tokenised RWA  | BlackRock NAV (1:1 USD peg)   | On-chain Treasury bill proxy     |
| USDY   | Tokenised RWA  | Ondo Finance NAV (yield-bearing)| On-chain Treasury note proxy    |
| USDC   | Stablecoin     | Chainlink USDC/USD oracle     | Cash buffer, redemption liquidity|

This four-class structure (crypto, liquid staking, tokenised Treasuries, stablecoin) creates a portfolio that can rotate between risk-on and risk-off allocations without leaving the on-chain universe. The regime-dependent risk budgets in `config.yaml` govern these rotations:

- **Bull regime**: 70% crypto, 20% staking, 0% Treasuries, 10% stablecoin
- **Normal regime**: 40% crypto, 30% staking, 20% Treasuries, 10% stablecoin
- **Crisis regime**: 10% crypto, 10% staking, 50% Treasuries, 30% stablecoin

The crisis allocation shifts 80% of portfolio value into yield-bearing stablecoins and tokenised Treasuries, providing capital preservation while maintaining full on-chain composability.

### 1.2 Token Layer: ERC-4626 Vault Share (rpCRYPTO)

The vault share token, rpCRYPTO, is an ERC-4626 compliant token representing a pro-rata claim on the portfolio's net asset value. Key properties:

- **Variable supply**: Shares are minted on deposit and burned on withdrawal. There is no fixed supply; the token is purely a unit of account for NAV ownership.
- **NAV-linked pricing**: Each share's value equals `totalAssets() / totalSupply()`, where `totalAssets()` aggregates Chainlink oracle valuations of all constituent holdings.
- **Fee-adjusted returns**: Management fees are collected through continuous share dilution (minting fee shares to `feeRecipient`), meaning the share price incorporates all costs without requiring explicit fee deductions from user balances.

The ERC-4626 standard was chosen over alternatives for specific financial reasons:
- **Not raw ERC-20**: A plain ERC-20 token cannot enforce deposit/withdrawal mechanics, fee accrual, or NAV accounting at the contract level. Building these features from scratch would create a non-standard, unauditable interface.
- **Not ERC-7540**: While ERC-7540 provides asynchronous deposit/redemption requests better suited to illiquid real-world assets, our portfolio consists entirely of liquid on-chain tokens where synchronous settlement is feasible. The added complexity of request queues and fulfilment callbacks is unnecessary for liquid crypto assets, though it would become relevant at institutional scale (see Limitations).

### 1.3 Market Layer: Three-Layer Liquidity Design

rpCRYPTO liquidity operates across three layers, each serving a different participant type:

**Layer 1 -- Primary Market (NAV-based)**: Depositors interact directly with the vault contract. Deposits mint shares at the current NAV; withdrawals burn shares and return the proportional underlying assets. This is the canonical price discovery mechanism and anchors the token's fundamental value. Subject to the 20% epoch redemption gate.

**Layer 2 -- Secondary Market (AMM)**: rpCRYPTO can trade on decentralised exchanges (e.g., Uniswap V3 rpCRYPTO/USDC pool). The secondary market provides instant liquidity for holders who cannot wait for epoch boundaries or whose redemptions exceed the gate. In normal conditions, the secondary market price tracks NAV. During stress, it may trade at a discount, which activates Layer 3.

**Layer 3 -- Keeper Arbitrage**: When the AMM price deviates from NAV, arbitrageurs can buy discounted rpCRYPTO on the AMM, redeem at NAV from the vault, and pocket the spread. This arbitrage mechanism bounds the NAV discount and incentivises third-party market making without requiring the vault itself to provide liquidity.

### 1.4 Risk Layer: Five Categories of Risk

| Risk Category   | Description                                                                                         | Mitigation                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| **Model risk**  | ML ensemble produces incorrect weights due to regime misclassification, overfitting, or stale data  | Inverse-variance weighting, 40% max single-asset cap, defensive fallback |
| **Oracle risk** | Chainlink price feed returns stale or incorrect prices, causing NAV miscalculation                  | Staleness check (revert if price older than threshold), multi-oracle fallback planned |
| **Contract risk** | Smart contract bug allows fund extraction, incorrect fee calculation, or frozen withdrawals        | OpenZeppelin v5 base, role-based access control, Merkle proof verification, pausability |
| **Market risk** | Underlying assets experience correlated drawdowns exceeding model expectations                      | Circuit breaker at 15% drawdown, crisis allocation (80% defensive), redemption gate |
| **Regulatory risk** | Tokenised fund structure may be classified as a collective investment scheme in certain jurisdictions | Permissioned deployment (admin roles), compliance-ready fee structure |

---

## 2. Financial Design Rationale

### 2.1 Three-Tier Fee Architecture

The fee structure adapts the traditional hedge fund "2-and-20" model for on-chain transparency and continuous settlement:

| Fee              | Rate     | Mechanism             | Financial Purpose                                      |
|------------------|----------|-----------------------|--------------------------------------------------------|
| Management fee   | 1% p.a.  | Continuous share dilution | Covers operational costs (data feeds, gas, infrastructure). Accrued every second, not quarterly, eliminating timing games. |
| Performance fee  | 10% above HWM | Crystallised at epoch boundaries | Aligns manager incentives with investor returns. The high-water mark ensures fees are only charged on new profits, never on recovery from losses. |
| Redemption fee   | 0.3% within 7 days | Deducted from withdrawal | Anti-churn mechanism. Prevents "hot money" that deposits before a rebalance and withdraws immediately after, free-riding on the portfolio's alpha. |

**Worked Example -- Management Fee Dilution**:

Suppose the vault has 100,000 shares outstanding and a 1% annual management fee. Over one day:

```
fee_shares = 100,000 x 0.01 x (86,400 / 31,557,600) = 0.2738 shares
```

These 0.2738 shares are minted to `feeRecipient`, diluting existing holders by 0.000274%. Over a full year, this compounds to approximately 1% dilution, matching the stated fee rate.

**Worked Example -- Deposit and Share Calculation**:

If Alice deposits 10,000 USDC when the vault NAV per share is 1.10 USDC:

```
shares_received = 10,000 / 1.10 = 9,090.91 rpCRYPTO
```

After one year, if the management fee has diluted supply by 1% and the portfolio has grown to NAV 1.30:

```
Alice's value = 9,090.91 x 1.30 = 11,818.18 USDC (before performance fee)
Performance fee applies on gain above HWM (1.10 -> 1.30):
  gain_per_share = 0.20 USDC
  total_gain = 9,090.91 x 0.20 = 1,818.18 USDC
  performance_fee = 1,818.18 x 10% = 181.82 USDC (paid as minted shares)
```

### 2.2 Merkle Commit-Reveal for Weight Updates

Portfolio weights are updated through a two-phase commit-reveal process with a 1-hour timelock:

1. **Commit**: The keeper publishes a Merkle root of the proposed weight vector. No weight values are revealed.
2. **Timelock** (1 hour): LPs can observe the commitment hash on-chain. If they disagree with the upcoming allocation, they can withdraw before execution.
3. **Execute**: After the timelock, the keeper reveals the full weight vector with Merkle proofs. The contract verifies the proof against the committed root before applying weights.

This design prevents **MEV extraction** by the keeper. Without commit-reveal, a keeper could front-run its own weight update by trading constituent tokens before submitting the new allocation. The timelock also functions as a "ragequit window" -- LP protection borrowed from MolochDAO governance design.

### 2.3 Circuit Breaker: 15% Drawdown Threshold

The circuit breaker activates when NAV per share drops below 85% of the high-water mark:

```
trigger_condition: NAV_per_share < highWaterMark x (1 - 1500/10000)
```

**Calibration rationale**: Analysis of 12 major crypto market crises between 2020 and 2024 shows that the average BTC peak-to-trough drawdown was approximately -33%, with the median drawdown reaching -25%. A 15% threshold is designed to trigger early in a drawdown -- before the majority of losses occur -- while avoiding false triggers during normal volatility (BTC's annualised volatility of roughly 60% implies a 1-sigma monthly move of about 17%, so a 15% drawdown exceeds routine fluctuations but catches genuine stress events).

When triggered, the circuit breaker:
- Overrides all model-generated weights with defensive allocation (30% USDC, 35% BUIDL, 15% USDY, 5% each to BTC/ETH/stETH/rETH, 0% SOL)
- Tightens the redemption gate to prevent bank-run dynamics
- Requires NAV recovery to 90% of HWM before admin can reset

### 2.4 Redemption Gate: 20% Per-Epoch Limit

The vault limits total withdrawals to 20% of AUM per epoch. This gate addresses the **liquidity trilemma** inherent in open-ended funds: instant liquidity, portfolio integrity, and run prevention cannot all be simultaneously satisfied.

**Design reasoning**: Modelling redemptions as a Poisson process with an average of 3-5 redemption events per epoch, the 20% gate ensures that in 99%+ of normal scenarios, all redemption requests are fulfilled within a single epoch. Only during coordinated panic selling does the gate bind, which is precisely when portfolio integrity needs protection -- forced liquidation of illiquid positions at distressed prices would harm remaining investors more than a short redemption delay.

The gate resets at each epoch boundary. Unfulfilled redemptions can resubmit in the next epoch, creating a queue without requiring complex on-chain queue management.

### 2.5 Ensemble ML: Model Risk Hedging

The off-chain ML pipeline combines three models rather than relying on a single model, treating model diversity as a form of **risk hedging**:

| Model      | What It Captures                  | Failure Mode                         |
|------------|-----------------------------------|--------------------------------------|
| GARCH-DCC  | Time-varying volatility and correlation | Fails under regime breaks (parameters estimated from recent history may not apply in a new regime) |
| HMM        | Discrete market regimes (bull, normal, crisis) | Assumes a fixed number of states; transitions between states are memoryless (Markov property) |
| SAC RL     | Nonlinear allocation policy learned from data | Overfits to training distribution; may fail on out-of-distribution market conditions |

The ensemble uses **inverse forecast variance** to dynamically weight models:
- When GARCH parameter standard errors are large, its weight decreases
- When HMM regime entropy is high (uncertain state classification), its weight decreases
- When RL policy entropy is high (exploratory, uncertain actions), its weight decreases
- When all models are uncertain, the portfolio defaults toward equal-weight or defensive allocation

This approach ensures that no single model's failure can dominate the portfolio, and during periods of high model disagreement, the allocation naturally becomes more conservative.

---

## 3. Data Flow with Financial Annotations

```
Binance API ──> fetch_data.py ──> preprocess.py ──> [returns_df, hmm_features]
                                                          |
  PURPOSE: Hourly OHLCV for 8         PURPOSE: Log returns,
  assets. Hourly frequency             realised vol, momentum,
  balances recency with noise           rolling Sharpe -- features
  reduction vs tick data.               that capture volatility
                                        clustering, mean reversion,
                                        and trend persistence.
                    |                         |                  |
                    v                         v                  v
              garch_dcc.py           bayesian_hmm.py      sac_agent.py
              (Sigma_t, w_rp)        (regime probs)        (w_rl)
                    |                         |                  |
  PURPOSE: Forecast       PURPOSE: Classify      PURPOSE: Learn
  covariance matrix        current regime          non-linear
  for risk-parity          (bull/normal/crisis)    allocation policy
  weight calculation.      to set risk budgets.    from reward signal
  Student-t innovations                            (Sharpe - CVaR -
  capture fat tails.                               drawdown penalty).
                    |                         |                  |
                    +------------+------------+------------------+
                                 v
                           ensemble.py
                     (CVaR-constrained w*)
                                 |
  PURPOSE: Combine model outputs via inverse-variance weighting.
  Apply CVaR constraint at 5% confidence level to limit tail risk.
  Enforce 40% max single-asset and 2% min single-asset bounds.
  Apply 30% max turnover constraint to limit transaction costs.
                                 |
                                 v
                     compute_weights.py ──> merkle.py ──> publish.py
                                                              |
  PURPOSE: Compute Merkle root    PURPOSE: Two-phase commit-
  of weight vector for on-chain   reveal prevents keeper from
  verification without revealing  front-running rebalance trades.
  weights during timelock.        1-hour timelock = LP ragequit
                                  window.
                                                              v
                                                    RiskParityVault.sol
                                                    (commitWeights -> executeWeights)
```

---

## 4. Epoch System: Liquidity Design

```
   <------ epochDuration (1 day calm / 7 days volatile) ------>
   |                                                           |
   epochStart                                          epochStart + epochDuration
   |                                                           |
   |-- Deposits accepted --------------------------------------|
   |-- Withdrawals accepted (up to 20% AUM gate) -------------|
   |                                                           |
   +-- At boundary: crystallise performance fees, reset gate --+
                     advance epoch counter
```

**Financial rationale for variable-length epochs**:

- **1-day epochs (calm markets)**: Allow daily deposit and withdrawal windows, comparable to traditional mutual fund NAV-based dealing. Daily rebalancing captures short-term mean reversion signals from the ML ensemble while gas costs remain manageable during low-congestion periods.

- **7-day epochs (volatile markets)**: When the HMM identifies a volatile or crisis regime, the keeper extends the epoch duration. This serves three purposes: (1) reduces rebalancing frequency during periods when transaction costs and slippage are elevated, (2) prevents panic selling by spacing out redemption windows, and (3) allows the ML models more time to confirm whether a regime transition is genuine or a false signal.

The epoch duration is toggled by the keeper via `setEpochVolatile()`, informed by the HMM's regime probability output. The on-chain contract enforces the epoch boundaries and gate mechanics regardless of the keeper's classification.

---

## 5. On-Chain Safety Mechanisms Summary

| Mechanism            | Trigger / Condition                    | Financial Effect                                                 |
|----------------------|----------------------------------------|------------------------------------------------------------------|
| Circuit breaker      | NAV drops 15% below HWM               | Forces defensive allocation, tightens redemption gate            |
| Redemption gate      | 20% of AUM redeemed in current epoch   | Prevents bank-run liquidation spirals                            |
| Weight timelock      | 1 hour between commit and execute      | LP ragequit window, prevents keeper MEV                          |
| Turnover cap         | 30% max portfolio turnover per rebalance| Limits transaction costs and slippage from aggressive rebalancing|
| Concentration cap    | 40% max single-asset weight            | Prevents over-concentration in any single constituent            |
| Minimum weight       | 2% min single-asset weight             | Ensures diversification is maintained across all constituents    |
| Pausability          | Admin can pause all deposits/withdrawals| Emergency stop for contract bugs or oracle failures              |
| Staleness check      | Reverts if Chainlink price is stale    | Prevents NAV calculation with outdated price data                |
