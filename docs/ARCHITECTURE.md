# Architecture

## System Overview

The system has two planes that interact via a keeper bridge:

### Off-Chain Plane (Python ML Pipeline)
Runs on your local machine (or Colab for GPU tasks). Responsible for:
1. **Data ingestion** — Fetches hourly OHLCV from Binance, generates synthetic Treasury/stablecoin data
2. **Feature engineering** — Log returns, realised vol, momentum, rolling Sharpe, HMM features
3. **Model fitting** — GARCH-DCC (covariance), HMM (regimes), SAC RL (allocation policy)
4. **Ensemble combination** — Inverse-variance weighting + CVaR constraint + circuit breaker
5. **Weight publication** — Computes Merkle root, calls vault contract via web3.py

### On-Chain Plane (Solidity Smart Contracts)
Lives on Ethereum (Sepolia testnet). Responsible for:
1. **Asset custody** — Holds USDC deposits and constituent tokens
2. **Share accounting** — ERC-4626 mint/burn logic for vault shares (rpCRYPTO)
3. **Fee enforcement** — Management (dilution), performance (HWM), redemption (anti-churn)
4. **Weight integrity** — Merkle commit-reveal prevents keeper front-running
5. **Safety mechanisms** — Circuit breaker, redemption gate, epoch controls, pause

### Keeper Bridge
The `weight_publisher/` module connects the two planes:
- `compute_weights.py` runs the full ML pipeline → outputs weight vector + Merkle root
- `merkle.py` computes the cryptographic commitment
- `publish.py` sends transactions to the vault contract via web3.py

## Data Flow

```
Binance API ──► fetch_data.py ──► preprocess.py ──► [returns_df, hmm_features]
                                                          │
                    ┌─────────────────────────────────────┤
                    ▼                   ▼                  ▼
              garch_dcc.py      bayesian_hmm.py     sac_agent.py
              (Σ_t, w_rp)      (regime probs)       (w_rl)
                    │                   │                  │
                    └───────────┬───────┘──────────────────┘
                                ▼
                          ensemble.py
                    (CVaR-constrained w*)
                                │
                                ▼
                    compute_weights.py ──► merkle.py ──► publish.py
                                                            │
                                                            ▼
                                                    RiskParityVault.sol
                                                    (commitWeights → executeWeights)
```

## Model Interaction

The four models don't run independently — they form a pipeline:

1. **GARCH-DCC** runs first → produces covariance matrix Σ_t
2. **HMM** runs second → produces regime probabilities (uses different features, independent of GARCH)
3. **SAC RL** runs third → uses GARCH vols + HMM probs as STATE inputs
4. **Ensemble** runs last → combines all three outputs, applies CVaR constraint

The ensemble uses **inverse forecast variance** to weight models:
- If GARCH parameter standard errors are large → GARCH gets lower weight
- If HMM regime entropy is high (uncertain classification) → HMM gets lower weight
- If RL entropy coefficient is high (exploratory policy) → RL gets lower weight
- When all models are uncertain → portfolio defaults toward equal weight or defensive

## Fee Mechanics

```
Management Fee (1% annual):
  Accrued continuously via share dilution.
  Every second: fee_shares = totalSupply × 0.01 × Δt / SECONDS_PER_YEAR
  Minted to feeRecipient, diluting all other holders equally.

Performance Fee (10% above HWM):
  Crystallised at epoch boundaries.
  If current_NAV_per_share > highWaterMark:
    gain = (NAV - HWM) × totalSupply
    fee = gain × 10%
    Mint fee shares worth `fee` to feeRecipient
    Update HWM = current NAV
  HWM never decreases — fees only charged on NEW highs.

Redemption Fee (0.3% within 7 days):
  Anti-churn mechanism. If withdraw within 7 days of deposit:
    fee = withdrawal_amount × 0.003
    Sent to feeRecipient
  After 7 days: no fee.
```

## Circuit Breaker Logic

```
                  NAV
                   │
    HWM ──────────┤
                   │
    85% of HWM ───┤──── TRIGGER: circuitBreakerActive = true
                   │              All weights → defensive allocation
                   │              Redemption gate tightened
                   │
    90% of HWM ───┤──── RESET ELIGIBLE: admin can call resetCircuitBreaker()
                   │
                   ▼
```

## Epoch System

```
   ◄────── epochDuration (1 day calm / 7 days volatile) ──────►
   │                                                           │
   epochStart                                          epochStart + epochDuration
   │                                                           │
   ├── Deposits accepted ──────────────────────────────────────┤
   ├── Withdrawals accepted (up to 20% AUM gate) ─────────────┤
   │                                                           │
   └── At boundary: crystallise performance fees, reset gate ──┘
                     advance epoch counter
```
