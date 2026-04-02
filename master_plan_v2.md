# MASTER PLAN v3 — Regime-Switching Risk-Parity Crypto Index Vault
## IFTE0007 Individual Coursework — Asset Tokenisation Design (60%)
### Tamer Atesyakar | UCL MSc Digital Finance & Banking | Deadline: 10 April 2026, 5:00 PM

---

## PROJECT STATUS (Audited 2 April 2026)

### Code — Written

| Component | Status | Verified Lines | Location |
|---|---|---|---|
| Smart contract (ERC-4626 + mocks + interfaces) | ✅ Complete | 1,026 | `contracts/` |
| Deployment scripts (Hardhat) | ✅ Complete | 387 | `scripts/` |
| Test suite (18 tests) | ✅ Complete | 218 | `test/` |
| Data pipeline (fetch + preprocess) | ✅ Complete | 230 | `ml/data/` |
| GARCH-DCC model | ✅ Complete | 262 | `ml/models/garch_dcc.py` |
| Bayesian HMM model | ✅ Complete | 209 | `ml/models/bayesian_hmm.py` |
| SAC RL agent + Gymnasium env | ✅ Complete | 378 | `ml/models/sac_agent.py` + `ml/environment/` |
| Ensemble meta-model | ✅ Complete | 255 | `ml/models/ensemble.py` |
| Portfolio optimizer (7 methods) | ✅ Complete | 357 | `ml/models/portfolio_optimizer.py` |
| Risk analyzer (VaR/CVaR/drawdown) | ✅ Complete | 241 | `ml/models/risk_analyzer.py` |
| Correlation analyzer (EWMA/regime) | ✅ Complete | 144 | `ml/models/correlation_analyzer.py` |
| Walk-forward backtest + 9 charts | ✅ Complete | 513 | `ml/backtest/walk_forward.py` |
| Monte Carlo stress testing | ✅ Complete | 257 | `ml/backtest/monte_carlo.py` |
| Benchmarks (4 strategies) | ✅ Complete | 190 | `ml/backtest/benchmarks.py` |
| Metrics (18+ ratios) | ✅ Complete | 313 | `ml/backtest/metrics.py` |
| Transaction costs (verified Q1 2026) | ✅ Complete | 224 | `ml/backtest/transaction_costs.py` |
| Crisis events (12 events) | ✅ Complete | 196 | `ml/backtest/crisis_events.py` |
| Weight publisher (Merkle + web3) | ✅ Complete | 387 | `ml/weight_publisher/` |
| README (professional, badges) | ✅ Complete | 594 | `README.md` |
| Docs (architecture, dev, expansion) | ✅ Complete | 276 | `docs/` |
| Makefile | ✅ Complete | 51 | `Makefile` |
| **TOTAL CODE (verified)** | | **~5,350** | |

### Execution — TODO (ALL COMPULSORY)

| Task | Status | Run Where | Est. Time |
|---|---|---|---|
| **Deploy to Sepolia** | ❌ TODO | Local terminal | 30 min |
| **Complete SAC training notebook** | ❌ TODO | Local (write cells) | 30 min |
| **Fetch OHLCV data** | ❌ TODO | Colab (GPU runtime) | 10 min |
| **Train SAC on Colab GPU** | ❌ TODO | Colab (T4 GPU) | 3-5 hrs |
| **Run full ML pipeline** | ❌ TODO | Colab (GPU runtime) | 15 min |
| **Run walk-forward backtest** | ❌ TODO | Colab (GPU runtime) | 1-2 hrs |
| **Run Monte Carlo stress test** | ❌ TODO | Colab (GPU runtime) | 30 min |
| **Compute + publish weights on-chain** | ❌ TODO | Local terminal | 15 min |
| **Push backtest charts to GitHub** | ❌ TODO | Local terminal | 5 min |
| **Write 2,400-word report** | ❌ TODO | Local | 8 hrs |
| **Create 2 diagrams** | ❌ TODO | Figma/draw.io | 1 hr |
| **Final assembly + submit** | ❌ TODO | UCL portal | 15 min |

> **Nothing is optional. Every task above must be completed before submission.**
> **All compute-heavy ML tasks (data fetch, SAC training, backtest, Monte Carlo) run on Google Colab to save time.**

---

## TABLE OF CONTENTS

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Step A: Environment Setup](#step-a-environment-setup)
3. [Step B: Smart Contracts — Understanding & Deploying](#step-b-smart-contracts)
4. [Step C: ML Pipeline — Running Every Model](#step-c-ml-pipeline)
5. [Step D: SAC RL Training on Colab GPU](#step-d-sac-rl-training-on-colab)
6. [Step E: Backtesting — Walk-Forward & Monte Carlo](#step-e-backtesting)
7. [Step F: Weight Publication — ML to On-Chain](#step-f-weight-publication)
8. [Step G: Report Writing — 2,400 Words](#step-g-report-writing)
9. [Step H: Diagrams](#step-h-diagrams)
10. [Step I: Final Assembly & Submission](#step-i-final-assembly)
11. [Compliance Checklist](#compliance-checklist)
12. [Timeline & Prioritisation](#timeline--prioritisation)
13. [Full Architecture Reference](#full-architecture-reference)

---

# 1. PROJECT OVERVIEW & ARCHITECTURE

## 1.1 What This Project Is

A tokenised quantitative crypto index fund. Investors deposit USDC into an ERC-4626 vault and receive rpCRYPTO shares representing pro-rata NAV ownership. The portfolio spans 8 assets across 4 asset classes, dynamically rebalanced by an ensemble of ML models.

## 1.2 The Asset → Token → Market → Risk Chain

```
ASSET                          TOKEN                        MARKET                         RISK
─────────────────────────────  ───────────────────────────  ─────────────────────────────  ─────────────────────────
Multi-asset crypto portfolio   ERC-20 vault share           Primary: mint/redeem at NAV    Model: GARCH error, HMM
managed by ensemble ML         (ERC-4626 standard)          Secondary: Uniswap V3 pool     misclassification, RL drift
engine. 4 value sources:       Pro-rata NAV claim           Arbitrage: keeper bots          Oracle: staleness, manip
appreciation, staking yield,   Variable supply              maintaining NAV peg             Contract: inflation attack
Treasury coupon, alpha         Embedded 3-tier fees                                         Market: liquidity spiral,
                                                                                            correlation breakdown
                                                                                            Regulatory: FCA s.235, MiCA
```

## 1.3 Portfolio Composition

| # | Asset | Type | Role | Value Source | Rebalance Venue | Fee (bps) |
|---|---|---|---|---|---|---|
| 1 | BTC | Spot crypto | Core crypto beta | Capital appreciation | Binance Spot | 7.5 |
| 2 | ETH | Spot crypto | Smart contract platform | Capital appreciation | Binance Spot | 7.5 |
| 3 | SOL | Spot crypto | High-beta DeFi/infra | Capital appreciation | Binance Spot | 7.5 |
| 4 | stETH | Liquid staking | ETH + staking yield | Staking ~3.5% | Curve L1 | 4 + $1.50 gas |
| 5 | rETH | Liquid staking | Decentralised ETH staking | Staking ~3.2% | Uniswap L1 | 5 + $1.50 gas |
| 6 | BUIDL | Tokenised Treasury | Risk-free anchor | Coupon ~4.5% | Institutional NAV | 0 |
| 7 | USDY | Tokenised Treasury | Alt risk-free anchor | Coupon ~5.0% | Institutional NAV | 0 |
| 8 | USDC | Stablecoin | Defensive position | Stability | Binance (0-fee) | 0 |

## 1.4 System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   OFF-CHAIN (Python ML Pipeline)                     │
│                                                                      │
│  fetch_data.py ──► preprocess.py ──► [returns_df, hmm_features]     │
│                                           │                          │
│              ┌────────────────────────────┤                          │
│              ▼              ▼              ▼                          │
│       garch_dcc.py   bayesian_hmm.py  sac_agent.py                  │
│       (Σ_t, w_rp)   (regime probs)   (w_rl) ← Colab GPU            │
│              │              │              │                          │
│              └──────┬───────┘──────────────┘                         │
│                     ▼                                                │
│              ensemble.py                                             │
│     ┌─ Inverse-variance weighting                                   │
│     ├─ Kelly confidence scaling (¼ Kelly)                           │
│     ├─ BL with HMM regime views                                    │
│     ├─ 7-method optimizer (HRP/RP/BL/MVO/InvVol/MaxDiv/CVaR)      │
│     └─ Circuit breaker (15% drawdown → defensive)                  │
│              │                                                       │
│              ▼                                                       │
│     compute_weights.py ──► merkle.py ──► publish.py                 │
│              │                                │                      │
└──────────────┼────────────────────────────────┼──────────────────────┘
               │ commitWeights(root)            │ executeWeights(w[])
               ▼                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   ON-CHAIN (Solidity · Sepolia)                      │
│                                                                      │
│  RiskParityVault.sol (ERC-4626)                                     │
│  ├── Deposit USDC → mint rpCRYPTO shares at oracle NAV              │
│  ├── Withdraw → burn shares, return USDC (20% gate)                 │
│  ├── 3-tier fees: management 1%, performance 10% HWM, exit 0.3%    │
│  ├── Dynamic epochs: 1d calm / 7d volatile                          │
│  ├── Merkle commit-reveal (1hr timelock)                            │
│  ├── Circuit breaker (15% drawdown → defensive allocation)          │
│  └── AccessControl: ADMIN + KEEPER roles                            │
└──────────────────────────────────────────────────────────────────────┘
```

## 1.5 Stat-Arb Integration

Components adapted from [Crypto-Statistical-Arbitrage](https://github.com/abailey81/Crypto-Statistical-Arbitrage) (226K lines, 184 files, Sharpe 1.61/5.81):

| Component Pulled | Source File | Adapted Into |
|---|---|---|
| HRP, Risk Parity, BL, MVO, MaxDiv, InvVol | `portfolio/optimization.py` | `ml/models/portfolio_optimizer.py` |
| VaR/CVaR (3 methods), drawdown, risk decomp | `portfolio/risk_analysis.py` | `ml/models/risk_analyzer.py` |
| Rolling/EWMA correlation, regime detection | `portfolio/correlation_analysis.py` | `ml/models/correlation_analyzer.py` |
| Almgren-Chriss slippage, venue-specific fees | `backtesting/transaction_costs.py` | `ml/backtest/transaction_costs.py` |
| 12 crisis event definitions | `backtesting/analysis/crisis_analyzer.py` | `ml/backtest/crisis_events.py` |
| Omega, Ulcer, Pain, Burke, Tail ratios | `backtesting/analysis/advanced_metrics.py` | `ml/backtest/metrics.py` |
| Kelly criterion confidence scaling | `backtesting/analysis/position_sizing_engine.py` | `ml/models/ensemble.py` |
| Monthly heatmap, risk dashboard | `backtesting/visualization.py` | `ml/backtest/walk_forward.py` |

---

# STEP A: ENVIRONMENT SETUP

## A.1 Prerequisites

| Tool | Version | Purpose | Install |
|---|---|---|---|
| Node.js | 18+ | Hardhat, Solidity | nodejs.org |
| Python | 3.10+ | ML pipeline | python.org |
| VS Code | Latest | IDE + Colab extension | code.visualstudio.com |
| MetaMask | Latest | Wallet (testnet only) | metamask.io |
| Git | Latest | Version control | git-scm.com |

## A.2 Project Setup

```bash
# 1. Unzip and enter
unzip riskparity-vault.zip
cd riskparity-vault

# 2. Install Node.js dependencies
npm install
# Installs: hardhat, @openzeppelin/contracts, ethers, dotenv, chai

# 3. Verify Solidity compiles
npx hardhat compile
# Expected: "Compiled 8 Solidity files successfully"

# 4. Run contract tests
npx hardhat test
# Expected: "30 passing"

# 5. Set up Python virtual environment
cd ml
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Installs: arch, hmmlearn, stable-baselines3, cvxpy, web3, ccxt, torch, etc.
cd ..

# 6. Set up environment variables
cp .env.example .env
# Edit .env:
#   SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
#   DEPLOYER_PRIVATE_KEY=0xYOUR_TESTNET_PRIVATE_KEY
#   ETHERSCAN_API_KEY=YOUR_KEY
```

## A.3 Getting API Keys

| Service | URL | What You Get | Time |
|---|---|---|---|
| Alchemy | alchemy.com/signup | Sepolia RPC URL | 2 min |
| MetaMask | metamask.io | Wallet address + private key | 5 min |
| Etherscan | etherscan.io/register → API Keys | Contract verification key | 2 min |
| Sepolia ETH | cloud.google.com/application/web3/faucet/ethereum/sepolia | 0.05 ETH per request | 1 min |

---

# STEP B: SMART CONTRACTS

## B.1 What the Contracts Do

### RiskParityVault.sol (835 lines)

The core contract. Inherits from OpenZeppelin v5 `ERC4626`, `AccessControl`, `ReentrancyGuard`, `Pausable`.

**Key functions:**

| Function | What It Does |
|---|---|
| `deposit(assets, receiver)` | Accept USDC, mint rpCRYPTO shares at oracle NAV. Checks epoch is open. |
| `withdraw(assets, receiver, owner)` | Burn shares, return USDC. Checks epoch + 20% gate. Applies 0.3% exit fee if < 7 days. |
| `totalAssets()` | Compute NAV: Σ(balance_i × price_i) for all constituents via Chainlink. |
| `commitWeights(merkleRoot)` | Keeper commits hash of proposed weights. Starts timelock. |
| `executeWeights(tokens[], weights[], proof[])` | After timelock: verify weights match committed root, update allocations. |
| `accrueManagementFee()` | Mint fee shares to feeRecipient. 1% annual via continuous dilution. |
| `crystallisePerformanceFee()` | At epoch boundary: if NAV > HWM, charge 10% of gains, update HWM. |
| `triggerCircuitBreaker()` | If NAV < 85% of HWM: freeze model weights, force defensive allocation. |
| `advanceEpoch()` | Move to next epoch. Process fee crystallisation. Reset gate. |

**Fee mechanics:**
```
Management Fee (1% annual):
  Every second: fee_shares = totalSupply × 0.01 × Δt / 31,536,000
  Minted to feeRecipient. Dilutes all other holders equally.

Performance Fee (10% above HWM):
  At epoch boundary: if NAV_per_share > highWaterMark:
    gain = (NAV - HWM) × totalSupply
    fee_shares = gain × 0.10 / NAV_per_share
    Mint to feeRecipient. Update HWM = NAV.
  HWM never decreases.

Redemption Fee (0.3% within 7 days):
  If block.timestamp < depositTimestamp[user] + 7 days:
    fee = withdrawal_amount × 0.003
```

### MockERC20.sol
Mintable ERC-20 for testnet. Has `faucet(amount)` so anyone can mint test tokens. Used for USDC, WBTC, WETH, etc.

### MockPriceFeed.sol
Simulates Chainlink AggregatorV3. Owner calls `updatePrice(int256 price)`. Stores price history. Can compute TWAP over configurable window.

## B.2 Deploy to Sepolia

```bash
# Step 1: Deploy 8 mock tokens + 8 mock price feeds (~3 min)
npx hardhat run scripts/deploy_mocks.js --network sepolia
# Output:
#   MockUSDC deployed to: 0x...
#   MockWBTC deployed to: 0x...
#   ... (8 tokens + 8 feeds)
# ★ COPY ALL ADDRESSES ★

# Step 2: Update deploy.js with the mock addresses
# Edit scripts/deploy.js lines 15-30 with the addresses from Step 1

# Step 3: Deploy the vault (~2 min)
npx hardhat run scripts/deploy.js --network sepolia
# Output:
#   RiskParityVault deployed to: 0x...
# ★★★ THIS IS YOUR CONTRACT ADDRESS — PUT IN REPORT ★★★

# Step 4: Run the lifecycle demo (~3 min)
npx hardhat run scripts/interact.js --network sepolia
# This does: approve USDC → deposit 10,000 → commit weights → execute → withdraw 2,000
# Creates visible transactions on Etherscan

# Step 5: Verify on Etherscan (optional but professional)
npx hardhat verify --network sepolia <VAULT_ADDRESS> <CONSTRUCTOR_ARGS>
```

## B.3 After Deployment

1. Update `README.md` line 9 with the contract address
2. Go to `https://sepolia.etherscan.io/address/<YOUR_ADDRESS>` — bookmark this
3. Take a screenshot showing the verified contract (for report if needed)
4. The `interact.js` script creates 3+ visible transactions — these prove the contract works

---

# STEP C: ML PIPELINE

## C.1 Data Fetching (`ml/data/fetch_data.py`)

```bash
cd ml
source venv/bin/activate
python -m data.fetch_data
```

**What it does:**
- Connects to Binance via ccxt library
- Downloads hourly OHLCV candles for BTC, ETH, SOL, stETH, rETH (real market data)
- Generates synthetic price series for BUIDL (4.5% annual yield + micro-noise) and USDY (5.0%)
- Generates USDC at constant $1.00
- Caches everything as parquet files in `data/cache/` (only downloads once)
- Takes ~10 minutes on first run

**Output:** `data/cache/{symbol}_hourly.parquet` for each asset

## C.2 Preprocessing (`ml/data/preprocess.py`)

Called automatically by downstream modules. Computes:
- Log returns for all assets
- Realised volatility at 3 windows (1d, 5d, 20d)
- Rolling 60-day Sharpe ratio
- Momentum (5d, 20d, 60d)
- Market breadth (fraction of assets with positive 24h returns)
- HMM-specific features: [log_returns_mean_24h, realised_vol_120h, market_breadth]

**Output:** `data/cache/prices_aligned.parquet`, `returns.parquet`, `hmm_features.parquet`

## C.3 GARCH-DCC Model (`ml/models/garch_dcc.py`)

**Mathematical specification:**
```
Univariate: r_{i,t} = μ_i + σ_{i,t} · z_{i,t},  z ~ Student-t(ν)
            σ²_{i,t} = ω + α · ε²_{t-1} + β · σ²_{t-1}

DCC:        Q_t = (1-a-b) · Q̄ + a · z_{t-1} · z'_{t-1} + b · Q_{t-1}
            R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

Covariance: Σ_t = D_t · R_t · D_t   where D_t = diag(σ_{1,t},...,σ_{n,t})
```

**What it does:**
1. Fits a GARCH(1,1) with Student-t innovations for each of the 8 assets (using `arch` library)
2. Extracts standardised residuals z_{i,t}
3. Estimates DCC parameters (a, b) via quasi-maximum likelihood
4. Forecasts the full 8×8 covariance matrix Σ_{t+1|t}
5. Computes risk-parity weights (equal risk contribution from each asset)

**To run standalone:**
```python
from ml.models.garch_dcc import StudentTGarchDCC
garch = StudentTGarchDCC(p=1, q=1, distribution="studentst")
garch.fit(returns_df)
sigma = garch.forecast_covariance()
rp_weights = garch.get_risk_parity_weights(sigma)
uncertainty = garch.get_uncertainty()
```

**Output:** Σ_t (8×8 matrix), risk-parity weights (8×1), uncertainty scalar

## C.4 Bayesian HMM (`ml/models/bayesian_hmm.py`)

**Mathematical specification:**
```
Hidden states: S_t ∈ {Bull, Normal, Crisis}
Observations: X_t = [log_returns_mean, realised_vol, market_breadth]
Emissions:    X_t | S_t=k ~ N(μ_k, Σ_k)
Transitions:  A_{ij} = P(S_t=j | S_{t-1}=i), with Dirichlet sticky prior
Output:       γ_t(k) = P(S_t=k | X_1,...,X_T)  — SOFT posteriors
```

**What it does:**
1. Fits 3-state Gaussian HMM using `hmmlearn`
2. Sets sticky transition prior (high diagonal = persistent regimes)
3. Sorts states by mean return: bull (highest) > normal > crisis (lowest)
4. Outputs soft posterior probabilities [P(bull), P(normal), P(crisis)]
5. These soft probabilities enable continuous risk budget blending (no hard switching)

**To run standalone:**
```python
from ml.models.bayesian_hmm import BayesianRegimeHMM
hmm = BayesianRegimeHMM(n_states=3, n_iter=200)
hmm.fit(hmm_features)
regime_probs = hmm.predict_proba(hmm_features)  # (T, 3)
regime_labels = hmm.predict_regime(hmm_features)  # bull/normal/crisis
persistence = hmm.get_regime_persistence()  # Expected duration per regime
trans_matrix = hmm.get_transition_matrix()  # 3×3 transition probabilities
```

**Output:** Regime probabilities (T×3), transition matrix, persistence stats

## C.5 SAC Deep RL Agent (`ml/models/sac_agent.py` + `ml/environment/portfolio_env.py`)

### The Environment

**State space (dim=38):**
```
Per-asset features (8 assets × 3 features = 24):
  - 5-day returns, 20-day returns, 20-day volatility
GARCH volatility forecasts (8)
HMM regime probabilities (3)
Portfolio state (3):
  - cumulative return, current drawdown, days since rebalance
```

**Action space (dim=8):**
- Continuous [0,1]^8 → softmax → target weights summing to 1

**Reward function:**
```
r_t = sharpe_coeff × portfolio_return
    - cvar_penalty × max(0, CVaR_threshold - rolling_CVaR)
    - drawdown_penalty × max(0, drawdown - 0.10)
    - turnover_penalty × |w_new - w_old|_1
```

### The Agent

Uses Soft Actor-Critic (Haarnoja et al., 2018) from stable-baselines3:
- Entropy-regularised actor-critic
- Automatic entropy coefficient tuning
- Multi-seed training (5 seeds), best seed selected

### Training (REQUIRES GPU — USE COLAB)

See [Step D](#step-d-sac-rl-training-on-colab) for full Colab instructions.

### Inference (CPU — fast)

```python
from ml.models.sac_agent import SACAllocator
sac = SACAllocator(config)
sac.load("models/saved/sac_best")
weights = sac.predict(observation)  # instant on CPU
```

### SAC Training Is Compulsory

SAC training must be completed on Colab T4 GPU. The full ensemble requires all three model outputs (GARCH-DCC, HMM, SAC) to demonstrate the inverse-variance weighting mechanism described in the report. Without a trained SAC model, the ensemble degrades to a two-model system and the RL contribution cannot be evidenced in backtest charts.

## C.6 Portfolio Optimizer (`ml/models/portfolio_optimizer.py`)

Seven methods available, adapted from the stat-arb:

| Method | Function | Mathematical Basis |
|---|---|---|
| Risk Parity | `optimizer.risk_parity(cov)` | Equal risk contribution: w_i × (Σw)_i = (1/n) × w'Σw |
| HRP | `optimizer.hrp(cov, corr)` | Hierarchical clustering + recursive bisection (López de Prado) |
| Black-Litterman | `optimizer.black_litterman(cov, mkt_w, views)` | BL posterior: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [...] |
| Mean-Variance | `optimizer.mean_variance(mu, cov)` | Max Sharpe: w = Σ⁻¹(μ - r_f) / 1'Σ⁻¹(μ - r_f) |
| Inverse Vol | `optimizer.inverse_volatility(cov)` | w_i ∝ 1/σ_i |
| Max Diversification | `optimizer.max_diversification(cov)` | Max: (Σ w_i σ_i) / √(w'Σw) |
| CVaR-Constrained | `optimizer.cvar_constrained(target, cov, current)` | CVXPY: min ‖w - target‖² s.t. vol ≤ bound, turnover ≤ 30% |

**To compare all methods:**
```python
from ml.models.portfolio_optimizer import MultiMethodOptimizer
opt = MultiMethodOptimizer(min_weight=0.02, max_weight=0.40, max_turnover=0.30)
comparison = opt.compare_methods(cov_matrix, returns_df)
print(comparison)  # Table with vol, CVaR, div ratio, eff N for each method
```

## C.7 Ensemble Meta-Model (`ml/models/ensemble.py`)

The orchestrator. Combines all model outputs in 7 steps:

```
Step 1: Update high-water mark
Step 2: Circuit breaker check (15% drawdown → force defensive)
Step 3: Regime-blended risk budgets from HMM soft probabilities
        bull:   70% crypto, 20% staking, 0% Treasury, 10% stable
        normal: 40% crypto, 30% staking, 20% Treasury, 10% stable
        crisis: 10% crypto, 10% staking, 50% Treasury, 30% stable
        Blended: Σ P(regime) × budget(regime) for each bucket
Step 4: Multi-method optimizer (HRP, RP, BL, etc.) on GARCH covariance
Step 5: Inverse-variance weighting of 3 model outputs
        model_weight_i = (1/uncertainty_i²) / Σ(1/uncertainty_j²)
Step 6: Kelly confidence scaling
        When uncertainty is HIGH: blend toward 1/N equal weight
        When uncertainty is LOW: trust model weights fully
        Uses quarter-Kelly (0.25x) for conservative sizing
Step 7: CVaR-constrained optimisation via CVXPY
        Enforces: vol ≤ 25%, weights 2-40%, turnover ≤ 30%
```

**To run the full pipeline:**
```python
result = ensemble.combine(
    garch_rp_weights=garch_weights,
    hmm_regime_probs=np.array([0.2, 0.5, 0.3]),
    rl_weights=rl_weights,
    covariance_matrix=sigma,
    current_weights=current_weights,
    current_nav=1.05,
    garch_uncertainty=0.3,
    hmm_uncertainty=0.4,
    rl_uncertainty=0.8,
    optimization_method="hrp",
)
print(result["weights"])     # Final weight vector
print(result["regime"])      # Dominant regime
print(result["kelly_scale"]) # Kelly confidence factor
```

## C.8 Risk Analyzer (`ml/models/risk_analyzer.py`)

Three VaR methods + drawdown + risk decomposition + BTC correlation:

```python
from ml.models.risk_analyzer import VaultRiskAnalyzer
analyzer = VaultRiskAnalyzer(returns_df, weights=final_weights)

# VaR comparison
var_table = analyzer.var_comparison(confidence=0.95, horizon=24)

# Top-5 drawdown events with dates and recovery
max_dd, events = analyzer.analyze_drawdowns(top_n=5)

# Risk decomposition (marginal risk per asset)
decomp = analyzer.decompose_risk()

# BTC correlation and beta
btc_stats = analyzer.btc_correlation(btc_returns)

# Full risk limit check
limits = analyzer.check_risk_limits(var_limit=0.03, dd_limit=0.20)
```

## C.9 Correlation Analyzer (`ml/models/correlation_analyzer.py`)

EWMA correlations + regime classification + breakdown detection:

```python
from ml.models.correlation_analyzer import CorrelationAnalyzer
corr = CorrelationAnalyzer(returns_df)

# Current correlation regime (LOW/NORMAL/HIGH/CRISIS)
regime = corr.current_regime(window=1440)

# Detect if diversification is breaking down
breakdown = corr.detect_correlation_breakdown(short_window=168, long_window=1440)

# Average correlation as HMM feature
avg_corr = corr.average_correlation(window=1440)
```

---

# STEP D: SAC RL TRAINING ON COLAB

**This is the only step that requires GPU. Everything else runs on CPU.**

## D.1 Why Colab

The SAC agent trains for 500K timesteps × 5 seeds. On CPU this takes ~20+ hours. On a T4 GPU via Colab, it takes ~3-5 hours.

## D.2 Setup (One-Time)

1. Open VS Code
2. Install extension: **Google Colab** (search in Extensions marketplace)
3. Open `ml/notebooks/train_sac.ipynb`
4. Click **Select Kernel** (top right)
5. Select **Colab**
6. Choose **T4 GPU** runtime

## D.3 What the Notebook Does

The notebook has these cells:

```
Cell 1: Install dependencies (pip install stable-baselines3 gymnasium torch)
Cell 2: Import modules and load config.yaml
Cell 3: Load preprocessed data (returns, features, regime probs, GARCH vols)
Cell 4: Create the PortfolioEnv gymnasium environment
Cell 5: Train SAC for 5 seeds × 500K timesteps each
         - Each seed takes ~40-60 minutes
         - Total: ~3-5 hours
Cell 6: Evaluate each seed on one full episode
Cell 7: Select best seed by total reward
Cell 8: Save to ml/models/saved/sac_best.zip
```

## D.4 After Training

1. Switch kernel back to **local Python**
2. The saved model is at `ml/models/saved/sac_best.zip`
3. Re-run the backtest with the `--rl` flag:
   ```bash
   python -m backtest.walk_forward --config config.yaml --rl
   ```
4. This produces updated charts that include SAC's contribution to the ensemble

## D.5 After Training — Verify Model Saved

```bash
# Check the trained model exists:
ls -la ml/models/saved/sac_best.zip
# Should be ~5-15 MB. If missing, training failed — re-run on Colab.
```

Download `sac_best.zip` from Colab to `ml/models/saved/` on your local machine before running the backtest.

After SAC training completes on Colab, re-run the backtest with the `--rl` flag to generate updated charts that include SAC's contribution to the ensemble. These charts go in the GitHub repo and support the report's claims about three-model ensemble weighting.

---

# STEP E: BACKTESTING

## E.1 Walk-Forward Backtest (Run on Colab)

All backtest compute runs on Google Colab to save time. Upload the `ml/` folder to Colab or mount Google Drive.

```python
# In a Colab notebook cell (after SAC training completes):
%cd /content/ml
!python -m backtest.walk_forward --config config.yaml --rl
```

The `--rl` flag is **compulsory** — it includes the trained SAC agent in the ensemble.

**Protocol:**
- Train window: 180 days (4,320 hours)
- Test window: 30 days (720 hours)
- Roll forward: 30 days
- Each fold: refit GARCH + HMM, run ensemble, simulate returns
- Compare against 4 benchmarks: equal weight, BTC only, 60/40, market-cap weighted
- Transaction costs: venue-specific (verified Q1 2026)

**Output (9 charts + CSV):**

| File | What It Shows |
|---|---|
| `results/equity_curves.png` | Log-scale performance vs 4 benchmarks |
| `results/drawdown_plot.png` | Drawdown time series comparison |
| `results/regime_timeline.png` | HMM regime classification over time |
| `results/weight_evolution.png` | Stacked area chart of portfolio weights |
| `results/rolling_sharpe.png` | 90-day rolling Sharpe comparison |
| `results/monthly_returns_heatmap.png` | Year × Month performance grid |
| `results/risk_dashboard.png` | 4-panel: rolling vol, CVaR, drawdown, VaR |
| `results/monte_carlo_fan_6m.png` | 10K-path fan chart (6-month) |
| `results/monte_carlo_fan_12m.png` | Fan chart (12-month) |
| `results/performance_summary.csv` | All 18+ metrics for all strategies |

## E.2 Metrics Computed

| Metric | What It Measures | Source |
|---|---|---|
| Annualised Return | Geometric annual return | Standard |
| Annualised Volatility | σ × √8760 | Standard |
| Sharpe Ratio | (return - rf) / vol | Standard |
| Sortino Ratio | (return - rf) / downside vol | Standard |
| Calmar Ratio | return / |max drawdown| | Standard |
| Max Drawdown | Worst peak-to-trough | Standard |
| CVaR 5% | Average of worst 5% returns | Standard |
| VaR 5% | 5th percentile return | Standard |
| Omega Ratio | Gains above 0 / losses below 0 | Stat-arb |
| Ulcer Index | RMS of drawdowns | Stat-arb |
| Pain Index | Mean absolute drawdown | Stat-arb |
| Burke Ratio | Return / √(Σ dd²) | Stat-arb |
| Tail Ratio | |95th pct| / |5th pct| | Stat-arb |
| Gain-to-Pain | Total return / Σ|losses| | Stat-arb |
| Tracking Error | vs equal-weight benchmark | Standard |
| Information Ratio | Excess return / tracking error | Standard |
| Average Turnover | L1 norm of weight changes | Standard |
| Skewness / Kurtosis | Distribution shape | Standard |

## E.3 Monte Carlo Stress Testing

Runs automatically as part of the backtest. 10,000 simulated paths using regime-conditioned block bootstrap from the HMM transition matrix.

**Reports:**
- Terminal value distribution (median, 5th, 95th percentile)
- P(loss), P(loss > 20%), P(gain > 50%), P(Sharpe > 1), P(MDD > 30%)

## E.4 Crisis-Stratified Analysis

Uses `crisis_events.py` to report metrics separately during 12 documented crypto crises:
- COVID Crash (Mar 2020, -52% BTC)
- UST/Luna Collapse (May 2022, -40%)
- FTX Collapse (Nov 2022, -27%)
- SVB/USDC Depeg (Mar 2023, -10%)
- Yen Carry Unwind (Aug 2024, -18%)
- ... and 7 more

---

# STEP F: WEIGHT PUBLICATION

## F.1 Compute Weights

```bash
cd ml
python -m weight_publisher.compute_weights --config config.yaml
```

**What it does:**
1. Loads preprocessed data
2. Fits GARCH-DCC → Σ_t, risk-parity weights
3. Fits HMM → regime probabilities
4. (Optional) Loads SAC model → RL weights
5. Runs ensemble combiner → final weights w*
6. Computes Merkle root from (token_address, weight_bps) pairs
7. Saves to `results/latest_weights.json`

## F.2 Push On-Chain

```bash
python -m weight_publisher.publish --vault 0xYOUR_VAULT_ADDRESS --weights results/latest_weights.json
```

**What it does:**
1. Connects to Sepolia via web3.py
2. Calls `vault.commitWeights(merkleRoot)` — Phase 1
3. Waits for 1-hour timelock (on testnet, can be reduced)
4. Calls `vault.executeWeights(tokens[], weights[], proof[])` — Phase 2

---

# STEP G: REPORT WRITING (2,400 WORDS)

## G.1 Title

```
Tokenised Regime-Switching Risk-Parity Crypto Index:
An ERC-4626 Vault Design for Democratised Quantitative Portfolio Management

Contract Address (Sepolia): 0x...
GitHub: https://github.com/abailey81/Regime-Switching-Risk-Parity-Crypto-Index-Vault
```

## G.2 Word Budget

| Section | Weight | Words | Key Question |
|---|---|---|---|
| 1. Asset Description | 15% | ~360 | WHAT is the asset, HOW does it create value? |
| 2. Rationale | 15% | ~360 | WHY tokenise, WHO benefits? |
| 3. Token Structure | 20% | ~480 | HOW is the token designed, WHAT rights/fees/rules? |
| 4. Market Access | 20% | ~480 | HOW is liquidity provided, WHAT's the pricing? |
| 5. Risk Analysis | 20% | ~480 | WHAT can go wrong, HOW is it mitigated, WHAT's residual? |
| 6. DeFi Infrastructure | 10% | ~240 | WHICH protocols, WHY blockchain? |
| **Total** | | **~2,400** | |

## G.3 Section-by-Section Content Plans

### Section 1: Asset Description & Value Source (~360 words)

**Paragraph 1 — What the asset is (~100 words):**
- Dynamically managed multi-asset crypto portfolio
- 8 constituents across 4 asset classes
- Managed by ensemble of quantitative models

**Paragraph 2 — Four value sources (~130 words):**
- Capital appreciation from BTC/ETH/SOL
- Staking yield (~3-4%) from stETH, rETH — accrues to NAV automatically
- Treasury coupon (~4.5-5%) from BUIDL, USDY — risk-free anchor
- Active management alpha from regime-conditional rebalancing
- Diversification benefit: imperfectly correlated assets → portfolio risk < individual risk

**Paragraph 3 — NAV computation (~80 words):**
- On-chain via Chainlink + 30-min TWAP
- Share price = NAV / totalSupply
- Contrast: traditional funds have T+1 admin-computed NAV

**Paragraph 4 — Economic significance (~50 words):**
- Spans DeFi yield + traditional fixed income + crypto beta — all on-chain
- Multi-source value distinguishes from static baskets

**References:** Markowitz (1952), Engle (2002), Maillard et al. (2010)

---

### Section 2: Rationale for Tokenisation (~360 words)

**Paragraph 1 — Problems (~120 words):**
- High minimums ($100K-$1M), opaque 2/20 fees, T+1-T+3 settlement, quarterly disclosures, geographic barriers

**Paragraph 2 — Solutions (~120 words):**
- Access: $50 minimum. Fee transparency: Solidity-enforced. Settlement: hours. Transparency: real-time on-chain. Accessibility: permissionless, 24/7.

**Paragraph 3 — Target holders (~70 words):**
- Retail seeking diversified risk-managed exposure
- DeFi-native seeking yield-bearing collateral (Aave)
- Institutional testing on-chain fund structures

**Paragraph 4 — Regulatory (~50 words):**
- FCA CP25/28 Blueprint model. Aligns with trajectory, identifies friction.

**References:** FCA (2025), BlackRock BUIDL, WEF (2024)

---

### Section 3: Token Structure & Design (~480 words)

**Paragraph 1 — FT justification (~80 words):**
- ERC-20 because each share = identical fractional NAV
- ERC-4626 standard for deposit/withdraw/share accounting

**Paragraph 2 — Rights and obligations (~100 words):**
- Pro-rata NAV claim. No governance. No dividends (accumulating). Accept fees + epoch + gate.

**Paragraph 3 — Fee architecture (~100 words):**
- Management 1% (dilution), Performance 10% (HWM), Redemption 0.3% (anti-churn)
- All in Solidity — transparent, verifiable, non-discretionary

**Paragraph 4 — Supply rules (~100 words):**
- Variable supply (mint/burn). Dynamic epochs. 20% gate. Queued if exceeded. Transferable.

**Paragraph 5 — Weight integrity (~60 words):**
- Merkle commit-reveal, 1hr timelock, circuit breaker 15%

**[INSERT DIAGRAM 1]**

**References:** EIP-4626 (2022), OpenZeppelin (2024), Rockafellar & Uryasev (2000)

---

### Section 4: Market Access & Liquidity (~480 words)

**Paragraph 1 — Three-layer intro (~60 words):**
- Analogous to ETF creation/redemption (Ben-David et al., 2018)

**Paragraph 2 — Primary market (~120 words):**
- Deposit USDC → shares at NAV. Dynamic epochs. 20% gate. TWAP prevents manipulation.

**Paragraph 3 — Secondary market (~120 words):**
- Uniswap V3 concentrated liquidity. ±1.5% of NAV. Instant. 24/7.

**Paragraph 4 — Arbitrage stabilisation (~100 words):**
- Keeper monitors spread. Discount → buy+redeem. Premium → mint+sell.
- Contrast: GBTC traded at -48% discount (no redemption mechanism)

**Paragraph 5 — Pricing (~80 words):**
- NAV: deterministic. Secondary: market-driven + arbitrage-anchored.

**References:** Ben-David et al. (2018), Adams et al. (2021)

---

### Section 5: Risk Analysis & Limitations (~480 words)

**THIS SECTION DETERMINES YOUR GRADE. Be analytical, not descriptive.**

**Paragraph 1 — Model risk (~120 words):**
- GARCH estimation error (kurtosis ~8-15, Student-t helps but doesn't eliminate)
- HMM lag (2-5 obs; UST/Luna: bull→crisis in 48hrs while model stays bull)
- RL distributional shift (novel dynamics outside training)
- Ensemble correlation (if all models wrong, error amplifies)

**Paragraph 2 — Technical risk (~100 words):**
- Oracle staleness (1hr heartbeat; flash crash → NAV miscalculation)
- ERC-4626 inflation attack (mitigated by virtual share offset)
- MEV sandwich on rebalance swaps (turnover limits + Flashbots)

**Paragraph 3 — Market/structural (~140 words):**
- **Liquidity spiral**: crash → mass redemptions → 20% gate → 80% locked → trust collapses → secondary discount widens → arbitrageurs can't close because redemption also gated. **This is a FUNDAMENTAL LIQUIDITY TRILEMMA: instant liquidity, portfolio integrity, and run prevention cannot coexist. This limitation is structural and cannot be fully resolved.**
- Correlation breakdown: crisis → all correlations → 1.0 → diversification eliminated. Regime rotation mitigates but constrained by turnover + detection lag.

**Paragraph 4 — Regulatory (~80 words):**
- s.235 FSMA → FCA authorisation needed. MiCA → MiFID II security. DSS pathway. Off-chain ML = centralised trust point.

**[INSERT DIAGRAM 2: Risk taxonomy]**

**References:** Hansen (1994), Artzner et al. (1999), FSB (2024), FCA (2025), MiCA (2023)

---

### Section 6: DeFi Infrastructure (~240 words)

**Paragraph 1 — Protocol mapping (~120 words):**
- Smart contracts: ERC-4626 accounting + fees + circuit breaker
- Oracles: Chainlink for NAV + TWAP
- DEXs: Uniswap V3 for secondary market
- Yield: Lido stETH, Rocket Pool rETH

**Paragraph 2 — Composability (~80 words):**
- ERC-20 → Aave V3 collateral → leveraged risk-parity exposure
- Recursive composability impossible in traditional funds

**Paragraph 3 — Settlement (~40 words):**
- On-chain finality. No counterparty risk. Full audit trail. vs T+2 traditional.

**References:** Adams et al. (2021), Chainlink, Aave V3

---

## G.4 Full Reference List

```
Adams, H., Zinsmeister, N. and Robinson, D. (2021) 'Uniswap v3 Core', Uniswap Labs.
Artzner, P. et al. (1999) 'Coherent Measures of Risk', Mathematical Finance, 9(3).
Ben-David, I., Franzoni, F. and Moussawi, R. (2018) 'Do ETFs Increase Volatility?', JoF, 73(6).
Engle, R. (2002) 'Dynamic Conditional Correlation', JBES, 20(3).
European Parliament (2023) MiCA Regulation (EU) 2023/1114.
FCA (2025) CP25/28: Progressing Fund Tokenisation.
FSB (2024) Financial Stability Implications of Tokenisation.
Haarnoja, T. et al. (2018) 'Soft Actor-Critic', ICML.
Hansen, B.E. (1994) 'Autoregressive Conditional Density', IER, 35(3).
López de Prado, M. (2016) 'Building Diversified Portfolios', JPM, 42(4).
Maillard, S. et al. (2010) 'Equally Weighted Risk Contribution Portfolios', JPM, 36(4).
Markowitz, H. (1952) 'Portfolio Selection', JoF, 7(1).
OpenZeppelin (2024) ERC-4626 Implementation Documentation.
Rockafellar, R.T. and Uryasev, S. (2000) 'Optimization of CVaR', J. Risk, 2(3).
EIP-4626 (2022) Tokenized Vault Standard.
```

---

# STEP H: DIAGRAMS

## Diagram 1: Architecture Flow

Show the full asset → token → market chain. Include:
- LEFT: 8 portfolio assets grouped by class + ML ensemble engine below
- CENTRE: ERC-4626 token with fee/epoch/Merkle/circuit breaker labels
- RIGHT: Three-layer liquidity (primary/secondary/arbitrage)
- Tool: Figma, draw.io, or PowerPoint. 300 DPI PNG.

## Diagram 2: Risk Taxonomy

Three-column hierarchical visual:
- Column 1: Model Risk (GARCH error, HMM lag, RL drift, ensemble correlation)
- Column 2: Technical Risk (oracle, inflation attack, MEV, reentrancy)
- Column 3: Market/Structural (liquidity spiral, correlation breakdown, regulatory, trust gap)

---

# STEP I: FINAL ASSEMBLY & SUBMISSION

```
□ Word count: ≤ 2,400 (excluding refs, diagrams, contract address, GitHub link)
□ Contract address visible in report header
□ GitHub link visible in report header
□ GitHub repo is PUBLIC
□ References in Harvard format
□ 2 diagrams labelled (Figure 1, Figure 2)
□ 12pt font, 1.5 spacing, standard margins
□ Sepolia contract verified on Etherscan
□ README updated with contract address
□ At least 1 deposit + 1 rebalance + 1 withdraw TX on Etherscan
□ All 9 backtest charts pushed to GitHub in results/
□ Monte Carlo fan charts pushed to GitHub in results/
□ SAC trained model (sac_best.zip) exists in ml/models/saved/
□ Weights published on-chain (commit + execute TX visible on Etherscan)
```

---

# COMPLIANCE CHECKLIST

```
HARD REQUIREMENTS:
[  ] ONE underlying asset (managed portfolio = one asset)
[  ] FT chosen and justified (ERC-20 via ERC-4626)
[  ] Contract address on testnet IN REPORT
[  ] GitHub link IN REPORT
[  ] ≤ 2,400 words
[  ] Individual work

EXECUTION REQUIREMENTS (ALL COMPULSORY):
[  ] Sepolia deployment with visible transactions on Etherscan
[  ] SAC trained on Colab T4 GPU (sac_best.zip saved)
[  ] Walk-forward backtest run with --rl flag (all 9 charts generated)
[  ] Monte Carlo stress test run (fan charts generated)
[  ] Weights computed and published on-chain
[  ] All charts pushed to GitHub results/ folder
[  ] 2 diagrams created (architecture + risk taxonomy)

GRADE CRITERIA:
[  ] Asset → Token → Market → Risk chain CLEAR + CONSISTENT
[  ] Economic reasoning behind every design choice
[  ] Critical reflection on trade-offs
[  ] Independent thinking
[  ] Financial design focus (not just technical complexity)

SOPHISTICATION (top band):
[  ] Multi-asset portfolio (4 asset classes)
[  ] Ensemble ML (GARCH + HMM + SAC + meta-model) — ALL THREE TRAINED
[  ] 7 optimisation methods (HRP, RP, BL, MVO, InvVol, MaxDiv, CVaR)
[  ] 3-tier fees, 3-layer liquidity, Merkle commit-reveal
[  ] Circuit breaker, correlation monitoring
[  ] Walk-forward + Monte Carlo + 12 crisis events
[  ] 18+ performance metrics
[  ] Stat-arb integration (credited, linked)
[  ] Professional README
```

---

# TIMELINE & PRIORITISATION

## Execution Order — ALL COMPULSORY

```
PHASE 1: INFRASTRUCTURE (Day 1)
1. Deploy to Sepolia → get contract address               [30 min]
2. Complete SAC training notebook (write missing cells)    [30 min]
3. Push to GitHub                                          [10 min]

PHASE 2: COMPUTE ON COLAB (Day 1-2)
4. Fetch OHLCV data on Colab                               [10 min]
5. Train SAC on Colab T4 GPU (leave running overnight)     [3-5 hrs]
6. Run walk-forward backtest on Colab                      [1-2 hrs]
7. Run Monte Carlo stress test on Colab                    [30 min]
8. Compute ensemble weights on Colab                       [15 min]

PHASE 3: ON-CHAIN BRIDGE (Day 2)
9. Publish weights on-chain via keeper                     [15 min]
10. Push all charts + results to GitHub                     [10 min]

PHASE 4: REPORT (Days 3-6)
11. Write 2,400-word report                                 [8 hrs]
12. Create 2 diagrams                                       [1 hr]

PHASE 5: POLISH + SUBMIT (Days 7-8)
13. Revision passes, formatting, citations                  [3 hrs]
14. Final compliance check + submit                         [30 min]
```

## Day-by-Day Schedule (April 2-10)

```
Day 1 (Wed Apr 2 — TODAY):
  PM: Deploy mocks + vault to Sepolia, record contract address
  PM: Complete SAC training notebook cells
  PM: Start data fetch on Colab
  EVE: Start SAC training on Colab T4 GPU (leave overnight)

Day 2 (Thu Apr 3):
  AM: SAC training finishes → download sac_best.zip
  AM: Run walk-forward backtest on Colab (with --rl flag)
  PM: Run Monte Carlo stress test on Colab
  PM: Compute weights → publish on-chain
  EVE: Push ALL charts + results to GitHub

Day 3 (Fri Apr 4):
  AM: Review charts + metrics. Understand backtest results.
  PM: Write Section 1 (Asset Description) = 360 words
  EVE: Write Section 2 (Rationale) = 360 words

Day 4 (Sat Apr 5):
  AM: Write Section 3 (Token Structure) = 480 words
  PM: Write Section 5 (Risk Analysis) = 480 words ★ MOST IMPORTANT ★
  EVE: Write Section 4 (Market Access) = 480 words

Day 5 (Sun Apr 6):
  AM: Write Section 6 (DeFi Infrastructure) = 240 words. Draft complete.
  PM: Create Diagram 1 (Architecture flow) + Diagram 2 (Risk taxonomy)
  EVE: Full revision pass — logic flow, word count, citations

Day 6 (Mon Apr 7):
  AM: Second revision — fresh eyes, read aloud
  PM: Format references (Harvard). Verify contract + GitHub links.
  EVE: Push any final code changes to GitHub

Day 7 (Tue Apr 8):
  AM: Final compliance checklist pass
  PM: BUFFER — fix anything broken
  EVE: Prepare submission PDF

Day 8 (Wed Apr 9):
  AM: Final review with fresh eyes
  PM: BUFFER — emergency fixes
  EVE: Format final PDF

Day 9 (Thu Apr 10 — DEADLINE 5:00 PM):
  AM: Final review
  NOON: SUBMIT via UCL portal
  1 PM: Verify submission received
  2-5 PM: Emergency buffer (4 hours)
```

---

# FULL ARCHITECTURE REFERENCE

```
Regime-Switching-Risk-Parity-Crypto-Index-Vault/ (~5,350 lines total, verified)
│
├── contracts/                         1,026 lines Solidity
│   ├── RiskParityVault.sol              835 — Core ERC-4626 vault
│   ├── mocks/MockERC20.sol              43  — Mintable test tokens
│   ├── mocks/MockPriceFeed.sol          126 — Simulated Chainlink + TWAP
│   └── interfaces/IChainlinkAggregator.sol  22
│
├── scripts/                           387 lines JavaScript
│   ├── deploy_mocks.js (114)            Deploy 8 tokens + 8 feeds
│   ├── deploy.js (120)                  Deploy vault, register constituents
│   └── interact.js (153)               Full lifecycle demo
│
├── test/RiskParityVault.test.js       218 lines, 18 test cases
│
├── ml/                                ~3,500 lines Python
│   ├── config.yaml (52)                 ALL hyperparameters
│   ├── data/
│   │   ├── fetch_data.py (87)           Binance OHLCV + synthetic
│   │   └── preprocess.py (143)          Feature engineering
│   ├── models/
│   │   ├── garch_dcc.py (262)           Student-t GARCH-DCC
│   │   ├── bayesian_hmm.py (209)        3-state soft-posterior HMM
│   │   ├── sac_agent.py (170)           SAC RL (train on Colab)
│   │   ├── ensemble.py (255)            Meta-model + Kelly + BL
│   │   ├── portfolio_optimizer.py (357) 7 methods (HRP/RP/BL/MVO/...)
│   │   ├── risk_analyzer.py (241)       VaR/CVaR/drawdown/BTC corr
│   │   └── correlation_analyzer.py (144) EWMA corr + breakdown
│   ├── environment/
│   │   └── portfolio_env.py (208)       Custom Gymnasium env
│   ├── backtest/
│   │   ├── walk_forward.py (513)        Engine + 9 charts
│   │   ├── monte_carlo.py (257)         10K-path stress test
│   │   ├── metrics.py (313)             18+ ratios
│   │   ├── benchmarks.py (190)          4 comparison strategies
│   │   ├── transaction_costs.py (224)   Verified Q1 2026 fees
│   │   └── crisis_events.py (196)       12 crypto crises
│   ├── weight_publisher/
│   │   ├── compute_weights.py (176)     Full pipeline
│   │   ├── merkle.py (65)              Keccak-256 tree
│   │   └── publish.py (146)            Web3.py keeper
│   └── notebooks/
│       └── train_sac.ipynb              Colab GPU training (MUST COMPLETE)
│
├── docs/                              276 lines
│   ├── ARCHITECTURE.md (119)            System design
│   ├── DEVELOPMENT.md (93)              Setup + workflow
│   └── EXPANSION.md (64)               Improvement ideas
│
├── README.md (594)                    Professional with badges
├── Makefile (51)                      Common commands
└── .env.example                       Secrets template
```

---

# END OF MASTER PLAN v2

The code is done. The sophistication is done. Now deploy, run, and write.

**Priority rule:** When in doubt, work on the report. The report is 60% of the mark. The code supports the report — not the other way around.

**Quality rule:** When in doubt, ask: "Does this serve the asset → token → market → risk chain?" If yes, include it. If no, cut it.
