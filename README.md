<div align="center">

<img src="https://img.shields.io/badge/%E2%9A%A1-Regime--Switching%20Risk--Parity%20Crypto%20Index%20Vault-000000?style=for-the-badge&labelColor=000000" alt="Project Title" />

<br />

### Regime-Switching Risk-Parity Crypto Index Vault
**ML-Driven ERC-4626 Tokenised Fund with GARCH-DCC, Bayesian HMM & Deep RL Ensemble**

<br />

[![Solidity](https://img.shields.io/badge/Solidity-0.8.24-363636?style=for-the-badge&logo=solidity&logoColor=white)](https://soliditylang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![OpenZeppelin](https://img.shields.io/badge/OpenZeppelin-v5-4E5EE4?style=for-the-badge&logo=openzeppelin&logoColor=white)](https://docs.openzeppelin.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-SAC_RL-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br />

[![ERC-4626](https://img.shields.io/badge/Standard-ERC--4626-blue?style=flat-square)](https://eips.ethereum.org/EIPS/eip-4626)
[![Hardhat](https://img.shields.io/badge/Framework-Hardhat-FFF100?style=flat-square&logo=hardhat&logoColor=black)](https://hardhat.org/)
[![Network](https://img.shields.io/badge/Network-Sepolia_Testnet-7B3FE4?style=flat-square)](https://sepolia.etherscan.io/)
[![stable-baselines3](https://img.shields.io/badge/RL-SAC_(SB3)-FF6F00?style=flat-square)](https://stable-baselines3.readthedocs.io/)
[![CVXPY](https://img.shields.io/badge/Optimisation-CVXPY-E74C3C?style=flat-square)](https://www.cvxpy.org/)
[![GARCH](https://img.shields.io/badge/Volatility-GARCH--DCC-2ECC71?style=flat-square)](https://arch.readthedocs.io/)
[![HMM](https://img.shields.io/badge/Regimes-Bayesian_HMM-9B59B6?style=flat-square)](https://hmmlearn.readthedocs.io/)

<br />

[Architecture](#-architecture) · [ML Engine](#-ml-ensemble-engine) · [Smart Contract](#-smart-contract) · [Quick Start](#-quick-start) · [Backtesting](#-backtesting) · [Risk Framework](#-risk-framework)

---

<br />

> **IFTE0007 — Decentralised Finance & Blockchain**
> UCL Institute of Finance & Technology · MSc Digital Finance & Banking · 2025/26
>
> **Tamer Atesyakar** · [GitHub](https://github.com/abailey81) · [LinkedIn](https://linkedin.com/in/tamerates)

<br />

</div>

---

## Contract Deployment

| | |
|---|---|
| **Network** | Ethereum Sepolia Testnet |
| **Vault Address** | `0x_TO_BE_UPDATED_AFTER_DEPLOYMENT` |
| **Share Token** | rpCRYPTO (ERC-20 via ERC-4626) |
| **Underlying** | USDC |
| **Total Code** | 12,381 lines across 18 Python + 4 Solidity + 4 JS files |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          OFF-CHAIN  (Python ML Pipeline)                     │
│                                                                              │
│   ┌──────────────┐    ┌────────────────┐    ┌────────────────┐              │
│   │  GARCH-DCC   │    │  Bayesian HMM  │    │  SAC Deep RL   │              │
│   │  Student-t   │    │  3-State Soft   │    │ (Colab T4 GPU) │              │
│   │  Σ_t, w_rp   │    │  Regime Probs   │    │  Policy π(s)   │              │
│   └──────┬───────┘    └───────┬────────┘    └───────┬────────┘              │
│          │                    │                      │                        │
│          └────────────────────┼──────────────────────┘                       │
│                               ▼                                              │
│                ┌────────────────────────────────┐                            │
│                │     ENSEMBLE META-MODEL        │                            │
│                │                                │                            │
│                │   • Inverse-variance model     │    ┌─────────────────────┐ │
│                │     weighting                  │    │  Multi-Method       │ │
│                │   • Kelly confidence           │    │  Optimiser          │ │
│                │     scaling (¼ Kelly)          │────│  HRP · Risk Parity  │ │
│                │   • Regime-blended risk        │    │  Black-Litterman    │ │
│                │     budgets                    │    │  Max Diversification│ │
│                │   • Circuit breaker (15% DD)   │    │  CVaR Constraint    │ │
│                │                                │    └─────────────────────┘ │
│                └───────────────┬────────────────┘                            │
│                                ▼                                             │
│                ┌────────────────────────────────┐                            │
│                │  WEIGHT PUBLISHER              │                            │
│                │  w* → BPS → Merkle Root → TX   │                            │
│                └───────────────┬────────────────┘                            │
└────────────────────────────────┼─────────────────────────────────────────────┘
                                 │  commitWeights(root)
                                 │  [1-hour timelock]
                                 │  executeWeights(w[], proof[])
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ON-CHAIN  (Solidity · Sepolia)                        │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐     │
│   │                   RiskParityVault.sol  (ERC-4626)                  │     │
│   │                                                                    │     │
│   │   DEPOSIT ────► Mint rpCRYPTO shares at oracle NAV                │     │
│   │   WITHDRAW ──► Burn shares, receive USDC (20% epoch gate)         │     │
│   │                                                                    │     │
│   │   ┌───────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │     │
│   │   │  3-Tier   │  │   Epoch      │  │  Merkle    │  │ Circuit  │  │     │
│   │   │   Fees    │  │   System     │  │  Commit-   │  │ Breaker  │  │     │
│   │   │           │  │              │  │  Reveal    │  │          │  │     │
│   │   │ Mgmt  1%  │  │ 1d  calm    │  │  1hr lock  │  │ 15% DD   │  │     │
│   │   │ Perf 10%  │  │ 7d  volatile│  │  keccak256 │  │ → safe   │  │     │
│   │   │ Exit 0.3% │  │ 20% gate    │  │  verify    │  │   mode   │  │     │
│   │   └───────────┘  └──────────────┘  └────────────┘  └──────────┘  │     │
│   └────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Asset → Token → Market → Risk Logic Chain

| Layer | Design | Rationale |
|:---|:---|:---|
| **Asset** | 8-asset crypto portfolio across 4 classes (spot, staking, treasuries, stablecoin) | Four distinct value sources: capital appreciation, staking yield (~3–4%), Treasury coupon (~4.5–5%), defensive stability. Imperfectly correlated → portfolio risk < individual asset risk |
| **Token** | ERC-4626 vault share (rpCRYPTO) — fungible, variable supply, pro-rata NAV claim | Fungible because each share represents identical fractional NAV. ERC-4626 standard ensures composability with Aave, Yearn, and other DeFi protocols |
| **Market** | Three-layer liquidity: (1) primary mint/redeem at NAV, (2) secondary AMM (Uniswap V3), (3) keeper arbitrage | Primary for institutional entry/exit. Secondary for retail trading. Arbitrage keeps secondary price within ~1% of NAV |
| **Risk** | Model risk (ensemble hedging), oracle risk (Chainlink + staleness), contract risk (reentrancy guard, pause), market risk (circuit breaker, redemption gate) | Each risk layer has a specific mitigation mechanism embedded in the smart contract or ML pipeline |

---

## Portfolio Constituents

| Asset Class | Asset | Ticker | Value Source | Rebalance Venue | Fee (bps) |
|:---|:---|:---|:---|:---|---:|
| **Spot Crypto** | Bitcoin | BTC | Capital appreciation | Binance Spot | 7.5 |
| | Ethereum | ETH | Capital appreciation | Binance Spot | 7.5 |
| | Solana | SOL | Capital appreciation | Binance Spot | 7.5 |
| **Liquid Staking** | Lido Staked ETH | stETH | Staking yield ~3.5% | Curve (L1) | 4.0 + gas |
| | Rocket Pool ETH | rETH | Staking yield ~3.2% | Uniswap V3 (L1) | 5.0 + gas |
| **Tokenised Treasuries** | BlackRock USD Inst | BUIDL | Treasury coupon ~4.5% | Institutional NAV | 0.0 |
| | Ondo US Dollar Yield | USDY | Treasury coupon ~5.0% | Institutional NAV | 0.0 |
| **Stablecoin** | USD Coin | USDC | Defensive allocation | Binance (0-fee) | 0.0 |

> Transaction costs verified against Binance fee schedule and Etherscan gas tracker, Q1 2026. Ethereum L1 gas: ~0.17 gwei ($0.05–$2 per swap post-Dencun). All costs include Almgren-Chriss impact slippage model.

---

## ML Ensemble Engine

<table>
<tr><td>

### Model 1 — Student-t GARCH(1,1)-DCC

`ml/models/garch_dcc.py`

Estimates the **time-varying covariance matrix** $\Sigma_t$ across all 8 assets. Student-t innovations capture crypto's fat tails (kurtosis ~8–15, vs Gaussian ~3). DCC layer models how cross-asset correlations shift dynamically — critical because crypto correlations spike toward 1.0 during crises.

$$\sigma^2_{i,t} = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

$$Q_t = (1-a-b)\bar{Q} + a \cdot z_{t-1}z'_{t-1} + b \cdot Q_{t-1}$$

**Output**: 8x8 covariance matrix $\rightarrow$ risk-parity weights where each asset contributes equally to portfolio risk.

</td></tr>
<tr><td>

### Model 2 — Bayesian HMM (3-State)

`ml/models/bayesian_hmm.py`

Classifies the market into **bull / normal / crisis** using hidden Markov chains with sticky transition priors ($P(\text{stay}) = 0.90$). Outputs **soft posterior probabilities** (e.g., 60% crisis, 25% normal, 15% bull) instead of hard labels, enabling continuous risk budget blending without whipsaw.

**Output**: $[P(\text{bull}),\ P(\text{normal}),\ P(\text{crisis})]$ $\rightarrow$ regime-blended risk budgets across 4 asset classes.

</td></tr>
<tr><td>

### Model 3 — SAC Deep RL Agent

`ml/models/sac_agent.py` + `ml/environment/portfolio_env.py`

Soft Actor-Critic agent trained on a custom Gymnasium environment (38-dim state) with composite reward function:

$$R = \lambda_1 \cdot r_t - \lambda_2 \cdot \max(0,\ \text{CVaR}_{5\%} - 2\%) - \lambda_3 \cdot \max(0,\ \text{DD} - 10\%) - \lambda_4 \cdot \|\Delta w\|_1$$

**Training**: Requires GPU. VS Code Colab extension $\rightarrow$ T4 GPU $\rightarrow$ ~3–5 hours for 500K timesteps $\times$ 5 seeds. See [Colab Workflow](#colab-gpu-workflow).

**Output**: Target portfolio weights from learned policy $\pi(s)$.

</td></tr>
<tr><td>

### Model 4 — Ensemble Meta-Model

`ml/models/ensemble.py`

Combines all three model outputs through a multi-stage pipeline:

| Stage | Mechanism |
|:---|:---|
| **1. Inverse-variance weighting** | More certain models get more influence: $w_i \propto 1/\sigma_i^2$ |
| **2. Kelly confidence scaling** | Quarter-Kelly blend toward $1/N$ when uncertain |
| **3. Black-Litterman with HMM views** | Regime posteriors $\rightarrow$ BL expected return views |
| **4. CVaR-constrained optimisation** | Enforces vol target, min/max weights, turnover limits |
| **5. Circuit breaker** | 15% drawdown from HWM $\rightarrow$ immediate defensive allocation |

</td></tr>
</table>

---

## Multi-Method Optimiser

The ensemble selects from **8 optimisation methods** per regime:

| Method | Best For | Source |
|:---|:---|:---|
| **HRP** (Hierarchical Risk Parity) | Crisis — robust to estimation error | Lopez de Prado (2016) |
| **Risk Parity** (Equal Risk Contribution) | Normal — balanced risk allocation | Maillard et al. (2010) |
| **Black-Litterman** (HMM views) | Bull — view-driven alpha capture | Black & Litterman (1992) |
| **Mean-Variance** (Max Sharpe) | Low-vol — when estimates are reliable | Markowitz (1952) |
| **Inverse Volatility** | Baseline — simple, robust | — |
| **Max Diversification** | High-corr — maximises diversification ratio | Choueifaty (2008) |
| **Minimum Variance** | Risk-averse — global minimum portfolio variance | Markowitz (1952) |
| **CVaR-Constrained** | Always — enforces hard risk limits | Rockafellar & Uryasev (2000) |

---

## Smart Contract

**`contracts/RiskParityVault.sol`** — 1,294 lines of Solidity 0.8.24 (NatSpec documented)

<table>
<tr>
<th width="200">Feature</th>
<th>Implementation</th>
</tr>
<tr>
<td><strong>ERC-4626 Vault</strong></td>
<td>OpenZeppelin v5 <code>ERC4626</code> with virtual share offset (inflation-attack resistant)</td>
</tr>
<tr>
<td><strong>Management Fee</strong></td>
<td>1% annual via continuous share dilution — mints fee shares to <code>feeRecipient</code> every second</td>
</tr>
<tr>
<td><strong>Performance Fee</strong></td>
<td>10% of gains above high-water mark, crystallised at epoch boundaries</td>
</tr>
<tr>
<td><strong>Redemption Fee</strong></td>
<td>0.3% for withdrawals within 7 days of deposit (anti-churn)</td>
</tr>
<tr>
<td><strong>Epoch System</strong></td>
<td>Dynamic: 1-day epochs in calm markets, 7-day in volatile. Configurable by keeper</td>
</tr>
<tr>
<td><strong>Redemption Gate</strong></td>
<td>Maximum 20% of total AUM withdrawable per epoch (bank-run protection)</td>
</tr>
<tr>
<td><strong>Merkle Commit-Reveal</strong></td>
<td>Keeper commits <code>keccak256(abi.encode(tokens[], weights[]))</code>, then reveals after 1hr timelock</td>
</tr>
<tr>
<td><strong>Circuit Breaker</strong></td>
<td>Automatic when NAV drops 15% from all-time high. Shifts to defensive weights</td>
</tr>
<tr>
<td><strong>Access Control</strong></td>
<td>OpenZeppelin <code>AccessControl</code>: ADMIN_ROLE (fees, pause) + KEEPER_ROLE (weights, epochs)</td>
</tr>
<tr>
<td><strong>Safety</strong></td>
<td><code>ReentrancyGuard</code>, <code>Pausable</code>, Chainlink oracle with 1hr staleness check</td>
</tr>
</table>

### Fee Architecture

```
Management Fee (1% annual)
  ► Accrued continuously via share dilution
  ► fee_shares = totalSupply × 0.01 × Δt / SECONDS_PER_YEAR
  ► Minted to feeRecipient → dilutes all holders equally

Performance Fee (10% above HWM)
  ► Crystallised at epoch boundaries
  ► gain = (NAV - HWM) × totalSupply
  ► HWM never decreases — fees only charged on NEW highs

Redemption Fee (0.3% within 7 days)
  ► Anti-churn mechanism
  ► After 7 days: no fee
```

### Circuit Breaker Logic

```
                  NAV
                   │
    HWM ──────────┤
                   │
    85% of HWM ───┤──── TRIGGER: circuitBreakerActive = true
                   │              All weights → defensive allocation
                   │
    90% of HWM ───┤──── RESET ELIGIBLE: admin can call resetCircuitBreaker()
                   │
                   ▼
```

---

## Financial Design Highlights

| Design Choice | Financial Rationale |
|:---|:---|
| **3-Tier Fee Structure** | Management fee (1%) compensates ongoing operation costs. Performance fee (10% above HWM) aligns manager incentives — fees only charged on NEW gains. Redemption fee (0.3% within 7 days) discourages short-term arbitrage that extracts value from long-term holders |
| **Merkle Commit-Reveal** | Prevents keeper from front-running rebalance trades. Without this, keeper could buy assets about to be weighted up, then execute weights for risk-free profit. 1-hour timelock makes this uneconomical |
| **Circuit Breaker (15% DD)** | Calibrated from 12 historical crypto crises (2020–2024) where average BTC drawdown was -33%. Triggers at 15% to catch drawdowns early, shifting 90% to USDC/Treasuries before the worst losses materialise |
| **20% Redemption Gate** | Prevents liquidity spirals: if 100% of investors try to exit simultaneously, the gate ensures orderly processing over 5 days. Modelled on traditional fund gates (e.g., Woodford Capital, 2019) |
| **Inverse-Variance Ensemble** | When one model (e.g., GARCH) has high parameter uncertainty, its weight automatically decreases. This is formally equivalent to a Bayesian model averaging approach |

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|:---|:---|:---|
| Node.js | 18+ | Hardhat, Solidity compilation |
| Python | 3.10+ | ML pipeline |
| VS Code | Latest | Development + Colab extension |

### 1. Smart Contracts

```bash
git clone https://github.com/abailey81/Regime-Switching-Risk-Parity-Crypto-Index-Vault.git
cd Regime-Switching-Risk-Parity-Crypto-Index-Vault
npm install
cp .env.example .env          # Fill in RPC URL + private key

npx hardhat compile            # Compile contracts
npx hardhat test               # Run 30+ tests

npx hardhat run scripts/deploy_mocks.js --network sepolia   # Deploy test tokens
npx hardhat run scripts/deploy.js --network sepolia          # Deploy vault
npx hardhat run scripts/interact.js --network sepolia        # Full lifecycle demo
```

### 2. ML Pipeline (CPU)

```bash
cd ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m data.fetch_data                          # Fetch OHLCV data (~10 min)
python -m weight_publisher.compute_weights         # Compute latest weights (~5 min)
python -m backtest.walk_forward                    # Walk-forward backtest (~1.5 hrs)
```

### 3. SAC RL Training (GPU — Colab)

See [Colab GPU Workflow](#colab-gpu-workflow) below.

---

## Colab GPU Workflow

The SAC deep RL agent is the only component that requires GPU training. Everything else runs on CPU.

**Setup** (one-time):
1. Install the **Google Colab** extension in VS Code
2. Open `ml/notebooks/train_sac.ipynb`
3. Click **Select Kernel** → **Colab** → choose **T4 GPU** runtime

**Training** (~3–5 hours):
- The notebook trains 5 seeds × 500K timesteps each
- Best seed is selected by total reward
- Model saves locally to `ml/models/saved/sac_best.zip`

**After training**: Switch kernel back to local Python. The backtest and weight publisher automatically load the saved model.

```
VS Code
│
├── Local Python kernel (CPU)
│   ├── Data fetching, preprocessing
│   ├── GARCH-DCC fitting
│   ├── HMM fitting
│   ├── Ensemble weight computation
│   ├── Walk-forward backtest
│   └── Sepolia deployment
│
└── Colab kernel (T4 GPU) — ONLY for:
    └── ml/notebooks/train_sac.ipynb
        ├── SAC training (3–5 hrs)
        └── Saves to ml/models/saved/
```

---

## Backtesting

### Walk-Forward Protocol

| Parameter | Value |
|:---|:---|
| Train window | 180 days (4,320 hours) |
| Test window | 30 days (720 hours) |
| Step size | 30 days (rolling) |
| Transaction costs | Venue-specific (verified Q1 2026) |
| Benchmarks | Equal Weight, BTC Only, 60/40 BTC/USDC, Market-Cap, Risk Parity Static, Min Variance |

### Generated Outputs (9 Charts + Metrics)

| Chart | Description |
|:---|:---|
| `equity_curves.png` | Log-scale performance: ensemble vs 6 benchmarks |
| `drawdown_plot.png` | Drawdown comparison over time |
| `regime_timeline.png` | HMM regime classification (colour-coded bull/normal/crisis) |
| `weight_evolution.png` | Stacked area chart of portfolio weights over time |
| `rolling_sharpe.png` | 90-day rolling Sharpe ratio comparison |
| `monthly_returns_heatmap.png` | Year × Month grid of monthly returns |
| `risk_dashboard.png` | 4-panel: rolling vol, CVaR, drawdown, VaR histogram |
| `monte_carlo_fan_6m.png` | 10K-path fan chart (6-month horizon) |
| `monte_carlo_fan_12m.png` | 10K-path fan chart (12-month horizon) |

### Crisis Stress Testing

The backtest evaluates performance across **12 major crypto crises** (2020–2024):

| Crisis | Period | Type | BTC Drawdown |
|:---|:---|:---|---:|
| COVID Crash | Mar–Apr 2020 | Macro | -52% |
| May 2021 Crash | May–Jun 2021 | Regulatory | -54% |
| UST/Luna Collapse | May–Jun 2022 | Protocol failure | -40% |
| 3AC Liquidation | Jun–Jul 2022 | Contagion | -25% |
| FTX Collapse | Nov–Dec 2022 | Exchange fraud | -27% |
| SVB/USDC Depeg | Mar–Apr 2023 | Banking crisis | -10% |
| Yen Carry Unwind | Aug 2024 | Macro flash crash | -18% |

> Metrics are reported separately for crisis and non-crisis periods with stratified analysis.

---

## Risk Framework

<table>
<tr>
<th width="120">Category</th>
<th width="250">Risk</th>
<th>Mitigation</th>
</tr>
<tr>
<td rowspan="4"><strong>Model</strong></td>
<td>GARCH estimation error under fat tails</td>
<td>Student-t distribution; rolling refit each fold</td>
</tr>
<tr>
<td>HMM regime detection lag (2–5 obs)</td>
<td>Soft posterior probabilities; continuous blending</td>
</tr>
<tr>
<td>RL policy distributional shift</td>
<td>Ensemble averaging; Kelly confidence scaling</td>
</tr>
<tr>
<td>Covariance estimation noise</td>
<td>HRP avoids matrix inversion; robust to noise</td>
</tr>
<tr>
<td rowspan="2"><strong>Oracle</strong></td>
<td>Price manipulation via flash loans</td>
<td>TWAP oracle; Chainlink staleness check</td>
</tr>
<tr>
<td>Stale price data</td>
<td>1-hour freshness requirement on all feeds</td>
</tr>
<tr>
<td rowspan="2"><strong>Contract</strong></td>
<td>ERC-4626 inflation attack</td>
<td>OpenZeppelin virtual share offset (1e8)</td>
</tr>
<tr>
<td>Front-running keeper rebalance</td>
<td>Merkle commit-reveal with 1-hour timelock</td>
</tr>
<tr>
<td rowspan="3"><strong>Market</strong></td>
<td>Liquidity spiral from mass redemptions</td>
<td>20% epoch gate; queued withdrawals</td>
</tr>
<tr>
<td>Correlation breakdown (all assets → 1.0)</td>
<td>EWMA correlation monitoring; regime rotation to Treasuries</td>
</tr>
<tr>
<td>MEV sandwich attacks on DEX rebalance</td>
<td>Turnover limits; Flashbots Protect</td>
</tr>
<tr>
<td rowspan="2"><strong>Regulatory</strong></td>
<td>FCA s.235 FSMA classification</td>
<td>Digital Securities Sandbox pathway; KYC whitelist</td>
</tr>
<tr>
<td>MiCA crypto-asset classification</td>
<td>EU compliance via ARTs/EMTs framework</td>
</tr>
</table>

---

## Repository Structure

```
Regime-Switching-Risk-Parity-Crypto-Index-Vault/
│
├── contracts/                          Solidity (1,641 lines)
│   ├── RiskParityVault.sol               Core ERC-4626 vault (1,294 lines)
│   ├── mocks/
│   │   ├── MockERC20.sol                 Mintable test tokens with faucet
│   │   └── MockPriceFeed.sol             Simulated Chainlink (TWAP + history)
│   └── interfaces/
│       └── IChainlinkAggregator.sol      Oracle interface
│
├── scripts/                            Deployment & interaction (387 lines)
│   ├── deploy_mocks.js                   Deploy 8 test tokens + 8 price feeds
│   ├── deploy.js                         Deploy vault, register constituents
│   └── interact.js                       Full lifecycle: deposit → rebalance → withdraw
│
├── test/
│   └── RiskParityVault.test.js           30+ tests, 809 lines (fees, epochs, circuit breaker, Merkle)
│
├── ml/                                 Python ML pipeline (9,532 lines)
│   ├── config.yaml                       ALL hyperparameters (single source of truth)
│   ├── data/
│   │   ├── fetch_data.py                 Binance OHLCV + synthetic Treasury/stablecoin
│   │   └── preprocess.py                 Feature engineering (returns, vol, momentum, Sharpe)
│   ├── models/
│   │   ├── garch_dcc.py                  Student-t GARCH(1,1)-DCC covariance
│   │   ├── bayesian_hmm.py               3-state HMM with soft posteriors
│   │   ├── sac_agent.py                  SAC RL agent (train on Colab, infer on CPU)
│   │   ├── ensemble.py                   Meta-model: Kelly + multi-method + CVaR
│   │   ├── portfolio_optimizer.py        7 methods: HRP, RP, BL, MVO, InvVol, MaxDiv, CVaR
│   │   ├── risk_analyzer.py              VaR/CVaR (3 methods), drawdown, BTC correlation
│   │   └── correlation_analyzer.py       EWMA correlation, regime detection, breakdown alerts
│   ├── environment/
│   │   └── portfolio_env.py              Custom Gymnasium env (38-dim state, composite reward)
│   ├── backtest/
│   │   ├── walk_forward.py               Walk-forward engine + 9 auto-generated charts
│   │   ├── monte_carlo.py                10K-path regime-conditioned block bootstrap
│   │   ├── benchmarks.py                 4 comparison strategies
│   │   ├── metrics.py                    18+ metrics (Sharpe, Omega, Ulcer, CVaR, Burke, ...)
│   │   ├── transaction_costs.py          Venue-specific costs (verified Q1 2026)
│   │   └── crisis_events.py              12 crypto crises with stratified analysis
│   ├── weight_publisher/
│   │   ├── compute_weights.py            Full pipeline: data → models → Merkle root
│   │   ├── merkle.py                     Keccak-256 commitment tree
│   │   └── publish.py                    Web3.py keeper: commit → timelock → execute
│   └── notebooks/
│       └── train_sac.ipynb               Colab GPU training notebook
│
├── docs/
│   ├── ARCHITECTURE.md                   System design, data flow, fee mechanics
│   ├── DEVELOPMENT.md                    Setup guide, daily workflow, troubleshooting
│   └── EXPANSION.md                      Improvement ideas + dissertation connection
│
├── Makefile                            Common commands (make compile, make ml-backtest, ...)
├── hardhat.config.js                   Solidity compiler + Sepolia network config
├── package.json                        Node.js dependencies
└── .env.example                        Template for secrets (RPC URL, private key)
```

---

## Related Work

This vault adapts portfolio construction and risk management components from the author's **[Crypto-Statistical-Arbitrage](https://github.com/abailey81/Crypto-Statistical-Arbitrage)** system — a multi-venue quantitative trading platform achieving Sharpe 1.61 (altcoin) / 5.81 (BTC futures) on walk-forward out-of-sample tests.

**Components adapted from prior work:**
- HRP, Risk Parity, Black-Litterman optimisers
- VaR/CVaR risk decomposition framework
- Almgren-Chriss transaction cost model
- Crisis event definitions

**All other components are original to this project:** ERC-4626 vault design, ensemble orchestration, HMM regime detection, SAC RL environment, Merkle commit-reveal, circuit breaker.

---

## References

| Paper | Relevance |
|:---|:---|
| Engle, R. (2002). Dynamic Conditional Correlation. *JBES* 20(3). | GARCH-DCC model |
| Hansen, B. (1994). Autoregressive Conditional Density. *IER* 35(3). | Student-t innovations |
| Lopez de Prado, M. (2016). Building Diversified Portfolios. *JPM* 42(4). | HRP optimisation |
| Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *FAJ* 48(5). | BL with regime views |
| Rockafellar & Uryasev (2000). Optimization of CVaR. *J. Risk* 2(3). | CVaR constraint |
| Haarnoja et al. (2018). Soft Actor-Critic. *ICML*. | SAC RL agent |
| EIP-4626 (2022). Tokenized Vault Standard. | Vault architecture |
| FCA (2025). CP25/28: Progressing Fund Tokenisation. | Regulatory framework |

---

## License

MIT — see [LICENSE](LICENSE)

---

*This project was developed as the individual coursework (60%) for IFTE0007 — Decentralised Finance & Blockchain, UCL Institute of Finance & Technology, MSc Digital Finance & Banking, 2025/26.*

<div align="center">

<br />

Built by [**Tamer Atesyakar**](https://github.com/abailey81)

<br />

</div>
