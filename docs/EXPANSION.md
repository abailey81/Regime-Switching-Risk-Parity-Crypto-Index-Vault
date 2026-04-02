# Expansion & Improvement Ideas

## Priority 1: Immediate Improvements (Before Submission)

### Smart Contract
- [ ] Add more comprehensive NatSpec documentation to all functions
- [ ] Add events for epoch duration changes and fee updates
- [ ] Implement a proper redemption queue with ERC-721 position tracking
- [ ] Add a `totalConstituentValue()` function that iterates over all holdings

### ML Pipeline
- [ ] Add EGARCH model variant (captures asymmetric volatility — crashes increase vol more than rallies)
- [ ] Add regime persistence analysis to HMM output (expected duration of each regime)
- [ ] Implement proper walk-forward RL retraining (currently uses single pre-trained model)
- [ ] Add factor attribution: decompose returns into regime alpha, risk-parity alpha, RL alpha

### Backtest
- [ ] Add rolling window analysis (3-month, 6-month, 1-year performance windows)
- [ ] Add regime-stratified metrics (Sharpe during bull vs crisis)
- [ ] Add cost sensitivity analysis (how do results change with 5/10/20/50 bps costs?)
- [ ] Add statistical significance tests (paired t-test vs benchmarks, bootstrap confidence intervals)

## Priority 2: Post-Submission Enhancements

### Advanced Models
- [ ] **Attention-based temporal model**: Replace or augment HMM with a Transformer encoder that processes return sequences — captures complex temporal patterns beyond Markov assumption
- [ ] **Graph Neural Network**: Model cross-asset dependencies as a dynamic graph where edge weights = time-varying correlations from DCC
- [ ] **Distributional RL (IQN-SAC)**: From your dissertation — learn the full return quantile function instead of just expected returns, enabling direct CVaR optimisation within the RL reward
- [ ] **Multi-objective RL**: Use Pareto-optimal policy search to find the frontier of Sharpe vs CVaR vs turnover trade-offs, letting investors choose their preferred point

### Smart Contract Upgrades
- [ ] **ERC-7540 async extension**: For production, replace epoch-based access with ERC-7540 asynchronous deposit/withdrawal requests — the standard designed for RWA vaults
- [ ] **Chainlink Functions**: Replace the keeper pattern with Chainlink Functions for decentralised off-chain computation — GARCH/HMM could run in a Chainlink DON
- [ ] **Cross-chain deployment**: Deploy on multiple L2s (Arbitrum, Base, Optimism) with LayerZero/Wormhole bridge for vault shares
- [ ] **On-chain weight verification**: Store acceptable weight bounds on-chain; reject keeper updates that violate them even without Merkle proof verification
- [ ] **Streaming fees via Superfluid**: Replace epoch-based management fee with continuous Superfluid streaming for real-time fee accrual

### Data & Infrastructure
- [ ] **Real-time data pipeline**: Replace batch processing with Kafka/Redis streaming for live weight computation
- [ ] **Alternative data**: Add on-chain metrics (DEX volume, TVL changes, whale movements) as HMM features
- [ ] **Sentiment features**: NLP analysis of crypto Twitter/Reddit for regime detection augmentation
- [ ] **Multi-exchange execution**: Implement smart order routing across Uniswap, Curve, Balancer for rebalancing

### Dashboard
- [ ] Build React frontend reading from Sepolia contract
- [ ] Real-time NAV chart from on-chain events
- [ ] Regime indicator with Bayesian probability pie chart
- [ ] Weight allocation bar chart
- [ ] Historical fee tracking
- [ ] Deploy to Vercel or GitHub Pages

## Priority 3: Research Extensions (Dissertation Connection)

### LLM-Driven Reward Engineering (Your Dissertation Topic)
The SAC agent in this vault uses a hand-designed reward function. Your dissertation proposes using LLMs to automatically design reward functions. You could:
- [ ] Implement the Eureka-style loop: LLM generates reward code → train SAC → evaluate → feedback to LLM → iterate
- [ ] Compare hand-designed vs LLM-designed rewards on this exact vault environment
- [ ] Use IQN-SAC (distributional RL) instead of standard SAC — feed quantile function stats back to the LLM
- [ ] This creates a direct bridge: coursework vault = evaluation environment for dissertation research

### Academic Publication Potential
- [ ] The regime-switching ensemble for on-chain portfolio management is publishable
- [ ] Target: ACM DeFi workshop, IEEE Blockchain, or Financial Innovation journal
- [ ] Contribution: First on-chain fund with ML-driven dynamic allocation + formal risk framework
