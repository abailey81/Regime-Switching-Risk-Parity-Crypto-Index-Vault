# Development Workflow

## Initial Setup

```bash
# 1. Clone the repo
git clone https://github.com/abailey81/riskparity-vault.git
cd riskparity-vault

# 2. Install Node dependencies (for Solidity)
npm install

# 3. Set up Python environment
cd ml/
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..

# 4. Copy environment file and fill in your keys
cp .env.example .env
# Edit .env with your Alchemy RPC URL, private key, Etherscan key
```

## Daily Development Workflow

### Smart Contracts (local, no GPU needed)
```bash
npm run compile                   # Compile Solidity
npm test                          # Run all tests
npm run test:gas                  # Run with gas reporting
```

### ML Pipeline (local CPU for everything except SAC training)
```bash
cd ml/

# Fetch data (first time only, ~10 min)
python -m data.fetch_data

# Run full backtest (CPU, ~1.5 hours)
python -m backtest.walk_forward --config config.yaml

# Compute latest weights (CPU, ~5 min)
python -m weight_publisher.compute_weights --config config.yaml
```

### SAC RL Training (Colab GPU via VS Code extension)
1. Open `ml/notebooks/train_sac.ipynb` in VS Code
2. Click "Select Kernel" → "Colab" → T4 GPU
3. Run all cells (~3-5 hours)
4. Model saves to `ml/models/saved/sac_best.zip`
5. Switch kernel back to local Python for everything else

### Deployment (requires Sepolia ETH)
```bash
npm run deploy:mocks              # Deploy test tokens + price feeds
# Update addresses in scripts/deploy.js
npm run deploy:vault              # Deploy vault
npm run interact                  # Run lifecycle demo
```

## Testing Strategy

### Solidity Tests
- Unit tests for each feature (fees, epochs, circuit breaker, Merkle)
- Integration test: full deposit → rebalance → withdraw lifecycle
- Edge cases: first depositor, zero amounts, gate overflow

### Python Tests
- Each model has internal validation (GARCH convergence, HMM log-likelihood)
- Walk-forward backtest IS the test — if it runs end-to-end, the pipeline works
- Monte Carlo validates distributional properties

## Git Workflow
```bash
git add -A
git commit -m "descriptive message"
git push origin main
```

Keep commits atomic: one feature per commit. Good messages:
- "Add GARCH-DCC model with Student-t innovations"
- "Implement epoch-based access control in vault"
- "Fix CVaR constraint in portfolio optimizer"

## Troubleshooting

**Hardhat compile fails**: Check Solidity version in hardhat.config.js matches contracts
**GARCH doesn't converge**: Try reducing rolling_window in config.yaml, or use EWMA fallback
**HMM produces degenerate states**: Increase transmat_concentration for stickier regimes
**SAC training diverges**: Reduce learning_rate, increase batch_size
**Web3 connection fails**: Check SEPOLIA_RPC_URL in .env, ensure you have Sepolia ETH
