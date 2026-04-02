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
2. Click "Select Kernel" -> "Colab" -> T4 GPU
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

## VS Code + Google Colab Extension Workflow

The SAC reinforcement learning agent requires GPU training (~500,000 timesteps across 5 seeds). This project uses the **VS Code Colab Extension** to run GPU workloads on Google Colab while maintaining a local development experience.

### Setup

1. Install the [Google Colab extension](https://marketplace.visualstudio.com/items?itemName=GoogleColab.colab-kernel) in VS Code.
2. Sign in with your Google account (Colab Pro recommended for longer training runs and priority GPU access).
3. Ensure the repository is pushed to GitHub, as the Colab runtime clones from the remote.

### Workflow

1. **Open the training notebook** in VS Code: `ml/notebooks/train_sac.ipynb`
2. **Select Kernel**: Click the kernel selector in the top right -> "Colab" -> select a T4 GPU runtime.
3. **The Colab runtime automatically clones the repo** from GitHub into its environment. Any changes you want reflected in Colab must be pushed first.
4. **Run all cells**: The notebook handles environment setup (pip installs), data loading, training across 5 random seeds, and model evaluation.
5. **Training artefacts** (the best model checkpoint `sac_best.zip`, training curves, evaluation metrics) are saved within the Colab runtime.
6. **Download results back to local**: The final notebook cells download `sac_best.zip` to `ml/models/saved/` on your local machine.
7. **Switch kernel back to local Python** for all subsequent work (backtesting, weight computation, Monte Carlo). Only SAC training requires the GPU.

### What Runs Where

| Task                          | Environment       | Approx. Time  |
|-------------------------------|-------------------|----------------|
| Data fetching (Binance API)   | Local CPU         | ~10 min        |
| GARCH-DCC fitting             | Local CPU         | ~5 min         |
| HMM fitting                   | Local CPU         | ~2 min         |
| SAC RL training (5 seeds)     | Colab T4 GPU      | ~3-5 hours     |
| Walk-forward backtest         | Local CPU         | ~1.5 hours     |
| Monte Carlo stress test       | Local CPU         | ~20 min        |
| Weight computation + publish  | Local CPU         | ~5 min         |
| Solidity compile + test       | Local (Hardhat)   | ~30 sec        |
| Sepolia deployment            | Local (Hardhat)   | ~2 min         |

### Troubleshooting Colab

- **Colab disconnects mid-training**: Colab Pro reduces this. The notebook includes checkpointing every 50,000 steps, so training can resume from the latest checkpoint.
- **Package version mismatch**: The first notebook cell pins `stable-baselines3`, `torch`, and `gymnasium` versions to match the local `requirements.txt`.
- **"No GPU available"**: Free Colab has limited GPU availability. Try again later, or switch to Colab Pro for guaranteed T4 access.

## Testing Strategy

### Solidity Tests
- Unit tests for each feature (fees, epochs, circuit breaker, Merkle)
- Integration test: full deposit -> rebalance -> withdraw lifecycle
- Edge cases: first depositor, zero amounts, gate overflow

### Python Tests
- Each model has internal validation (GARCH convergence, HMM log-likelihood)
- Walk-forward backtest IS the test -- if it runs end-to-end, the pipeline works
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
