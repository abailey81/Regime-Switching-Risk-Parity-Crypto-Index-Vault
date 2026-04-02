.PHONY: help compile test deploy-mocks deploy-vault ml-setup ml-data ml-backtest ml-weights clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Solidity ───
compile: ## Compile smart contracts
	npx hardhat compile

test: ## Run smart contract tests
	npx hardhat test

test-gas: ## Run tests with gas reporting
	REPORT_GAS=true npx hardhat test

deploy-mocks: ## Deploy mock tokens to Sepolia
	npx hardhat run scripts/deploy_mocks.js --network sepolia

deploy-vault: ## Deploy vault to Sepolia
	npx hardhat run scripts/deploy.js --network sepolia

interact: ## Run vault interaction demo on Sepolia
	npx hardhat run scripts/interact.js --network sepolia

# ─── Python ML ───
ml-setup: ## Set up Python environment
	cd ml && python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

ml-data: ## Fetch and preprocess data
	cd ml && python -m data.fetch_data

ml-backtest: ## Run walk-forward backtest
	cd ml && python -m backtest.walk_forward --config config.yaml

ml-monte-carlo: ## Run Monte Carlo stress test
	cd ml && python -m backtest.monte_carlo

ml-weights: ## Compute latest portfolio weights
	cd ml && python -m weight_publisher.compute_weights --config config.yaml

# ─── Utility ───
clean: ## Clean build artifacts
	npx hardhat clean
	rm -rf cache artifacts
	find ml -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

loc: ## Count lines of code
	@echo "Solidity: $$(find contracts -name '*.sol' -exec cat {} + | wc -l) lines"
	@echo "Python:   $$(find ml -name '*.py' ! -name '__init__.py' -exec cat {} + | wc -l) lines"
	@echo "JS:       $$(find scripts test -name '*.js' -exec cat {} + | wc -l) lines"
