/**
 * @title Deploy RiskParity Vault
 * @notice Deploys the main RiskParityVault contract on Sepolia.
 *         Run deploy_mocks.js FIRST to get token and feed addresses.
 *
 * Usage: npx hardhat run scripts/deploy.js --network sepolia
 *
 * IMPORTANT: Update the addresses below with your deployed mock addresses.
 */
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying RiskParityVault with:", deployer.address);
  console.log("Balance:", hre.ethers.formatEther(await hre.ethers.provider.getBalance(deployer.address)), "ETH\n");

  // ═══════════════════════════════════════════════════
  //      UPDATE THESE WITH YOUR DEPLOYED ADDRESSES
  // ═══════════════════════════════════════════════════

  // Mock token addresses (from deploy_mocks.js output)
  const MOCK_USDC  = "0x_REPLACE_WITH_MUSDC_ADDRESS";
  const MOCK_WBTC  = "0x_REPLACE_WITH_MWBTC_ADDRESS";
  const MOCK_WETH  = "0x_REPLACE_WITH_MWETH_ADDRESS";
  const MOCK_SOL   = "0x_REPLACE_WITH_MSOL_ADDRESS";
  const MOCK_STETH = "0x_REPLACE_WITH_MSTETH_ADDRESS";
  const MOCK_RETH  = "0x_REPLACE_WITH_MRETH_ADDRESS";
  const MOCK_BUIDL = "0x_REPLACE_WITH_MBUIDL_ADDRESS";
  const MOCK_USDY  = "0x_REPLACE_WITH_MUSDY_ADDRESS";

  // Mock price feed addresses (from deploy_mocks.js output)
  const FEED_BTC   = "0x_REPLACE_WITH_BTC_FEED";
  const FEED_ETH   = "0x_REPLACE_WITH_ETH_FEED";
  const FEED_SOL   = "0x_REPLACE_WITH_SOL_FEED";
  const FEED_STETH = "0x_REPLACE_WITH_STETH_FEED";
  const FEED_RETH  = "0x_REPLACE_WITH_RETH_FEED";
  const FEED_BUIDL = "0x_REPLACE_WITH_BUIDL_FEED";
  const FEED_USDY  = "0x_REPLACE_WITH_USDY_FEED";
  const FEED_USDC  = "0x_REPLACE_WITH_USDC_FEED";

  // ═══════════════════════════════════════════════════
  //                  DEPLOY VAULT
  // ═══════════════════════════════════════════════════

  console.log("Deploying RiskParityVault...");
  const RiskParityVault = await hre.ethers.getContractFactory("RiskParityVault");

  const vault = await RiskParityVault.deploy(
    MOCK_USDC,                          // underlying asset (USDC)
    "RiskParity Crypto Index",          // share token name
    "rpCRYPTO",                         // share token symbol
    deployer.address,                   // admin
    deployer.address,                   // keeper (use deployer for testnet)
    deployer.address                    // fee recipient
  );

  await vault.waitForDeployment();
  const vaultAddress = await vault.getAddress();
  console.log(`\n  Vault deployed at: ${vaultAddress}`);

  // ═══════════════════════════════════════════════════
  //              REGISTER CONSTITUENTS
  // ═══════════════════════════════════════════════════

  console.log("\nRegistering constituents...");

  const constituents = [
    { token: MOCK_WBTC,  feed: FEED_BTC,   weight: 1500, name: "WBTC"  },  // 15%
    { token: MOCK_WETH,  feed: FEED_ETH,   weight: 1500, name: "WETH"  },  // 15%
    { token: MOCK_SOL,   feed: FEED_SOL,   weight: 1000, name: "SOL"   },  // 10%
    { token: MOCK_STETH, feed: FEED_STETH, weight: 1000, name: "stETH" },  // 10%
    { token: MOCK_RETH,  feed: FEED_RETH,  weight: 1000, name: "rETH"  },  // 10%
    { token: MOCK_BUIDL, feed: FEED_BUIDL, weight: 1500, name: "BUIDL" },  // 15%
    { token: MOCK_USDY,  feed: FEED_USDY,  weight: 1000, name: "USDY"  },  // 10%
    { token: MOCK_USDC,  feed: FEED_USDC,  weight: 1500, name: "USDC"  },  // 15%
  ];                                              // Total: 10000 = 100%

  for (const c of constituents) {
    const tx = await vault.addConstituent(c.token, c.feed, c.weight);
    await tx.wait();
    console.log(`  Added ${c.name}: weight=${c.weight/100}%, feed=${c.feed.slice(0, 10)}...`);
  }

  // ═══════════════════════════════════════════════════
  //           SET DEFENSIVE WEIGHTS
  // ═══════════════════════════════════════════════════

  console.log("\nSetting defensive weights (circuit breaker mode)...");
  const defTokens = constituents.map(c => c.token);
  const defWeights = [500, 500, 0, 500, 500, 3500, 1500, 3000]; // Crisis allocation

  const tx = await vault.setDefensiveWeights(defTokens, defWeights);
  await tx.wait();
  console.log("  Defensive weights set");

  // ═══════════════════════════════════════════════════
  //              DEPLOYMENT COMPLETE
  // ═══════════════════════════════════════════════════

  console.log("\n" + "═".repeat(60));
  console.log("  VAULT DEPLOYMENT COMPLETE");
  console.log("═".repeat(60));
  console.log(`\n  Contract Address: ${vaultAddress}`);
  console.log(`  Network:          Sepolia (Chain ID: 11155111)`);
  console.log(`  Underlying:       USDC (${MOCK_USDC})`);
  console.log(`  Share Token:      rpCRYPTO`);
  console.log(`  Constituents:     ${constituents.length}`);
  console.log(`  Admin:            ${deployer.address}`);
  console.log(`  Keeper:           ${deployer.address}`);
  console.log(`\n  Etherscan: https://sepolia.etherscan.io/address/${vaultAddress}`);
  console.log("\n  *** INCLUDE THIS ADDRESS IN YOUR REPORT ***");
  console.log("═".repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
