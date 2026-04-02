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
  const MOCK_USDC  = "0x3dF734eb2cfae0eA510D5C0bf2c16f632Af3B000";
  const MOCK_WBTC  = "0x0fFd700DB7b7B148a225B749534db32e4C04BEe7";
  const MOCK_WETH  = "0x12907575569f6B9817DCbaD1824CbEc52e2DaFAd";
  const MOCK_SOL   = "0x74a7934248006Bb8e0e8C740D0fEc336a4cd8DB7";
  const MOCK_STETH = "0x7e198A3F2eed9Cabb53d4C245A73273942b985Ed";
  const MOCK_RETH  = "0x50Ad88734F9E64ce1e594744F837792fe372F8c2";
  const MOCK_BUIDL = "0xA6FA0bFFb602D9d7edc9605Ff8FA34c7b0165c48";
  const MOCK_USDY  = "0x9d7372473C01Fe2713B70ef4EA6DcEF2c637e819";

  // Mock price feed addresses (deployed 2 April 2026)
  const FEED_BTC   = "0x83752190Ce54db8D0Df5167E2c8b0c0649C8e41C";
  const FEED_ETH   = "0x87C349B95ABD2Bb4b6a173c892dAaF0FB4Da2A5e";
  const FEED_SOL   = "0x39d8908669ec3d4aE60B9B745B4523eB2eb6F86D";
  const FEED_STETH = "0x52fb2aEb8EfA080FF39f077E2fF334EaF6D32897";
  const FEED_RETH  = "0xcd4D2Ca5d8c4e131E688986C96300edda0f02BF7";
  const FEED_BUIDL = "0xE24c2b4fe00b028e19edc49c2eda51B89600cE07";
  const FEED_USDY  = "0x20121F6B496D1F95B3326A00cB0d8E1f0ebB4867";
  const FEED_USDC  = "0x11D579F65d653Cc88B6a806bd146d691470A0884";

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
