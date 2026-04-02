/**
 * @title Deploy Mock Tokens
 * @notice Deploys MockERC20 tokens and MockPriceFeeds on Sepolia for testing.
 *         Run this BEFORE deploying the vault.
 *
 * Usage: npx hardhat run scripts/deploy_mocks.js --network sepolia
 */
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying mock tokens with:", deployer.address);
  console.log("Balance:", hre.ethers.formatEther(await hre.ethers.provider.getBalance(deployer.address)), "ETH\n");

  // ═══════════════════════════════════════════════════
  //                  MOCK ERC-20 TOKENS
  // ═══════════════════════════════════════════════════

  const MockERC20 = await hre.ethers.getContractFactory("MockERC20");

  const tokens = [
    { name: "Mock USDC",           symbol: "mUSDC",  decimals: 6  },
    { name: "Mock Wrapped BTC",    symbol: "mWBTC",  decimals: 8  },
    { name: "Mock Wrapped ETH",    symbol: "mWETH",  decimals: 18 },
    { name: "Mock SOL",            symbol: "mSOL",   decimals: 18 },
    { name: "Mock Lido stETH",     symbol: "mstETH", decimals: 18 },
    { name: "Mock Rocket Pool ETH",symbol: "mrETH",  decimals: 18 },
    { name: "Mock BlackRock BUIDL",symbol: "mBUILD", decimals: 18 },
    { name: "Mock Ondo USDY",      symbol: "mUSDY",  decimals: 18 },
  ];

  const deployed = {};

  for (const t of tokens) {
    const contract = await MockERC20.deploy(t.name, t.symbol, t.decimals);
    await contract.waitForDeployment();
    const addr = await contract.getAddress();
    deployed[t.symbol] = addr;
    console.log(`  ${t.symbol.padEnd(8)} deployed at: ${addr}`);
  }

  // Mint test tokens to deployer
  console.log("\nMinting test tokens to deployer...");
  const mintAmounts = {
    mUSDC:  hre.ethers.parseUnits("1000000", 6),   // 1M USDC
    mWBTC:  hre.ethers.parseUnits("10", 8),         // 10 BTC
    mWETH:  hre.ethers.parseUnits("100", 18),       // 100 ETH
    mSOL:   hre.ethers.parseUnits("1000", 18),      // 1000 SOL
    mstETH: hre.ethers.parseUnits("50", 18),        // 50 stETH
    mrETH:  hre.ethers.parseUnits("50", 18),        // 50 rETH
    mBUILD: hre.ethers.parseUnits("500000", 18),    // 500K BUIDL
    mUSDY:  hre.ethers.parseUnits("500000", 18),    // 500K USDY
  };

  for (const [symbol, amount] of Object.entries(mintAmounts)) {
    const contract = await hre.ethers.getContractAt("MockERC20", deployed[symbol]);
    const tx = await contract.mint(deployer.address, amount);
    await tx.wait();
    console.log(`  Minted ${symbol}: ${hre.ethers.formatUnits(amount, symbol === "mUSDC" ? 6 : symbol === "mWBTC" ? 8 : 18)}`);
  }

  // ═══════════════════════════════════════════════════
  //                 MOCK PRICE FEEDS
  // ═══════════════════════════════════════════════════

  console.log("\nDeploying mock price feeds...");
  const MockPriceFeed = await hre.ethers.getContractFactory("MockPriceFeed");

  const feeds = [
    { desc: "BTC/USD",   decimals: 8, price: 8400000000000n  },  // $84,000
    { desc: "ETH/USD",   decimals: 8, price: 190000000000n   },  // $1,900
    { desc: "SOL/USD",   decimals: 8, price: 13000000000n    },  // $130
    { desc: "stETH/USD", decimals: 8, price: 189500000000n   },  // $1,895
    { desc: "rETH/USD",  decimals: 8, price: 205000000000n   },  // $2,050
    { desc: "BUIDL/USD", decimals: 8, price: 100000000n      },  // $1.00
    { desc: "USDY/USD",  decimals: 8, price: 100500000n      },  // $1.005
    { desc: "USDC/USD",  decimals: 8, price: 100000000n      },  // $1.00
  ];

  const feedAddresses = {};
  for (const f of feeds) {
    const contract = await MockPriceFeed.deploy(f.desc, f.decimals, f.price);
    await contract.waitForDeployment();
    const addr = await contract.getAddress();
    feedAddresses[f.desc] = addr;
    console.log(`  ${f.desc.padEnd(10)} feed at: ${addr}  (price: $${Number(f.price) / 1e8})`);
  }

  // ═══════════════════════════════════════════════════
  //              DEPLOYMENT SUMMARY
  // ═══════════════════════════════════════════════════

  console.log("\n" + "═".repeat(60));
  console.log("  DEPLOYMENT SUMMARY — Save these addresses!");
  console.log("═".repeat(60));
  console.log("\nTokens:");
  for (const [symbol, addr] of Object.entries(deployed)) {
    console.log(`  ${symbol}: ${addr}`);
  }
  console.log("\nPrice Feeds:");
  for (const [desc, addr] of Object.entries(feedAddresses)) {
    console.log(`  ${desc}: ${addr}`);
  }
  console.log("\n" + "═".repeat(60));

  return { deployed, feedAddresses };
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
