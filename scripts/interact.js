/**
 * @title Interact with RiskParity Vault
 * @notice Demonstrates the full vault lifecycle: deposit → rebalance → withdraw.
 *         Run AFTER deploy.js and update the vault address below.
 *
 * Usage: npx hardhat run scripts/interact.js --network sepolia
 */
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  // ═══════════════════════════════════════════════════
  //         UPDATE WITH YOUR DEPLOYED ADDRESSES
  // ═══════════════════════════════════════════════════

  const VAULT_ADDRESS = "0x_REPLACE_WITH_VAULT_ADDRESS";
  const MUSDC_ADDRESS = "0x_REPLACE_WITH_MUSDC_ADDRESS";

  const vault = await hre.ethers.getContractAt("RiskParityVault", VAULT_ADDRESS);
  const usdc = await hre.ethers.getContractAt("MockERC20", MUSDC_ADDRESS);

  console.log("═".repeat(60));
  console.log("  VAULT INTERACTION DEMO");
  console.log("═".repeat(60));
  console.log(`  Vault:    ${VAULT_ADDRESS}`);
  console.log(`  User:     ${deployer.address}`);
  console.log(`  USDC:     ${MUSDC_ADDRESS}`);

  // ═══════════════════════════════════════════════════
  //               1. CHECK INITIAL STATE
  // ═══════════════════════════════════════════════════

  console.log("\n─── 1. Initial State ───");
  const metrics = await vault.getVaultMetrics();
  console.log(`  NAV per share:     ${hre.ethers.formatUnits(metrics.nav, 18)}`);
  console.log(`  High Water Mark:   ${hre.ethers.formatUnits(metrics.hwm, 18)}`);
  console.log(`  Epoch:             ${metrics.epoch}`);
  console.log(`  Circuit Breaker:   ${metrics.cbActive}`);
  console.log(`  Total Shares:      ${hre.ethers.formatUnits(metrics.totalShares, 18)}`);
  console.log(`  Total Assets:      ${hre.ethers.formatUnits(metrics.totalVal, 6)}`);

  // ═══════════════════════════════════════════════════
  //              2. DEPOSIT 10,000 USDC
  // ═══════════════════════════════════════════════════

  console.log("\n─── 2. Deposit 10,000 USDC ───");
  const depositAmount = hre.ethers.parseUnits("10000", 6);

  // Approve vault to spend USDC
  let tx = await usdc.approve(VAULT_ADDRESS, depositAmount);
  await tx.wait();
  console.log("  Approved USDC spending");

  // Deposit
  tx = await vault.deposit(depositAmount, deployer.address);
  const receipt = await tx.wait();
  console.log(`  Deposited 10,000 USDC (tx: ${receipt.hash})`);

  // Check shares received
  const shares = await vault.balanceOf(deployer.address);
  console.log(`  Shares received:   ${hre.ethers.formatUnits(shares, 18)} rpCRYPTO`);

  // Check vault state
  const postDepositAssets = await vault.totalAssets();
  console.log(`  Vault total assets: ${hre.ethers.formatUnits(postDepositAssets, 6)} USDC`);

  // ═══════════════════════════════════════════════════
  //         3. COMMIT + EXECUTE WEIGHT UPDATE
  // ═══════════════════════════════════════════════════

  console.log("\n─── 3. Weight Update (Commit-Reveal) ───");

  // Get current constituents
  const constituentAddrs = await vault.getConstituents();
  console.log(`  Constituents: ${constituentAddrs.length}`);

  // New weights (simulating ML ensemble output)
  const newWeights = [2000, 1800, 800, 1200, 800, 1500, 900, 1000]; // sum = 10000

  // Compute Merkle leaf (simplified for testnet: just hash all data together)
  const leaf = hre.ethers.keccak256(
    hre.ethers.AbiCoder.defaultAbiCoder().encode(
      ["address[]", "uint256[]"],
      [constituentAddrs, newWeights]
    )
  );

  // For testnet: Merkle root = leaf (single-leaf tree)
  const merkleRoot = leaf;
  const proof = []; // Empty proof for single-leaf tree

  // Phase 1: Commit
  tx = await vault.commitWeights(merkleRoot);
  await tx.wait();
  console.log(`  Committed weights (root: ${merkleRoot.slice(0, 18)}...)`);

  // Phase 2: Wait for timelock (on testnet, we can use hardhat time manipulation)
  console.log("  Waiting for timelock (1 hour)...");

  // In production, wait 1 hour. For testnet demo, we can advance time in hardhat local:
  // await hre.network.provider.send("evm_increaseTime", [3600]);
  // await hre.network.provider.send("evm_mine");
  // On actual Sepolia, you'd need to wait or reduce timelock.

  console.log("  NOTE: On Sepolia, wait 1 hour before calling executeWeights.");
  console.log("        Or redeploy with a shorter timelock for demo purposes.");

  // ═══════════════════════════════════════════════════
  //              4. PARTIAL WITHDRAWAL
  // ═══════════════════════════════════════════════════

  console.log("\n─── 4. Withdraw 2,000 USDC ───");
  const withdrawAmount = hre.ethers.parseUnits("2000", 6);

  tx = await vault.withdraw(withdrawAmount, deployer.address, deployer.address);
  const withdrawReceipt = await tx.wait();
  console.log(`  Withdrew 2,000 USDC (tx: ${withdrawReceipt.hash})`);

  const remainingShares = await vault.balanceOf(deployer.address);
  console.log(`  Remaining shares:  ${hre.ethers.formatUnits(remainingShares, 18)} rpCRYPTO`);

  const remainingAssets = await vault.totalAssets();
  console.log(`  Vault assets:      ${hre.ethers.formatUnits(remainingAssets, 6)} USDC`);

  // ═══════════════════════════════════════════════════
  //              5. FINAL STATE
  // ═══════════════════════════════════════════════════

  console.log("\n─── 5. Final State ───");
  const finalMetrics = await vault.getVaultMetrics();
  console.log(`  NAV per share:     ${hre.ethers.formatUnits(finalMetrics.nav, 18)}`);
  console.log(`  Epoch:             ${finalMetrics.epoch}`);
  console.log(`  Gate used:         ${hre.ethers.formatUnits(finalMetrics.gateUsed, 6)}`);
  console.log(`  Gate max:          ${hre.ethers.formatUnits(finalMetrics.gateMax, 6)}`);
  console.log(`  Total shares:      ${hre.ethers.formatUnits(finalMetrics.totalShares, 18)}`);

  const feeSummary = await vault.getFeeSummary();
  console.log(`\n  Fees collected:`);
  console.log(`    Management:      ${hre.ethers.formatUnits(feeSummary.totalMgmtCollected, 18)} shares`);
  console.log(`    Performance:     ${hre.ethers.formatUnits(feeSummary.totalPerfCollected, 18)} shares`);

  console.log("\n" + "═".repeat(60));
  console.log("  INTERACTION DEMO COMPLETE");
  console.log("═".repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
