const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time, loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

describe("RiskParityVault", function () {
  // ═══════════════════════════════════════════════════════════════════
  //                         FIXTURES
  // ═══════════════════════════════════════════════════════════════════

  async function deployFixture() {
    const [admin, keeper, user1, user2, feeRecipient] = await ethers.getSigners();

    // Deploy mock tokens
    const MockERC20 = await ethers.getContractFactory("MockERC20");
    const usdc = await MockERC20.deploy("Mock USDC", "mUSDC", 6);
    const wbtc = await MockERC20.deploy("Mock WBTC", "mWBTC", 8);
    const weth = await MockERC20.deploy("Mock WETH", "mWETH", 18);
    const sol_ = await MockERC20.deploy("Mock SOL", "mSOL", 18);
    await Promise.all([
      usdc.waitForDeployment(),
      wbtc.waitForDeployment(),
      weth.waitForDeployment(),
      sol_.waitForDeployment(),
    ]);

    // Deploy mock price feeds (BTC ~$84k, ETH ~$1900, SOL ~$130, USDC ~$1)
    const MockPriceFeed = await ethers.getContractFactory("MockPriceFeed");
    const btcFeed = await MockPriceFeed.deploy("BTC/USD", 8, 8400000000000n);
    const ethFeed = await MockPriceFeed.deploy("ETH/USD", 8, 190000000000n);
    const solFeed = await MockPriceFeed.deploy("SOL/USD", 8, 13000000000n);
    const usdcFeed = await MockPriceFeed.deploy("USDC/USD", 8, 100000000n);
    await Promise.all([
      btcFeed.waitForDeployment(),
      ethFeed.waitForDeployment(),
      solFeed.waitForDeployment(),
      usdcFeed.waitForDeployment(),
    ]);

    // Deploy vault
    const Vault = await ethers.getContractFactory("RiskParityVault");
    const vault = await Vault.deploy(
      await usdc.getAddress(),
      "RiskParity Crypto Index",
      "rpCRYPTO",
      admin.address,
      keeper.address,
      feeRecipient.address
    );
    await vault.waitForDeployment();

    // Register constituents (BTC 30%, ETH 30%, SOL 20%, USDC 20%)
    await vault.connect(admin).addConstituent(await wbtc.getAddress(), await btcFeed.getAddress(), 3000);
    await vault.connect(admin).addConstituent(await weth.getAddress(), await ethFeed.getAddress(), 3000);
    await vault.connect(admin).addConstituent(await sol_.getAddress(), await solFeed.getAddress(), 2000);
    await vault.connect(admin).addConstituent(await usdc.getAddress(), await usdcFeed.getAddress(), 2000);

    // Set defensive weights (risk-off: 5% BTC, 5% ETH, 0% SOL, 90% USDC)
    const defTokens = [
      await wbtc.getAddress(),
      await weth.getAddress(),
      await sol_.getAddress(),
      await usdc.getAddress(),
    ];
    await vault.connect(admin).setDefensiveWeights(defTokens, [500, 500, 0, 9000]);

    // Disable portfolio valuation for simpler testing (testnet mode)
    await vault.connect(admin).setUsePortfolioValuation(false);

    // Mint USDC to test users
    const mint = ethers.parseUnits("100000", 6);
    await usdc.mint(user1.address, mint);
    await usdc.mint(user2.address, mint);

    return {
      vault, usdc, wbtc, weth, sol: sol_,
      btcFeed, ethFeed, solFeed, usdcFeed,
      admin, keeper, user1, user2, feeRecipient,
    };
  }

  // ═══════════════════════════════════════════════════════════════════
  //                       1. DEPLOYMENT
  // ═══════════════════════════════════════════════════════════════════

  describe("Deployment", function () {
    it("should deploy with correct name, symbol, and roles", async function () {
      const { vault, admin, keeper } = await loadFixture(deployFixture);
      expect(await vault.name()).to.equal("RiskParity Crypto Index");
      expect(await vault.symbol()).to.equal("rpCRYPTO");
      expect(await vault.hasRole(await vault.ADMIN_ROLE(), admin.address)).to.be.true;
      expect(await vault.hasRole(await vault.KEEPER_ROLE(), keeper.address)).to.be.true;
    });

    it("should initialise fees and epoch correctly", async function () {
      const { vault } = await loadFixture(deployFixture);
      expect(await vault.managementFeeBps()).to.equal(100);
      expect(await vault.performanceFeeBps()).to.equal(1000);
      expect(await vault.redemptionFeeBps()).to.equal(30);
      expect(await vault.currentEpoch()).to.equal(1);
      expect(await vault.constituentCount()).to.equal(4);
      expect(await vault.highWaterMark()).to.equal(ethers.parseEther("1"));
    });

    it("should revert on zero admin address", async function () {
      const [, keeper, , , feeRecipient] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const usdc = await MockERC20.deploy("Mock USDC", "mUSDC", 6);
      await usdc.waitForDeployment();
      const Vault = await ethers.getContractFactory("RiskParityVault");
      await expect(
        Vault.deploy(await usdc.getAddress(), "Test", "TST", ethers.ZeroAddress, keeper.address, feeRecipient.address)
      ).to.be.revertedWithCustomError(Vault, "ZeroAddress");
    });

    it("should revert on zero keeper address", async function () {
      const [admin, , , , feeRecipient] = await ethers.getSigners();
      const MockERC20 = await ethers.getContractFactory("MockERC20");
      const usdc = await MockERC20.deploy("Mock USDC", "mUSDC", 6);
      await usdc.waitForDeployment();
      const Vault = await ethers.getContractFactory("RiskParityVault");
      await expect(
        Vault.deploy(await usdc.getAddress(), "Test", "TST", admin.address, ethers.ZeroAddress, feeRecipient.address)
      ).to.be.revertedWithCustomError(Vault, "ZeroAddress");
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                       2. DEPOSITS
  // ═══════════════════════════════════════════════════════════════════

  describe("Deposit", function () {
    it("should mint shares on USDC deposit", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      const amt = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), amt);
      await vault.connect(user1).deposit(amt, user1.address);
      expect(await vault.balanceOf(user1.address)).to.be.gt(0);
      expect(await vault.totalAssets()).to.equal(amt);
    });

    it("should track deposit time for redemption fee window", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      await usdc.connect(user1).approve(await vault.getAddress(), ethers.parseUnits("1000", 6));
      await vault.connect(user1).deposit(ethers.parseUnits("1000", 6), user1.address);
      expect(await vault.lastDepositTime(user1.address)).to.be.gt(0);
    });

    it("should revert when paused", async function () {
      const { vault, usdc, admin, user1 } = await loadFixture(deployFixture);
      await vault.connect(admin).pause();
      await usdc.connect(user1).approve(await vault.getAddress(), 1000);
      await expect(
        vault.connect(user1).deposit(1000, user1.address)
      ).to.be.revertedWithCustomError(vault, "EnforcedPause");
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                       3. WITHDRAWALS
  // ═══════════════════════════════════════════════════════════════════

  describe("Withdraw", function () {
    it("should return assets on withdrawal after early window", async function () {
      const { vault, usdc, user1, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      // Wait past early redemption window (7 days)
      await time.increase(8 * 86400);
      // Refresh mock price feeds so they are not stale
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);
      await vault.connect(user1).withdraw(ethers.parseUnits("1000", 6), user1.address, user1.address);
      // Vault should have less than the original deposit
      expect(await vault.totalAssets()).to.be.lt(dep);
    });

    it("should enforce redemption gate with custom error", async function () {
      const { vault, usdc, user1, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("100000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      await time.increase(8 * 86400);
      // Refresh price feeds
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);
      // Gate is 20% of 100k = 20k, try to withdraw 25k
      await expect(
        vault.connect(user1).withdraw(ethers.parseUnits("25000", 6), user1.address, user1.address)
      ).to.be.revertedWithCustomError(vault, "RedemptionGateExceeded");
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                  4. EARLY REDEMPTION FEE
  // ═══════════════════════════════════════════════════════════════════

  describe("Early Redemption Fee", function () {
    it("should charge fee for withdrawal within early window", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);

      // Withdraw immediately (within 7-day window) — should trigger fee
      const withdrawAmt = ethers.parseUnits("1000", 6);
      const tx = await vault.connect(user1).withdraw(withdrawAmt, user1.address, user1.address);
      const receipt = await tx.wait();

      // Check RedemptionFeeCharged event was emitted
      const feeEvents = receipt.logs.filter((log) => {
        try {
          return vault.interface.parseLog(log)?.name === "RedemptionFeeCharged";
        } catch { return false; }
      });
      expect(feeEvents.length).to.equal(1);
    });

    it("should NOT charge fee after early window expires", async function () {
      const { vault, usdc, user1, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);

      // Wait past the 7-day window
      await time.increase(8 * 86400);
      // Refresh price feeds
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);

      const withdrawAmt = ethers.parseUnits("1000", 6);
      const tx = await vault.connect(user1).withdraw(withdrawAmt, user1.address, user1.address);
      const receipt = await tx.wait();

      // No RedemptionFeeCharged event expected
      const feeEvents = receipt.logs.filter((log) => {
        try {
          return vault.interface.parseLog(log)?.name === "RedemptionFeeCharged";
        } catch { return false; }
      });
      expect(feeEvents.length).to.equal(0);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                       5. FEES
  // ═══════════════════════════════════════════════════════════════════

  describe("Fees", function () {
    it("should accrue management fees over time via share dilution", async function () {
      const { vault, usdc, user1, user2, feeRecipient, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      await usdc.connect(user1).approve(await vault.getAddress(), ethers.parseUnits("50000", 6));
      await vault.connect(user1).deposit(ethers.parseUnits("50000", 6), user1.address);

      // Advance 1 year
      await time.increase(365 * 86400);
      // Refresh mock price feeds so they are not stale
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);

      // Trigger fee accrual via a deposit from user2 (user1 may not have balance left)
      const triggerAmt = ethers.parseUnits("100", 6);
      await usdc.connect(user2).approve(await vault.getAddress(), triggerAmt);
      await vault.connect(user2).deposit(triggerAmt, user2.address);

      expect(await vault.balanceOf(feeRecipient.address)).to.be.gt(0);
      expect(await vault.totalManagementFeesCollected()).to.be.gt(0);
    });

    it("should reject excessive fee parameters", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await expect(vault.connect(admin).setFees(600, 1000, 30)).to.be.revertedWith("Mgmt fee > 5%");
      await expect(vault.connect(admin).setFees(100, 3500, 30)).to.be.revertedWith("Perf fee > 30%");
      await expect(vault.connect(admin).setFees(100, 1000, 250)).to.be.revertedWith("Redemption fee > 2%");
    });

    it("should update fee parameters correctly", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await vault.connect(admin).setFees(200, 2000, 50);
      expect(await vault.managementFeeBps()).to.equal(200);
      expect(await vault.performanceFeeBps()).to.equal(2000);
      expect(await vault.redemptionFeeBps()).to.equal(50);
    });

    it("should return correct fee summary", async function () {
      const { vault, feeRecipient } = await loadFixture(deployFixture);
      const summary = await vault.getFeeSummary();
      expect(summary.mgmtBps).to.equal(100);
      expect(summary.perfBps).to.equal(1000);
      expect(summary.redemptBps).to.equal(30);
      expect(summary.recipient).to.equal(feeRecipient.address);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                       6. EPOCHS
  // ═══════════════════════════════════════════════════════════════════

  describe("Epoch", function () {
    it("should advance after duration expires", async function () {
      const { vault } = await loadFixture(deployFixture);
      await time.increase(2 * 86400);
      await vault.advanceEpoch();
      expect(await vault.currentEpoch()).to.equal(2);
    });

    it("should revert with EpochNotEnded if epoch not ended", async function () {
      const { vault } = await loadFixture(deployFixture);
      await expect(vault.advanceEpoch()).to.be.revertedWithCustomError(vault, "EpochNotEnded");
    });

    it("should reset redemption counter on epoch advance", async function () {
      const { vault, usdc, user1, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      await usdc.connect(user1).approve(await vault.getAddress(), ethers.parseUnits("100000", 6));
      await vault.connect(user1).deposit(ethers.parseUnits("100000", 6), user1.address);
      await time.increase(8 * 86400);
      // Refresh price feeds
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);
      await vault.connect(user1).withdraw(ethers.parseUnits("5000", 6), user1.address, user1.address);
      await time.increase(86400);
      // Refresh price feeds again before epoch advance (navPerShare is called internally)
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);
      await vault.advanceEpoch();
      expect((await vault.getVaultMetrics()).gateUsed).to.equal(0);
    });

    it("should toggle volatile epoch mode", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      // Switch to volatile
      await vault.connect(keeper).setEpochVolatile(true);
      expect(await vault.volatileEpochMode()).to.be.true;
      expect(await vault.epochDuration()).to.equal(7 * 86400);

      // Switch back to calm
      await vault.connect(keeper).setEpochVolatile(false);
      expect(await vault.volatileEpochMode()).to.be.false;
      expect(await vault.epochDuration()).to.equal(86400);
    });

    it("should emit EpochModeChanged event", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      await expect(vault.connect(keeper).setEpochVolatile(true))
        .to.emit(vault, "EpochModeChanged")
        .withArgs(true, 7 * 86400);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //               7. WEIGHT COMMITMENT (MERKLE)
  // ═══════════════════════════════════════════════════════════════════

  describe("Weight Commitment", function () {
    it("should accept commitment from keeper", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      const root = ethers.keccak256(ethers.toUtf8Bytes("weights"));
      await vault.connect(keeper).commitWeights(root);
      expect(await vault.commitPending()).to.be.true;
      expect(await vault.committedWeightsRoot()).to.equal(root);
    });

    it("should reject commitment from non-keeper", async function () {
      const { vault, user1 } = await loadFixture(deployFixture);
      await expect(
        vault.connect(user1).commitWeights(ethers.keccak256(ethers.toUtf8Bytes("x")))
      ).to.be.reverted;
    });

    it("should enforce timelock with custom error", async function () {
      const { vault, keeper, wbtc, weth, sol, usdc } = await loadFixture(deployFixture);
      const tokens = [
        await wbtc.getAddress(), await weth.getAddress(),
        await sol.getAddress(), await usdc.getAddress(),
      ];
      const newW = [2500, 2500, 2500, 2500];
      const leaf = ethers.keccak256(
        ethers.AbiCoder.defaultAbiCoder().encode(["address[]", "uint256[]"], [tokens, newW])
      );
      await vault.connect(keeper).commitWeights(leaf);
      await expect(
        vault.connect(keeper).executeWeights(tokens, newW, [])
      ).to.be.revertedWithCustomError(vault, "TimelockNotElapsed");
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                    8. CANCEL COMMIT
  // ═══════════════════════════════════════════════════════════════════

  describe("Cancel Commit", function () {
    it("should cancel a pending commitment", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      const root = ethers.keccak256(ethers.toUtf8Bytes("weights"));
      await vault.connect(keeper).commitWeights(root);
      expect(await vault.commitPending()).to.be.true;

      await vault.connect(keeper).cancelCommit();
      expect(await vault.commitPending()).to.be.false;
      expect(await vault.committedWeightsRoot()).to.equal(ethers.ZeroHash);
      expect(await vault.commitTimestamp()).to.equal(0);
    });

    it("should emit WeightCommitCancelled event", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      const root = ethers.keccak256(ethers.toUtf8Bytes("weights"));
      await vault.connect(keeper).commitWeights(root);

      await expect(vault.connect(keeper).cancelCommit())
        .to.emit(vault, "WeightCommitCancelled")
        .withArgs(root, (await time.latest()) + 1); // next block timestamp
    });

    it("should revert if no commit is pending", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      await expect(
        vault.connect(keeper).cancelCommit()
      ).to.be.revertedWithCustomError(vault, "NoCommitPending");
    });

    it("should reject cancel from non-keeper", async function () {
      const { vault, keeper, user1 } = await loadFixture(deployFixture);
      const root = ethers.keccak256(ethers.toUtf8Bytes("weights"));
      await vault.connect(keeper).commitWeights(root);
      await expect(vault.connect(user1).cancelCommit()).to.be.reverted;
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                    9. CIRCUIT BREAKER
  // ═══════════════════════════════════════════════════════════════════

  describe("Circuit Breaker", function () {
    it("should not trigger when no shares exist", async function () {
      const { vault } = await loadFixture(deployFixture);
      await vault.checkCircuitBreaker();
      expect(await vault.circuitBreakerActive()).to.be.false;
    });

    it("should revert resetCircuitBreaker when not active", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await expect(
        vault.connect(admin).resetCircuitBreaker()
      ).to.be.revertedWithCustomError(vault, "CircuitBreakerState");
    });

    it("should record circuitBreakerThresholdBps correctly", async function () {
      const { vault } = await loadFixture(deployFixture);
      expect(await vault.circuitBreakerThresholdBps()).to.equal(1500);
    });

    it("should have defensive weights set correctly", async function () {
      const { vault, usdc } = await loadFixture(deployFixture);
      const usdcDefensiveWeight = await vault.defensiveWeights(await usdc.getAddress());
      expect(usdcDefensiveWeight).to.equal(9000);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //              10. PORTFOLIO VALUATION (totalAssets)
  // ═══════════════════════════════════════════════════════════════════

  describe("Portfolio Valuation", function () {
    it("should return only USDC balance when portfolio valuation is off", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      // With portfolio valuation off, totalAssets = USDC balance only
      expect(await vault.totalAssets()).to.equal(dep);
    });

    it("should include constituent values when portfolio valuation is on", async function () {
      const { vault, usdc, wbtc, btcFeed, admin, user1 } = await loadFixture(deployFixture);
      // Deposit USDC
      const dep = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);

      // Send some WBTC to the vault (simulating a rebalance)
      await wbtc.mint(await vault.getAddress(), 100000n); // 0.001 BTC (8 decimals)

      // Enable portfolio valuation
      await vault.connect(admin).setUsePortfolioValuation(true);

      // totalAssets should now include the BTC value
      const totalWithPortfolio = await vault.totalAssets();
      expect(totalWithPortfolio).to.be.gt(dep);
    });

    it("should toggle portfolio valuation and emit event", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      // Currently off (set in fixture)
      expect(await vault.usePortfolioValuation()).to.be.false;

      await expect(vault.connect(admin).setUsePortfolioValuation(true))
        .to.emit(vault, "PortfolioValuationToggled")
        .withArgs(true);

      expect(await vault.usePortfolioValuation()).to.be.true;
    });

    it("should compute portfolio value from Chainlink oracles", async function () {
      const { vault, wbtc } = await loadFixture(deployFixture);
      // Send 1 BTC to vault (1e8 smallest units)
      await wbtc.mint(await vault.getAddress(), 100000000n);

      // computePortfolioValue should reflect BTC price
      const portfolioValue = await vault.computePortfolioValue();
      // 1 BTC at $84,000 = 84,000e18 in 18-decimal
      expect(portfolioValue).to.be.gt(0);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                  11. CUSTOM ERRORS
  // ═══════════════════════════════════════════════════════════════════

  describe("Custom Errors", function () {
    it("should revert with AlreadyConstituent on duplicate registration", async function () {
      const { vault, admin, wbtc, btcFeed } = await loadFixture(deployFixture);
      await expect(
        vault.connect(admin).addConstituent(await wbtc.getAddress(), await btcFeed.getAddress(), 1000)
      ).to.be.revertedWithCustomError(vault, "AlreadyConstituent");
    });

    it("should revert with ZeroAddress on null token", async function () {
      const { vault, admin, btcFeed } = await loadFixture(deployFixture);
      await expect(
        vault.connect(admin).addConstituent(ethers.ZeroAddress, await btcFeed.getAddress(), 1000)
      ).to.be.revertedWithCustomError(vault, "ZeroAddress");
    });

    it("should revert with ZeroAddress on null fee recipient", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await expect(
        vault.connect(admin).setFeeRecipient(ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(vault, "ZeroAddress");
    });

    it("should revert with NotConstituent for unregistered token in defensive weights", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      const fakeToken = "0x0000000000000000000000000000000000000001";
      await expect(
        vault.connect(admin).setDefensiveWeights([fakeToken], [10000])
      ).to.be.revertedWithCustomError(vault, "NotConstituent");
    });

    it("should revert with WeightSumMismatch when defensive weights do not sum to 10000", async function () {
      const { vault, admin, wbtc, weth, sol, usdc } = await loadFixture(deployFixture);
      const tokens = [
        await wbtc.getAddress(), await weth.getAddress(),
        await sol.getAddress(), await usdc.getAddress(),
      ];
      await expect(
        vault.connect(admin).setDefensiveWeights(tokens, [2500, 2500, 2500, 2000])
      ).to.be.revertedWithCustomError(vault, "WeightSumMismatch");
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                   12. ACCESS CONTROL
  // ═══════════════════════════════════════════════════════════════════

  describe("Access Control", function () {
    it("should restrict admin functions from non-admin users", async function () {
      const { vault, user1, btcFeed } = await loadFixture(deployFixture);
      await expect(
        vault.connect(user1).addConstituent(ethers.ZeroAddress, await btcFeed.getAddress(), 1000)
      ).to.be.reverted;
      await expect(vault.connect(user1).pause()).to.be.reverted;
      await expect(vault.connect(user1).setFees(50, 500, 10)).to.be.reverted;
    });

    it("should restrict keeper functions from non-keeper users", async function () {
      const { vault, user1 } = await loadFixture(deployFixture);
      await expect(
        vault.connect(user1).setEpochVolatile(true)
      ).to.be.reverted;
    });

    it("should restrict setUsePortfolioValuation to admin", async function () {
      const { vault, user1 } = await loadFixture(deployFixture);
      await expect(
        vault.connect(user1).setUsePortfolioValuation(true)
      ).to.be.reverted;
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                   13. VIEW FUNCTIONS
  // ═══════════════════════════════════════════════════════════════════

  describe("View Functions", function () {
    it("should return correct vault metrics and weights", async function () {
      const { vault } = await loadFixture(deployFixture);
      const metrics = await vault.getVaultMetrics();
      expect(metrics.epoch).to.equal(1);
      expect(metrics.cbActive).to.be.false;

      const [tokens, w] = await vault.getCurrentWeights();
      expect(tokens.length).to.equal(4);
      let total = 0n;
      for (const wt of w) total += wt;
      expect(total).to.equal(10000n);
    });

    it("should return portfolio breakdown", async function () {
      const { vault } = await loadFixture(deployFixture);
      const breakdown = await vault.getPortfolioBreakdown();
      expect(breakdown.tokens.length).to.equal(4);
      expect(breakdown.weightsBps.length).to.equal(4);
      expect(breakdown.prices.length).to.equal(4);

      // All prices should be positive
      for (const p of breakdown.prices) {
        expect(p).to.be.gt(0);
      }
    });

    it("should return correct constituent count", async function () {
      const { vault } = await loadFixture(deployFixture);
      expect(await vault.constituentCount()).to.equal(4);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                  14. MOCK PRICE FEED
  // ═══════════════════════════════════════════════════════════════════

  describe("MockPriceFeed", function () {
    it("should simulate volatility and create price history", async function () {
      const { btcFeed, admin } = await loadFixture(deployFixture);
      const initialLen = await btcFeed.priceHistoryLength();
      await btcFeed.connect(admin).simulateVolatility(500, 10); // 5% amplitude, 10 periods
      expect(await btcFeed.priceHistoryLength()).to.equal(initialLen + 10n);
    });

    it("should return correct return series", async function () {
      const { btcFeed, admin } = await loadFixture(deployFixture);
      // Generate some price history
      await btcFeed.connect(admin).updatePrice(8500000000000n);
      await btcFeed.connect(admin).updatePrice(8300000000000n);
      await btcFeed.connect(admin).updatePrice(8600000000000n);

      const returns = await btcFeed.getReturnSeries(4);
      expect(returns.length).to.equal(3); // 4 observations = 3 returns
    });

    it("should return empty array for insufficient history", async function () {
      const MockPriceFeed = await ethers.getContractFactory("MockPriceFeed");
      const feed = await MockPriceFeed.deploy("TEST/USD", 8, 100000000n);
      await feed.waitForDeployment();
      // Only 1 observation, need 2 for returns
      const returns = await feed.getReturnSeries(10);
      expect(returns.length).to.equal(0);
    });

    it("should allow setting custom updatedAt for staleness tests", async function () {
      const { btcFeed, admin } = await loadFixture(deployFixture);
      const oldTimestamp = (await time.latest()) - 7200; // 2 hours ago
      await btcFeed.connect(admin).setUpdatedAt(oldTimestamp);

      const roundData = await btcFeed.latestRoundData();
      expect(roundData.updatedAt).to.equal(oldTimestamp);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                  15. FULL LIFECYCLE
  // ═══════════════════════════════════════════════════════════════════

  describe("Full Lifecycle", function () {
    it("deposit -> time -> fee accrual -> epoch advance -> withdraw", async function () {
      const { vault, usdc, user1, user2, feeRecipient, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("50000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      const initialShares = await vault.balanceOf(user1.address);

      // 30 days pass (keeps fees reasonable and avoids price staleness complexity)
      await time.increase(30 * 86400);
      // Refresh mock price feeds
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);

      // Trigger fee accrual via user2 deposit
      await usdc.connect(user2).approve(await vault.getAddress(), ethers.parseUnits("100", 6));
      await vault.connect(user2).deposit(ethers.parseUnits("100", 6), user2.address);

      // Fee recipient should have received management fees
      expect(await vault.balanceOf(feeRecipient.address)).to.be.gt(0);

      // Advance epoch (already past 1-day base duration)
      // Refresh feeds again since advanceEpoch calls navPerShare internally
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);
      await vault.advanceEpoch();

      // Refresh feeds for withdrawal
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);

      // Use share-based redemption (avoids ERC4626 rounding edge case in withdraw)
      const sharesToRedeem = (await vault.balanceOf(user1.address)) / 10n;
      await vault.connect(user1).redeem(sharesToRedeem, user1.address, user1.address);

      // User should have fewer shares after redemption
      expect(await vault.balanceOf(user1.address)).to.be.lt(initialShares);
    });

    it("commit -> cancel -> recommit flow", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      const root1 = ethers.keccak256(ethers.toUtf8Bytes("weights-v1"));
      const root2 = ethers.keccak256(ethers.toUtf8Bytes("weights-v2"));

      // Commit initial weights
      await vault.connect(keeper).commitWeights(root1);
      expect(await vault.commitPending()).to.be.true;

      // Cancel
      await vault.connect(keeper).cancelCommit();
      expect(await vault.commitPending()).to.be.false;

      // Recommit with different weights
      await vault.connect(keeper).commitWeights(root2);
      expect(await vault.commitPending()).to.be.true;
      expect(await vault.committedWeightsRoot()).to.equal(root2);
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                  16. REDEMPTION GATE EVENT
  // ═══════════════════════════════════════════════════════════════════

  describe("Redemption Gate Event", function () {
    it("should emit RedemptionGateHit when gate utilisation exceeds 80%", async function () {
      const { vault, usdc, user1, btcFeed, ethFeed, solFeed, usdcFeed, admin } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("100000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);

      // Wait past early redemption window
      await time.increase(8 * 86400);
      // Refresh price feeds
      await btcFeed.connect(admin).updatePrice(8400000000000n);
      await ethFeed.connect(admin).updatePrice(190000000000n);
      await solFeed.connect(admin).updatePrice(13000000000n);
      await usdcFeed.connect(admin).updatePrice(100000000n);

      // Gate is 20% of 100k = 20k USDC
      // 80% of gate = 16k. Withdraw 17k to exceed 80% threshold
      const withdrawAmt = ethers.parseUnits("17000", 6);
      await expect(
        vault.connect(user1).withdraw(withdrawAmt, user1.address, user1.address)
      ).to.emit(vault, "RedemptionGateHit");
    });
  });

  // ═══════════════════════════════════════════════════════════════════
  //                  17. ADMIN PARAMETER UPDATES
  // ═══════════════════════════════════════════════════════════════════

  describe("Admin Parameter Updates", function () {
    it("should update redemption gate within valid range", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await vault.connect(admin).setRedemptionGate(3000);
      expect(await vault.redemptionGateBps()).to.equal(3000);
    });

    it("should reject redemption gate outside valid range", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await expect(vault.connect(admin).setRedemptionGate(400)).to.be.revertedWith("Gate out of range");
      await expect(vault.connect(admin).setRedemptionGate(5100)).to.be.revertedWith("Gate out of range");
    });

    it("should update fee recipient", async function () {
      const { vault, admin, user2 } = await loadFixture(deployFixture);
      await vault.connect(admin).setFeeRecipient(user2.address);
      expect(await vault.feeRecipient()).to.equal(user2.address);
    });

    it("should pause and unpause correctly", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await vault.connect(admin).pause();
      expect(await vault.paused()).to.be.true;
      await vault.connect(admin).unpause();
      expect(await vault.paused()).to.be.false;
    });
  });
});
