const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time, loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

describe("RiskParityVault", function () {
  async function deployFixture() {
    const [admin, keeper, user1, user2, feeRecipient] = await ethers.getSigners();
    const MockERC20 = await ethers.getContractFactory("MockERC20");
    const usdc = await MockERC20.deploy("Mock USDC", "mUSDC", 6);
    const wbtc = await MockERC20.deploy("Mock WBTC", "mWBTC", 8);
    const weth = await MockERC20.deploy("Mock WETH", "mWETH", 18);
    const sol_ = await MockERC20.deploy("Mock SOL", "mSOL", 18);
    await Promise.all([usdc.waitForDeployment(), wbtc.waitForDeployment(), weth.waitForDeployment(), sol_.waitForDeployment()]);

    const MockPriceFeed = await ethers.getContractFactory("MockPriceFeed");
    const btcFeed = await MockPriceFeed.deploy("BTC/USD", 8, 8400000000000n);
    const ethFeed = await MockPriceFeed.deploy("ETH/USD", 8, 190000000000n);
    const solFeed = await MockPriceFeed.deploy("SOL/USD", 8, 13000000000n);
    const usdcFeed = await MockPriceFeed.deploy("USDC/USD", 8, 100000000n);
    await Promise.all([btcFeed.waitForDeployment(), ethFeed.waitForDeployment(), solFeed.waitForDeployment(), usdcFeed.waitForDeployment()]);

    const Vault = await ethers.getContractFactory("RiskParityVault");
    const vault = await Vault.deploy(await usdc.getAddress(), "RiskParity Crypto Index", "rpCRYPTO", admin.address, keeper.address, feeRecipient.address);
    await vault.waitForDeployment();

    await vault.connect(admin).addConstituent(await wbtc.getAddress(), await btcFeed.getAddress(), 3000);
    await vault.connect(admin).addConstituent(await weth.getAddress(), await ethFeed.getAddress(), 3000);
    await vault.connect(admin).addConstituent(await sol_.getAddress(), await solFeed.getAddress(), 2000);
    await vault.connect(admin).addConstituent(await usdc.getAddress(), await usdcFeed.getAddress(), 2000);

    const defTokens = [await wbtc.getAddress(), await weth.getAddress(), await sol_.getAddress(), await usdc.getAddress()];
    await vault.connect(admin).setDefensiveWeights(defTokens, [500, 500, 0, 9000]);

    const mint = ethers.parseUnits("100000", 6);
    await usdc.mint(user1.address, mint);
    await usdc.mint(user2.address, mint);

    return { vault, usdc, wbtc, weth, sol: sol_, btcFeed, ethFeed, solFeed, usdcFeed, admin, keeper, user1, user2, feeRecipient };
  }

  describe("Deployment", function () {
    it("should deploy with correct name, symbol, roles", async function () {
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
    });
  });

  describe("Deposit", function () {
    it("should mint shares on USDC deposit", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      const amt = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), amt);
      await vault.connect(user1).deposit(amt, user1.address);
      expect(await vault.balanceOf(user1.address)).to.be.gt(0);
      expect(await vault.totalAssets()).to.equal(amt);
    });

    it("should track deposit time", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      await usdc.connect(user1).approve(await vault.getAddress(), ethers.parseUnits("1000", 6));
      await vault.connect(user1).deposit(ethers.parseUnits("1000", 6), user1.address);
      expect(await vault.lastDepositTime(user1.address)).to.be.gt(0);
    });

    it("should revert when paused", async function () {
      const { vault, usdc, admin, user1 } = await loadFixture(deployFixture);
      await vault.connect(admin).pause();
      await usdc.connect(user1).approve(await vault.getAddress(), 1000);
      await expect(vault.connect(user1).deposit(1000, user1.address)).to.be.revertedWithCustomError(vault, "EnforcedPause");
    });
  });

  describe("Withdraw", function () {
    it("should return assets on withdrawal", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("10000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      await time.increase(8 * 86400);
      await vault.connect(user1).withdraw(ethers.parseUnits("5000", 6), user1.address, user1.address);
      expect(await vault.totalAssets()).to.be.closeTo(ethers.parseUnits("5000", 6), ethers.parseUnits("10", 6));
    });

    it("should enforce redemption gate", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("100000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      await time.increase(8 * 86400);
      await expect(
        vault.connect(user1).withdraw(ethers.parseUnits("25000", 6), user1.address, user1.address)
      ).to.be.revertedWith("Redemption gate exceeded");
    });
  });

  describe("Fees", function () {
    it("should accrue management fees over time", async function () {
      const { vault, usdc, user1, feeRecipient } = await loadFixture(deployFixture);
      await usdc.connect(user1).approve(await vault.getAddress(), ethers.parseUnits("100000", 6));
      await vault.connect(user1).deposit(ethers.parseUnits("100000", 6), user1.address);
      await time.increase(365 * 86400);
      await usdc.connect(user1).approve(await vault.getAddress(), 1);
      await vault.connect(user1).deposit(1, user1.address);
      expect(await vault.balanceOf(feeRecipient.address)).to.be.gt(0);
    });

    it("should reject excessive fee parameters", async function () {
      const { vault, admin } = await loadFixture(deployFixture);
      await expect(vault.connect(admin).setFees(600, 1000, 30)).to.be.revertedWith("Mgmt fee > 5%");
      await expect(vault.connect(admin).setFees(100, 3500, 30)).to.be.revertedWith("Perf fee > 30%");
    });
  });

  describe("Epoch", function () {
    it("should advance after duration expires", async function () {
      const { vault } = await loadFixture(deployFixture);
      await time.increase(2 * 86400);
      await vault.advanceEpoch();
      expect(await vault.currentEpoch()).to.equal(2);
    });

    it("should revert if epoch not ended", async function () {
      const { vault } = await loadFixture(deployFixture);
      await expect(vault.advanceEpoch()).to.be.revertedWith("Current epoch not ended");
    });

    it("should reset redemption counter", async function () {
      const { vault, usdc, user1 } = await loadFixture(deployFixture);
      await usdc.connect(user1).approve(await vault.getAddress(), ethers.parseUnits("100000", 6));
      await vault.connect(user1).deposit(ethers.parseUnits("100000", 6), user1.address);
      await time.increase(8 * 86400);
      await vault.connect(user1).withdraw(ethers.parseUnits("5000", 6), user1.address, user1.address);
      await time.increase(86400);
      await vault.advanceEpoch();
      expect((await vault.getVaultMetrics()).gateUsed).to.equal(0);
    });
  });

  describe("Weight Commitment", function () {
    it("should accept commitment from keeper", async function () {
      const { vault, keeper } = await loadFixture(deployFixture);
      const root = ethers.keccak256(ethers.toUtf8Bytes("weights"));
      await vault.connect(keeper).commitWeights(root);
      expect(await vault.commitPending()).to.be.true;
    });

    it("should reject from non-keeper", async function () {
      const { vault, user1 } = await loadFixture(deployFixture);
      await expect(vault.connect(user1).commitWeights(ethers.keccak256(ethers.toUtf8Bytes("x")))).to.be.reverted;
    });

    it("should enforce timelock", async function () {
      const { vault, keeper, wbtc, weth, sol, usdc } = await loadFixture(deployFixture);
      const tokens = [await wbtc.getAddress(), await weth.getAddress(), await sol.getAddress(), await usdc.getAddress()];
      const newW = [2500, 2500, 2500, 2500];
      const leaf = ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(["address[]", "uint256[]"], [tokens, newW]));
      await vault.connect(keeper).commitWeights(leaf);
      await expect(vault.connect(keeper).executeWeights(tokens, newW, [])).to.be.revertedWith("Timelock not elapsed");
    });
  });

  describe("Access Control", function () {
    it("should restrict admin functions", async function () {
      const { vault, user1, btcFeed } = await loadFixture(deployFixture);
      await expect(vault.connect(user1).addConstituent(ethers.ZeroAddress, await btcFeed.getAddress(), 1000)).to.be.reverted;
      await expect(vault.connect(user1).pause()).to.be.reverted;
      await expect(vault.connect(user1).setFees(50, 500, 10)).to.be.reverted;
    });
  });

  describe("View Functions", function () {
    it("should return correct metrics and weights", async function () {
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
  });

  describe("Full Lifecycle", function () {
    it("deposit → time → fee accrual → epoch advance → withdraw", async function () {
      const { vault, usdc, user1, feeRecipient } = await loadFixture(deployFixture);
      const dep = ethers.parseUnits("50000", 6);
      await usdc.connect(user1).approve(await vault.getAddress(), dep);
      await vault.connect(user1).deposit(dep, user1.address);
      const initialShares = await vault.balanceOf(user1.address);

      await time.increase(180 * 86400);
      await time.increase(86400);
      await vault.advanceEpoch();

      const gateMax = (await vault.totalAssets() * 2000n) / 10000n;
      const maxW = await vault.maxWithdraw(user1.address);
      const withdrawAmt = maxW < gateMax ? maxW : gateMax;
      await vault.connect(user1).withdraw(withdrawAmt, user1.address, user1.address);

      expect(await vault.balanceOf(feeRecipient.address)).to.be.gt(0);
      expect(await vault.balanceOf(user1.address)).to.be.lt(initialShares);
    });
  });
});
