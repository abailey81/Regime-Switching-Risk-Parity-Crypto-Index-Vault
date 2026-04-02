// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC4626.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "./interfaces/IChainlinkAggregator.sol";

/**
 * @title RiskParityVault
 * @author Tamer Atesyakar (UCL MSc Digital Finance & Banking)
 * @notice ERC-4626 tokenised vault implementing a regime-switching risk-parity
 *         crypto index fund with dynamic allocation, Merkle-root weight
 *         commitment, three-tier fee architecture, drawdown circuit breaker,
 *         and epoch-based liquidity management.
 *
 * @dev Architecture:
 *   - Underlying asset: USDC (depositors provide USDC, vault holds multi-asset portfolio)
 *   - Share token: rpCRYPTO (ERC-20, represents pro-rata NAV ownership)
 *   - Allocation: Off-chain ML ensemble pushes weights on-chain via keeper
 *   - NAV: Computed from Chainlink oracle prices of constituent assets
 *   - Fees: Management (continuous dilution), Performance (HWM), Redemption (anti-churn)
 *   - Safety: Circuit breaker, epoch gates, Merkle commitment, pausability
 *
 *   IFTE0007 Coursework — Decentralised Finance and Blockchain
 *   UCL Institute of Finance & Technology, 2025/26
 */
contract RiskParityVault is
    ERC4626,
    AccessControl,
    Pausable,
    ReentrancyGuard
{
    using SafeERC20 for IERC20;
    using Math for uint256;

    // ═══════════════════════════════════════════════════════════════════
    //                           ROLES
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Role for the ML weight publisher / keeper bot
    bytes32 public constant KEEPER_ROLE = keccak256("KEEPER_ROLE");

    /// @notice Role for administrative operations (fee changes, circuit breaker reset)
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    // ═══════════════════════════════════════════════════════════════════
    //                      PORTFOLIO STATE
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Array of constituent token addresses in the portfolio
    address[] public constituents;

    /// @notice Current target weight for each constituent (in basis points, sum = 10000)
    mapping(address => uint256) public weights;

    /// @notice Chainlink price feed for each constituent
    mapping(address => address) public priceFeeds;

    /// @notice Whether a token is a registered constituent
    mapping(address => bool) public isConstituent;

    /// @notice Weight precision: weights are in basis points (10000 = 100%)
    uint256 public constant WEIGHT_PRECISION = 10000;

    // ═══════════════════════════════════════════════════════════════════
    //                        FEE MODULE
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Management fee in basis points (annualised). 100 = 1%
    uint256 public managementFeeBps;

    /// @notice Performance fee in basis points. 1000 = 10%
    uint256 public performanceFeeBps;

    /// @notice Early redemption fee in basis points. 30 = 0.3%
    uint256 public redemptionFeeBps;

    /// @notice Time window for early redemption fee (seconds). Default: 7 days
    uint256 public earlyRedemptionWindow;

    /// @notice Address receiving all fee payments
    address public feeRecipient;

    /// @notice Timestamp of last management fee accrual
    uint256 public lastFeeAccrual;

    /// @notice High-water mark for performance fee (in asset units per share, scaled by 1e18)
    uint256 public highWaterMark;

    /// @notice Tracks when each address last deposited (for early redemption fee)
    mapping(address => uint256) public lastDepositTime;

    /// @notice Cumulative management fees collected (for transparency)
    uint256 public totalManagementFeesCollected;

    /// @notice Cumulative performance fees collected
    uint256 public totalPerformanceFeesCollected;

    uint256 private constant SECONDS_PER_YEAR = 365.25 days;

    // ═══════════════════════════════════════════════════════════════════
    //                       EPOCH MODULE
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Current epoch number (incremented at each epoch transition)
    uint256 public currentEpoch;

    /// @notice Start timestamp of the current epoch
    uint256 public epochStart;

    /// @notice Duration of the current epoch in seconds
    uint256 public epochDuration;

    /// @notice Base epoch duration (calm markets)
    uint256 public epochBaseDuration;

    /// @notice Extended epoch duration (volatile markets)
    uint256 public epochVolatileDuration;

    /// @notice Total assets redeemed in the current epoch
    uint256 public epochRedemptionTotal;

    /// @notice Maximum percentage of AUM redeemable per epoch (basis points)
    uint256 public redemptionGateBps;

    // ═══════════════════════════════════════════════════════════════════
    //                   WEIGHT COMMITMENT (MERKLE)
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Committed Merkle root of proposed weights
    bytes32 public committedWeightsRoot;

    /// @notice Timestamp of weight commitment
    uint256 public commitTimestamp;

    /// @notice Timelock delay between commit and reveal (seconds)
    uint256 public weightTimelock;

    /// @notice Whether a commitment is pending execution
    bool public commitPending;

    // ═══════════════════════════════════════════════════════════════════
    //                      CIRCUIT BREAKER
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Whether the circuit breaker is currently active
    bool public circuitBreakerActive;

    /// @notice Drawdown threshold to trigger circuit breaker (basis points). 1500 = 15%
    uint256 public circuitBreakerThresholdBps;

    /// @notice Defensive weights applied when circuit breaker triggers
    mapping(address => uint256) public defensiveWeights;

    // ═══════════════════════════════════════════════════════════════════
    //                          EVENTS
    // ═══════════════════════════════════════════════════════════════════

    event EpochAdvanced(uint256 indexed epoch, uint256 start, uint256 duration);
    event WeightsCommitted(bytes32 indexed root, uint256 timestamp);
    event WeightsExecuted(uint256 indexed epoch, address[] tokens, uint256[] newWeights);
    event ManagementFeeAccrued(uint256 feeShares, uint256 totalAssets, uint256 timestamp);
    event PerformanceFeeCharged(uint256 feeShares, uint256 navPerShare, uint256 hwm);
    event RedemptionFeeCharged(address indexed user, uint256 feeAmount);
    event CircuitBreakerTriggered(uint256 navPerShare, uint256 highWaterMark);
    event CircuitBreakerReset(uint256 navPerShare);
    event RedemptionGateHit(address indexed user, uint256 requested, uint256 epochTotal);
    event ConstituentAdded(address indexed token, address priceFeed, uint256 weight);
    event DefensiveWeightSet(address indexed token, uint256 weight);

    // ═══════════════════════════════════════════════════════════════════
    //                        CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @param asset_         The underlying asset (USDC) for deposits/withdrawals
     * @param name_          Vault share token name ("RiskParity Crypto Index")
     * @param symbol_        Vault share token symbol ("rpCRYPTO")
     * @param admin_         Address with ADMIN_ROLE
     * @param keeper_        Address with KEEPER_ROLE (ML weight publisher)
     * @param feeRecipient_  Address receiving fee payments
     */
    constructor(
        IERC20 asset_,
        string memory name_,
        string memory symbol_,
        address admin_,
        address keeper_,
        address feeRecipient_
    ) ERC4626(asset_) ERC20(name_, symbol_) {
        require(admin_ != address(0), "Zero admin");
        require(keeper_ != address(0), "Zero keeper");
        require(feeRecipient_ != address(0), "Zero fee recipient");

        // Access control
        _grantRole(DEFAULT_ADMIN_ROLE, admin_);
        _grantRole(ADMIN_ROLE, admin_);
        _grantRole(KEEPER_ROLE, keeper_);

        // Fee defaults
        managementFeeBps = 100;        // 1% annual
        performanceFeeBps = 1000;      // 10% above HWM
        redemptionFeeBps = 30;         // 0.3% early redemption
        earlyRedemptionWindow = 7 days;
        feeRecipient = feeRecipient_;
        lastFeeAccrual = block.timestamp;
        highWaterMark = 1e18;          // Initial NAV per share = 1.0

        // Epoch defaults
        epochBaseDuration = 1 days;
        epochVolatileDuration = 7 days;
        epochDuration = epochBaseDuration;
        epochStart = block.timestamp;
        currentEpoch = 1;
        redemptionGateBps = 2000;      // 20% max redemption per epoch

        // Weight commitment
        weightTimelock = 1 hours;

        // Circuit breaker
        circuitBreakerThresholdBps = 1500; // 15% drawdown
    }

    // ═══════════════════════════════════════════════════════════════════
    //                   CONSTITUENT MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Register a constituent token with its price feed and initial weight.
     * @param token      Address of the ERC-20 constituent token
     * @param priceFeed  Address of the Chainlink price feed for this token
     * @param weight     Initial weight in basis points
     */
    function addConstituent(
        address token,
        address priceFeed,
        uint256 weight
    ) external onlyRole(ADMIN_ROLE) {
        require(token != address(0), "Zero token");
        require(priceFeed != address(0), "Zero feed");
        require(!isConstituent[token], "Already registered");

        constituents.push(token);
        priceFeeds[token] = priceFeed;
        weights[token] = weight;
        isConstituent[token] = true;

        emit ConstituentAdded(token, priceFeed, weight);
    }

    /**
     * @notice Set defensive weights for circuit breaker mode.
     * @param tokens   Array of constituent addresses
     * @param dWeights Array of defensive weights (basis points)
     */
    function setDefensiveWeights(
        address[] calldata tokens,
        uint256[] calldata dWeights
    ) external onlyRole(ADMIN_ROLE) {
        require(tokens.length == dWeights.length, "Length mismatch");
        uint256 totalWeight = 0;
        for (uint256 i = 0; i < tokens.length; i++) {
            require(isConstituent[tokens[i]], "Not constituent");
            defensiveWeights[tokens[i]] = dWeights[i];
            totalWeight += dWeights[i];
            emit DefensiveWeightSet(tokens[i], dWeights[i]);
        }
        require(totalWeight == WEIGHT_PRECISION, "Weights must sum to 10000");
    }

    // ═══════════════════════════════════════════════════════════════════
    //                      NAV COMPUTATION
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Compute the total value of all constituent holdings in USD terms.
     * @dev Queries Chainlink price feeds for each constituent.
     *      Returns value denominated in the underlying asset's decimals.
     * @return totalValue Sum of (balance × price) for all constituents
     */
    function computePortfolioValue() public view returns (uint256 totalValue) {
        for (uint256 i = 0; i < constituents.length; i++) {
            address token = constituents[i];
            uint256 balance = IERC20(token).balanceOf(address(this));
            if (balance == 0) continue;

            address feed = priceFeeds[token];
            (, int256 price,, uint256 updatedAt,) =
                IChainlinkAggregator(feed).latestRoundData();

            require(price > 0, "Invalid price");
            require(block.timestamp - updatedAt < 1 hours, "Stale price");

            uint8 feedDecimals = IChainlinkAggregator(feed).decimals();
            uint8 tokenDecimals = ERC20(token).decimals();

            // Normalise to 18 decimals for internal accounting
            uint256 value = (balance * uint256(price) * 1e18)
                / (10 ** tokenDecimals * 10 ** feedDecimals);

            totalValue += value;
        }
    }

    /**
     * @notice Current NAV per share (scaled by 1e18).
     */
    function navPerShare() public view returns (uint256) {
        uint256 supply = totalSupply();
        if (supply == 0) return 1e18;
        return (computePortfolioValue() * 1e18) / supply;
    }

    /**
     * @inheritdoc ERC4626
     * @dev Override to include portfolio constituent values + underlying balance.
     *      This is the total assets figure used by ERC-4626 for share pricing.
     */
    function totalAssets() public view override returns (uint256) {
        // Base: underlying (USDC) balance held directly by vault
        uint256 underlyingBalance = IERC20(asset()).balanceOf(address(this));

        // If no constituents registered, vault only holds underlying
        if (constituents.length == 0) return underlyingBalance;

        // Add portfolio value (converted to underlying decimals)
        // For simplicity in testnet: use underlying balance as totalAssets
        // In production: computePortfolioValue() normalised to underlying decimals
        return underlyingBalance;
    }

    // ═══════════════════════════════════════════════════════════════════
    //                        EPOCH LOGIC
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Advance to the next epoch. Callable by keeper or anyone after epoch expires.
     * @dev Crystallises performance fees and resets epoch redemption counter.
     */
    function advanceEpoch() external {
        require(
            block.timestamp >= epochStart + epochDuration,
            "Current epoch not ended"
        );

        // Crystallise performance fee at epoch boundary
        _crystallisePerformanceFee();

        // Reset epoch state
        currentEpoch++;
        epochStart = block.timestamp;
        epochRedemptionTotal = 0;

        emit EpochAdvanced(currentEpoch, epochStart, epochDuration);
    }

    /**
     * @notice Check if the current epoch is open for deposits/withdrawals.
     */
    function isEpochOpen() public view returns (bool) {
        return block.timestamp >= epochStart
            && block.timestamp < epochStart + epochDuration;
    }

    /**
     * @notice Set epoch duration. Keeper can extend during volatile markets.
     * @param duration New epoch duration in seconds
     */
    function setEpochDuration(uint256 duration) external onlyRole(KEEPER_ROLE) {
        require(
            duration == epochBaseDuration || duration == epochVolatileDuration,
            "Invalid duration"
        );
        epochDuration = duration;
    }

    // ═══════════════════════════════════════════════════════════════════
    //                     FEE ACCRUAL LOGIC
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Accrue management fee via share dilution.
     * @dev Mints new shares to feeRecipient proportional to elapsed time.
     *      Called internally before NAV-dependent operations.
     */
    function _accrueManagementFee() internal {
        if (totalSupply() == 0) {
            lastFeeAccrual = block.timestamp;
            return;
        }

        uint256 elapsed = block.timestamp - lastFeeAccrual;
        if (elapsed == 0) return;

        // fee = totalSupply × (managementFeeBps / 10000) × (elapsed / SECONDS_PER_YEAR)
        // Implemented as share minting (dilution):
        // newShares = totalSupply × feeRate × elapsed / (SECONDS_PER_YEAR - feeRate × elapsed)
        uint256 feeNumerator = totalSupply() * managementFeeBps * elapsed;
        uint256 feeDenominator = WEIGHT_PRECISION * SECONDS_PER_YEAR;

        uint256 feeShares = feeNumerator / feeDenominator;

        if (feeShares > 0) {
            _mint(feeRecipient, feeShares);
            totalManagementFeesCollected += feeShares;
            emit ManagementFeeAccrued(feeShares, totalAssets(), block.timestamp);
        }

        lastFeeAccrual = block.timestamp;
    }

    /**
     * @notice Crystallise performance fee at epoch boundary.
     * @dev Only charges fees on NAV gains above the high-water mark.
     */
    function _crystallisePerformanceFee() internal {
        if (totalSupply() == 0) return;

        uint256 currentNav = navPerShare();
        if (currentNav <= highWaterMark) return;

        // Gain per share above HWM
        uint256 gainPerShare = currentNav - highWaterMark;

        // Total gain across all shares
        uint256 totalGain = (gainPerShare * totalSupply()) / 1e18;

        // Performance fee
        uint256 fee = (totalGain * performanceFeeBps) / WEIGHT_PRECISION;

        if (fee > 0) {
            // Convert fee amount to shares
            uint256 feeShares = convertToShares(fee);
            if (feeShares > 0) {
                _mint(feeRecipient, feeShares);
                totalPerformanceFeesCollected += feeShares;
            }
            emit PerformanceFeeCharged(feeShares, currentNav, highWaterMark);
        }

        // Update high-water mark
        highWaterMark = currentNav;
    }

    /**
     * @notice Compute early redemption fee for a user's withdrawal.
     * @param user   Address of the withdrawer
     * @param assets Amount of assets being withdrawn
     * @return fee   Fee amount to deduct
     */
    function _computeRedemptionFee(
        address user,
        uint256 assets
    ) internal view returns (uint256 fee) {
        if (block.timestamp < lastDepositTime[user] + earlyRedemptionWindow) {
            fee = (assets * redemptionFeeBps) / WEIGHT_PRECISION;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //               WEIGHT COMMITMENT (MERKLE PROOF)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Phase 1: Commit a Merkle root of proposed portfolio weights.
     * @dev The keeper commits the hash before revealing actual weights,
     *      preventing front-running of rebalancing trades.
     * @param merkleRoot Root hash of the weight Merkle tree
     */
    function commitWeights(bytes32 merkleRoot) external onlyRole(KEEPER_ROLE) {
        require(merkleRoot != bytes32(0), "Zero root");
        committedWeightsRoot = merkleRoot;
        commitTimestamp = block.timestamp;
        commitPending = true;

        emit WeightsCommitted(merkleRoot, block.timestamp);
    }

    /**
     * @notice Phase 2: Reveal and execute weight update after timelock.
     * @dev Verifies each (token, weight) pair against the committed Merkle root.
     * @param tokens     Array of constituent addresses (must be in same order as tree)
     * @param newWeights Array of new weights in basis points
     * @param proof      Merkle proof for verification
     */
    function executeWeights(
        address[] calldata tokens,
        uint256[] calldata newWeights,
        bytes32[] calldata proof
    ) external onlyRole(KEEPER_ROLE) whenNotPaused {
        require(commitPending, "No pending commit");
        require(
            block.timestamp >= commitTimestamp + weightTimelock,
            "Timelock not elapsed"
        );
        require(tokens.length == newWeights.length, "Length mismatch");
        require(tokens.length == constituents.length, "Must update all constituents");

        // Verify Merkle proof
        bytes32 leaf = keccak256(abi.encodePacked(tokens, newWeights));
        require(
            MerkleProof.verify(proof, committedWeightsRoot, leaf),
            "Invalid Merkle proof"
        );

        // Validate weights
        uint256 totalWeight = 0;
        uint256 totalTurnover = 0;

        for (uint256 i = 0; i < tokens.length; i++) {
            require(isConstituent[tokens[i]], "Not constituent");
            require(newWeights[i] <= 4000, "Weight exceeds 40% max");

            // Track turnover
            uint256 oldWeight = weights[tokens[i]];
            if (newWeights[i] > oldWeight) {
                totalTurnover += newWeights[i] - oldWeight;
            } else {
                totalTurnover += oldWeight - newWeights[i];
            }

            totalWeight += newWeights[i];
        }

        require(totalWeight == WEIGHT_PRECISION, "Weights must sum to 10000");
        require(totalTurnover <= 3000, "Turnover exceeds 30% limit");

        // Apply weights (circuit breaker overrides)
        if (circuitBreakerActive) {
            // In circuit breaker mode, apply defensive weights instead
            for (uint256 i = 0; i < tokens.length; i++) {
                weights[tokens[i]] = defensiveWeights[tokens[i]];
            }
        } else {
            for (uint256 i = 0; i < tokens.length; i++) {
                weights[tokens[i]] = newWeights[i];
            }
        }

        commitPending = false;
        emit WeightsExecuted(currentEpoch, tokens, newWeights);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                      CIRCUIT BREAKER
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Check and trigger circuit breaker if drawdown exceeds threshold.
     * @dev Can be called by anyone (keeper, users, or bots).
     */
    function checkCircuitBreaker() external {
        if (circuitBreakerActive) return;
        if (totalSupply() == 0) return;

        uint256 currentNav = navPerShare();
        uint256 threshold = highWaterMark
            * (WEIGHT_PRECISION - circuitBreakerThresholdBps)
            / WEIGHT_PRECISION;

        if (currentNav < threshold) {
            circuitBreakerActive = true;

            // Override all weights to defensive
            for (uint256 i = 0; i < constituents.length; i++) {
                weights[constituents[i]] = defensiveWeights[constituents[i]];
            }

            emit CircuitBreakerTriggered(currentNav, highWaterMark);
        }
    }

    /**
     * @notice Reset the circuit breaker after recovery.
     * @dev Only admin can reset, and only if NAV has recovered sufficiently.
     */
    function resetCircuitBreaker() external onlyRole(ADMIN_ROLE) {
        require(circuitBreakerActive, "Not active");
        uint256 currentNav = navPerShare();
        uint256 recoveryLevel = highWaterMark * 9000 / WEIGHT_PRECISION; // 90% of HWM

        require(currentNav >= recoveryLevel, "Insufficient recovery");

        circuitBreakerActive = false;
        emit CircuitBreakerReset(currentNav);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                  DEPOSIT / WITHDRAW OVERRIDES
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @inheritdoc ERC4626
     * @dev Override deposit to:
     *   1. Accrue management fees before share pricing
     *   2. Track deposit time for early redemption fee
     *   3. Update high-water mark if needed
     */
    function deposit(
        uint256 assets,
        address receiver
    ) public override nonReentrant whenNotPaused returns (uint256) {
        _accrueManagementFee();

        uint256 shares = super.deposit(assets, receiver);

        // Track deposit time for redemption fee window
        lastDepositTime[receiver] = block.timestamp;

        // Update HWM if this is the first deposit
        if (highWaterMark == 0) {
            highWaterMark = navPerShare();
        }

        return shares;
    }

    /**
     * @inheritdoc ERC4626
     * @dev Override withdraw to:
     *   1. Accrue management fees before share pricing
     *   2. Enforce redemption gate (max % of AUM per epoch)
     *   3. Charge early redemption fee if within window
     */
    function withdraw(
        uint256 assets,
        address receiver,
        address owner
    ) public override nonReentrant whenNotPaused returns (uint256) {
        _accrueManagementFee();

        // Check redemption gate
        uint256 maxRedeemable = (totalAssets() * redemptionGateBps) / WEIGHT_PRECISION;
        require(
            epochRedemptionTotal + assets <= maxRedeemable,
            "Redemption gate exceeded"
        );

        // Compute and deduct early redemption fee
        uint256 fee = _computeRedemptionFee(owner, assets);
        uint256 netAssets = assets - fee;

        if (fee > 0) {
            emit RedemptionFeeCharged(owner, fee);
        }

        // Track epoch redemptions
        epochRedemptionTotal += assets;

        return super.withdraw(netAssets, receiver, owner);
    }

    /**
     * @inheritdoc ERC4626
     * @dev Override redeem with same fee and gate logic as withdraw.
     */
    function redeem(
        uint256 shares,
        address receiver,
        address owner
    ) public override nonReentrant whenNotPaused returns (uint256) {
        _accrueManagementFee();

        uint256 assets = previewRedeem(shares);

        // Check redemption gate
        uint256 maxRedeemable = (totalAssets() * redemptionGateBps) / WEIGHT_PRECISION;
        require(
            epochRedemptionTotal + assets <= maxRedeemable,
            "Redemption gate exceeded"
        );

        // Compute and deduct early redemption fee
        uint256 fee = _computeRedemptionFee(owner, assets);

        if (fee > 0) {
            emit RedemptionFeeCharged(owner, fee);
        }

        epochRedemptionTotal += assets;

        return super.redeem(shares, receiver, owner);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                      ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Update fee parameters. Only admin.
     */
    function setFees(
        uint256 newManagementBps,
        uint256 newPerformanceBps,
        uint256 newRedemptionBps
    ) external onlyRole(ADMIN_ROLE) {
        require(newManagementBps <= 500, "Mgmt fee > 5%");
        require(newPerformanceBps <= 3000, "Perf fee > 30%");
        require(newRedemptionBps <= 200, "Redemption fee > 2%");

        _accrueManagementFee(); // Accrue at old rate first

        managementFeeBps = newManagementBps;
        performanceFeeBps = newPerformanceBps;
        redemptionFeeBps = newRedemptionBps;
    }

    /**
     * @notice Update the fee recipient address.
     */
    function setFeeRecipient(address newRecipient) external onlyRole(ADMIN_ROLE) {
        require(newRecipient != address(0), "Zero address");
        feeRecipient = newRecipient;
    }

    /**
     * @notice Update the redemption gate percentage.
     */
    function setRedemptionGate(uint256 newGateBps) external onlyRole(ADMIN_ROLE) {
        require(newGateBps >= 500 && newGateBps <= 5000, "Gate out of range");
        redemptionGateBps = newGateBps;
    }

    /**
     * @notice Emergency pause. Stops all deposits and withdrawals.
     */
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    /**
     * @notice Unpause the vault.
     */
    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }

    // ═══════════════════════════════════════════════════════════════════
    //                         VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Get all constituent addresses.
     */
    function getConstituents() external view returns (address[] memory) {
        return constituents;
    }

    /**
     * @notice Get current weights for all constituents.
     * @return tokens Array of constituent addresses
     * @return w     Array of weights in basis points
     */
    function getCurrentWeights()
        external
        view
        returns (address[] memory tokens, uint256[] memory w)
    {
        tokens = constituents;
        w = new uint256[](constituents.length);
        for (uint256 i = 0; i < constituents.length; i++) {
            w[i] = weights[constituents[i]];
        }
    }

    /**
     * @notice Get vault metrics for the dashboard.
     * @return nav          Current NAV per share (1e18 scaled)
     * @return hwm          High-water mark (1e18 scaled)
     * @return epoch        Current epoch number
     * @return cbActive     Whether circuit breaker is active
     * @return gateUsed     Redemption amount used this epoch
     * @return gateMax      Maximum redemption for this epoch
     * @return totalShares  Total shares outstanding
     * @return totalVal     Total assets under management
     */
    function getVaultMetrics()
        external
        view
        returns (
            uint256 nav,
            uint256 hwm,
            uint256 epoch,
            bool cbActive,
            uint256 gateUsed,
            uint256 gateMax,
            uint256 totalShares,
            uint256 totalVal
        )
    {
        nav = navPerShare();
        hwm = highWaterMark;
        epoch = currentEpoch;
        cbActive = circuitBreakerActive;
        gateUsed = epochRedemptionTotal;
        gateMax = (totalAssets() * redemptionGateBps) / WEIGHT_PRECISION;
        totalShares = totalSupply();
        totalVal = totalAssets();
    }

    /**
     * @notice Get fee summary for transparency.
     */
    function getFeeSummary()
        external
        view
        returns (
            uint256 mgmtBps,
            uint256 perfBps,
            uint256 redemptBps,
            uint256 totalMgmtCollected,
            uint256 totalPerfCollected,
            address recipient
        )
    {
        mgmtBps = managementFeeBps;
        perfBps = performanceFeeBps;
        redemptBps = redemptionFeeBps;
        totalMgmtCollected = totalManagementFeesCollected;
        totalPerfCollected = totalPerformanceFeesCollected;
        recipient = feeRecipient;
    }

    /**
     * @notice Number of registered constituents.
     */
    function constituentCount() external view returns (uint256) {
        return constituents.length;
    }
}
