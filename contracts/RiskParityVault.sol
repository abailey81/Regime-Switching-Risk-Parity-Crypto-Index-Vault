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
 *   Design rationale: The vault separates portfolio intelligence (off-chain ML ensemble
 *   computing GARCH-DCC, HMM, SAC RL, CVaR-optimal weights) from on-chain execution
 *   (weight verification via Merkle proofs, NAV accounting via Chainlink oracles).
 *   This hybrid architecture ensures gas efficiency while maintaining verifiability.
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
    //                        CUSTOM ERRORS
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Thrown when a Chainlink price feed returns a stale price beyond the staleness threshold
    /// @param feed The address of the Chainlink price feed that returned the stale price
    /// @param updatedAt The timestamp of the last price update from the feed
    error StalePrice(address feed, uint256 updatedAt);

    /// @notice Thrown when a Chainlink price feed returns a non-positive price
    /// @param feed The address of the Chainlink price feed
    /// @param price The invalid price value returned
    error InvalidPrice(address feed, int256 price);

    /// @notice Thrown when a redemption request exceeds the per-epoch redemption gate
    /// @param requested The total epoch redemption amount including this request
    /// @param remaining The remaining redeemable amount for the current epoch
    error RedemptionGateExceeded(uint256 requested, uint256 remaining);

    /// @notice Thrown when a proposed constituent weight exceeds the per-asset maximum
    /// @param weight The proposed weight in basis points
    /// @param max The maximum allowed weight in basis points
    error InvalidWeight(uint256 weight, uint256 max);

    /// @notice Thrown when aggregate portfolio turnover exceeds the per-rebalance limit
    /// @param turnover The computed turnover in basis points
    /// @param max The maximum allowed turnover in basis points
    error TurnoverExceeded(uint256 turnover, uint256 max);

    /// @notice Thrown when a zero address is provided where a non-zero address is required
    /// @param param Human-readable identifier for the parameter that was zero
    error ZeroAddress(string param);

    /// @notice Thrown when an operation requires a pending weight commitment but none exists
    error NoCommitPending();

    /// @notice Thrown when the weight timelock period has not yet elapsed
    /// @param commitTime The timestamp when the commitment was made
    /// @param requiredWait The timelock duration in seconds
    error TimelockNotElapsed(uint256 commitTime, uint256 requiredWait);

    /// @notice Thrown when the proposed weights do not sum to WEIGHT_PRECISION (10000 bps)
    /// @param actual The actual sum of proposed weights
    /// @param expected The expected sum (WEIGHT_PRECISION)
    error WeightSumMismatch(uint256 actual, uint256 expected);

    /// @notice Thrown when an operation references a token that is not a registered constituent
    /// @param token The address of the unregistered token
    error NotConstituent(address token);

    /// @notice Thrown when attempting to register a token that is already a constituent
    /// @param token The address of the already-registered token
    error AlreadyConstituent(address token);

    /// @notice Thrown when attempting to advance an epoch before the current one has ended
    /// @param epochEnd The timestamp when the current epoch ends
    /// @param currentTime The current block timestamp
    error EpochNotEnded(uint256 epochEnd, uint256 currentTime);

    /// @notice Thrown when the circuit breaker is not in the expected state
    /// @param expected Whether the circuit breaker was expected to be active
    error CircuitBreakerState(bool expected);

    /// @notice Thrown when NAV has not recovered sufficiently to reset the circuit breaker
    /// @param currentNav The current NAV per share
    /// @param requiredNav The minimum NAV required for reset
    error InsufficientRecovery(uint256 currentNav, uint256 requiredNav);

    // ═══════════════════════════════════════════════════════════════════
    //                           ROLES
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Role for the ML weight publisher / keeper bot that submits
    ///         portfolio weight updates from the off-chain ensemble
    bytes32 public constant KEEPER_ROLE = keccak256("KEEPER_ROLE");

    /// @notice Role for administrative operations (fee changes, circuit breaker reset,
    ///         constituent management, emergency pause)
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    // ═══════════════════════════════════════════════════════════════════
    //                      PORTFOLIO STATE
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Ordered array of constituent token addresses in the portfolio.
    /// @dev Constituents are append-only; removal is not supported to preserve
    ///      weight history integrity. New constituents are added via addConstituent().
    address[] public constituents;

    /// @notice Current target weight for each constituent in basis points (sum = 10000).
    /// @dev Updated via executeWeights() after Merkle proof verification.
    ///      During circuit breaker mode, defensive weights override these values.
    mapping(address => uint256) public weights;

    /// @notice Chainlink AggregatorV3 price feed address for each constituent.
    /// @dev Used by computePortfolioValue() to compute NAV from on-chain oracles.
    mapping(address => address) public priceFeeds;

    /// @notice Lookup mapping to verify whether a token is a registered constituent.
    /// @dev Prevents duplicate registration and validates weight update arrays.
    mapping(address => bool) public isConstituent;

    /// @notice Weight precision constant: weights are in basis points where 10000 = 100%.
    /// @dev All weight arithmetic uses this denominator for consistency.
    uint256 public constant WEIGHT_PRECISION = 10000;

    // ═══════════════════════════════════════════════════════════════════
    //                        FEE MODULE
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Management fee in basis points (annualised). Example: 100 = 1% per year.
    /// @dev Accrued continuously via share dilution in _accrueManagementFee().
    ///      Capped at 500 bps (5%) by setFees().
    uint256 public managementFeeBps;

    /// @notice Performance fee in basis points, charged on gains above the high-water mark.
    /// @dev Example: 1000 = 10%. Crystallised at epoch boundaries.
    ///      Capped at 3000 bps (30%) by setFees().
    uint256 public performanceFeeBps;

    /// @notice Early redemption fee in basis points, applied to withdrawals within
    ///         earlyRedemptionWindow of the depositor's last deposit. Example: 30 = 0.3%.
    /// @dev Anti-churn mechanism to discourage hot-money deposits.
    uint256 public redemptionFeeBps;

    /// @notice Time window in seconds during which the early redemption fee applies.
    /// @dev Default: 7 days (604800 seconds). Measured from lastDepositTime[user].
    uint256 public earlyRedemptionWindow;

    /// @notice Address receiving all fee payments (management, performance, redemption).
    /// @dev Can be updated by admin via setFeeRecipient(). Must be non-zero.
    address public feeRecipient;

    /// @notice Timestamp of the last management fee accrual event.
    /// @dev Updated each time _accrueManagementFee() runs. Used to compute elapsed time.
    uint256 public lastFeeAccrual;

    /// @notice High-water mark for performance fee in asset units per share (scaled by 1e18).
    /// @dev Performance fees are only charged on NAV gains above this mark.
    ///      Initialised to 1e18 (i.e., 1.0 NAV per share).
    uint256 public highWaterMark;

    /// @notice Records the block timestamp of each address's most recent deposit.
    /// @dev Used by _computeRedemptionFee() to determine if the early redemption
    ///      fee applies to a withdrawal.
    mapping(address => uint256) public lastDepositTime;

    /// @notice Cumulative management fees collected in share units (for transparency dashboards).
    uint256 public totalManagementFeesCollected;

    /// @notice Cumulative performance fees collected in share units.
    uint256 public totalPerformanceFeesCollected;

    /// @dev Seconds per year including leap year correction (365.25 * 86400).
    uint256 private constant SECONDS_PER_YEAR = 365.25 days;

    // ═══════════════════════════════════════════════════════════════════
    //                       EPOCH MODULE
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Current epoch number, incremented at each epoch transition.
    /// @dev Starts at 1. Used to index weight history events and gate resets.
    uint256 public currentEpoch;

    /// @notice Start timestamp of the current epoch.
    /// @dev Set to block.timestamp in constructor and at each advanceEpoch() call.
    uint256 public epochStart;

    /// @notice Duration of the current epoch in seconds.
    /// @dev Toggled between epochBaseDuration and epochVolatileDuration by the keeper
    ///      based on market regime detection from the off-chain ML ensemble.
    uint256 public epochDuration;

    /// @notice Base epoch duration for calm market regimes (default: 1 day = 86400s).
    uint256 public epochBaseDuration;

    /// @notice Extended epoch duration for volatile market regimes (default: 7 days = 604800s).
    /// @dev Longer epochs reduce rebalancing frequency during high-volatility periods,
    ///      lowering transaction costs and slippage.
    uint256 public epochVolatileDuration;

    /// @notice Whether the vault is currently in volatile epoch mode.
    /// @dev Toggled by the keeper via setEpochVolatile(). When true, epochDuration
    ///      is set to epochVolatileDuration; when false, to epochBaseDuration.
    bool public volatileEpochMode;

    /// @notice Total assets redeemed in the current epoch (in underlying asset units).
    /// @dev Reset to zero at each epoch transition. Compared against the redemption gate.
    uint256 public epochRedemptionTotal;

    /// @notice Maximum percentage of AUM redeemable per epoch in basis points.
    /// @dev Default: 2000 (20%). Enforced in withdraw() and redeem().
    uint256 public redemptionGateBps;

    // ═══════════════════════════════════════════════════════════════════
    //                   WEIGHT COMMITMENT (MERKLE)
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Committed Merkle root of proposed weights awaiting execution.
    /// @dev Set by commitWeights(), verified by executeWeights(). Prevents
    ///      front-running of rebalancing trades by separating commit from reveal.
    bytes32 public committedWeightsRoot;

    /// @notice Timestamp when the current weight commitment was made.
    /// @dev Used to enforce the timelock delay before executeWeights() can proceed.
    uint256 public commitTimestamp;

    /// @notice Timelock delay between weight commitment and execution (default: 1 hour).
    /// @dev Gives LPs time to observe the commitment and exit if they disagree
    ///      with the proposed allocation before it takes effect.
    uint256 public weightTimelock;

    /// @notice Whether a weight commitment is pending execution.
    /// @dev Set to true by commitWeights(), cleared by executeWeights() or cancelCommit().
    bool public commitPending;

    // ═══════════════════════════════════════════════════════════════════
    //                      CIRCUIT BREAKER
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Whether the circuit breaker is currently active (defensive mode).
    /// @dev When true, all weight updates are overridden with defensiveWeights
    ///      and the vault enters a risk-off posture.
    bool public circuitBreakerActive;

    /// @notice Drawdown threshold to trigger the circuit breaker in basis points.
    /// @dev Default: 1500 (15%). Triggered when NAV/share drops below
    ///      highWaterMark * (1 - threshold/10000).
    uint256 public circuitBreakerThresholdBps;

    /// @notice Defensive weights applied when the circuit breaker triggers.
    /// @dev Typically allocates heavily to stablecoins (e.g., 90% USDC).
    ///      Set by admin via setDefensiveWeights(). Must sum to WEIGHT_PRECISION.
    mapping(address => uint256) public defensiveWeights;

    // ═══════════════════════════════════════════════════════════════════
    //                     PORTFOLIO VALUATION
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Toggle for portfolio valuation in totalAssets().
    /// @dev When true (production), totalAssets() includes Chainlink-based constituent
    ///      valuations. When false (testnet), only the underlying USDC balance is used.
    ///      Default: true. Toggle via setUsePortfolioValuation().
    bool public usePortfolioValuation = true;

    // ═══════════════════════════════════════════════════════════════════
    //                          EVENTS
    // ═══════════════════════════════════════════════════════════════════

    /// @notice Emitted when a new epoch begins after the previous one expires.
    /// @param epoch The new epoch number
    /// @param start The start timestamp of the new epoch
    /// @param duration The duration of the new epoch in seconds
    event EpochAdvanced(uint256 indexed epoch, uint256 start, uint256 duration);

    /// @notice Emitted when the keeper commits a Merkle root of proposed weights.
    /// @param root The Merkle root hash of the proposed weight tree
    /// @param timestamp The block timestamp of the commitment
    event WeightsCommitted(bytes32 indexed root, uint256 timestamp);

    /// @notice Emitted when committed weights are verified and applied to the portfolio.
    /// @param epoch The epoch during which weights were executed
    /// @param tokens The array of constituent addresses updated
    /// @param newWeights The array of new weights in basis points
    event WeightsExecuted(uint256 indexed epoch, address[] tokens, uint256[] newWeights);

    /// @notice Emitted when the keeper cancels a pending weight commitment.
    /// @param root The Merkle root that was cancelled
    /// @param timestamp The block timestamp of the cancellation
    event WeightCommitCancelled(bytes32 indexed root, uint256 timestamp);

    /// @notice Emitted when management fees are accrued via share dilution.
    /// @param feeShares Number of new shares minted to the fee recipient
    /// @param totalAssets_ Total assets at the time of accrual
    /// @param timestamp Block timestamp of the accrual
    event ManagementFeeAccrued(uint256 feeShares, uint256 totalAssets_, uint256 timestamp);

    /// @notice Emitted when performance fees are crystallised at an epoch boundary.
    /// @param feeShares Number of new shares minted to the fee recipient
    /// @param navPerShare_ The NAV per share at crystallisation
    /// @param hwm The high-water mark at the time of charging
    event PerformanceFeeCharged(uint256 feeShares, uint256 navPerShare_, uint256 hwm);

    /// @notice Emitted when an early redemption fee is deducted from a withdrawal.
    /// @param user The address of the withdrawer being charged
    /// @param feeAmount The fee amount deducted in underlying asset units
    event RedemptionFeeCharged(address indexed user, uint256 feeAmount);

    /// @notice Emitted when the circuit breaker triggers due to excessive drawdown.
    /// @param navPerShare_ The NAV per share that triggered the breaker
    /// @param highWaterMark_ The high-water mark at the time of triggering
    event CircuitBreakerTriggered(uint256 navPerShare_, uint256 highWaterMark_);

    /// @notice Emitted when the circuit breaker is reset after sufficient NAV recovery.
    /// @param navPerShare_ The NAV per share at the time of reset
    event CircuitBreakerReset(uint256 navPerShare_);

    /// @notice Emitted when a redemption request approaches or hits the epoch gate limit.
    /// @param user The address whose redemption hit the gate
    /// @param requested The total epoch redemption amount including this request
    /// @param epochTotal The epoch redemption total after this request
    event RedemptionGateHit(address indexed user, uint256 requested, uint256 epochTotal);

    /// @notice Emitted when a new constituent token is registered in the portfolio.
    /// @param token The address of the newly added ERC-20 constituent
    /// @param priceFeed The Chainlink price feed address for the constituent
    /// @param weight The initial weight in basis points
    event ConstituentAdded(address indexed token, address priceFeed, uint256 weight);

    /// @notice Emitted when defensive weights are set or updated for a constituent.
    /// @param token The constituent address
    /// @param weight The defensive weight in basis points
    event DefensiveWeightSet(address indexed token, uint256 weight);

    /// @notice Emitted when the epoch mode switches between calm and volatile.
    /// @param volatile_ Whether the vault is now in volatile epoch mode
    /// @param newDuration The new epoch duration in seconds
    event EpochModeChanged(bool volatile_, uint256 newDuration);

    /// @notice Emitted when the portfolio valuation toggle is changed.
    /// @param enabled Whether portfolio valuation is now enabled
    event PortfolioValuationToggled(bool enabled);

    // ═══════════════════════════════════════════════════════════════════
    //                        CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Deploy a new RiskParityVault with the given configuration.
     * @dev Initialises all fee parameters, epoch settings, weight commitment
     *      timelock, and circuit breaker thresholds to sensible defaults.
     *      The deployer must grant KEEPER_ROLE to the ML weight publisher bot
     *      and ADMIN_ROLE to the fund administrator.
     * @param asset_         The underlying asset (USDC) for deposits/withdrawals
     * @param name_          Vault share token name ("RiskParity Crypto Index")
     * @param symbol_        Vault share token symbol ("rpCRYPTO")
     * @param admin_         Address with ADMIN_ROLE (must be non-zero)
     * @param keeper_        Address with KEEPER_ROLE — ML weight publisher (must be non-zero)
     * @param feeRecipient_  Address receiving fee payments (must be non-zero)
     */
    constructor(
        IERC20 asset_,
        string memory name_,
        string memory symbol_,
        address admin_,
        address keeper_,
        address feeRecipient_
    ) ERC4626(asset_) ERC20(name_, symbol_) {
        if (admin_ == address(0)) revert ZeroAddress("admin");
        if (keeper_ == address(0)) revert ZeroAddress("keeper");
        if (feeRecipient_ == address(0)) revert ZeroAddress("feeRecipient");

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
     * @notice Register a constituent token with its Chainlink price feed and initial weight.
     * @dev Only callable by ADMIN_ROLE. Constituents are append-only.
     *      The token must not already be registered, and both addresses must be non-zero.
     *      Note: adding a constituent does NOT automatically rebalance the portfolio;
     *      weights for all constituents must still sum to WEIGHT_PRECISION after a
     *      subsequent executeWeights() call.
     * @param token      Address of the ERC-20 constituent token to add
     * @param priceFeed  Address of the Chainlink AggregatorV3 price feed for this token
     * @param weight     Initial weight in basis points (e.g., 3000 = 30%)
     */
    function addConstituent(
        address token,
        address priceFeed,
        uint256 weight
    ) external onlyRole(ADMIN_ROLE) {
        if (token == address(0)) revert ZeroAddress("token");
        if (priceFeed == address(0)) revert ZeroAddress("priceFeed");
        if (isConstituent[token]) revert AlreadyConstituent(token);

        constituents.push(token);
        priceFeeds[token] = priceFeed;
        weights[token] = weight;
        isConstituent[token] = true;

        emit ConstituentAdded(token, priceFeed, weight);
    }

    /**
     * @notice Set defensive weights applied when the circuit breaker triggers.
     * @dev Only callable by ADMIN_ROLE. Defensive weights must sum to WEIGHT_PRECISION
     *      (10000 bps). These override normal weights during drawdown events to
     *      shift the portfolio to a risk-off posture (typically heavy stablecoin allocation).
     * @param tokens   Array of constituent addresses (must all be registered)
     * @param dWeights Array of defensive weights in basis points (must sum to 10000)
     */
    function setDefensiveWeights(
        address[] calldata tokens,
        uint256[] calldata dWeights
    ) external onlyRole(ADMIN_ROLE) {
        require(tokens.length == dWeights.length, "Length mismatch");
        uint256 totalWeight = 0;
        for (uint256 i = 0; i < tokens.length; i++) {
            if (!isConstituent[tokens[i]]) revert NotConstituent(tokens[i]);
            defensiveWeights[tokens[i]] = dWeights[i];
            totalWeight += dWeights[i];
            emit DefensiveWeightSet(tokens[i], dWeights[i]);
        }
        if (totalWeight != WEIGHT_PRECISION) revert WeightSumMismatch(totalWeight, WEIGHT_PRECISION);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                      NAV COMPUTATION
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Compute the total value of all constituent holdings in USD terms.
     * @dev Iterates through all registered constituents, queries each token's
     *      balance held by the vault, fetches the latest Chainlink price, and
     *      normalises to 18-decimal internal accounting. Reverts if any price
     *      feed returns stale or invalid data.
     *
     *      Value calculation per constituent:
     *        value = (balance * price * 1e18) / (10^tokenDecimals * 10^feedDecimals)
     *
     * @return totalValue Sum of (balance x price) for all constituents, in 18-decimal USD
     */
    function computePortfolioValue() public view returns (uint256 totalValue) {
        for (uint256 i = 0; i < constituents.length; i++) {
            address token = constituents[i];
            uint256 balance = IERC20(token).balanceOf(address(this));
            if (balance == 0) continue;

            address feed = priceFeeds[token];
            (, int256 price,, uint256 updatedAt,) =
                IChainlinkAggregator(feed).latestRoundData();

            if (price <= 0) revert InvalidPrice(feed, price);
            if (block.timestamp - updatedAt >= 1 hours) revert StalePrice(feed, updatedAt);

            uint8 feedDecimals = IChainlinkAggregator(feed).decimals();
            uint8 tokenDecimals = ERC20(token).decimals();

            // Normalise to 18 decimals for internal accounting
            uint256 value = (balance * uint256(price) * 1e18)
                / (10 ** tokenDecimals * 10 ** feedDecimals);

            totalValue += value;
        }
    }

    /**
     * @notice Current NAV per share scaled by 1e18 (i.e., 1.0 = 1e18).
     * @dev Returns 1e18 when no shares are outstanding (initial state).
     *      Uses computePortfolioValue() which queries Chainlink oracles.
     * @return The net asset value per share in 18-decimal precision
     */
    function navPerShare() public view returns (uint256) {
        uint256 supply = totalSupply();
        if (supply == 0) return 1e18;
        return (computePortfolioValue() * 1e18) / supply;
    }

    /**
     * @inheritdoc ERC4626
     * @notice Returns the total assets under management by the vault.
     * @dev Override to include portfolio constituent values alongside the underlying
     *      USDC balance. This is the total assets figure used by ERC-4626 for
     *      share pricing (deposit/withdraw/mint/redeem conversions).
     *
     *      When usePortfolioValuation is true (production mode), the return value
     *      includes Chainlink-derived constituent valuations converted to underlying
     *      decimals. When false (testnet mode), only the raw USDC balance is returned.
     *
     *      Portfolio value is normalised from 18-decimal internal precision to the
     *      underlying asset's decimal precision (e.g., 6 for USDC).
     * @return Total assets in underlying asset decimals (e.g., USDC with 6 decimals)
     */
    function totalAssets() public view override returns (uint256) {
        // Base: underlying (USDC) balance held directly by vault
        uint256 underlyingBalance = IERC20(asset()).balanceOf(address(this));

        // If no constituents registered or portfolio valuation disabled, use underlying only
        if (!usePortfolioValuation || constituents.length == 0) {
            return underlyingBalance;
        }

        // Production mode: add portfolio value (converted from 18-decimal to underlying decimals)
        // computePortfolioValue() returns 18-decimal USD value
        // We convert to underlying decimals (e.g., 6 for USDC)
        uint256 portfolioValue18 = computePortfolioValue();
        uint8 underlyingDecimals = ERC20(asset()).decimals();
        uint256 portfolioValueUnderlying = portfolioValue18 / (10 ** (18 - underlyingDecimals));

        return underlyingBalance + portfolioValueUnderlying;
    }

    /**
     * @notice Toggle portfolio valuation mode for totalAssets() computation.
     * @dev When enabled (true), totalAssets() includes Chainlink-derived portfolio
     *      constituent values. When disabled (false), only the underlying USDC balance
     *      is used (simpler for testnet deployment where price feeds may be unreliable).
     * @param enabled Whether to enable full portfolio valuation
     */
    function setUsePortfolioValuation(bool enabled) external onlyRole(ADMIN_ROLE) {
        usePortfolioValuation = enabled;
        emit PortfolioValuationToggled(enabled);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                        EPOCH LOGIC
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Advance to the next epoch. Callable by anyone after the current epoch expires.
     * @dev Performs the following at epoch boundary:
     *      1. Crystallises any accrued performance fees above the high-water mark
     *      2. Increments the epoch counter
     *      3. Resets the per-epoch redemption gate counter to zero
     *      This is a permissionless function — any address can trigger it once the
     *      epoch duration has elapsed, ensuring liveness even if the keeper is offline.
     */
    function advanceEpoch() external {
        if (block.timestamp < epochStart + epochDuration) {
            revert EpochNotEnded(epochStart + epochDuration, block.timestamp);
        }

        // Crystallise performance fee at epoch boundary
        _crystallisePerformanceFee();

        // Reset epoch state
        currentEpoch++;
        epochStart = block.timestamp;
        epochRedemptionTotal = 0;

        emit EpochAdvanced(currentEpoch, epochStart, epochDuration);
    }

    /**
     * @notice Check whether the current epoch is still within its active window.
     * @dev Returns true if the current block timestamp falls within [epochStart, epochStart + epochDuration).
     * @return True if the epoch is currently open, false otherwise
     */
    function isEpochOpen() public view returns (bool) {
        return block.timestamp >= epochStart
            && block.timestamp < epochStart + epochDuration;
    }

    /**
     * @notice Set epoch duration directly. Keeper can extend during volatile markets.
     * @dev Only accepts epochBaseDuration or epochVolatileDuration as valid values.
     *      For the recommended approach, use setEpochVolatile() instead which also
     *      updates the volatileEpochMode flag.
     * @param duration New epoch duration in seconds (must match base or volatile)
     */
    function setEpochDuration(uint256 duration) external onlyRole(KEEPER_ROLE) {
        require(
            duration == epochBaseDuration || duration == epochVolatileDuration,
            "Invalid duration"
        );
        epochDuration = duration;
    }

    /**
     * @notice Toggle between calm and volatile epoch durations based on market regime.
     * @dev Called by the keeper when the off-chain ML ensemble detects a regime change.
     *      Volatile mode uses longer epochs (default 7 days vs 1 day) to reduce
     *      rebalancing frequency during turbulent markets, lowering transaction costs.
     * @param volatile_ True to enter volatile mode (longer epochs), false for calm mode
     */
    function setEpochVolatile(bool volatile_) external onlyRole(KEEPER_ROLE) {
        volatileEpochMode = volatile_;
        epochDuration = volatile_ ? epochVolatileDuration : epochBaseDuration;
        emit EpochModeChanged(volatile_, epochDuration);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                     FEE ACCRUAL LOGIC
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Accrue management fee via share dilution.
     * @dev Mints new shares to feeRecipient proportional to elapsed time since
     *      the last accrual. The fee is computed as:
     *        feeShares = totalSupply * managementFeeBps * elapsed / (10000 * SECONDS_PER_YEAR)
     *
     *      This implements continuous dilution: existing shareholders' proportional
     *      ownership decreases as new fee shares are minted. Called internally before
     *      any NAV-dependent operation (deposit, withdraw, redeem).
     */
    function _accrueManagementFee() internal {
        if (totalSupply() == 0) {
            lastFeeAccrual = block.timestamp;
            return;
        }

        uint256 elapsed = block.timestamp - lastFeeAccrual;
        if (elapsed == 0) return;

        // fee = totalSupply * (managementFeeBps / 10000) * (elapsed / SECONDS_PER_YEAR)
        // Implemented as share minting (dilution):
        // newShares = totalSupply * feeRate * elapsed / (SECONDS_PER_YEAR - feeRate * elapsed)
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
     * @dev Only charges fees on NAV gains above the high-water mark (HWM).
     *      The fee is computed as:
     *        totalGain = (currentNav - HWM) * totalSupply / 1e18
     *        fee = totalGain * performanceFeeBps / 10000
     *      Fee shares are minted to feeRecipient. HWM is updated to currentNav
     *      regardless of whether a fee was charged (to track the peak).
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
     * @dev Returns a non-zero fee if the user deposited within the earlyRedemptionWindow.
     *      Fee = assets * redemptionFeeBps / WEIGHT_PRECISION.
     *      This anti-churn mechanism discourages short-term deposits that increase
     *      rebalancing costs for long-term holders.
     * @param user   Address of the withdrawer
     * @param assets Amount of assets being withdrawn (in underlying decimals)
     * @return fee   Fee amount to deduct (in underlying decimals), zero if outside window
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
     *      preventing front-running of rebalancing trades. After commitment,
     *      a timelock period (weightTimelock) must elapse before execution.
     *
     *      This implements a commit-reveal scheme:
     *      1. Keeper computes new weights off-chain and builds a Merkle tree
     *      2. Keeper calls commitWeights(root) to lock the commitment on-chain
     *      3. After timelock, keeper calls executeWeights() with proof
     *
     *      If a commit is already pending, a new commit replaces it (restarting the timelock).
     * @param merkleRoot Root hash of the weight Merkle tree (must be non-zero)
     */
    function commitWeights(bytes32 merkleRoot) external onlyRole(KEEPER_ROLE) {
        require(merkleRoot != bytes32(0), "Zero root");
        committedWeightsRoot = merkleRoot;
        commitTimestamp = block.timestamp;
        commitPending = true;

        emit WeightsCommitted(merkleRoot, block.timestamp);
    }

    /**
     * @notice Cancel a pending weight commitment before execution.
     * @dev Allows the keeper to abort a proposed weight update if market conditions
     *      have changed during the timelock period or if the commitment was made
     *      in error. Clears the committed root, timestamp, and pending flag.
     *      Only callable when a commitment is actually pending.
     */
    function cancelCommit() external onlyRole(KEEPER_ROLE) {
        if (!commitPending) revert NoCommitPending();

        bytes32 cancelledRoot = committedWeightsRoot;
        committedWeightsRoot = bytes32(0);
        commitTimestamp = 0;
        commitPending = false;

        emit WeightCommitCancelled(cancelledRoot, block.timestamp);
    }

    /**
     * @notice Phase 2: Reveal and execute weight update after timelock.
     * @dev Verifies each (token, weight) pair against the committed Merkle root,
     *      validates that weights sum to WEIGHT_PRECISION, enforces per-asset max
     *      weight (40%) and per-rebalance max turnover (30%), then applies the new
     *      weights. In circuit breaker mode, defensive weights override the proposed ones.
     *
     *      The Merkle leaf is computed as:
     *        keccak256(abi.encode(tokens, newWeights))
     *
     * @param tokens     Array of constituent addresses (must match constituents array length)
     * @param newWeights Array of new weights in basis points (must sum to 10000)
     * @param proof      Merkle proof path for verification against committedWeightsRoot
     */
    function executeWeights(
        address[] calldata tokens,
        uint256[] calldata newWeights,
        bytes32[] calldata proof
    ) external onlyRole(KEEPER_ROLE) whenNotPaused {
        if (!commitPending) revert NoCommitPending();
        if (block.timestamp < commitTimestamp + weightTimelock) {
            revert TimelockNotElapsed(commitTimestamp, weightTimelock);
        }
        require(tokens.length == newWeights.length, "Length mismatch");
        require(tokens.length == constituents.length, "Must update all constituents");

        // Verify Merkle proof
        bytes32 leaf = keccak256(abi.encode(tokens, newWeights));
        require(
            MerkleProof.verify(proof, committedWeightsRoot, leaf),
            "Invalid Merkle proof"
        );

        // Validate weights
        uint256 totalWeight = 0;
        uint256 totalTurnover = 0;

        for (uint256 i = 0; i < tokens.length; i++) {
            if (!isConstituent[tokens[i]]) revert NotConstituent(tokens[i]);
            if (newWeights[i] > 4000) revert InvalidWeight(newWeights[i], 4000);

            // Track turnover
            uint256 oldWeight = weights[tokens[i]];
            if (newWeights[i] > oldWeight) {
                totalTurnover += newWeights[i] - oldWeight;
            } else {
                totalTurnover += oldWeight - newWeights[i];
            }

            totalWeight += newWeights[i];
        }

        if (totalWeight != WEIGHT_PRECISION) revert WeightSumMismatch(totalWeight, WEIGHT_PRECISION);
        if (totalTurnover > 3000) revert TurnoverExceeded(totalTurnover, 3000);

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
     * @notice Check and trigger the circuit breaker if portfolio drawdown exceeds threshold.
     * @dev Permissionless function — can be called by anyone (keeper, users, bots).
     *      Compares current NAV per share against the high-water mark:
     *        threshold = HWM * (10000 - circuitBreakerThresholdBps) / 10000
     *      If NAV drops below this threshold, the circuit breaker activates,
     *      overriding all weights with defensive allocations.
     *
     *      The circuit breaker is a key safety mechanism that shifts the portfolio
     *      to a risk-off posture (e.g., 90% stablecoins) during severe drawdowns,
     *      protecting depositors from further losses.
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
     * @notice Reset the circuit breaker after sufficient NAV recovery.
     * @dev Only ADMIN_ROLE can reset. Requires NAV to have recovered to at least
     *      90% of the high-water mark to prevent premature re-entry into risk-on mode.
     *      After reset, the keeper can resume normal weight updates via executeWeights().
     */
    function resetCircuitBreaker() external onlyRole(ADMIN_ROLE) {
        if (!circuitBreakerActive) revert CircuitBreakerState(true);
        uint256 currentNav = navPerShare();
        uint256 recoveryLevel = highWaterMark * 9000 / WEIGHT_PRECISION; // 90% of HWM

        if (currentNav < recoveryLevel) revert InsufficientRecovery(currentNav, recoveryLevel);

        circuitBreakerActive = false;
        emit CircuitBreakerReset(currentNav);
    }

    // ═══════════════════════════════════════════════════════════════════
    //                  DEPOSIT / WITHDRAW OVERRIDES
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @inheritdoc ERC4626
     * @notice Deposit underlying assets (USDC) and receive vault shares (rpCRYPTO).
     * @dev Override deposit to:
     *   1. Accrue management fees before share pricing (ensures fair NAV)
     *   2. Track deposit time for early redemption fee calculation
     *
     *   The number of shares minted is determined by the ERC-4626 standard:
     *     shares = assets * totalSupply / totalAssets
     * @param assets   Amount of underlying asset (USDC) to deposit
     * @param receiver Address to receive the minted vault shares
     * @return Number of vault shares minted to the receiver
     */
    function deposit(
        uint256 assets,
        address receiver
    ) public override nonReentrant whenNotPaused returns (uint256) {
        _accrueManagementFee();

        uint256 shares = super.deposit(assets, receiver);

        // Track deposit time for redemption fee window
        lastDepositTime[receiver] = block.timestamp;

        return shares;
    }

    /**
     * @inheritdoc ERC4626
     * @notice Withdraw underlying assets by burning vault shares.
     * @dev Override withdraw to:
     *   1. Accrue management fees before share pricing
     *   2. Enforce the per-epoch redemption gate (max % of AUM redeemable per epoch)
     *   3. Charge early redemption fee if within earlyRedemptionWindow
     *   4. Emit RedemptionGateHit if gate utilisation exceeds 80% of maximum
     *
     *   The redemption gate prevents bank-run scenarios by limiting total
     *   withdrawals per epoch to redemptionGateBps of totalAssets.
     * @param assets   Amount of underlying asset to withdraw
     * @param receiver Address to receive the withdrawn assets
     * @param owner    Address whose shares are burned
     * @return Number of shares burned
     */
    function withdraw(
        uint256 assets,
        address receiver,
        address owner
    ) public override nonReentrant whenNotPaused returns (uint256) {
        _accrueManagementFee();

        // Check redemption gate
        uint256 maxRedeemable = (totalAssets() * redemptionGateBps) / WEIGHT_PRECISION;
        if (epochRedemptionTotal + assets > maxRedeemable) {
            revert RedemptionGateExceeded(
                epochRedemptionTotal + assets,
                maxRedeemable - epochRedemptionTotal
            );
        }

        // Emit gate warning if utilisation exceeds 80%
        if (epochRedemptionTotal + assets > (maxRedeemable * 8000) / WEIGHT_PRECISION) {
            emit RedemptionGateHit(owner, assets, epochRedemptionTotal + assets);
        }

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
     * @notice Redeem vault shares for underlying assets.
     * @dev Override redeem with same fee and gate logic as withdraw().
     *      The assets returned are computed by the ERC-4626 standard:
     *        assets = shares * totalAssets / totalSupply
     *      Early redemption fee and redemption gate are applied after conversion.
     * @param shares   Number of vault shares to redeem
     * @param receiver Address to receive the underlying assets
     * @param owner    Address whose shares are burned
     * @return Amount of underlying assets returned to receiver
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
        if (epochRedemptionTotal + assets > maxRedeemable) {
            revert RedemptionGateExceeded(
                epochRedemptionTotal + assets,
                maxRedeemable - epochRedemptionTotal
            );
        }

        // Emit gate warning if utilisation exceeds 80%
        if (epochRedemptionTotal + assets > (maxRedeemable * 8000) / WEIGHT_PRECISION) {
            emit RedemptionGateHit(owner, assets, epochRedemptionTotal + assets);
        }

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
     * @notice Update fee parameters. Only callable by ADMIN_ROLE.
     * @dev Accrues fees at the old rate before applying the new rates.
     *      Each parameter is individually capped to prevent abusive fees:
     *      - Management: max 500 bps (5% annual)
     *      - Performance: max 3000 bps (30%)
     *      - Redemption: max 200 bps (2%)
     * @param newManagementBps  New management fee in basis points (max 500)
     * @param newPerformanceBps New performance fee in basis points (max 3000)
     * @param newRedemptionBps  New redemption fee in basis points (max 200)
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
     * @dev All future fee payments (management, performance, redemption) will be
     *      directed to the new recipient. Must be non-zero.
     * @param newRecipient The new fee recipient address
     */
    function setFeeRecipient(address newRecipient) external onlyRole(ADMIN_ROLE) {
        if (newRecipient == address(0)) revert ZeroAddress("feeRecipient");
        feeRecipient = newRecipient;
    }

    /**
     * @notice Update the per-epoch redemption gate percentage.
     * @dev Controls the maximum percentage of AUM that can be redeemed in a single epoch.
     *      Bounded between 500 bps (5%) and 5000 bps (50%) for safety.
     * @param newGateBps New redemption gate in basis points (500-5000)
     */
    function setRedemptionGate(uint256 newGateBps) external onlyRole(ADMIN_ROLE) {
        require(newGateBps >= 500 && newGateBps <= 5000, "Gate out of range");
        redemptionGateBps = newGateBps;
    }

    /**
     * @notice Emergency pause. Stops all deposits, withdrawals, and weight executions.
     * @dev Only callable by ADMIN_ROLE. Use in response to critical vulnerabilities,
     *      oracle failures, or black swan events. Unpause with unpause().
     */
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    /**
     * @notice Resume vault operations after an emergency pause.
     * @dev Only callable by ADMIN_ROLE. Verify that the root cause of the pause
     *      has been resolved before calling.
     */
    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }

    // ═══════════════════════════════════════════════════════════════════
    //                         VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════

    /**
     * @notice Get all constituent token addresses registered in the portfolio.
     * @dev Returns the full constituents array. Order is preserved from registration.
     * @return Array of constituent ERC-20 token addresses
     */
    function getConstituents() external view returns (address[] memory) {
        return constituents;
    }

    /**
     * @notice Get current target weights for all constituents.
     * @dev Weights are in basis points (10000 = 100%). During circuit breaker mode,
     *      these reflect the defensive weights rather than the ML-optimised ones.
     * @return tokens Array of constituent addresses
     * @return w     Array of weights in basis points (same order as tokens)
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
     * @notice Get a detailed breakdown of the portfolio showing per-constituent metrics.
     * @dev For each constituent, returns the token address, current target weight,
     *      latest Chainlink price, token balance held by the vault, USD value of the
     *      holding, and percentage of total portfolio value. Useful for dashboard
     *      rendering and transparency reporting.
     *
     *      Note: If a price feed is stale or invalid, the function will revert.
     *      Use computePortfolioValue() first to check for oracle issues.
     *
     * @return tokens       Array of constituent token addresses
     * @return weightsBps   Array of current weights in basis points
     * @return prices       Array of latest Chainlink prices (in feed decimals)
     * @return balances     Array of token balances held by the vault
     * @return values       Array of USD values in 18-decimal precision
     * @return percentages  Array of portfolio percentages in basis points (10000 = 100%)
     */
    function getPortfolioBreakdown()
        external
        view
        returns (
            address[] memory tokens,
            uint256[] memory weightsBps,
            int256[] memory prices,
            uint256[] memory balances,
            uint256[] memory values,
            uint256[] memory percentages
        )
    {
        uint256 len = constituents.length;
        tokens = new address[](len);
        weightsBps = new uint256[](len);
        prices = new int256[](len);
        balances = new uint256[](len);
        values = new uint256[](len);
        percentages = new uint256[](len);

        uint256 totalValue = 0;

        for (uint256 i = 0; i < len; i++) {
            address token = constituents[i];
            tokens[i] = token;
            weightsBps[i] = weights[token];
            balances[i] = IERC20(token).balanceOf(address(this));

            address feed = priceFeeds[token];
            (, int256 price,, uint256 updatedAt,) =
                IChainlinkAggregator(feed).latestRoundData();

            if (price <= 0) revert InvalidPrice(feed, price);
            if (block.timestamp - updatedAt >= 1 hours) revert StalePrice(feed, updatedAt);

            prices[i] = price;

            if (balances[i] > 0) {
                uint8 feedDecimals = IChainlinkAggregator(feed).decimals();
                uint8 tokenDecimals = ERC20(token).decimals();
                values[i] = (balances[i] * uint256(price) * 1e18)
                    / (10 ** tokenDecimals * 10 ** feedDecimals);
            }
            totalValue += values[i];
        }

        // Compute percentages (in basis points)
        if (totalValue > 0) {
            for (uint256 i = 0; i < len; i++) {
                percentages[i] = (values[i] * WEIGHT_PRECISION) / totalValue;
            }
        }
    }

    /**
     * @notice Get vault metrics for the dashboard.
     * @dev Returns a comprehensive snapshot of the vault's current state including
     *      NAV, high-water mark, epoch info, circuit breaker status, gate utilisation,
     *      and share/asset totals. All values are current as of the block.
     * @return nav          Current NAV per share (1e18 scaled)
     * @return hwm          High-water mark (1e18 scaled)
     * @return epoch        Current epoch number
     * @return cbActive     Whether the circuit breaker is currently active
     * @return gateUsed     Redemption amount used this epoch (underlying decimals)
     * @return gateMax      Maximum redemption for this epoch (underlying decimals)
     * @return totalShares  Total shares outstanding
     * @return totalVal     Total assets under management (underlying decimals)
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
     * @notice Get fee configuration and cumulative collection summary.
     * @dev Returns current fee parameters and total fees collected for transparency.
     *      Fee shares can be converted to underlying value via convertToAssets().
     * @return mgmtBps             Current management fee in basis points
     * @return perfBps             Current performance fee in basis points
     * @return redemptBps          Current redemption fee in basis points
     * @return totalMgmtCollected  Total management fee shares collected
     * @return totalPerfCollected  Total performance fee shares collected
     * @return recipient           Current fee recipient address
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
     * @notice Get the number of registered constituent tokens.
     * @dev Useful for off-chain iteration without fetching the full array.
     * @return Number of constituents in the portfolio
     */
    function constituentCount() external view returns (uint256) {
        return constituents.length;
    }
}
