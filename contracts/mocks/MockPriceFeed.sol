// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../interfaces/IChainlinkAggregator.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title MockPriceFeed
 * @author Tamer Atesyakar (UCL MSc Digital Finance & Banking)
 * @notice Simulates a Chainlink AggregatorV3 price feed for Sepolia testnet.
 * @dev Owner can update the price to simulate market movements.
 *      Stores price history for TWAP computation and volatility simulation testing.
 *      This mock is NOT intended for production use — it provides deterministic
 *      price control for integration testing of the RiskParityVault.
 */
contract MockPriceFeed is IChainlinkAggregator, Ownable {
    /// @notice Number of decimals in the price feed (e.g., 8 for USD pairs)
    uint8 private _decimals;

    /// @notice Human-readable description of the price pair (e.g., "BTC/USD")
    string private _description;

    /// @notice Current price value in feed decimals
    int256 private _price;

    /// @notice Timestamp of the last price update
    uint256 private _updatedAt;

    /// @notice Monotonically increasing round identifier
    uint80 private _roundId;

    /// @notice Historical price observation for TWAP computation
    /// @param price The price at the observation time
    /// @param timestamp The block timestamp of the observation
    struct PriceObservation {
        int256 price;
        uint256 timestamp;
    }

    /// @notice Ordered array of all price observations since deployment
    PriceObservation[] public priceHistory;

    /// @notice Emitted when the price is updated by the owner
    /// @param newPrice The new price value
    /// @param timestamp The block timestamp of the update
    /// @param roundId The new round identifier
    event PriceUpdated(int256 newPrice, uint256 timestamp, uint80 roundId);

    /// @notice Emitted when simulateVolatility generates price movements
    /// @param periods Number of price observations generated
    /// @param finalPrice The price after all simulated movements
    event VolatilitySimulated(uint256 periods, int256 finalPrice);

    /**
     * @notice Deploy a new MockPriceFeed with an initial price.
     * @param description_ Human-readable pair description (e.g., "BTC/USD")
     * @param decimals_    Number of decimals in price values (typically 8)
     * @param initialPrice_ Starting price in feed decimals (must be positive)
     */
    constructor(
        string memory description_,
        uint8 decimals_,
        int256 initialPrice_
    ) Ownable(msg.sender) {
        _description = description_;
        _decimals = decimals_;
        _price = initialPrice_;
        _updatedAt = block.timestamp;
        _roundId = 1;

        priceHistory.push(PriceObservation({
            price: initialPrice_,
            timestamp: block.timestamp
        }));
    }

    /**
     * @notice Return the latest round data in Chainlink AggregatorV3 format.
     * @dev All timestamp fields return the same value (_updatedAt) since this
     *      is a mock that doesn't distinguish between round start and answer times.
     * @return roundId The current round identifier
     * @return answer The current price in feed decimals
     * @return startedAt Timestamp of the current round (same as updatedAt)
     * @return updatedAt Timestamp of the last price update
     * @return answeredInRound The round in which the answer was computed (same as roundId)
     */
    function latestRoundData()
        external
        view
        override
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        )
    {
        return (_roundId, _price, _updatedAt, _updatedAt, _roundId);
    }

    /**
     * @notice Return the number of decimals in the price feed.
     * @return Number of decimals (e.g., 8 for most USD pairs)
     */
    function decimals() external view override returns (uint8) {
        return _decimals;
    }

    /**
     * @notice Return the human-readable description of this price feed.
     * @return Description string (e.g., "BTC/USD")
     */
    function description() external view override returns (string memory) {
        return _description;
    }

    /**
     * @notice Update the price feed to a new value. Only owner.
     * @dev Simulates a Chainlink oracle update. Increments the round ID,
     *      records the price in history, and updates the timestamp.
     * @param newPrice New price value in feed decimals (must be positive)
     */
    function updatePrice(int256 newPrice) external onlyOwner {
        require(newPrice > 0, "Price must be positive");
        _roundId++;
        _price = newPrice;
        _updatedAt = block.timestamp;

        priceHistory.push(PriceObservation({
            price: newPrice,
            timestamp: block.timestamp
        }));

        emit PriceUpdated(newPrice, block.timestamp, _roundId);
    }

    /**
     * @notice Force-set the updatedAt timestamp for staleness testing.
     * @dev Allows tests to simulate stale price feeds by backdating the
     *      last update time without changing the price value.
     * @param timestamp The timestamp to set as the last update time
     */
    function setUpdatedAt(uint256 timestamp) external onlyOwner {
        _updatedAt = timestamp;
    }

    /**
     * @notice Get the number of price observations stored.
     * @dev Useful for iterating through history off-chain.
     * @return Number of observations in the priceHistory array
     */
    function priceHistoryLength() external view returns (uint256) {
        return priceHistory.length;
    }

    /**
     * @notice Simulate realistic price volatility by generating a series of
     *         price movements with controlled amplitude.
     * @dev Creates `periods` new price observations, each deviating from the
     *      previous by up to `amplitude` basis points. The direction alternates
     *      to simulate oscillation (odd periods go up, even go down).
     *
     *      This is useful for testing:
     *      - Circuit breaker triggering under volatile conditions
     *      - TWAP computation with multiple observations
     *      - NAV fluctuation impact on performance fees
     *
     *      Example: simulateVolatility(500, 10) creates 10 price points each
     *      varying up to 5% from the previous.
     *
     * @param amplitude Maximum price deviation per period in basis points (e.g., 500 = 5%)
     * @param periods   Number of price observations to generate
     */
    function simulateVolatility(uint256 amplitude, uint256 periods) external onlyOwner {
        require(amplitude > 0 && amplitude <= 5000, "Amplitude 1-5000 bps");
        require(periods > 0 && periods <= 100, "Periods 1-100");

        int256 currentPrice = _price;

        for (uint256 i = 0; i < periods; i++) {
            // Alternate direction: odd periods up, even periods down
            // Use period index + roundId for deterministic pseudo-randomness
            uint256 seed = uint256(keccak256(abi.encodePacked(i, _roundId, block.timestamp)));
            // Scale amplitude by a factor derived from seed (50%-100% of amplitude)
            uint256 scaledAmp = amplitude * (5000 + (seed % 5001)) / 10000;

            int256 delta = (currentPrice * int256(scaledAmp)) / 10000;

            if (i % 2 == 0) {
                currentPrice = currentPrice + delta;
            } else {
                currentPrice = currentPrice - delta;
            }

            // Ensure price stays positive
            if (currentPrice <= 0) {
                currentPrice = 1;
            }

            _roundId++;
            _price = currentPrice;
            _updatedAt = block.timestamp;

            priceHistory.push(PriceObservation({
                price: currentPrice,
                timestamp: block.timestamp
            }));
        }

        emit PriceUpdated(currentPrice, block.timestamp, _roundId);
        emit VolatilitySimulated(periods, currentPrice);
    }

    /**
     * @notice Compute a series of price returns (changes) over a rolling window.
     * @dev Returns an array of price differences: return[i] = price[n-window+i+1] - price[n-window+i].
     *      Useful for off-chain analysis of simulated price paths and volatility estimation.
     *
     *      If the window exceeds the available history, the full history is used.
     *      Returns an empty array if fewer than 2 observations exist.
     *
     * @param window Number of observations to include (from the most recent)
     * @return returns_ Array of price changes (can be negative), length = min(window, historyLength) - 1
     */
    function getReturnSeries(uint256 window) external view returns (int256[] memory returns_) {
        uint256 len = priceHistory.length;
        if (len < 2) {
            returns_ = new int256[](0);
            return returns_;
        }

        // Clamp window to available history
        uint256 effectiveWindow = window > len ? len : window;
        uint256 startIdx = len - effectiveWindow;
        uint256 returnCount = effectiveWindow - 1;

        returns_ = new int256[](returnCount);
        for (uint256 i = 0; i < returnCount; i++) {
            returns_[i] = priceHistory[startIdx + i + 1].price - priceHistory[startIdx + i].price;
        }
    }

    /**
     * @notice Compute TWAP (Time-Weighted Average Price) over a given window.
     * @dev Iterates backwards through price history, weighting each observation
     *      by the duration it was active. Used for testing TWAP oracle integrations.
     *
     *      The TWAP is computed as:
     *        sum(price_i * duration_i) / sum(duration_i)
     *      where duration_i is the time between observation i and observation i+1.
     *
     * @param window Duration in seconds to compute TWAP over (from current timestamp)
     * @return twap Time-weighted average price in feed decimals
     */
    function computeTWAP(uint256 window) external view returns (int256 twap) {
        require(priceHistory.length > 0, "No price history");
        uint256 cutoff = block.timestamp - window;
        int256 weightedSum = 0;
        uint256 totalDuration = 0;

        for (uint256 i = priceHistory.length; i > 0; i--) {
            PriceObservation memory obs = priceHistory[i - 1];
            if (obs.timestamp < cutoff && i < priceHistory.length) break;

            uint256 start = obs.timestamp < cutoff ? cutoff : obs.timestamp;
            uint256 end_;
            if (i < priceHistory.length) {
                end_ = priceHistory[i].timestamp;
            } else {
                end_ = block.timestamp;
            }

            uint256 duration = end_ - start;
            weightedSum += obs.price * int256(duration);
            totalDuration += duration;
        }

        require(totalDuration > 0, "No observations in window");
        twap = weightedSum / int256(totalDuration);
    }
}
