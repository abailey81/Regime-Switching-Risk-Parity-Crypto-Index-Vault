// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../interfaces/IChainlinkAggregator.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title MockPriceFeed
 * @notice Simulates a Chainlink AggregatorV3 price feed for Sepolia testnet.
 * @dev Owner can update the price to simulate market movements.
 *      Stores price history for TWAP computation testing.
 */
contract MockPriceFeed is IChainlinkAggregator, Ownable {
    uint8 private _decimals;
    string private _description;

    int256 private _price;
    uint256 private _updatedAt;
    uint80 private _roundId;

    /// @notice Historical price observations for TWAP
    struct PriceObservation {
        int256 price;
        uint256 timestamp;
    }
    PriceObservation[] public priceHistory;

    event PriceUpdated(int256 newPrice, uint256 timestamp, uint80 roundId);

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

    function decimals() external view override returns (uint8) {
        return _decimals;
    }

    function description() external view override returns (string memory) {
        return _description;
    }

    /**
     * @notice Update the price feed. Only owner (simulates oracle update).
     * @param newPrice New price (in feed decimals, e.g., 8 for USD pairs)
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
     * @notice Get the number of price observations stored.
     */
    function priceHistoryLength() external view returns (uint256) {
        return priceHistory.length;
    }

    /**
     * @notice Compute TWAP over a given window (for testing).
     * @param window Duration in seconds to compute TWAP over
     * @return twap Time-weighted average price
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
