// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IChainlinkAggregator
 * @notice Minimal interface for Chainlink V3 price feed aggregators.
 */
interface IChainlinkAggregator {
    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );

    function decimals() external view returns (uint8);
    function description() external view returns (string memory);
}
