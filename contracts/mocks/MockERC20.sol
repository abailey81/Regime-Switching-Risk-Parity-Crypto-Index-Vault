// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title MockERC20
 * @notice Mintable ERC-20 token for Sepolia testnet deployment.
 * @dev Used to simulate USDC, WBTC, WETH, SOL, stETH, rETH, BUIDL, USDY
 *      as vault constituents on testnet.
 */
contract MockERC20 is ERC20, Ownable {
    uint8 private _decimals;

    constructor(
        string memory name_,
        string memory symbol_,
        uint8 decimals_
    ) ERC20(name_, symbol_) Ownable(msg.sender) {
        _decimals = decimals_;
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }

    /**
     * @notice Mint tokens to any address. Only for testnet use.
     * @param to Recipient address
     * @param amount Amount to mint (in smallest unit)
     */
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    /**
     * @notice Convenience: mint tokens to msg.sender.
     */
    function faucet(uint256 amount) external {
        _mint(msg.sender, amount);
    }
}
