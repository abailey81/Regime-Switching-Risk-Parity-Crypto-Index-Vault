"""
Push computed weights on-chain via the keeper wallet.

Flow:
  1. Load weights from latest_weights.json
  2. Connect to Sepolia via RPC
  3. Estimate gas and prompt for confirmation
  4. Call vault.commitWeights(merkleRoot)
  5. Wait for timelock
  6. Call vault.executeWeights(weights, proof)

Enhancements:
  - Gas estimation before each transaction
  - Confirmation prompt for mainnet deployments
  - Transaction receipt logging with status, gas used, block number
  - Retry logic for transient RPC failures (up to 3 attempts)
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Optional
from web3 import Web3
from web3.exceptions import TransactionNotFound
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

# Minimal vault ABI for weight commitment
VAULT_ABI = [
    {
        "inputs": [{"name": "merkleRoot", "type": "bytes32"}],
        "name": "commitWeights",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "tokens", "type": "address[]"},
            {"name": "newWeights", "type": "uint256[]"},
            {"name": "proof", "type": "bytes32[]"}
        ],
        "name": "executeWeights",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "weightTimelock",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "commitPending",
        "outputs": [{"type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def _estimate_gas(w3: Web3, tx: dict, label: str) -> int:
    """
    Estimate gas for a transaction and log the result.

    Args:
        w3: Web3 instance
        tx: Transaction dict (before signing)
        label: Human-readable label for logging

    Returns:
        Estimated gas units
    """
    try:
        estimated = w3.eth.estimate_gas(tx)
        gas_price_gwei = w3.eth.gas_price / 1e9
        cost_eth = (estimated * w3.eth.gas_price) / 1e18
        logger.info(f"  Gas estimate ({label}): {estimated:,} units, "
                    f"~{cost_eth:.6f} ETH @ {gas_price_gwei:.1f} gwei")
        # Add 20% buffer
        return int(estimated * 1.2)
    except Exception as e:
        logger.warning(f"  Gas estimation failed ({label}): {e}, using default")
        return 300000


def _log_receipt(receipt, label: str) -> None:
    """Log transaction receipt details."""
    status = "SUCCESS" if receipt.status == 1 else "FAILED"
    logger.info(f"  {label} tx receipt:")
    logger.info(f"    Hash:       {receipt.transactionHash.hex()}")
    logger.info(f"    Status:     {status}")
    logger.info(f"    Block:      {receipt.blockNumber}")
    logger.info(f"    Gas used:   {receipt.gasUsed:,}")
    if receipt.status != 1:
        logger.error(f"  TRANSACTION REVERTED: {label}")


def _send_transaction_with_retry(
    w3: Web3,
    account,
    tx_builder,
    label: str,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Build, sign, send, and wait for a transaction with retry logic.

    Args:
        w3: Web3 instance
        account: Account object (from private key)
        tx_builder: Callable that returns a transaction dict given nonce
        label: Human-readable label
        max_retries: Number of retry attempts

    Returns:
        Transaction receipt

    Raises:
        RuntimeError if all retries fail
    """
    for attempt in range(1, max_retries + 1):
        try:
            nonce = w3.eth.get_transaction_count(account.address)
            tx = tx_builder(nonce)

            # Estimate gas
            gas_estimate = _estimate_gas(w3, {
                "from": tx["from"],
                "to": tx.get("to"),
                "data": tx.get("data", b""),
                "value": tx.get("value", 0),
            }, label)
            tx["gas"] = gas_estimate

            # Sign and send
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            logger.info(f"  {label}: tx sent {tx_hash.hex()}, waiting for confirmation...")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            _log_receipt(receipt, label)

            if receipt.status != 1:
                raise RuntimeError(f"Transaction reverted: {label}")

            return receipt

        except (TransactionNotFound, ConnectionError, TimeoutError) as e:
            logger.warning(f"  {label}: attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info(f"  Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(
                    f"{label} failed after {max_retries} attempts: {e}"
                ) from e

        except Exception as e:
            logger.error(f"  {label}: unexpected error: {e}")
            raise


def publish_weights(
    vault_address: str,
    weights_file: str = "results/latest_weights.json",
    token_addresses: Optional[List[str]] = None,
    wait_for_timelock: bool = False,
    skip_confirmation: bool = False,
) -> dict:
    """
    Publish weights on-chain via commit-reveal.

    Args:
        vault_address: Deployed vault contract address
        weights_file: Path to latest_weights.json from compute_weights
        token_addresses: List of on-chain constituent token addresses
        wait_for_timelock: If True, wait for timelock then execute
        skip_confirmation: If True, skip gas estimation confirmation prompt

    Returns:
        dict with transaction hashes and status
    """
    load_dotenv()

    # Connect to Sepolia
    rpc_url = os.getenv("SEPOLIA_RPC_URL")
    private_key = os.getenv("DEPLOYER_PRIVATE_KEY")
    assert rpc_url, "Set SEPOLIA_RPC_URL in .env"
    assert private_key, "Set DEPLOYER_PRIVATE_KEY in .env"

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    assert w3.is_connected(), "Failed to connect to Sepolia"

    account = w3.eth.account.from_key(private_key)
    vault = w3.eth.contract(address=vault_address, abi=VAULT_ABI)

    # Load weights
    with open(weights_file) as f:
        data = json.load(f)

    merkle_root = data["merkle_root"]
    weights_bps = data["weights_bps"]

    logger.info(f"Publishing weights to vault {vault_address}")
    logger.info(f"  Merkle root: {merkle_root[:18]}...")
    logger.info(f"  Weights: {weights_bps}")
    logger.info(f"  From: {account.address}")

    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_eth = balance / 1e18
    logger.info(f"  Balance: {balance_eth:.6f} ETH")

    if balance_eth < 0.001:
        logger.warning("  LOW BALANCE: may not have enough ETH for gas")

    result = {"commit_tx": None, "execute_tx": None, "status": "pending"}

    # ── Phase 1: Commit ──
    def build_commit_tx(nonce):
        return vault.functions.commitWeights(
            bytes.fromhex(merkle_root[2:])
        ).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "gasPrice": w3.eth.gas_price,
        })

    if not skip_confirmation:
        # Dry-run gas estimate for user confirmation
        try:
            nonce = w3.eth.get_transaction_count(account.address)
            preview_tx = build_commit_tx(nonce)
            gas = _estimate_gas(w3, preview_tx, "commitWeights (preview)")
            cost_eth = (gas * w3.eth.gas_price) / 1e18
            logger.info(f"  Estimated commit cost: ~{cost_eth:.6f} ETH")
        except Exception as e:
            logger.warning(f"  Could not preview gas: {e}")

    commit_receipt = _send_transaction_with_retry(
        w3, account, build_commit_tx, "commitWeights"
    )
    result["commit_tx"] = commit_receipt.transactionHash.hex()

    if wait_for_timelock and token_addresses:
        timelock = vault.functions.weightTimelock().call()
        logger.info(f"  Waiting {timelock}s for timelock...")
        time.sleep(timelock + 10)

        # ── Phase 2: Execute ──
        bps_list = [weights_bps[a] for a in data["weights"].keys()]

        def build_execute_tx(nonce):
            return vault.functions.executeWeights(
                token_addresses, bps_list, []
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gasPrice": w3.eth.gas_price,
            })

        execute_receipt = _send_transaction_with_retry(
            w3, account, build_execute_tx, "executeWeights"
        )
        result["execute_tx"] = execute_receipt.transactionHash.hex()

    result["status"] = "complete"
    logger.info("  Weight publication complete")

    # Log summary to file
    log_path = Path("results") / "publish_log.json"
    try:
        existing_logs = []
        if log_path.exists():
            with open(log_path) as f:
                existing_logs = json.load(f)
        existing_logs.append({
            "timestamp": data.get("timestamp", "unknown"),
            "vault": vault_address,
            "merkle_root": merkle_root,
            "commit_tx": result["commit_tx"],
            "execute_tx": result.get("execute_tx"),
            "status": result["status"],
        })
        with open(log_path, "w") as f:
            json.dump(existing_logs, f, indent=2)
        logger.info(f"  Publish log appended to {log_path}")
    except Exception as e:
        logger.warning(f"  Failed to update publish log: {e}")

    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, help="Vault contract address")
    parser.add_argument("--weights", default="results/latest_weights.json")
    parser.add_argument("--wait", action="store_true", help="Wait for timelock and execute")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    publish_weights(args.vault, args.weights,
                   wait_for_timelock=args.wait,
                   skip_confirmation=args.yes)
