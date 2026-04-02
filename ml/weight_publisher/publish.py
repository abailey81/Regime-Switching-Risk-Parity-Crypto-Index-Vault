"""
Push computed weights on-chain via the keeper wallet.

Flow:
  1. Load weights from latest_weights.json
  2. Connect to Sepolia via RPC
  3. Call vault.commitWeights(merkleRoot)
  4. Wait for timelock
  5. Call vault.executeWeights(weights, proof)
"""
import json
import time
import logging
from pathlib import Path
from web3 import Web3
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


def publish_weights(
    vault_address: str,
    weights_file: str = "results/latest_weights.json",
    token_addresses: list = None,
    wait_for_timelock: bool = False,
):
    """
    Publish weights on-chain via commit-reveal.

    Args:
        vault_address: Deployed vault contract address
        weights_file: Path to latest_weights.json from compute_weights
        token_addresses: List of on-chain constituent token addresses
        wait_for_timelock: If True, wait for timelock then execute
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

    # Phase 1: Commit
    nonce = w3.eth.get_transaction_count(account.address)
    tx = vault.functions.commitWeights(bytes.fromhex(merkle_root[2:])).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 100000,
        "gasPrice": w3.eth.gas_price,
    })
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    logger.info(f"  Commit tx: {receipt.transactionHash.hex()}")

    if wait_for_timelock:
        timelock = vault.functions.weightTimelock().call()
        logger.info(f"  Waiting {timelock}s for timelock...")
        time.sleep(timelock + 10)

        # Phase 2: Execute
        if token_addresses:
            bps_list = [weights_bps[a] for a in data["weights"].keys()]
            nonce = w3.eth.get_transaction_count(account.address)
            tx = vault.functions.executeWeights(
                token_addresses, bps_list, []
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": w3.eth.gas_price,
            })
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info(f"  Execute tx: {receipt.transactionHash.hex()}")

    logger.info("  Weight publication complete")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, help="Vault contract address")
    parser.add_argument("--weights", default="results/latest_weights.json")
    parser.add_argument("--wait", action="store_true", help="Wait for timelock and execute")
    args = parser.parse_args()

    publish_weights(args.vault, args.weights, wait_for_timelock=args.wait)
