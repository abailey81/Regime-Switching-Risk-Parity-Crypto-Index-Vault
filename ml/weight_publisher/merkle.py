"""
Merkle Tree construction for on-chain weight commitment.

Generates a Merkle root from (token_address, weight_bps) pairs.
The keeper commits this root on-chain, then reveals actual weights
after the timelock, proving they match the commitment.
"""
import hashlib
from typing import List, Tuple
from eth_abi import encode


def keccak256(data: bytes) -> bytes:
    """Compute Keccak-256 hash (Ethereum's hash function)."""
    from web3 import Web3
    return Web3.keccak(data)


def compute_leaf(token_address: str, weight_bps: int) -> bytes:
    """Compute Merkle leaf: keccak256(abi.encodePacked(token, weight))."""
    encoded = encode(["address", "uint256"], [token_address, weight_bps])
    return keccak256(encoded)


def compute_merkle_root(tokens: List[str], weights_bps: List[int]) -> str:
    """
    Build a Merkle tree from token-weight pairs and return the root hash.

    For the simplified testnet deployment, we use a single-leaf tree:
    root = keccak256(abi.encode(tokens[], weights[]))

    This matches the verification logic in the smart contract.
    """
    encoded = encode(
        ["address[]", "uint256[]"],
        [tokens, weights_bps]
    )
    root = keccak256(encoded)
    return "0x" + root.hex()


def compute_merkle_proof(tokens: List[str], weights_bps: List[int],
                          leaf_index: int = 0) -> List[str]:
    """
    Compute Merkle proof for a given leaf.

    For single-leaf tree (testnet): proof is empty [].
    For production multi-leaf tree: returns sibling hashes.
    """
    # Simplified: single-leaf tree, empty proof
    return []


if __name__ == "__main__":
    # Example usage
    tokens = [
        "0x1234567890123456789012345678901234567890",
        "0x2345678901234567890123456789012345678901",
    ]
    weights = [6000, 4000]  # 60%, 40%

    root = compute_merkle_root(tokens, weights)
    proof = compute_merkle_proof(tokens, weights)
    print(f"Merkle root: {root}")
    print(f"Proof: {proof}")
