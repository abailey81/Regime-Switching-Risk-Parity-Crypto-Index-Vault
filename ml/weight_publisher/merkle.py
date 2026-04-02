"""
Merkle Tree construction for on-chain weight commitment.

Generates a Merkle root from (token_address, weight_bps) pairs.
The keeper commits this root on-chain, then reveals actual weights
after the timelock, proving they match the commitment.

Supports:
  - Multi-leaf Merkle tree with proper sibling hash proofs
  - Sorted pair hashing (Solidity-compatible, OpenZeppelin style)
  - Local proof verification before on-chain submission
  - Single-leaf fallback for testnet simplicity
"""
import logging
from typing import Dict, List, Tuple
from eth_abi import encode

logger = logging.getLogger(__name__)


def keccak256(data: bytes) -> bytes:
    """Compute Keccak-256 hash (Ethereum's hash function)."""
    from web3 import Web3
    return Web3.keccak(data)


def _sorted_hash_pair(a: bytes, b: bytes) -> bytes:
    """
    Hash two nodes in sorted order (OpenZeppelin MerkleProof standard).

    This ensures the tree is deterministic regardless of insertion order,
    matching Solidity: keccak256(abi.encodePacked(min(a,b), max(a,b))).
    """
    if a <= b:
        return keccak256(a + b)
    else:
        return keccak256(b + a)


def compute_leaf(token_address: str, weight_bps: int) -> bytes:
    """Compute Merkle leaf: keccak256(abi.encodePacked(token, weight))."""
    encoded = encode(["address", "uint256"], [token_address, weight_bps])
    return keccak256(encoded)


def compute_merkle_tree(tokens: List[str], weights_bps: List[int]) -> Dict:
    """
    Build a proper multi-leaf Merkle tree from token-weight pairs.

    Each leaf = keccak256(abi.encode(["address", "uint256"], [token, weight_bps])).
    Internal nodes use sorted pair hashing (OpenZeppelin-compatible).

    Args:
        tokens: List of token addresses (checksummed or lowercase hex)
        weights_bps: List of weights in basis points (must sum to 10000)

    Returns:
        dict with:
          - root: hex string of the Merkle root
          - proofs: list of proof arrays (one per leaf, each proof is list of hex strings)
          - leaves: list of leaf hashes (hex strings)
          - tree: full tree layers (for debugging)
    """
    assert len(tokens) == len(weights_bps), "Token and weight lists must match"

    n = len(tokens)

    if n == 0:
        return {
            "root": "0x" + "0" * 64,
            "proofs": [],
            "leaves": [],
            "tree": [],
        }

    # ── Compute leaves ──
    leaves = [compute_leaf(tokens[i], weights_bps[i]) for i in range(n)]

    # ── Build tree bottom-up ──
    # Pad to next power of 2 with duplicate of last leaf (standard practice)
    tree_size = 1
    while tree_size < n:
        tree_size *= 2

    current_layer = list(leaves)
    while len(current_layer) < tree_size:
        current_layer.append(current_layer[-1])  # Duplicate last leaf

    layers = [current_layer[:]]  # Layer 0 = leaves

    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left
            parent = _sorted_hash_pair(left, right)
            next_layer.append(parent)
        layers.append(next_layer)
        current_layer = next_layer

    root = current_layer[0]

    # ── Compute proofs for each original leaf ──
    proofs = []
    for leaf_idx in range(n):
        proof = []
        idx = leaf_idx

        for layer in layers[:-1]:  # All layers except root
            # Sibling index
            if idx % 2 == 0:
                sibling_idx = idx + 1
            else:
                sibling_idx = idx - 1

            if sibling_idx < len(layer):
                proof.append("0x" + layer[sibling_idx].hex())

            idx = idx // 2

        proofs.append(proof)

    return {
        "root": "0x" + root.hex(),
        "proofs": proofs,
        "leaves": ["0x" + leaf.hex() for leaf in leaves],
        "tree": [["0x" + node.hex() for node in layer] for layer in layers],
    }


def compute_merkle_root(tokens: List[str], weights_bps: List[int]) -> str:
    """
    Build a Merkle tree and return just the root hash.

    Backward-compatible wrapper around compute_merkle_tree().

    Args:
        tokens: List of token addresses
        weights_bps: List of weights in basis points

    Returns:
        Hex string of the Merkle root
    """
    result = compute_merkle_tree(tokens, weights_bps)
    return result["root"]


def compute_merkle_proof(tokens: List[str], weights_bps: List[int],
                          leaf_index: int = 0) -> List[str]:
    """
    Compute Merkle proof for a given leaf.

    Args:
        tokens: List of token addresses
        weights_bps: List of weights in basis points
        leaf_index: Index of the leaf to prove

    Returns:
        List of sibling hashes (hex strings) forming the proof
    """
    result = compute_merkle_tree(tokens, weights_bps)
    if leaf_index < len(result["proofs"]):
        return result["proofs"][leaf_index]
    return []


def verify_proof(
    leaf: str,
    proof: List[str],
    root: str,
) -> bool:
    """
    Verify a Merkle proof locally before on-chain submission.

    Recomputes the root from the leaf and proof path using sorted pair
    hashing, then checks against the expected root.

    Args:
        leaf: Hex string of the leaf hash (e.g. "0xabc...")
        proof: List of sibling hashes (hex strings)
        root: Expected root hash (hex string)

    Returns:
        True if proof is valid, False otherwise
    """
    # Strip 0x prefix and convert to bytes
    current = bytes.fromhex(leaf[2:]) if leaf.startswith("0x") else bytes.fromhex(leaf)
    expected_root = bytes.fromhex(root[2:]) if root.startswith("0x") else bytes.fromhex(root)

    for sibling_hex in proof:
        sibling = bytes.fromhex(sibling_hex[2:]) if sibling_hex.startswith("0x") else bytes.fromhex(sibling_hex)
        current = _sorted_hash_pair(current, sibling)

    return current == expected_root


def verify_all_proofs(tokens: List[str], weights_bps: List[int]) -> bool:
    """
    Build tree and verify all proofs are valid (self-test).

    Args:
        tokens: List of token addresses
        weights_bps: List of weights in basis points

    Returns:
        True if all proofs verify correctly
    """
    result = compute_merkle_tree(tokens, weights_bps)
    root = result["root"]

    for i, (leaf, proof) in enumerate(zip(result["leaves"], result["proofs"])):
        if not verify_proof(leaf, proof, root):
            logger.error(f"Proof verification FAILED for leaf {i}")
            return False

    logger.info(f"All {len(tokens)} proofs verified successfully")
    return True


if __name__ == "__main__":
    # Example usage and self-test
    tokens = [
        "0x1234567890123456789012345678901234567890",
        "0x2345678901234567890123456789012345678901",
        "0x3456789012345678901234567890123456789012",
        "0x4567890123456789012345678901234567890123",
    ]
    weights = [4000, 3000, 2000, 1000]  # 40%, 30%, 20%, 10%

    result = compute_merkle_tree(tokens, weights)
    print(f"Merkle root: {result['root']}")
    print(f"Leaves: {len(result['leaves'])}")
    print(f"Tree depth: {len(result['tree'])} layers")

    for i, proof in enumerate(result["proofs"]):
        valid = verify_proof(result["leaves"][i], proof, result["root"])
        print(f"  Leaf {i} proof: {len(proof)} siblings, valid={valid}")

    # Full verification
    assert verify_all_proofs(tokens, weights), "Self-test failed!"
    print("\nAll proofs verified successfully.")
