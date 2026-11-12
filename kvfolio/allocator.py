"""Physical block pool allocator with O(1) free-list and reference counts.

External fragmentation is strictly zero: all blocks are fixed size.
Internal fragmentation is confined to the last partially-filled block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PhysicalBlock:
    """Metadata for one physical KV slab slot."""

    block_id: int
    ref_count: int = 0
    num_tokens: int = 0  # valid tokens written into this block [0, block_size]
    token_hash: Optional[int] = None  # optional radix-cache key


class BlockAllocator:
    """
    Preallocated physical KV storage with free-list allocation.

    Layout of ``storage``:
        [2, num_blocks, num_heads, block_size, head_dim]
        axis-0: 0 = keys, 1 = values
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float64,
    ) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be >= 1")

        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dtype = np.dtype(dtype)

        self.storage = np.zeros(
            (2, self.num_blocks, self.num_heads, self.block_size, self.head_dim),
            dtype=self.dtype,
        )
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i) for i in range(self.num_blocks)
        ]
        self._free: List[int] = list(range(self.num_blocks))
        # radix prefix cache: token-chunk hash -> physical block id
        self._radix: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def free_count(self) -> int:
        return len(self._free)

    def used_count(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self) -> int:
        """Pop one free physical block. Raises RuntimeError on OOM."""
        if not self._free:
            raise RuntimeError(
                f"KV block pool exhausted ({self.num_blocks} blocks of size "
                f"{self.block_size}); cannot allocate"
            )
        bid = self._free.pop()
        meta = self.blocks[bid]
        meta.ref_count = 1
        meta.num_tokens = 0
        meta.token_hash = None
        # zero for determinism / security hygiene
        self.storage[:, bid].fill(0.0)
        return bid

    def retain(self, block_id: int) -> None:
        """Increment reference count (CoW share)."""
        self._check(block_id)
        self.blocks[block_id].ref_count += 1

    def release(self, block_id: int) -> None:
        """Decrement reference count; free when it hits zero."""
        self._check(block_id)
        meta = self.blocks[block_id]
        if meta.ref_count <= 0:
            raise RuntimeError(f"double-free on block {block_id}")
        meta.ref_count -= 1
        if meta.ref_count == 0:
            if meta.token_hash is not None:
                self._radix.pop(meta.token_hash, None)
            meta.num_tokens = 0
            meta.token_hash = None
            self._free.append(block_id)

    def clone_block(self, src_id: int) -> int:
        """Allocate a new block and deep-copy contents from ``src_id`` (CoW break)."""
        self._check(src_id)
        dst_id = self.allocate()
        self.storage[:, dst_id] = self.storage[:, src_id].copy()
        src = self.blocks[src_id]
        dst = self.blocks[dst_id]
        dst.num_tokens = src.num_tokens
        # cloned block is private; do not inherit radix hash
        dst.token_hash = None
        return dst_id

    # ------------------------------------------------------------------
    # KV write / read
    # ------------------------------------------------------------------

    def write_token(
        self,
        block_id: int,
        offset: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """
        Write one token's K/V into ``block_id`` at local ``offset``.

        key, value: shape (num_heads, head_dim)
        """
        self._check(block_id)
        if not (0 <= offset < self.block_size):
            raise IndexError(f"offset {offset} out of block_size {self.block_size}")
        key = np.asarray(key, dtype=self.dtype)
        value = np.asarray(value, dtype=self.dtype)
        if key.shape != (self.num_heads, self.head_dim):
            raise ValueError(f"key shape {key.shape} != {(self.num_heads, self.head_dim)}")
        if value.shape != (self.num_heads, self.head_dim):
            raise ValueError(f"value shape {value.shape} != {(self.num_heads, self.head_dim)}")

        meta = self.blocks[block_id]
        if meta.ref_count != 1:
            raise RuntimeError(
                f"write into shared block {block_id} (ref_count={meta.ref_count}); "
                "caller must CoW-clone first"
            )
        self.storage[0, block_id, :, offset, :] = key
        self.storage[1, block_id, :, offset, :] = value
        meta.num_tokens = max(meta.num_tokens, offset + 1)

    def read_block_kv(self, block_id: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (K, V, num_tokens) views/slices for a physical block."""
        self._check(block_id)
        n = self.blocks[block_id].num_tokens
        # K, V: (num_heads, n, head_dim)
        k = self.storage[0, block_id, :, :n, :]
        v = self.storage[1, block_id, :, :n, :]
        return k, v, n

    # ------------------------------------------------------------------
    # Radix prefix cache
    # ------------------------------------------------------------------

    def cache_block(self, token_hash: int, block_id: int) -> None:
        self._check(block_id)
        self.blocks[block_id].token_hash = int(token_hash)
        self._radix[int(token_hash)] = block_id

    def lookup_cached(self, token_hash: int) -> Optional[int]:
        bid = self._radix.get(int(token_hash))
        if bid is None:
            return None
        if self.blocks[bid].ref_count <= 0:
            self._radix.pop(int(token_hash), None)
            return None
        return bid

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check(self, block_id: int) -> None:
        if not (0 <= block_id < self.num_blocks):
            raise IndexError(f"block_id {block_id} out of range [0, {self.num_blocks})")

    def memory_bytes(self) -> int:
        return int(self.storage.nbytes)

    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free": self.free_count(),
            "used": self.used_count(),
            "radix_entries": len(self._radix),
            "bytes": self.memory_bytes(),
        }
