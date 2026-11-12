"""High-level PagedAttention engine: sequences, page tables, CoW fork, decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .allocator import BlockAllocator
from .attention import dense_attention, gather_contiguous_kv, paged_attention_decode


@dataclass
class SequenceHandle:
    """Logical sequence: page table maps logical block index -> physical block id."""

    seq_id: str
    block_table: List[int] = field(default_factory=list)
    length: int = 0  # number of tokens written
    parent_id: Optional[str] = None


class PagedAttentionEngine:
    """
    vLLM-style PagedAttention runtime.

    - Logical tokens partitioned into blocks of ``block_size``.
    - Physical pool preallocated; page table translates logical -> physical.
    - ``fork_sequence`` shares blocks via reference counts (copy-on-write).
    - ``forward_decode`` runs online-softmax over non-contiguous physical blocks.
    """

    def __init__(
        self,
        num_blocks: int = 256,
        block_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._seqs: Dict[str, SequenceHandle] = {}

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def register_sequence(
        self,
        seq_id: str,
        prompt_len: int = 0,
        *,
        prompt_tokens: Optional[Sequence[int]] = None,
    ) -> SequenceHandle:
        """
        Register a new empty sequence. Optionally pre-size page table for prompt.

        ``prompt_tokens`` is only used for radix-hash keys (optional caching).
        """
        if seq_id in self._seqs:
            raise ValueError(f"sequence '{seq_id}' already exists")
        handle = SequenceHandle(seq_id=seq_id)
        self._seqs[seq_id] = handle
        # prompt tokens themselves are written via write_kv; we only track length here
        _ = prompt_len
        _ = prompt_tokens
        return handle

    def get_sequence(self, seq_id: str) -> SequenceHandle:
        if seq_id not in self._seqs:
            raise KeyError(f"unknown sequence '{seq_id}'")
        return self._seqs[seq_id]

    def free_sequence(self, seq_id: str) -> None:
        handle = self.get_sequence(seq_id)
        for bid in handle.block_table:
            self.allocator.release(bid)
        del self._seqs[seq_id]

    def fork_sequence(self, parent_id: str, child_id: str) -> SequenceHandle:
        """
        Copy-on-write branch: child shares parent's physical blocks (ref++).
        Subsequent writes that hit a shared block will clone it.
        """
        parent = self.get_sequence(parent_id)
        if child_id in self._seqs:
            raise ValueError(f"sequence '{child_id}' already exists")
        child = SequenceHandle(
            seq_id=child_id,
            block_table=list(parent.block_table),
            length=parent.length,
            parent_id=parent_id,
        )
        for bid in child.block_table:
            self.allocator.retain(bid)
        self._seqs[child_id] = child
        return child

    # ------------------------------------------------------------------
    # KV writes
    # ------------------------------------------------------------------

    def write_kv(
        self,
        seq_id: str,
        position: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """
        Write K/V for token at absolute ``position`` (0-indexed).

        Positions must be written in order (position == current length),
        or as the next contiguous slot. Random sparse writes are rejected.
        """
        handle = self.get_sequence(seq_id)
        if position != handle.length:
            raise ValueError(
                f"out-of-order write: expected position {handle.length}, got {position}"
            )

        logical_block = position // self.block_size
        offset = position % self.block_size

        # ensure page table has an entry for this logical block
        while len(handle.block_table) <= logical_block:
            bid = self.allocator.allocate()
            handle.block_table.append(bid)

        phys = handle.block_table[logical_block]
        # CoW break if shared
        if self.allocator.blocks[phys].ref_count > 1:
            new_bid = self.allocator.clone_block(phys)
            self.allocator.release(phys)  # drop this sequence's share of old
            handle.block_table[logical_block] = new_bid
            phys = new_bid

        self.allocator.write_token(phys, offset, key, value)
        handle.length = position + 1

    def append_kv(self, seq_id: str, key: np.ndarray, value: np.ndarray) -> int:
        """Append one token; returns the position written."""
        handle = self.get_sequence(seq_id)
        pos = handle.length
        self.write_kv(seq_id, pos, key, value)
        return pos

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _iter_block_kvs(self, seq_id: str):
        handle = self.get_sequence(seq_id)
        remaining = handle.length
        for bid in handle.block_table:
            if remaining <= 0:
                break
            k, v, n_stored = self.allocator.read_block_kv(bid)
            n = min(n_stored, remaining)
            yield k, v, n
            remaining -= n

    def forward_decode(self, seq_id: str, query: np.ndarray) -> np.ndarray:
        """
        Single-token decode attention against the sequence's paged KV cache.

        query: (num_heads, head_dim)
        returns: (num_heads, head_dim)
        """
        q = np.asarray(query, dtype=np.float64)
        if q.shape != (self.num_heads, self.head_dim):
            raise ValueError(
                f"query shape {q.shape} != {(self.num_heads, self.head_dim)}"
            )
        handle = self.get_sequence(seq_id)
        if handle.length == 0:
            return np.zeros((self.num_heads, self.head_dim), dtype=np.float64)
        return paged_attention_decode(q, list(self._iter_block_kvs(seq_id)))

    def forward_decode_dense_reference(self, seq_id: str, query: np.ndarray) -> np.ndarray:
        """Dense attention over gathered contiguous KV (oracle for tests)."""
        q = np.asarray(query, dtype=np.float64)
        handle = self.get_sequence(seq_id)
        if handle.length == 0:
            return np.zeros((self.num_heads, self.head_dim), dtype=np.float64)
        k, v = gather_contiguous_kv(list(self._iter_block_kvs(seq_id)))
        return dense_attention(q, k, v)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def memory_stats(self) -> dict:
        base = self.allocator.stats()
        base["num_sequences"] = len(self._seqs)
        base["total_tokens"] = sum(s.length for s in self._seqs.values())
        # naive contiguous reservation would pad each seq to max observed length
        # report effective utilization vs full-pool reservation
        used_token_slots = sum(
            len(s.block_table) * self.block_size for s in self._seqs.values()
        )
        base["logical_token_slots"] = used_token_slots
        base["external_fragmentation"] = 0.0  # by construction
        return base

    def page_table(self, seq_id: str) -> List[int]:
        return list(self.get_sequence(seq_id).block_table)

    def status_report(self) -> str:
        st = self.memory_stats()
        lines = [
            "KVFolio PagedAttention Engine",
            f"  blocks: {st['used']}/{st['num_blocks']} used "
            f"(free={st['free']}, block_size={st['block_size']})",
            f"  sequences: {st['num_sequences']}  tokens: {st['total_tokens']}",
            f"  pool bytes: {st['bytes']:,}  external_frag: {st['external_fragmentation']:.2%}",
            f"  radix cache entries: {st['radix_entries']}",
        ]
        for sid, seq in self._seqs.items():
            shared = sum(
                1 for b in seq.block_table if self.allocator.blocks[b].ref_count > 1
            )
            lines.append(
                f"  · {sid}: len={seq.length} blocks={len(seq.block_table)} "
                f"shared={shared} parent={seq.parent_id}"
            )
        return "\n".join(lines)
