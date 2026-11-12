"""KVFolio — PagedAttention & block-paged KV-cache virtual memory."""

from .engine import PagedAttentionEngine, SequenceHandle
from .attention import dense_attention, paged_attention_decode
from .allocator import BlockAllocator

__all__ = [
    "PagedAttentionEngine",
    "SequenceHandle",
    "BlockAllocator",
    "dense_attention",
    "paged_attention_decode",
]

__version__ = "0.1.0"
