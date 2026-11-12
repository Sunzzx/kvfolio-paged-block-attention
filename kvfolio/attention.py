"""Dense reference attention and block-paged online-softmax decode.

Paged decode is numerically equivalent to dense attention within float64 eps.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def dense_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    causal: bool = False,
) -> np.ndarray:
    """
    Standard scaled-dot-product attention (reference).

    query:  (H, D)           single decode query, or (H, Tq, D) for multi
    keys:   (H, Tk, D)
    values: (H, Tk, D)
    returns: same leading shape as query with last dim D
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    h, d = k.shape[0], k.shape[-1]
    scale = 1.0 / np.sqrt(d)

    single = q.ndim == 2
    if single:
        q = q[:, None, :]  # (H, 1, D)

    # scores: (H, Tq, Tk)
    scores = np.einsum("hqd,hkd->hqk", q, k) * scale
    if causal:
        tq, tk = scores.shape[1], scores.shape[2]
        mask = np.triu(np.ones((tq, tk), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    out = np.einsum("hqk,hkd->hqd", weights, v)
    if single:
        out = out[:, 0, :]
    return out


def paged_attention_decode(
    query: np.ndarray,
    block_kvs: Sequence[Tuple[np.ndarray, np.ndarray, int]],
) -> np.ndarray:
    """
    Online-softmax attention over non-contiguous physical KV blocks.

    query: (H, D)
    block_kvs: iterable of (K, V, n) where
        K, V have shape (H, block_size, D) or (H, n, D); only first n tokens used.

    Numerically stable online softmax (Milakov & Gimelshein / FlashAttention style):
        For each block with scores S:
            m_new = max(m_prev, max(S))
            alpha = exp(m_prev - m_new)
            beta  = exp(S - m_new)
            l_new = alpha * l_prev + sum(beta)
            acc   = alpha * acc + beta @ V
        o = acc / l_final
    """
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"query must be (H, D), got {q.shape}")
    h, d = q.shape
    scale = 1.0 / np.sqrt(d)

    # running state per head
    m = np.full(h, -np.inf, dtype=np.float64)
    l = np.zeros(h, dtype=np.float64)
    acc = np.zeros((h, d), dtype=np.float64)

    any_token = False
    for k_blk, v_blk, n in block_kvs:
        if n <= 0:
            continue
        any_token = True
        k = np.asarray(k_blk, dtype=np.float64)[:, :n, :]  # (H, n, D)
        v = np.asarray(v_blk, dtype=np.float64)[:, :n, :]
        if k.shape[0] != h or k.shape[-1] != d:
            raise ValueError(f"K shape {k.shape} incompatible with query {(h, d)}")

        # scores: (H, n)
        s = np.einsum("hd,hnd->hn", q, k) * scale
        m_blk = np.max(s, axis=-1)  # (H,)
        m_new = np.maximum(m, m_blk)

        # avoid 0 * (-inf) from exp(-inf - (-inf)) when m was -inf
        alpha = np.where(np.isfinite(m), np.exp(m - m_new), 0.0)
        beta = np.exp(s - m_new[:, None])  # (H, n)

        l = alpha * l + np.sum(beta, axis=-1)
        acc = alpha[:, None] * acc + np.einsum("hn,hnd->hd", beta, v)
        m = m_new

    if not any_token:
        return np.zeros((h, d), dtype=np.float64)

    # l should be > 0 for finite scores
    return acc / l[:, None]


def gather_contiguous_kv(
    block_kvs: Sequence[Tuple[np.ndarray, np.ndarray, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate paged blocks into contiguous (H, T, D) K/V for reference checks."""
    ks: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    for k_blk, v_blk, n in block_kvs:
        if n <= 0:
            continue
        ks.append(np.asarray(k_blk, dtype=np.float64)[:, :n, :])
        vs.append(np.asarray(v_blk, dtype=np.float64)[:, :n, :])
    if not ks:
        raise ValueError("no tokens to gather")
    return np.concatenate(ks, axis=1), np.concatenate(vs, axis=1)
