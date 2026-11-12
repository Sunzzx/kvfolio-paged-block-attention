"""Comprehensive tests for KVFolio PagedAttention engine."""

from __future__ import annotations

import numpy as np
import pytest

from kvfolio import PagedAttentionEngine, dense_attention, paged_attention_decode
from kvfolio.allocator import BlockAllocator
from kvfolio.attention import gather_contiguous_kv


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


def test_allocator_oom_and_free_cycle():
    alloc = BlockAllocator(num_blocks=3, block_size=4, num_heads=2, head_dim=8)
    ids = [alloc.allocate() for _ in range(3)]
    assert alloc.free_count() == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        alloc.allocate()
    alloc.release(ids[1])
    assert alloc.free_count() == 1
    new_id = alloc.allocate()
    assert new_id == ids[1]


def test_allocator_write_shape_guard():
    alloc = BlockAllocator(num_blocks=2, block_size=4, num_heads=2, head_dim=8)
    bid = alloc.allocate()
    bad = np.zeros((2, 7))
    with pytest.raises(ValueError, match="key shape"):
        alloc.write_token(bid, 0, bad, np.zeros((2, 8)))


def test_allocator_rejects_write_on_shared_block():
    alloc = BlockAllocator(num_blocks=4, block_size=4, num_heads=1, head_dim=4)
    bid = alloc.allocate()
    alloc.retain(bid)  # ref_count == 2
    k = np.ones((1, 4))
    v = np.ones((1, 4))
    with pytest.raises(RuntimeError, match="shared"):
        alloc.write_token(bid, 0, k, v)


def test_clone_block_deep_copy_independence():
    alloc = BlockAllocator(num_blocks=4, block_size=4, num_heads=2, head_dim=4)
    src = alloc.allocate()
    k = np.arange(8.0).reshape(2, 4)
    v = np.arange(8.0, 16.0).reshape(2, 4)
    alloc.write_token(src, 0, k, v)
    dst = alloc.clone_block(src)
    # mutate source
    alloc.write_token(src, 0, np.zeros((2, 4)), np.zeros((2, 4)))
    k2, v2, n = alloc.read_block_kv(dst)
    assert n == 1
    np.testing.assert_allclose(k2[:, 0, :], k)
    np.testing.assert_allclose(v2[:, 0, :], v)


# ---------------------------------------------------------------------------
# Online softmax numerical identity
# ---------------------------------------------------------------------------


def test_paged_matches_dense_single_block():
    rng = np.random.default_rng(0)
    h, d, n = 4, 16, 8
    q = rng.normal(size=(h, d))
    k = rng.normal(size=(h, n, d))
    v = rng.normal(size=(h, n, d))
    ref = dense_attention(q, k, v)
    out = paged_attention_decode(q, [(k, v, n)])
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_paged_matches_dense_multi_block_variable_fill():
    rng = np.random.default_rng(42)
    h, d, bs = 8, 32, 16
    lengths = [16, 16, 7]  # three physical blocks, last partial
    blocks = []
    ks, vs = [], []
    for n in lengths:
        k = rng.normal(size=(h, bs, d))
        v = rng.normal(size=(h, bs, d))
        blocks.append((k, v, n))
        ks.append(k[:, :n, :])
        vs.append(v[:, :n, :])
    k_cat = np.concatenate(ks, axis=1)
    v_cat = np.concatenate(vs, axis=1)
    q = rng.normal(size=(h, d))
    ref = dense_attention(q, k_cat, v_cat)
    out = paged_attention_decode(q, blocks)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_paged_empty_returns_zeros():
    q = np.ones((2, 4))
    out = paged_attention_decode(q, [])
    np.testing.assert_array_equal(out, np.zeros((2, 4)))


def test_dense_causal_masks_future():
    rng = np.random.default_rng(1)
    h, t, d = 2, 4, 8
    q = rng.normal(size=(h, t, d))
    k = rng.normal(size=(h, t, d))
    v = rng.normal(size=(h, t, d))
    out = dense_attention(q, k, v, causal=True)
    # position 0 can only see itself: equal to v[:,0] after attn on single key
    mono = dense_attention(q[:, 0, :], k[:, :1, :], v[:, :1, :])
    np.testing.assert_allclose(out[:, 0, :], mono, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Engine: page tables, CoW, decode equivalence
# ---------------------------------------------------------------------------


def _fill_seq(engine: PagedAttentionEngine, seq_id: str, n: int, rng: np.random.Generator):
    engine.register_sequence(seq_id)
    for t in range(n):
        k = rng.normal(size=(engine.num_heads, engine.head_dim))
        v = rng.normal(size=(engine.num_heads, engine.head_dim))
        engine.append_kv(seq_id, k, v)


def test_engine_decode_matches_dense_across_block_boundary():
    rng = np.random.default_rng(7)
    eng = PagedAttentionEngine(num_blocks=64, block_size=8, num_heads=4, head_dim=16)
    n = 23  # spans 3 blocks (8+8+7)
    _fill_seq(eng, "s0", n, rng)
    q = rng.normal(size=(4, 16))
    paged = eng.forward_decode("s0", q)
    dense = eng.forward_decode_dense_reference("s0", q)
    rel = np.max(np.abs(paged - dense)) / (np.max(np.abs(dense)) + 1e-15)
    assert rel < 1e-12
    np.testing.assert_allclose(paged, dense, rtol=1e-12, atol=1e-12)
    assert eng.get_sequence("s0").length == n
    assert len(eng.page_table("s0")) == 3


def test_engine_cow_fork_shares_then_diverges():
    rng = np.random.default_rng(99)
    eng = PagedAttentionEngine(num_blocks=32, block_size=4, num_heads=2, head_dim=8)
    _fill_seq(eng, "parent", 6, rng)  # 2 blocks
    parent_table = eng.page_table("parent")
    eng.fork_sequence("parent", "child")
    child_table = eng.page_table("child")
    assert parent_table == child_table
    # both blocks shared
    for bid in parent_table:
        assert eng.allocator.blocks[bid].ref_count == 2

    # append on child — last block may be partially filled (offset 2 in block 1)
    # writing position 6 goes to logical block 1 offset 2 — must CoW that block
    k = rng.normal(size=(2, 8))
    v = rng.normal(size=(2, 8))
    eng.append_kv("child", k, v)

    # parent unchanged length; child longer
    assert eng.get_sequence("parent").length == 6
    assert eng.get_sequence("child").length == 7

    # after CoW on the written block, tables diverge on that index
    pt = eng.page_table("parent")
    ct = eng.page_table("child")
    # first block still shared if not written
    assert pt[0] == ct[0]
    assert eng.allocator.blocks[pt[0]].ref_count == 2
    # second block should be private to child after write into it
    assert pt[1] != ct[1]
    assert eng.allocator.blocks[pt[1]].ref_count == 1
    assert eng.allocator.blocks[ct[1]].ref_count == 1

    # attention still valid on both
    q = rng.normal(size=(2, 8))
    out_p = eng.forward_decode("parent", q)
    out_c = eng.forward_decode("child", q)
    assert out_p.shape == (2, 8)
    assert out_c.shape == (2, 8)
    # outputs should differ because child has an extra token
    assert not np.allclose(out_p, out_c)


def test_engine_free_sequence_returns_blocks():
    rng = np.random.default_rng(3)
    eng = PagedAttentionEngine(num_blocks=16, block_size=4, num_heads=2, head_dim=4)
    free0 = eng.allocator.free_count()
    _fill_seq(eng, "tmp", 9, rng)
    used = free0 - eng.allocator.free_count()
    assert used == 3  # ceil(9/4)
    eng.free_sequence("tmp")
    assert eng.allocator.free_count() == free0


def test_engine_fork_then_free_parent_keeps_child_alive():
    rng = np.random.default_rng(5)
    eng = PagedAttentionEngine(num_blocks=16, block_size=4, num_heads=2, head_dim=4)
    _fill_seq(eng, "p", 5, rng)
    eng.fork_sequence("p", "c")
    eng.free_sequence("p")
    # child still decodable
    q = rng.normal(size=(2, 4))
    out = eng.forward_decode("c", q)
    ref = eng.forward_decode_dense_reference("c", q)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_out_of_order_write_rejected():
    eng = PagedAttentionEngine(num_blocks=8, block_size=4, num_heads=1, head_dim=4)
    eng.register_sequence("s")
    eng.append_kv("s", np.ones((1, 4)), np.ones((1, 4)))
    with pytest.raises(ValueError, match="out-of-order"):
        eng.write_kv("s", 5, np.ones((1, 4)), np.ones((1, 4)))


def test_external_fragmentation_is_zero():
    rng = np.random.default_rng(11)
    eng = PagedAttentionEngine(num_blocks=64, block_size=16, num_heads=4, head_dim=8)
    for i, n in enumerate([3, 17, 33, 1, 64]):
        _fill_seq(eng, f"s{i}", n, rng)
    st = eng.memory_stats()
    assert st["external_fragmentation"] == 0.0
    assert st["num_sequences"] == 5


def test_o1_block_allocation_amortized():
    """Allocation from free list is O(1); filling N tokens uses ceil(N/B) allocs."""
    eng = PagedAttentionEngine(num_blocks=128, block_size=16, num_heads=2, head_dim=8)
    eng.register_sequence("s")
    rng = np.random.default_rng(0)
    n = 100
    for _ in range(n):
        eng.append_kv(
            "s",
            rng.normal(size=(2, 8)),
            rng.normal(size=(2, 8)),
        )
    assert len(eng.page_table("s")) == (n + 15) // 16


def test_memory_pool_fixed_footprint():
    eng = PagedAttentionEngine(num_blocks=32, block_size=16, num_heads=8, head_dim=64)
    bytes0 = eng.allocator.memory_bytes()
    rng = np.random.default_rng(0)
    _fill_seq(eng, "a", 40, rng)
    _fill_seq(eng, "b", 40, rng)
    # pool is preallocated — nbytes must not grow
    assert eng.allocator.memory_bytes() == bytes0
    expected = 2 * 32 * 8 * 16 * 64 * 8  # float64
    assert bytes0 == expected


def test_status_report_smoke():
    eng = PagedAttentionEngine(num_blocks=8, block_size=4, num_heads=1, head_dim=4)
    eng.register_sequence("x")
    eng.append_kv("x", np.zeros((1, 4)), np.zeros((1, 4)))
    text = eng.status_report()
    assert "KVFolio" in text
    assert "x:" in text
