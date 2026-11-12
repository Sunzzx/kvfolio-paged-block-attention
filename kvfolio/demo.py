"""Interactive CLI demo for KVFolio PagedAttention."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from kvfolio import PagedAttentionEngine


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_visual(seed: int = 0) -> int:
    """Live visualization of page tables, CoW branching, and numerical identity."""
    rng = np.random.default_rng(seed)
    eng = PagedAttentionEngine(num_blocks=64, block_size=8, num_heads=4, head_dim=16)

    _banner("1. Prompt prefilling (on-demand block allocation)")
    eng.register_sequence("session")
    prompt_len = 20
    for t in range(prompt_len):
        eng.append_kv(
            "session",
            rng.normal(size=(4, 16)),
            rng.normal(size=(4, 16)),
        )
    print(f"  wrote {prompt_len} prompt tokens")
    print(f"  page table: {eng.page_table('session')}")
    print(eng.status_report())

    _banner("2. Copy-on-Write fork for parallel sampling (beam / n-best)")
    for branch in ("beam_0", "beam_1", "beam_2"):
        eng.fork_sequence("session", branch)
    print("  forked 3 beams sharing prompt blocks (ref_count += 1 each)")
    print(eng.status_report())

    _banner("3. Divergent decode — CoW breaks only on write")
    for step in range(5):
        for branch in ("beam_0", "beam_1", "beam_2"):
            eng.append_kv(
                branch,
                rng.normal(size=(4, 16)),
                rng.normal(size=(4, 16)),
            )
        print(f"  step {step + 1}: beams advanced; shared prompt blocks retained")
    print(eng.status_report())

    _banner("4. Numerical identity: paged online-softmax vs dense reference")
    q = rng.normal(size=(4, 16))
    for sid in ("session", "beam_0", "beam_1", "beam_2"):
        paged = eng.forward_decode(sid, q)
        dense = eng.forward_decode_dense_reference(sid, q)
        err = float(np.max(np.abs(paged - dense)))
        rel = err / (float(np.max(np.abs(dense))) + 1e-15)
        print(f"  {sid:10s}  max|Δ|={err:.3e}  rel={rel:.3e}  len={eng.get_sequence(sid).length}")
        assert rel < 1e-10

    _banner("5. Memory: fixed pool, zero external fragmentation")
    st = eng.memory_stats()
    print(f"  pool bytes          : {st['bytes']:,}")
    print(f"  blocks used/total   : {st['used']}/{st['num_blocks']}")
    print(f"  external frag       : {st['external_fragmentation']:.2%}")
    print(f"  logical token slots : {st['logical_token_slots']}")
    print(f"  live tokens         : {st['total_tokens']}")
    print()
    print("Demo complete — all invariants green.")
    return 0


def demo_bench(
    heads: int = 8,
    dim: int = 64,
    block_size: int = 16,
    batch_size: int = 8,
    max_len: int = 512,
    seed: int = 0,
) -> int:
    """Throughput / memory micro-benchmark across sequence lengths."""
    rng = np.random.default_rng(seed)
    # size pool for worst case: batch * ceil(max_len / block_size) * small slack
    blocks_needed = batch_size * ((max_len + block_size - 1) // block_size) + 16
    eng = PagedAttentionEngine(
        num_blocks=blocks_needed,
        block_size=block_size,
        num_heads=heads,
        head_dim=dim,
    )

    _banner(
        f"Benchmark  H={heads} D={dim} B={block_size} batch={batch_size} max_len={max_len}"
    )
    print(
        f"{'L':>6}  {'paged_ms':>10}  {'dense_ms':>10}  {'max|Δ|':>12}  "
        f"{'blocks':>7}  {'save%':>7}"
    )

    lengths = [64, 128, 256, 512]
    lengths = [L for L in lengths if L <= max_len]
    for L in lengths:
        # reset sequences
        for sid in list(eng._seqs.keys()):
            eng.free_sequence(sid)

        for b in range(batch_size):
            sid = f"b{b}"
            eng.register_sequence(sid)
            for _ in range(L):
                eng.append_kv(
                    sid,
                    rng.normal(size=(heads, dim)),
                    rng.normal(size=(heads, dim)),
                )

        q = rng.normal(size=(heads, dim))

        t0 = time.perf_counter()
        paged_outs = [eng.forward_decode(f"b{b}", q) for b in range(batch_size)]
        t_paged = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        dense_outs = [
            eng.forward_decode_dense_reference(f"b{b}", q) for b in range(batch_size)
        ]
        t_dense = (time.perf_counter() - t0) * 1e3

        err = max(
            float(np.max(np.abs(p - d))) for p, d in zip(paged_outs, dense_outs)
        )
        used = eng.memory_stats()["used"]
        # naive would reserve max_len contiguous per batch item in same head config
        naive_blocks = batch_size * ((max_len + block_size - 1) // block_size)
        save = 100.0 * (1.0 - used / max(naive_blocks, 1))
        print(
            f"{L:6d}  {t_paged:10.3f}  {t_dense:10.3f}  {err:12.3e}  "
            f"{used:7d}  {save:6.1f}%"
        )

    print()
    print(eng.status_report())
    return 0


def demo_verify() -> int:
    """Strict numerical equivalence sweep."""
    rng = np.random.default_rng(123)
    _banner("Verify: paged == dense across configs")
    configs = [
        (2, 8, 4, 7),
        (4, 16, 8, 33),
        (8, 32, 16, 100),
        (8, 64, 16, 257),
    ]
    worst = 0.0
    for h, d, bs, n in configs:
        eng = PagedAttentionEngine(num_blocks=64, block_size=bs, num_heads=h, head_dim=d)
        eng.register_sequence("s")
        for _ in range(n):
            eng.append_kv("s", rng.normal(size=(h, d)), rng.normal(size=(h, d)))
        q = rng.normal(size=(h, d))
        p = eng.forward_decode("s", q)
        r = eng.forward_decode_dense_reference("s", q)
        err = float(np.max(np.abs(p - r)))
        worst = max(worst, err)
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  H={h} D={d} B={bs} L={n:4d}  max|Δ|={err:.3e}  [{status}]")
        if err >= 1e-10:
            return 1
    print(f"\n  worst max|Δ| = {worst:.3e}")
    print("  all configs PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kvfolio",
        description="KVFolio — PagedAttention & block-paged KV virtual memory",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="interactive CoW / page-table demo")
    p_demo.add_argument("--seed", type=int, default=0)

    p_bench = sub.add_parser("bench", help="throughput and memory benchmark")
    p_bench.add_argument("--heads", type=int, default=8)
    p_bench.add_argument("--dim", type=int, default=64)
    p_bench.add_argument("--block-size", type=int, default=16)
    p_bench.add_argument("--batch-size", type=int, default=8)
    p_bench.add_argument("--max-len", type=int, default=512)
    p_bench.add_argument("--seed", type=int, default=0)

    sub.add_parser("verify", help="numerical equivalence sweep")
    sub.add_parser("status", help="print engine status on a tiny fixture")

    args = parser.parse_args(argv)
    cmd = args.cmd or "demo"

    if cmd == "demo":
        return demo_visual(seed=args.seed)
    if cmd == "bench":
        return demo_bench(
            heads=args.heads,
            dim=args.dim,
            block_size=args.block_size,
            batch_size=args.batch_size,
            max_len=args.max_len,
            seed=args.seed,
        )
    if cmd == "verify":
        return demo_verify()
    if cmd == "status":
        eng = PagedAttentionEngine()
        eng.register_sequence("idle")
        print(eng.status_report())
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
