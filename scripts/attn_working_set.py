# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimate the working-set size of one iteration of the online-softmax inner
loop in SpyreAttentionImpl (_create_compilable_page_attn).

Two corrections over the first version, both prompted by the lq=128 device sweep
that FALSIFIED the original spill prediction (.plans/06-does-tiling-help-working-set.md):

1. LIVENESS PEAK, not sum. The old model summed all ~12 transients as if
   simultaneously live. The allocator reuses storage once a buffer is dead, so
   the true residency figure is the maximum SIMULTANEOUSLY-live bytes across the
   iteration's dataflow. live_peak_bytes walks the i>0 branch for that.

2. PER-CORE, not whole-op, and NOT "2 MiB * 32". The always-on work_division
   pass splits each op's iteration space across up to NUM_CORES cores, so the
   2 MiB scratchpad applies PER CORE to a per-core SLICE. The split per dim is
   the largest DIVISOR of that dim's size <= cores-remaining (core_split), and
   the product over output dims is <= NUM_CORES. So a dim of size 8 yields at
   most 8 cores; the budget does not pool. Per-core bytes = live_peak /
   (core split on each tensor's dims).

CAVEATS (pure-arithmetic UPPER BOUND, no device/torch): ignores in-place reuse,
sub-buffer aliasing, allocator packing; treat crossovers as order-of-magnitude.
Coarse tiling is RESIDENCY, not parallelism (work_division maps cores
independently of any tiling hint) -- so there is no "corelets used" output.
NUM_CORES=32 per work_division's cost model; the old "64 corelets" is unverified.

Usage:
    python scripts/attn_working_set.py \\
        --head-size 128 --num-query-heads 32 --num-kv-heads 8 \\
        --block-size 128 --padded-query-len 32 \\
        --tile-kv-heads 1,2,4,8 --scratchpad-kib 2048 --num-cores 32
"""

import argparse
import math

_DTYPE_BYTES = 2  # float16


def _fmt(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} GiB"


def _shapes(hkv, qpk, pql, block_size, head_size):
    return {
        "S": (hkv, qpk, pql, block_size),  # scores / tile_probs
        "R": (hkv, qpk, pql, 1),  # per-row reductions
        "O": (hkv, qpk, pql, head_size),  # tile_output
        "KV": (hkv, 1, block_size, head_size),  # k_page_4d / v_page_4d
        "Q": (hkv, qpk, pql, head_size),  # q tile
        "M": (pql, block_size),  # mask_tile
        "A": (hkv, qpk, 1, block_size),  # alibi_bias_tile
    }


def carried_tensors(hkv, qpk, pql, block_size, head_size):
    """Accumulators that survive to the next iteration: tile_output/max/sum."""
    s = _shapes(hkv, qpk, pql, block_size, head_size)
    return {"tile_output": s["O"], "tile_max": s["R"], "tile_sum": s["R"]}


def peak_tensors(hkv, qpk, pql, block_size, head_size, has_alibi):
    """Every distinct buffer touched in the i>0 branch (for the BREAKDOWN only,
    NOT a simultaneously-live set -- use live_peak_bytes for residency)."""
    s = _shapes(hkv, qpk, pql, block_size, head_size)
    t = {
        "tile_output": s["O"], "tile_max": s["R"], "tile_sum": s["R"],
        "q": s["Q"], "k_page_4d": s["KV"], "v_page_4d": s["KV"],
        "mask_tile": s["M"], "scores": s["S"], "scores_max": s["R"],
        "tile_probs": s["S"], "new_max": s["R"], "rescale": s["R"],
    }
    if has_alibi:
        t["alibi_bias_tile"] = s["A"]
    return t


def working_set_bytes(tensors):
    return sum(math.prod(shape) * _DTYPE_BYTES for shape in tensors.values())


def _sz(shape):
    return math.prod(shape) * _DTYPE_BYTES


def core_split(size: int, max_cores: int) -> int:
    """Largest divisor of size that is <= max_cores (work_division.py:173-186).
    NOT min(size, max_cores): must divide evenly (size=8,cores=32 -> 8)."""
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i
    return 1


def op_core_splits(output_dim_sizes, max_cores):
    """Greedy per-dim core split over output dims; product <= max_cores
    (multi_dim_iteration_space_split output-dim pass, work_division.py:258-277)."""
    splits = {name: 1 for name, _ in output_dim_sizes}
    remaining = max_cores
    for name, size in output_dim_sizes:
        if remaining <= 1:
            break
        s = core_split(size, remaining)
        if s > 1:
            splits[name] = s
            remaining //= s
    return splits


_TENSOR_AXES = {
    "tile_output": ("kv_head", "qpk", "lq", "d"),
    "tile_max": ("kv_head", "qpk", "lq"),
    "tile_sum": ("kv_head", "qpk", "lq"),
    "q": ("kv_head", "qpk", "lq", "d"),
    "k_page_4d": ("kv_head", "blk", "d"),
    "v_page_4d": ("kv_head", "blk", "d"),
    "mask_tile": ("lq", "blk"),
    "scores": ("kv_head", "qpk", "lq", "blk"),
    "scores_max": ("kv_head", "qpk", "lq"),
    "new_max": ("kv_head", "qpk", "lq"),
    "rescale": ("kv_head", "qpk", "lq"),
    "tile_probs": ("kv_head", "qpk", "lq", "blk"),
    "weighted": ("kv_head", "qpk", "lq", "d"),
    "alibi_bias_tile": ("kv_head", "qpk", "blk"),
}


def _per_core_divisor(buf, splits):
    d = 1
    for ax in _TENSOR_AXES.get(buf, ()):
        d *= splits.get(ax, 1)
    return d


def _iteration_splits(hkv, qpk, pql, head_size, num_cores):
    """Core split over the output dims the carries live under (kv_head,qpk,lq),
    priority kv_head > qpk > lq. blk is omitted (PV-matmul reduction dim)."""
    return op_core_splits([("kv_head", hkv), ("qpk", qpk), ("lq", pql)], num_cores)


def live_peak_bytes(hkv, qpk, pql, block_size, head_size, has_alibi, num_cores=1):
    """Max simultaneously-live bytes across the i>0 dataflow (spyre_attn.py:505-519).
    num_cores>1 divides each buffer by the work_division core split on its dims,
    returning the PER-CORE peak. Returns (peak_bytes, step_label)."""
    s = _shapes(hkv, qpk, pql, block_size, head_size)
    O, R, S, KV, Q, M, A = s["O"], s["R"], s["S"], s["KV"], s["Q"], s["M"], s["A"]
    splits = _iteration_splits(hkv, qpk, pql, head_size, num_cores)
    raw = {
        "tile_output": _sz(O), "tile_max": _sz(R), "tile_sum": _sz(R),
        "q": _sz(Q), "k_page_4d": _sz(KV), "v_page_4d": _sz(KV),
        "mask_tile": _sz(M), "scores": _sz(S), "scores_max": _sz(R),
        "new_max": _sz(R), "rescale": _sz(R), "tile_probs": _sz(S),
        "weighted": _sz(O),
    }
    if has_alibi:
        raw["alibi_bias_tile"] = _sz(A)
    size = {b: max(1, raw[b] // _per_core_divisor(b, splits)) for b in raw}

    alibi_reads = ["alibi_bias_tile"] if has_alibi else []
    steps = [
        ("scores=q@kT+mask", ["scores"], ["q", "k_page_4d", "mask_tile", *alibi_reads]),
        ("scores_max=amax", ["scores_max"], []),
        ("new_max=max", ["new_max"], ["scores_max"]),
        ("rescale=exp", ["rescale"], []),
        ("tile_output*=rescale", [], []),
        ("tile_sum*=rescale", [], ["rescale"]),
        ("tile_probs=exp", ["tile_probs"], ["scores"]),
        ("weighted=probs@v", ["weighted"], ["v_page_4d"]),
        ("tile_output+=weighted", [], ["weighted"]),
        ("tile_sum+=sum(probs)", [], ["tile_probs"]),
        ("tile_max=new_max", [], ["new_max", "tile_max"]),
    ]
    live = {"tile_output", "tile_max", "tile_sum", "q", "k_page_4d",
            "v_page_4d", "mask_tile"}
    if has_alibi:
        live.add("alibi_bias_tile")
    peak = sum(size[b] for b in live)
    peak_label = "iter start (inputs+carries)"
    for label, born, dead in steps:
        for b in born:
            live.add(b)
        cur = sum(size[b] for b in live)
        if cur > peak:
            peak, peak_label = cur, label
        for b in dead:
            live.discard(b)
    return peak, peak_label


def carry_bytes(hkv, qpk, pql, block_size, head_size, num_cores=1):
    """Per-core loop-carried accumulator bytes (tile_output/max/sum)."""
    s = _shapes(hkv, qpk, pql, block_size, head_size)
    splits = _iteration_splits(hkv, qpk, pql, head_size, num_cores)
    raw = {"tile_output": _sz(s["O"]), "tile_max": _sz(s["R"]), "tile_sum": _sz(s["R"])}
    return sum(max(1, raw[b] // _per_core_divisor(b, splits)) for b in raw)


def report(head_size, num_query_heads, num_kv_heads, block_size, padded_query_len,
           tile_counts, has_alibi, scratchpad_bytes, num_cores):
    qpk = num_query_heads // num_kv_heads
    print(
        f"shape: head_size={head_size} num_query_heads={num_query_heads} "
        f"num_kv_heads={num_kv_heads} (qpk={qpk}) block_size={block_size} "
        f"padded_query_len={padded_query_len} alibi={has_alibi} dtype=fp16"
    )
    if scratchpad_bytes is not None:
        print(
            f"scratchpad budget: {_fmt(scratchpad_bytes)} PER CORE "
            f"(work_division across up to {num_cores} cores; does NOT pool)"
        )
    print()
    for n in tile_counts:
        if num_kv_heads % n != 0:
            print(f"tile_kv_heads={n}: SKIP (does not divide num_kv_heads={num_kv_heads})")
            continue
        hkv = num_kv_heads // n
        splits = _iteration_splits(hkv, qpk, padded_query_len, head_size, num_cores)
        cores_used = math.prod(splits.values())
        whole, _ = live_peak_bytes(
            hkv, qpk, padded_query_len, block_size, head_size, has_alibi, 1
        )
        per_core, label = live_peak_bytes(
            hkv, qpk, padded_query_len, block_size, head_size, has_alibi, num_cores
        )
        carry_pc = carry_bytes(hkv, qpk, padded_query_len, block_size, head_size, num_cores)
        split_str = "x".join(f"{k}:{v}" for k, v in splits.items() if v > 1) or "none"
        line = (
            f"tile_kv_heads={n}: hkv_per_tile={hkv}  cores_used={cores_used} "
            f"({split_str})  whole_op_peak={_fmt(whole)}  "
            f"per_core_peak={_fmt(per_core)} @ {label}  per_core_carry={_fmt(carry_pc)}"
        )
        if scratchpad_bytes is not None:
            fits = "FITS" if per_core <= scratchpad_bytes else "SPILLS"
            line += f"  (per_core {per_core / scratchpad_bytes * 100:.1f}% of 2MiB, {fits})"
        print(line)

    print("\nper-tensor breakdown (tile_kv_heads=1, all buffers in i>0 branch):")
    print("(distinct buffers, whole-op bytes; NOT simultaneously live)")
    base = peak_tensors(num_kv_heads, qpk, padded_query_len, block_size, head_size, has_alibi)
    carried_names = set(carried_tensors(num_kv_heads, qpk, padded_query_len, block_size, head_size))
    for name, shape in sorted(base.items(), key=lambda kv: -math.prod(kv[1])):
        b = math.prod(shape) * _DTYPE_BYTES
        tag = "carry" if name in carried_names else "transient"
        print(f"  {name:16s} {str(shape):28s} {_fmt(b):>10s}  [{tag}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--num-query-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--padded-query-len", type=int, default=32)
    ap.add_argument("--tile-kv-heads", type=str, default="1,2,4,8",
                    help="comma-separated tile_kv_heads values to compare")
    ap.add_argument("--alibi", action="store_true", help="include ALiBi bias tile")
    ap.add_argument("--scratchpad-kib", type=int, required=True,
                    help="per-core scratchpad budget in KiB (Spyre: 2048)")
    ap.add_argument("--num-cores", type=int, default=32,
                    help="cores work_division splits an op across (Spyre: 32)")
    args = ap.parse_args()
    tile_counts = [int(x) for x in args.tile_kv_heads.split(",") if x]
    scratchpad_bytes = args.scratchpad_kib * 1024 if args.scratchpad_kib else None
    report(
        head_size=args.head_size, num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads, block_size=args.block_size,
        padded_query_len=args.padded_query_len, tile_counts=tile_counts,
        has_alibi=args.alibi, scratchpad_bytes=scratchpad_bytes, num_cores=args.num_cores,
    )


if __name__ == "__main__":
    main()
