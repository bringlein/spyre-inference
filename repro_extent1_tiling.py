"""Minimal reproducer: coarse-tiling a head dim to per-tile EXTENT 1 silently
corrupts the output of a paged-attention loop that reads an indirectly-gathered,
slot-major K/V cache.

Bug (see .plans/05-tiling-resolution-summary.md):
  spyre_hint(tiles={"H": N}) with N == H divides the tiled head dim down to
  extent 1. Inductor's SqueezeView.squeezer unconditionally drops the extent-1
  dim from dep.index, so _general_tile_advance can no longer read it and falls
  back to squeezed_advance_per_read. That fallback's host_stride is a
  squeezed-iteration-space ranges product, NOT the buffer's memory stride; for
  the indirectly-gathered slot-major K/V page that substitute value is absent
  from the page's stride_map, so tiling_expr_to_device_expr picks the WRONG
  device axis and advances by the wrong amount. Every tile but tile 0 then reads
  the wrong head. No error, no warning -- silently wrong numbers.

  A tile count that leaves per-tile extent >= 2 (e.g. N == H // 2) is correct.

Expected result on the Spyre pod:
  [tiles=H/2  -> extent 2] PASS  (matches CPU reference)
  [tiles=H/1  -> extent 1] FAIL  (silently wrong; tile 0 ok, tiles 1..N-1 wrong)

Both compile caches are disabled below: without that, a second run silently
reuses the first run's artifact and reports the wrong tile count (the count
reaches the compiler through the spyre_hint side-channel, not the FX-graph
cache key). See the METHODOLOGY note in plan 05.

Run on the pod:
  cd /home/ngl/helion-experiments/spyre-inference
  /opt/spyre-inference/bin/python repro_extent1_tiling.py
"""

import math

import torch

# Defeat both compile caches -- mandatory, or a later candidate reuses an
# earlier artifact and reports THAT candidate's tile count (plan 05 methodology).
torch._inductor.config.force_disable_caches = True
torch.compiler.config.force_disable_caches = True

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import spyre_hint
from torch_spyre._inductor.wsr.propagate_named_dims import (
    declare_tensor_dim,
    name_tensor_dims,
)

# Small shape: ONE KV block per sequence so no other page dilutes the wrong head
# in the softmax denominator (the error is largest at a single page -- plan 05).
B = 1  # batch
H = 8  # kv heads == query heads (the dim we tile to extent 1)
D = 128  # head_size (stays on the stick)
kv_block_size = 64  # BS
Lq = 64  # query tokens (one block, <= q_block_size)
blocks = 1  # KV blocks per sequence
Lk = kv_block_size * blocks
Tk = blocks * B
cache_size = 8  # total pages in the paged cache


def paged_cpu(queries, key_cache, value_cache, block_table):
    """Plain online-softmax paged attention on CPU -- the reference."""
    scale = 1.0 / math.sqrt(D)
    q = queries.view(B, Lq, H, D).permute(0, 2, 1, 3).float()  # B,H,Lq,D
    out = torch.zeros(B, H, Lq, D)
    m = torch.full((B, H, Lq, 1), float("-inf"))
    l = torch.zeros((B, H, Lq, 1))
    for blk in range(blocks):
        page = block_table[:, blk]  # B
        k = key_cache[page].permute(0, 2, 1, 3).float()  # B,H,BS,D
        v = value_cache[page].permute(0, 2, 1, 3).float()  # B,H,BS,D
        s = torch.matmul(q, k.transpose(-2, -1)) * scale  # B,H,Lq,BS
        blk_max = torch.amax(s, dim=-1, keepdim=True)
        new_m = torch.maximum(m, blk_max)
        corr = torch.exp(m - new_m)
        p = torch.exp(s - new_m)
        out = out * corr + torch.matmul(p, v)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        m = new_m
    out = out / l  # B,H,Lq,D
    return out.permute(0, 2, 1, 3).reshape(Lq, H, D).half()


def make_paged_spyre(head_tiles: int):
    """Factory: paged attention with the head dim H split into `head_tiles`
    coarse tiles. head_tiles == H -> per-tile extent 1 (the bug)."""

    def paged_spyre(queries, key_cache, value_cache, block_table):
        scale = 1.0 / math.sqrt(D)
        q = queries.view(B, Lq, H, D).permute(0, 2, 1, 3)  # B,H,Lq,D
        out = torch.zeros(B, H, Lq, D, dtype=torch.float16, device=queries.device)
        m = torch.full(
            (B, H, Lq, 1), float("-inf"), dtype=torch.float16, device=queries.device
        )
        l = torch.zeros((B, H, Lq, 1), dtype=torch.float16, device=queries.device)

        # One tile scope over the whole (unrolled) block loop; tile the head dim H.
        with spyre_hint(tiles={"H": head_tiles}):
            for blk in range(blocks):
                page = block_table[:, blk]
                # INDIRECT ACCESS: gather the slot-major page, permute to head-major.
                k = key_cache[page].permute(0, 2, 1, 3)  # B,H,BS,D
                v = value_cache[page].permute(0, 2, 1, 3)  # B,H,BS,D
                s = torch.matmul(q, k.transpose(-2, -1)) * scale  # B,H,Lq,BS
                blk_max = torch.amax(s, dim=-1, keepdim=True)
                new_m = torch.maximum(m, blk_max)
                corr = torch.exp(m - new_m)
                p = torch.exp(s - new_m)
                out = out * corr + torch.matmul(p, v)
                l = l * corr + p.sum(dim=-1, keepdim=True)
                m = new_m
            out = out / l
        return out.permute(0, 2, 1, 3).reshape(Lq, H, D)

    return paged_spyre


def run_candidate(head_tiles, inputs_cpu, inputs_spyre, ref):
    extent = H // head_tiles
    tag = f"tiles=H/{H // head_tiles} (count={head_tiles}, per-tile extent={extent})"

    torch._dynamo.reset()  # fresh compile per candidate (caches already off)

    q_s, kc_s, vc_s, bt_s = inputs_spyre
    # Name the inputs so the head-dim hint can bind (PR #3674): a num_tiles hint
    # only tiles a dim propagation can trace to a named loop var on the inputs.
    declare_tensor_dim("B", B)
    declare_tensor_dim("H", H)
    declare_tensor_dim("Lq", Lq)
    declare_tensor_dim("D", D)
    declare_tensor_dim("BS", kv_block_size)
    declare_tensor_dim("CS", cache_size)
    declare_tensor_dim("blocks", blocks)
    name_tensor_dims(q_s, ["Lq", "H", "D"])
    name_tensor_dims(kc_s, ["CS", "BS", "H", "D"])
    name_tensor_dims(vc_s, ["CS", "BS", "H", "D"])
    name_tensor_dims(bt_s, ["B", "blocks"])

    fn = torch.compile(make_paged_spyre(head_tiles), dynamic=False)
    got = fn(q_s, kc_s, vc_s, bt_s).cpu()

    max_diff = (got.float() - ref.float()).abs().max().item()
    # Per-head max error exposes the "tile 0 ok, tiles 1..N-1 wrong" signature.
    per_head = (
        (got.float() - ref.float())
        .abs()
        .view(Lq, H, D)
        .amax(dim=(0, 2))
        .tolist()
    )
    ok = max_diff < 0.3
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: max_diff={max_diff:.4g}")
    print("        per-head max_diff = " + ", ".join(f"{x:.3g}" for x in per_head))
    return ok


def main():
    torch.manual_seed(0)
    queries = torch.randn(Lq, H * D, dtype=torch.float16)
    key_cache = torch.zeros(cache_size, kv_block_size, H, D, dtype=torch.float16)
    value_cache = torch.zeros(cache_size, kv_block_size, H, D, dtype=torch.float16)
    # Fill only the pages this sequence uses.
    block_table = torch.arange(B * blocks, dtype=torch.int64).reshape(B, blocks)
    for p in block_table.flatten().tolist():
        key_cache[p] = torch.randn(kv_block_size, H, D, dtype=torch.float16)
        value_cache[p] = torch.randn(kv_block_size, H, D, dtype=torch.float16)

    ref = paged_cpu(
        queries.view(Lq, H, D), key_cache, value_cache, block_table
    )

    # Slot-major device layout (production layout, torch-spyre#3705): cache &
    # block dims outermost, stick stays on D. dim_order [2,0,1,3] on
    # [CS, BS, H, D] -> device dims [CS*BS, H, D//64, 64].
    kv_layout = SpyreTensorLayout(
        [cache_size, kv_block_size, H, D],
        [kv_block_size * H * D, H * D, D, 1],
        torch.float16,
        [2, 0, 1, 3],
    )
    inputs_cpu = (queries.view(Lq, H, D), key_cache, value_cache, block_table)
    inputs_spyre = (
        queries.to(device="spyre").view(Lq, H, D),
        key_cache.to(device_layout=kv_layout),
        value_cache.to(device_layout=kv_layout),
        block_table.to(device="spyre"),
    )

    print(f"shape: B={B} H={H} D={D} BS={kv_block_size} blocks={blocks} Lq={Lq}")
    # Correct control: extent 2. Then the bug: extent 1.
    good = run_candidate(H // 2, inputs_cpu, inputs_spyre, ref)
    bad = run_candidate(H, inputs_cpu, inputs_spyre, ref)

    print()
    if good and not bad:
        print("REPRODUCED: extent>=2 correct, extent-1 silently wrong.")
    elif good and bad:
        print("NOT reproduced: extent-1 also correct on this torch-spyre pin.")
    else:
        print("UNEXPECTED: the extent>=2 control failed; check the harness/shape.")


if __name__ == "__main__":
    main()
