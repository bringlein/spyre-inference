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

"""Paged KV-cache attention backend for Spyre using a dense page tensor and online softmax."""

import bisect
import contextlib
import functools
import json
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import os

import torch

from spyre_inference.custom_ops.utils import convert

from vllm.config import CompilationMode, VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# When set, wraps forward(), _reshape_and_cache(), and _online_softmax_attention()
# in torch.profiler.record_function spans for kineto trace capture.
_ATTN_PROFILING = os.environ.get("SPYRE_ATTN_PROFILING", "0") == "1"


def _record_function(name: str):
    def decorator(fn):
        if not _ATTN_PROFILING:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


# TODO: Make these hyperparameters configurable
# KV length alignment: KV tensors are padded to the next multiple of this value.
# Because torch.compile treats shapes as static constants, every distinct kv_len
# triggers a full recompile. Aligning to 256 buckets sequence lengths into tiers
# (256, 512, 768, ...) so only the first request at each tier pays compilation cost,
# rather than recompiling on every decode step.
KV_LENGTH_ALIGNMENT = 256

# Query chunk size for padding - ensures consistent tensor sizes for Spyre compilation.
# TODO: decode sequences in a mixed batch still pad to this; only decode-only
# batches skip it.
QUERY_CHUNK_SIZE = 32

# Elements per stick for int32 (128-byte stick / 4 bytes). Page-index rows are
# padded to this width so each row starts on a stick boundary; see
# SpyreAttentionMetadata.page_index_tables.
INT32_ELEMS_PER_STICK = 32

# Bucketed decode fast path (#612): decode-only batches are batched into one 4D
# bmm and padded UP to the nearest (num_seqs, num_blocks) lattice point so one
# compiled kernel serves a family of real shapes. Below the smallest seqs bucket
# the batched matmul is memory-bound and regresses, so those batches keep the
# per-seq loop. No env vars: the lattice is derived from engine config
# (max_num_seqs / max_model_len) as powers of two.
_MIN_DECODE_BATCH = 8


def _pow2_buckets(limit: int, start: int = 1) -> tuple[int, ...]:
    """Powers of two from `start` up to and including the smallest pow2 >= limit."""
    if limit < start:
        return (start,)
    buckets = []
    b = start
    while b < limit:
        buckets.append(b)
        b *= 2
    buckets.append(b)
    return tuple(buckets)


def _bucket_up(n: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest bucket >= n via bisect, or None when n exceeds the top bucket.

    None means "outside the lattice" — the caller falls back to the per-seq
    path rather than compiling a wider kernel on the fly.
    """
    i = bisect.bisect_left(buckets, n)
    if i == len(buckets):
        return None
    return buckets[i]


# Directory of tuned coarse-tile configs, one JSON per attention shape signature
# (see _attn_tile_config_filename). Emitted by examples/tune_attn_tiling.py.
_TILE_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# kv_head tiling is enabled only at long prefill; below this padded_query_len it
# is measured HARMFUL. Device-time micro-benchmark (bs=1, head_size=128, 8 kv /
# 4 q-per-kv): query_len=512 regresses ~5%, query_len=1024 improves ~6% (tile=2),
# query_len=2048 improves ~4% (tile=4). The crossover is between 512 and 1024, so
# the gate is 1024. The win is from smaller on-chip transients (compute-kernel
# scheduling), not memory-op reduction (mem-op time stays ~0.3-1% throughout).
# `tile_q_heads` (qpk) has no such gate (it is a small, workload-fixed axis).
KV_HEAD_TILE_THRESHOLD = 1024


def _attn_tile_config_filename(
    head_size: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
    block_size: int,
) -> str:
    return (
        f"head_size={head_size},num_kv_heads={num_kv_heads},"
        f"num_queries_per_kv={num_queries_per_kv},block_size={block_size}.json"
    )


@functools.lru_cache(maxsize=None)
def _get_attn_tile_config(
    head_size: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
    block_size: int,
) -> dict:
    """Load the tuned coarse-tile config for this shape, or a no-op fallback.

    Fallback is {"tile_kv_heads": 1, "tile_q_heads": 1} (no tiling), so an absent
    or invalid config leaves the kernel byte-identical to the untuned path. Each
    tile count must divide its axis evenly (the compiler asserts even
    divisibility); a value that does not is dropped back to 1 with a warning.
    """
    default = {"tile_kv_heads": 1, "tile_q_heads": 1}
    fname = _attn_tile_config_filename(
        head_size, num_kv_heads, num_queries_per_kv, block_size
    )
    path = os.path.join(_TILE_CONFIG_DIR, fname)
    if not os.path.exists(path):
        logger.debug_once(f"No attention tile config at {path}; tiling disabled")
        return default
    with open(path) as f:
        cfg = json.load(f)

    def _validated(key: str, axis: int) -> int:
        val = int(cfg.get(key, 1))
        if val < 1 or axis % val != 0:
            logger.warning(
                f"Ignoring {key}={val} from {path}: must be >=1 and divide "
                f"{axis} evenly; disabling that axis"
            )
            return 1
        return val

    return {
        "tile_kv_heads": _validated("tile_kv_heads", num_kv_heads),
        "tile_q_heads": _validated("tile_q_heads", num_queries_per_kv),
    }


class SpyrePagedKVCache(NamedTuple):
    """Per-layer paged KV cache for the Spyre backend.

    Each field is one dense tensor of shape
    [num_blocks, block_size, num_kv_heads, head_size] on the Spyre device,
    matching `SpyreAttentionBackend.get_kv_cache_shape`.

    NamedTuple (not dataclass) because it is a tuple at runtime, so unpacking
    (`k_pages, v_pages = cache`) traces cleanly under Dynamo without relying on
    attribute access on a custom object.

    Allocated by `TorchSpyreModelRunner.initialize_kv_cache_tensors` and
    consumed by `SpyreAttentionImpl.forward`. vLLM's `bind_kv_cache` types
    the relay path as `dict[str, torch.Tensor]`; see the suppression at the
    `bind_kv_cache(...)` call site for why that type-hole is benign.
    """

    k_pages: torch.Tensor
    v_pages: torch.Tensor


def slot_major_kv_layout(num_slots: int, num_kv_heads: int, head_size: int, dtype: torch.dtype):
    """Slot-axis-outermost layout. The default tiled layout spreads the slot index
    across two device dims, so a 1-KV-head cache cannot be work-divided under the
    256 MB per-core span limit; it also makes the indirect store write to the wrong
    rows (torch-spyre#3705)."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def _maybe_compile(fn, compile_enabled: bool):
    """Compile `fn` when enabled. Attention compiles separately from the model's
    fullgraph capture, which can't hold its per-sequence Python loop.
    """
    if compile_enabled:
        return torch.compile(fn, dynamic=False)
    return fn


def _reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


# ---------------------------------------------------------------------------
# Compilable factory functions
# ---------------------------------------------------------------------------


def _clamp_tile_count(count: int, dim_size: int) -> int:
    """Largest divisor-friendly tile count leaving a per-tile extent >= 2.

    A tile count that divides an axis down to extent 1 gets squeezed out of the
    read index (SqueezeView.squeezer), which silently corrupts the strided read
    for every tile but the first (Plan 5). Halving keeps extent >= 2.
    """
    if count <= 1:
        return count
    while count > 1 and dim_size // count < 2:
        count //= 2
    return count


class _TileHints:
    """Coarse-tile the online-softmax ops along the head dims (kv_head, qpk).

    kv_head is a NATIVE page axis (k_page[kv_head, blk, d]); tiling it slices a
    real page dim, so with both pages hoisted to full-extent HBM buffers the
    read-copy is a clean slice. This is why kv_head compiles where the query-only
    axes (qpk-broadcast, lq) do not: the score matmul broadcasts the page over
    those, and the broadcast-expand fused into the gather makes the tiled page
    read-copy an unresolvable slot-major->head-major relayout.

    Binding a hint requires the tiled dim to be a named loop var on the ops in
    scope, so the tiled dims are named on the kernel inputs (declare_tensor_dim +
    name_tensor_dims in _name_attn_inputs) and on the matmul outputs (which
    inherit no names) — else the hint is silently dropped (PR #3674).

    Load-bearing (PR #3674): the whole unrolled page loop must sit inside ONE
    tile scope, not a fresh scope per iteration.

    All methods are no-ops when no dim is tiled (>1) or torch_spyre's hint API is
    unavailable (CPU test path), so the default kernel is byte-identical.
    """

    def __init__(
        self,
        tile_kv_heads: int,
        tile_q_heads: int,
        num_kv_heads: int,
        num_queries_per_kv: int,
    ):
        self.active = False
        self._spyre_hint = None
        self._tiles: list[tuple[str, int]] = []
        tile_kv_heads = _clamp_tile_count(tile_kv_heads, num_kv_heads)
        tile_q_heads = _clamp_tile_count(tile_q_heads, num_queries_per_kv)
        want = (tile_kv_heads > 1) or (tile_q_heads > 1)
        if not want:
            return
        try:
            from torch_spyre._inductor import spyre_hint
        except ImportError:
            return
        self._spyre_hint = spyre_hint
        if tile_kv_heads > 1:
            self._tiles.append(("kv_head", tile_kv_heads))
        if tile_q_heads > 1:
            self._tiles.append(("qpk", tile_q_heads))
        self.active = True

    @contextlib.contextmanager
    def tile(self):
        """Open nested coarse-tile loops (one per tiled head dim) around the
        whole page loop. One dim per spyre_hint() call; scopes nest.

        Explicit nested `with` rather than ExitStack: Dynamo traces the
        spyre_hint annotate context managers only as lexical `with` statements.
        At most two head dims are tiled (kv_head, qpk).
        """
        if not self.active:
            yield
            return
        hint = self._spyre_hint
        tiles = self._tiles
        if len(tiles) == 1:
            (n0, c0) = tiles[0]
            with hint(num_tiles_per_dim={n0: c0}):
                yield
        else:
            (n0, c0), (n1, c1) = tiles[0], tiles[1]
            with hint(num_tiles_per_dim={n0: c0}):
                with hint(num_tiles_per_dim={n1: c1}):
                    yield

    @contextlib.contextmanager
    def named(self, *names: str):
        """Name the enclosed op's output dims so the head-dim hints can bind."""
        if not self.active:
            yield
            return
        with self._spyre_hint(named_dims=list(names)):
            yield


def _name_attn_inputs(
    q,
    k_pages,
    v_pages,
    mask_tiles,
    alibi_bias_tiles,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
):
    """Name the traced kernel inputs so head-dim tile hints can bind (PR #3674).

    A `num_tiles_per_dim` hint only tiles a dim that propagation can trace back to
    a named loop var on the inputs; naming only intermediate outputs is not enough
    (the matmul-fed reductions otherwise carry _untracked_ names). No-op when the
    hint API is unavailable (CPU path). Called only when tiling is active.

    q:               [kv_head, qpk, lq, head_size]
    k_pages/v_pages: [block_pool, blk, kv_head, head_size]
    mask_tiles[i]:   [lq, blk] (broadcast against scores over the head dims)
    alibi_bias_tiles[i]: [kv_head, qpk, 1, blk]

    The mask stays 2D on purpose (giving it a head axis makes the coarse-tile pass
    slice it, returning the wrong slice for all but the first tile).
    """
    try:
        from torch_spyre._inductor.wsr.propagate_named_dims import (
            declare_tensor_dim,
            name_tensor_dims,
        )
    except ImportError:
        return
    declare_tensor_dim("kv_head", num_kv_heads)
    declare_tensor_dim("qpk", num_queries_per_kv)
    declare_tensor_dim("lq", q.shape[2])
    declare_tensor_dim("blk", block_size)
    declare_tensor_dim("d", head_size)
    declare_tensor_dim("one", 1)
    declare_tensor_dim("block_pool", k_pages.shape[0])
    name_tensor_dims(q, ["kv_head", "qpk", "lq", "d"])
    name_tensor_dims(k_pages, ["block_pool", "blk", "kv_head", "d"])
    name_tensor_dims(v_pages, ["block_pool", "blk", "kv_head", "d"])
    for mask_tile in mask_tiles:
        if mask_tile.dim() == 4:
            second = "qpk" if mask_tile.shape[1] == num_queries_per_kv else "one"
            name_tensor_dims(mask_tile, ["kv_head", second, "lq", "blk"])
        else:
            name_tensor_dims(mask_tile, ["lq", "blk"])
    if alibi_bias_tiles is not None:
        for bias in alibi_bias_tiles:
            name_tensor_dims(bias, ["kv_head", "qpk", "lq", "blk"])


def _create_compilable_page_attn(
    num_blocks: int,
    padded_query_len: int,
    num_heads: int,
    head_size: int,
    has_alibi: bool = False,
    logits_soft_cap: float = 0.0,
    tile_kv_heads: int = 1,
    tile_q_heads: int = 1,
    batched_decode: bool = False,
):
    """Create online softmax attention over a fixed number of pages for torch.compile.

    Dynamo unrolls the loop because num_blocks, padded_query_len, has_alibi,
    logits_soft_cap, tile_kv_heads, tile_q_heads, and batched_decode are closure
    constants.

    batched_decode (the #612 fast path) folds the whole decode batch into the
    leading axis: the kernel's per-block online-softmax math is identical, but
    K/V pages arrive PRE-GATHERED as Python lists (one 4D tensor per block, with
    the (num_seqs, num_kv_heads) pair already folded into the leading dim by the
    caller) instead of being gathered inside the kernel from `page_index_table`.
    Under this mode padded_query_len is 1 and alibi/soft-cap/tiling are off (the
    caller's preconditions guarantee it), so those branches are dead. Keeping one
    factory avoids duplicating the online-softmax recurrence.

    tile_kv_heads / tile_q_heads > 1 wrap the whole unrolled page loop in nested
    coarse-tile hints over the head axes (KV heads `kv_head`, queries-per-kv
    `qpk`), shrinking the score/prob/output transients so the fused chain stays
    resident and avoids spill/refill IO. Both == 1 emits no hints (byte-identical
    to the untuned kernel). Supplied by the tuned config (see _get_attn_tile_config).

    When tiling is active the page gather + head-major permute is HOISTED out of
    the tile scope for BOTH K and V. Inside the scope, coarse-tile read-copy
    insertion would have to fuse the permute into a copy of the slot-major cache
    (see slot_major_kv_layout) and no candidate layout can express that transpose
    — the beam search fails with "no mechanism to resolve stick incompatibility".
    Hoisting makes each page a plain full-extent buffer the copy can slice along
    the tiled kv_head dim. It costs one HBM round-trip per page; the score/prob
    transients still stay on-chip. Only matmul outputs are named (a matmul inherits
    no dim names from its inputs, PR #3674); naming q/k_pages/v_pages plus the two
    matmul outputs is sufficient for the head-dim hints to bind. The final
    normalization stays INSIDE the tile scope (finalize_layouts cannot restickify a
    per-tile-written accumulator into an untiled read); the caller does the output
    reshape (folding it into the kernel corrupts the tiled accumulator's layout).
    """

    def specialized_paged_attn_kernel(
        q,
        k_pages,
        v_pages,
        page_index_table,
        mask_tiles,
        scale,
        alibi_bias_tiles=None,
    ):
        """
        This kernels specializes for num_blocks and padded_query_len.

        Expected shapes (per-seq / non-batched):
            q: [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
            v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
            page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
                tensor, row i holding the i-th active block's page index at
                column 0.
            mask_tiles: [num_blocks]
            alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size]
                (only when has_alibi=True; None otherwise). The query-axis dim
                is 1 because softmax absorbs per-query-row constants — see
                the derivation at the bias-tile construction site in
                _online_softmax_attention.

        Batched-decode mode (batched_decode=True):
            q: [B_seqs * num_kv_heads, num_queries_per_kv, 1, head_size]
            k_pages / v_pages: length-num_blocks lists of
                [B_seqs * num_kv_heads, 1, block_size, head_size] on device.
            page_index_table: unused (None); pages are pre-gathered.
            mask_tiles: length-num_blocks list of
                [B_seqs * num_kv_heads, 1, 1, block_size] fp16.

        Returns [<leading>, num_queries_per_kv, padded_query_len, head_size]
        (the caller folds the head dims and moves the query axis out).
        """
        tile_max = None
        tile_sum = None
        tile_output = None

        th = _TileHints(tile_kv_heads, tile_q_heads, q.shape[0], q.shape[1])

        def gather_page(i):
            # index_select, not `k_pages[page_idx]`: subscripting lowers to
            # aten.index, which upcasts the int32 index to int64 and fails eager.
            page_idx = page_index_table[i, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            v_page = v_pages.index_select(0, page_idx)
            # Token-major page to head-major for the matmuls; permutes on device.
            return (
                k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1),
                v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1),
            )

        # When tiling, the page gather + head-major permute is hoisted OUT of the
        # tile scope for BOTH pages (see the factory docstring for why). It also
        # reassociates the untiled kernel's fusion groups, so it is applied only
        # when tiling is on. Batched-decode pages are already gathered lists.
        if batched_decode:
            pages = None
        else:
            pages = [gather_page(i) for i in range(num_blocks)] if th.active else None

        # One tile scope wraps the ENTIRE unrolled page loop (PR #3674): a fresh
        # scope per iteration makes the scheduler interleave blocks and
        # validate_coarse_tile_groups rejects the non-contiguous group.
        with th.tile():
            for i in range(num_blocks):
                if batched_decode:
                    # Pre-gathered, already head-major-folded per-block tensors.
                    k_page_4d = k_pages[i]
                    v_page_4d = v_pages[i]
                else:
                    k_page_4d, v_page_4d = pages[i] if pages is not None else gather_page(i)

                mask_tile = mask_tiles[i]

                # Matmul output carries no names -> name it so the head-dim hints
                # bind. Downstream elementwise/reductions inherit these names.
                with th.named("kv_head", "qpk", "lq", "blk"):
                    scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
                if logits_soft_cap > 0.0:
                    # Pull logits into (-cap, +cap) before the mask add so masked
                    # positions still map cleanly to -inf. Applied before the ALiBi
                    # bias so the positional term is not squashed by the tanh.
                    scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
                if has_alibi:
                    # ALiBi bias slope[h] * (kv_pos - context_len). The additive
                    # mask_tile below uses finfo.min for masked positions, so this
                    # bias cannot un-mask them.
                    assert alibi_bias_tiles is not None
                    scores = scores + alibi_bias_tiles[i]
                scores = scores + mask_tile
                scores_max = torch.amax(scores, dim=-1, keepdim=True)

                if i == 0:
                    tile_max = scores_max
                    tile_probs = torch.exp(scores - tile_max)
                    with th.named("kv_head", "qpk", "lq", "d"):
                        tile_output = torch.matmul(tile_probs, v_page_4d)
                    tile_sum = tile_probs.sum(dim=-1, keepdim=True)
                else:
                    # i > 0 only reachable after the i == 0 branch initialized these.
                    assert tile_max is not None
                    assert tile_sum is not None
                    assert tile_output is not None
                    new_max = torch.maximum(tile_max, scores_max)
                    rescale = torch.exp(tile_max - new_max)
                    tile_output = tile_output * rescale
                    tile_sum = tile_sum * rescale
                    tile_probs = torch.exp(scores - new_max)
                    with th.named("kv_head", "qpk", "lq", "d"):
                        weighted = torch.matmul(tile_probs, v_page_4d)
                    tile_output = tile_output + weighted
                    tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                    tile_max = new_max

            # The final normalization stays INSIDE the tile scope. Left outside, it
            # is an ungrouped op reading the group's full-extent accumulator, and
            # finalize_layouts cannot restickify that per-tile-written accumulator
            # into the untiled read the div wants.
            assert tile_output is not None and tile_sum is not None
            attn = tile_output / tile_sum

        # Return the raw 4D accumulator; the caller reshapes with .contiguous()
        # (folding the transpose into the compiled kernel corrupts the tiled
        # accumulator's device layout -> wrong output metadata / half-size storage).
        return attn

    return specialized_paged_attn_kernel




@dataclass
class _DecodeGroup:
    """One decode group: all sequences whose active-block count snaps to the same
    NUM_BLOCKS_BUCKETS rung. Run as ONE #612 batched call over the group's own
    block count — no cross-group carry (each sequence finalizes in one call).

    "jump off the train": grouping by context length means total KV-step × request
    work tracks the SUM of contexts (Σ_groups group_size × group_blocks), not
    batch × max, while every kernel call is an already-compiled #612 lattice point.
    """

    seqs_bucket: int  # leading batch axis (surviving count snapped up)
    blocks_bucket: int  # KV block count for this group (rung)
    slot_start: int  # first sorted-order slot in this group
    num_members: int  # real sequences in this group
    # Flat page-index buffer [seqs_bucket * blocks_bucket] int32; row j of the
    # logical (seqs_bucket, blocks_bucket) grid at [j*blocks_bucket:(j+1)*...].
    block_ids_cpu: torch.Tensor
    block_ids_dev: torch.Tensor | None = None
    # Combined kv-tail + padding mask [seqs_bucket, blocks_bucket, block_size] fp16.
    mask_cpu: torch.Tensor | None = None
    mask_dev: torch.Tensor | None = None


@dataclass
class _DecodeSchedule:
    """Host-side decode plan, computed once per step in build().

    Sequences are sorted ascending by block count and partitioned into groups by
    NUM_BLOCKS_BUCKETS rung. sort_perm maps sorted order -> original seq index;
    inv_perm is its inverse (for the final unsort scatter into output).
    """

    groups: list[_DecodeGroup]
    sort_perm: torch.Tensor  # [num_seqs] int64, original index for each sorted slot
    blocks_sorted: list[int]  # per sorted slot, its active-block count
    # Query row IDs in sorted order [num_seqs] int64 (row in the flat q buffer).
    query_row_ids_sorted_cpu: torch.Tensor
    query_row_ids_sorted_dev: torch.Tensor | None = None
    # Inverse permutation [num_seqs] int64: inv_perm[i] = sorted slot of original
    # seq i. Used to unsort outputs (output[i] = out_sorted[inv_perm[i]]).
    inv_perm_cpu: torch.Tensor | None = None


@dataclass
class SpyreAttentionMetadata(AttentionMetadata):
    """Metadata for paged online-softmax attention on Spyre."""

    # Total real (non-padding) tokens across all sequences. Used to slice
    # q/k/v to actual tokens before processing (input may have padding).
    num_actual_tokens: int

    # Number of sequences in this batch.
    num_seqs: int

    # Maximum query length among all sequences (raw, unaligned).
    max_query_len: int

    # Maximum KV sequence length among all sequences (raw, unaligned).
    max_seq_len: int

    # Per-sequence KV lengths. [num_seqs]
    seq_lens: torch.Tensor

    # Cumulative query lengths for varlen layout. query_start_loc[i]
    # is the start offset of sequence i in the flat q/k/v buffer.
    # [num_seqs + 1], last entry = total tokens.
    query_start_loc: torch.Tensor

    # Block table mapping logical blocks to physical pages.
    # [num_seqs, max_num_blocks_per_seq]
    block_table: torch.Tensor

    # Number of KV tokens per physical page.
    block_size: int

    # Flat mapping from token index to its position in the KV cache
    # (physical_block_index * block_size + block_offset). [num_actual_tokens]
    slot_mapping: torch.Tensor

    # True when causal masking is needed (prefill/mixed, i.e. max_query_len > 1).
    # Decode steps (max_query_len=1) don't need explicit causal masking because
    # the online softmax over KV pages naturally only attends to past tokens.
    apply_causal_mask: bool = False

    # Number of KV heads (for GQA).
    num_kv_heads: int = 0

    # Number of query heads.
    num_heads: int = 0

    # Pre-tiled additive attention mask. attention_mask_tiles[seq_idx][i]
    # gives the mask tile for the i-th ACTIVE block of one sequence (indexed
    # by position within active_block_indices[seq_idx], not by absolute block
    # index). Each tile: [aligned_max_query_len, block_size] on CPU. When
    # sliding_window is None, active == all blocks and the layout is
    # equivalent to indexing by absolute block index.
    attention_mask_tiles: list[list[torch.Tensor]] | None = None

    # For each sequence: absolute block indices whose mask is not fully
    # `-inf` (blocks that contribute to at least one query's attention).
    # None means all blocks are active (sliding_window is None, or the
    # window covers the whole sequence). When set, len(active_block_indices[s])
    # matches len(attention_mask_tiles[s]).
    active_block_indices: list[list[int]] | None = None

    # Global aligned query length for stable kernel compilation.
    # max_query_len rounded up to QUERY_CHUNK_SIZE (32). All queries are
    # padded to this length so the compiled attention kernel receives
    # consistent tensor shapes across steps and sequences.
    aligned_max_query_len: int = 0

    # Global aligned KV sequence length for stable kernel compilation.
    # max_seq_len rounded up to KV_LENGTH_ALIGNMENT (256). The KV mask
    # dimension is padded to this length so recompilation only happens
    # per 256-token tier, not per distinct sequence length.
    aligned_max_seq_len: int = 0

    # Gather indices for the paged attention loop, one row per active block:
    # [num_seqs, max_active_blocks, INT32_ELEMS_PER_STICK] int32 with the page
    # index at [s, b, 0]. Each index needs its own stick-wide row to compile,
    # which is why block_table cannot serve as the index. The device mirror is
    # filled by the first forward(), since the builder's device is CPU.
    # One tensor per sequence, materialized once per step: a compiled kernel reads
    # its inputs from offset 0, ignoring storage_offset (torch-spyre#3770).
    page_index_table_cpu: torch.Tensor | None = None
    page_index_tables: list[torch.Tensor] | None = None

    # Device mirror of slot_mapping, which vLLM hands us on the host.
    slot_mapping_device: torch.Tensor | None = None

    # Device mirror of attention_mask_tiles, filled once per step by forward().
    attention_mask_tiles_device: list[list[torch.Tensor]] | None = None

    # Bucketed / ragged decode fast path (#612 + #468). Populated by the builder
    # only for decode-only batches (max_query_len == 1, no sliding window) whose
    # num_seqs is within the seqs-bucket lattice. When None, callers fall back to
    # the per-seq loop. All host-side and layer-invariant per step: the forward()
    # pass mirrors the CPU tensors to device once, and every layer reuses them.
    #
    # decode_schedule holds the block-bucket groups ("jump off the train": each
    # group is one #612 batched call over its own block count), plus the
    # ascending-by-blocks sort permutation used to unsort outputs.
    decode_schedule: "_DecodeSchedule | None" = None

    @property
    def query_lens(self) -> torch.Tensor:
        """Per-sequence query lengths, derived from query_start_loc. [num_seqs]"""
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


class SpyreAttentionMetadataBuilder(AttentionMetadataBuilder[SpyreAttentionMetadata]):
    """Builds attention metadata — only the attention mask is precomputed."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.head_size = kv_cache_spec.head_size
        self.sliding_window = getattr(kv_cache_spec, "sliding_window", None)
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")

        # Validate block_size alignment: Spyre stick size is 128 bytes (64 fp16 elements).
        # block_size must be a multiple of 64 to avoid restickification errors during
        # torch.compile.
        if self.block_size % 64 != 0:
            raise ValueError(
                f"block_size must be a multiple of 64 for the Spyre paged attention "
                f"backend. Got block_size={self.block_size}, head_size={self.head_size}. "
            )

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        # `model_config.dtype` is typed `ModelDType | torch.dtype`, but
        # `TorchSpyrePlatform.check_and_update_config` rejects anything but
        # `torch.float16` upstream so it's always a real torch.dtype here.
        assert isinstance(model_config.dtype, torch.dtype)
        self.model_dtype: torch.dtype = model_config.dtype

        # Decode fast-path bucket lattice, derived from engine config as powers
        # of two (no env vars): seqs up to max_num_seqs, blocks up to the block
        # count for max_model_len. The smallest seqs bucket is _MIN_DECODE_BATCH;
        # below it the batched matmul is memory-bound and the per-seq loop wins.
        max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        max_model_len = int(model_config.max_model_len)
        max_blocks = (max_model_len + self.block_size - 1) // self.block_size
        self.num_seqs_buckets = _pow2_buckets(max(max_num_seqs, _MIN_DECODE_BATCH), _MIN_DECODE_BATCH)
        self.num_blocks_buckets = _pow2_buckets(max(max_blocks, 1), 1)

        # Shared zero tile reused for interior active blocks (fully inside the
        # window, so their mask is all-zeros). Allocated lazily on first use
        # and resized if aligned_max_query_len or block_size changes across
        # calls.
        self._zero_tile: torch.Tensor | None = None
        self._zero_tile_shape: tuple[int, int] = (0, 0)

    def _get_zero_tile(self, aligned_max_query_len: int) -> torch.Tensor:
        """Return (or create) the shared all-zero mask tile for interior blocks.

        The returned tensor is reused by reference across all interior blocks
        and sequences in a batch. Callers must treat it as read-only: any
        in-place mutation would corrupt every interior tile simultaneously.
        This is safe today because attention kernels only read mask tiles.
        """
        shape = (aligned_max_query_len, self.block_size)
        if self._zero_tile is None or self._zero_tile_shape != shape:
            self._zero_tile = torch.zeros(shape, dtype=self.model_dtype)
            self._zero_tile_shape = shape
        return self._zero_tile

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        apply_causal_mask: bool,
        max_query_len: int,
        aligned_max_query_len: int,
        aligned_max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive attention mask on Spyre for the non-sliding-window path.

        All sequences share the same aligned_max_query_len so every mask tile
        has a uniform query dimension — this avoids per-sequence kernel
        specializations.

        Sliding-window sequences take a different path: see
        _build_active_tiles_with_skip.

        Returns:
            - mask: [num_seqs, aligned_max_query_len, aligned_max_seq_len] additive mask
        """
        assert self.sliding_window is None
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_seqs = len(seq_lens)

        q_pos = torch.arange(max_query_len, device=device)
        kv_pos = torch.arange(aligned_max_seq_len, device=device)

        # Padding mask: valid positions are within actual sequence/query lengths
        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        # Causal mask: prevent attending to future tokens during generation
        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit
            attend = attend & causal_ok

        # Convert to additive mask: finfo.min for masked positions, 0 for valid
        mask_bool = ~attend  # [num_seqs, max_query_len, aligned_max_seq_len]

        if aligned_max_query_len > max_query_len:
            padding = torch.ones(
                num_seqs,
                aligned_max_query_len - max_query_len,
                aligned_max_seq_len,
                dtype=torch.bool,
                device=device,
            )
            mask_bool = torch.cat([mask_bool, padding], dim=1)

        mask_additive = torch.where(
            mask_bool,
            torch.tensor(torch.finfo(self.model_dtype).min, dtype=self.model_dtype, device=device),
            torch.tensor(0.0, dtype=self.model_dtype, device=device),
        )

        return mask_additive

    def _build_single_tile(
        self,
        block_idx: int,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_max_query_len: int,
        apply_causal_mask: bool,
    ) -> torch.Tensor:
        """Build the additive mask tile for one (sequence, block) pair.

        Returns a [aligned_max_query_len, block_size] CPU tensor.

        Only called for boundary blocks that require real mask content:
          - lower-boundary blocks (window-start cutoff falls inside them for
            at least one query), and
          - the upper-boundary block (last block: KV padding, plus causal
            during prefill).
        Interior blocks reuse the shared zero tile instead.
        """
        block_size = self.block_size
        mask_min = torch.finfo(self.model_dtype).min

        # KV positions covered by this block. May extend past kv_len (handled
        # by the kv_valid mask below).
        kv_start = block_idx * block_size
        kv_end = kv_start + block_size

        q_pos = torch.arange(aligned_max_query_len)  # [aligned_max_query_len]
        kv_pos = torch.arange(kv_start, kv_end)  # [block_size]

        # Padding mask: query rows beyond query_len are fully masked;
        # KV columns beyond kv_len are fully masked.
        q_valid = q_pos < query_len  # [aligned_max_query_len]
        kv_valid = kv_pos < kv_len  # [block_size]
        attend = q_valid.unsqueeze(1) & kv_valid.unsqueeze(0)  # [Q, B]

        # Causal mask (prefill only): query at absolute position
        # context_len + q_pos can only attend to KV positions <= that value.
        if apply_causal_mask:
            causal_limit = context_len + q_pos  # [aligned_max_query_len]
            attend = attend & (kv_pos.unsqueeze(0) <= causal_limit.unsqueeze(1))

        # Sliding window: per-query window_start.
        assert self.sliding_window is not None
        abs_q_pos = context_len + q_pos  # [aligned_max_query_len]
        window_start = (abs_q_pos - self.sliding_window + 1).clamp(min=0)
        attend = attend & (kv_pos.unsqueeze(0) >= window_start.unsqueeze(1))

        mask_bool = ~attend
        return torch.where(
            mask_bool,
            torch.tensor(mask_min, dtype=self.model_dtype),
            torch.tensor(0.0, dtype=self.model_dtype),
        )

    def _build_active_tiles_with_skip(
        self,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_max_query_len: int,
        apply_causal_mask: bool,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Return (active_block_indices, mask_tiles) using arithmetic block-skip.

        active_block_indices: absolute block indices whose mask contributes
        to at least one query's attention (i.e. inside the window of the
        earliest query).
        mask_tiles: one tile per active block, in the same order.

        Block classification:
          - [0, first_active):
                entirely outside every query's window; skipped.
          - [first_active, last_lower_boundary]:
                lower-boundary blocks — the window cutoff falls inside them
                for at least one query. Real tile with per-query-row cutoffs.
                In decode (query_len == 1) this collapses to a single block.
          - (last_lower_boundary, last_causal_interior]:
                interior blocks — fully inside every query's window AND fully
                below the earliest query's causal limit. Mask is all-zero.
          - (last_causal_interior, last_block):
                causal-boundary blocks — inside every window, but early
                queries have causal cutoffs falling inside them (prefill
                only). Real tile.
          - last_block:
                upper-boundary block — always has KV padding (and causal
                cutoffs during prefill). Real tile.

        When any of the boundary ranges overlap (short kv_len, single-block
        sequence, etc.) real tiles are built for the union — never zero tiles.
        """
        assert self.sliding_window is not None
        block_size = self.block_size
        num_blocks = (kv_len + block_size - 1) // block_size

        # Earliest query (q_pos=0) has window
        # [max(0, context_len - W + 1), context_len].
        # Latest query (q_pos=query_len-1) has window
        # [max(0, kv_len - W), kv_len - 1].
        # A block is fully outside every query's window when its highest KV
        # position is below the earliest query's window start.
        # NOTE: using the EARLIEST query's window (not the latest, kv_len - W)
        # is required for prefill correctness. In a prefill batch with
        # query_len > 1, early queries have earlier windows and their
        # in-window blocks would otherwise be incorrectly dropped. For decode
        # (query_len == 1) both formulas coincide.
        earliest_window_start = max(0, context_len - self.sliding_window + 1)
        latest_window_start = max(0, kv_len - self.sliding_window)

        first_active = earliest_window_start // block_size
        # Every block from first_active up to the block containing the
        # latest window start can have a per-query cutoff falling inside it.
        last_lower_boundary = latest_window_start // block_size
        # A block is fully below the earliest query's causal limit
        # (abs_pos = context_len) iff (b + 1) * block_size - 1 <= context_len.
        # For decode (no causal mask) all blocks satisfy this trivially.
        if apply_causal_mask:
            last_causal_interior = (context_len + 1) // block_size - 1
        else:
            last_causal_interior = num_blocks - 1
        last_block = num_blocks - 1

        active_bs = list(range(first_active, num_blocks))
        if not active_bs:
            return [], []

        zero_tile = self._get_zero_tile(aligned_max_query_len)
        tiles: list[torch.Tensor] = []

        for b in active_bs:
            is_lower_boundary = b <= last_lower_boundary
            is_upper_boundary = (b == last_block) and not is_lower_boundary
            is_causal_boundary = apply_causal_mask and b > last_causal_interior and b != last_block
            if is_lower_boundary or is_upper_boundary or is_causal_boundary:
                tiles.append(
                    self._build_single_tile(
                        b,
                        kv_len,
                        query_len,
                        context_len,
                        aligned_max_query_len,
                        apply_causal_mask,
                    )
                )
            else:
                # Interior block: entirely within every query's window,
                # entirely filled with valid KV tokens, and (for prefill)
                # entirely below the earliest query's causal limit.
                # Mask is all-zero.
                tiles.append(zero_tile)

        return active_bs, tiles

    def _build_decode_schedule(
        self,
        num_seqs: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        num_active: list[int],
        attention_mask_tiles: list[list[torch.Tensor]],
    ) -> "_DecodeSchedule | None":
        """Host-side decode plan (#612 + #468, group-by-block-bucket). Pure Python.

        Returns None when the batch is ineligible (too few seqs, or the longest
        sequence exceeds the block lattice), so callers keep the per-seq loop.

        Sorts sequences ascending by active-block count and partitions them into
        groups by NUM_BLOCKS_BUCKETS rung: group g holds every sequence whose block
        count snaps up to rung g. Each group runs ONE #612 batched call over its
        own block count (rung), so total work tracks Σ contexts, and each sequence
        finalizes in exactly one call — no cross-group carry.
        """
        if num_seqs < self.num_seqs_buckets[0]:
            return None
        max_blocks = max(num_active)
        if _bucket_up(max_blocks, self.num_blocks_buckets) is None:
            return None

        block_size = self.block_size

        # Sort ascending by block count. order[j] = original seq index at slot j.
        order = sorted(range(num_seqs), key=lambda s: num_active[s])
        blocks_sorted = [num_active[s] for s in order]
        sort_perm = torch.tensor(order, dtype=torch.int64)

        # Query row IDs: under Q=1, seq s's single token is at query_start_loc[s].
        qsl = query_start_loc[:num_seqs].to(torch.int64)
        query_row_ids_sorted = qsl[sort_perm].contiguous()

        # Per sorted slot, the block-bucket rung it snaps to. Group boundaries are
        # where the rung changes (blocks_sorted is ascending, so groups are
        # contiguous slot ranges).
        rung_per_slot = [_bucket_up(b, self.num_blocks_buckets) for b in blocks_sorted]

        groups: list[_DecodeGroup] = []
        slot = 0
        while slot < num_seqs:
            rung = rung_per_slot[slot]
            end = slot
            while end < num_seqs and rung_per_slot[end] == rung:
                end += 1
            num_members = end - slot
            blocks_bucket = rung
            seqs_bucket = _bucket_up(num_members, self.num_seqs_buckets)
            assert blocks_bucket is not None and seqs_bucket is not None

            # Flat page-index buffer [seqs_bucket * blocks_bucket]. Row j maps to
            # sorted slot (slot + j); its blocks are [0, blocks_bucket) clamped to
            # the seq's own block count. Padded rows/blocks hold 0 and -inf mask.
            block_ids = torch.zeros(seqs_bucket * blocks_bucket, dtype=torch.int32)
            mask = torch.full(
                (seqs_bucket, blocks_bucket, block_size),
                torch.finfo(self.model_dtype).min,
                dtype=self.model_dtype,
            )
            for j in range(num_members):
                s = order[slot + j]
                n_seq = num_active[s]
                for bb in range(min(blocks_bucket, n_seq)):
                    block_ids[j * blocks_bucket + bb] = block_table[s, bb]
                    # attention_mask_tiles[s][bb] is [aligned_max_query_len,
                    # block_size]; under Q=1 only row 0 carries the kv-tail mask.
                    mask[j, bb] = attention_mask_tiles[s][bb][0]

            groups.append(
                _DecodeGroup(
                    seqs_bucket=seqs_bucket,
                    blocks_bucket=blocks_bucket,
                    slot_start=slot,
                    num_members=num_members,
                    block_ids_cpu=block_ids,
                    mask_cpu=mask,
                )
            )
            slot = end

        if not groups:
            return None

        return _DecodeSchedule(
            groups=groups,
            sort_perm=sort_perm,
            blocks_sorted=blocks_sorted,
            query_row_ids_sorted_cpu=query_row_ids_sorted,
            inv_perm_cpu=torch.argsort(sort_perm).contiguous(),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionMetadata:
        """Build attention metadata from common metadata."""

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        max_seq_len = common_attn_metadata.max_seq_len
        max_query_len = common_attn_metadata.max_query_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.item())
        # Batch-level flag: True iff the batch contains at least one prefill
        # sequence (max_query_len > 1). For decode sequences (query_len == 1)
        # in a mixed batch, the causal constraint is subsumed by the KV
        # validity mask (the single query at position context_len can only
        # attend to KV positions [0, kv_len) = [0, context_len]), so applying
        # the causal mask to them is a correct no-op.
        apply_causal_mask = causal and max_query_len > 1

        # A decode-only batch needs no padding at all: every query_len is 1.
        if max_query_len == 1:
            aligned_max_query_len = 1
        else:
            aligned_max_query_len = (
                (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
            )
        aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1) // KV_LENGTH_ALIGNMENT * KV_LENGTH_ALIGNMENT
        )

        num_seqs = common_attn_metadata.num_reqs
        block_size = self.block_size
        attention_mask_tiles: list[list[torch.Tensor]] = []
        active_block_indices: list[list[int]] | None = None

        if self.sliding_window is None:
            # No sliding window: build the full additive mask and split it into
            # per-block tiles (one tile per absolute block index).
            mask_cpu = self._build_attention_mask(
                seq_lens,
                query_start_loc,
                apply_causal_mask,
                max_query_len,
                aligned_max_query_len,
                aligned_max_seq_len,
                torch.device("cpu"),
            )
            # Pre-tile the mask: split into per-block tiles.
            # Query dimension is uniform (aligned_max_query_len) for all sequences,
            # so tiling only follows the KV dimension.
            for s in range(num_seqs):
                seq_tiles: list[torch.Tensor] = []
                kv_len_s = int(seq_lens[s].item())
                num_blocks_s = (kv_len_s + block_size - 1) // block_size
                for b in range(num_blocks_s):
                    col_start = b * block_size
                    col_end = col_start + block_size
                    tile = mask_cpu[s, :aligned_max_query_len, col_start:col_end]
                    seq_tiles.append(tile.contiguous())
                attention_mask_tiles.append(seq_tiles)
            # active_block_indices stays None, so forward iterates all blocks.
        else:
            # Sliding window: arithmetic block-skip. Blocks entirely outside
            # every query's window are dropped; interior blocks share a
            # zero mask tile; only boundary blocks get real per-query cutoffs.
            active_block_indices = []
            query_lens_list = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
            seq_lens_list = seq_lens.tolist()

            for s in range(num_seqs):
                kv_len_s = int(seq_lens_list[s])
                query_len_s = int(query_lens_list[s])
                context_len_s = kv_len_s - query_len_s

                active_bs, tiles = self._build_active_tiles_with_skip(
                    kv_len_s,
                    query_len_s,
                    context_len_s,
                    aligned_max_query_len,
                    apply_causal_mask,
                )
                active_block_indices.append(active_bs)
                attention_mask_tiles.append(tiles)

        # Gather indices for the attention loop, one row per active block.
        num_active = [len(tiles) for tiles in attention_mask_tiles]
        page_index_table_cpu = torch.zeros(
            num_seqs, max(num_active), INT32_ELEMS_PER_STICK, dtype=torch.int32
        )
        for s, n in enumerate(num_active):
            blocks_s = slice(n) if active_block_indices is None else active_block_indices[s]
            page_index_table_cpu[s, :n, 0] = block_table[s, blocks_s]

        # Ragged decode schedule (#612 + #468): only for decode-only batches with
        # no sliding window. Pure host arithmetic; layer-invariant per step.
        decode_schedule = None
        if max_query_len == 1 and self.sliding_window is None:
            decode_schedule = self._build_decode_schedule(
                num_seqs,
                seq_lens,
                query_start_loc,
                block_table,
                num_active,
                attention_mask_tiles,
            )

        return SpyreAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_seqs=common_attn_metadata.num_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            block_size=self.block_size,
            slot_mapping=slot_mapping,
            apply_causal_mask=apply_causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attention_mask_tiles=attention_mask_tiles,
            active_block_indices=active_block_indices,
            page_index_table_cpu=page_index_table_cpu,
            aligned_max_query_len=aligned_max_query_len,
            aligned_max_seq_len=aligned_max_seq_len,
            decode_schedule=decode_schedule,
        )


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Spyre stick size is 128 bytes; tensors are transferred as float16 (2 bytes),
        # so block_size must be a multiple of 64 (= 128 / 2) to satisfy stick alignment.
        # This matches the constraint on head_size in supports_head_size().
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SpyreAttentionImpl"]:
        return SpyreAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SpyreAttentionMetadataBuilder"]:
        return SpyreAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # K and V are separate tensors in SpyrePagedKVCache, each with the same
        # shape. The base vLLM API expects a single tuple here; callers like
        # get_kv_cache_block_dim and KV-transfer code index into it directly.
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Spyre stick size is 128 bytes; tensors are transferred as float16 (2 bytes),
        # so head_size must be a multiple of 64 (= 128 / 2) to satisfy stick alignment.
        return head_size % 64 == 0

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes


class SpyreAttentionImpl(AttentionImpl[SpyreAttentionMetadata]):
    """Online-softmax paged attention iterating over KV pages.

    KV cache is a tuple (k_pages, v_pages) where each is one dense tensor of
    shape [num_blocks, block_size, num_kv_heads, head_size] on Spyre. Pages are
    read by indirect access, indexing the dense tensor with a device-resident
    page index. No gather masks.

    On Spyre, the per-page attention loop and reshape_and_cache are compiled
    via torch.compile with fixed iteration counts. A dict
    caches compiled variants per unique loop length.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        tile_kv_heads: int | None = None,
        tile_q_heads: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # Head-axis coarse-tile counts for the online-softmax kernel. None ->
        # resolved lazily per (block_size, padded_query_len) from the tuned config
        # (_get_attn_tile_config); an explicit value overrides the config (tests/
        # tuner). tile_kv_heads must divide num_kv_heads; tile_q_heads must divide
        # num_queries_per_kv.
        if tile_kv_heads is not None and (
            tile_kv_heads < 1 or self.num_kv_heads % tile_kv_heads != 0
        ):
            raise ValueError(
                f"tile_kv_heads={tile_kv_heads} must be >=1 and divide "
                f"num_kv_heads={self.num_kv_heads}"
            )
        if tile_q_heads is not None and (
            tile_q_heads < 1 or self.num_queries_per_kv % tile_q_heads != 0
        ):
            raise ValueError(
                f"tile_q_heads={tile_q_heads} must be >=1 and divide "
                f"num_queries_per_kv={self.num_queries_per_kv}"
            )
        self._tile_kv_heads_override = tile_kv_heads
        self._tile_q_heads_override = tile_q_heads

        # `== STOCK`, not `!= NONE`: a bare CompilationConfig (e.g. the unit-test
        # fixture) leaves mode unset (Python None), which `!= NONE` would wrongly
        # treat as compiled. The platform resolves compiled runs to STOCK.
        _mode = get_current_vllm_config().compilation_config.mode
        self._compile_attn = _mode == CompilationMode.STOCK_TORCH_COMPILE

        # ALiBi slopes: per-head linear-bias coefficients (BLOOM/MPT style).
        # Reshape once to [num_kv_heads, num_queries_per_kv, 1, 1] so the
        # per-block bias construction in _online_softmax_attention broadcasts
        # cleanly against the score-tile shape.
        if alibi_slopes is not None:
            slopes_t = torch.tensor(alibi_slopes, dtype=torch.float16)
            if slopes_t.numel() != num_heads:
                raise ValueError(
                    f"alibi_slopes must have length num_heads={num_heads}, got {slopes_t.numel()}"
                )
            self.alibi_slopes: torch.Tensor | None = slopes_t.view(
                num_kv_heads, self.num_queries_per_kv, 1, 1
            )
        else:
            self.alibi_slopes = None

        # Normalise the API's Optional[float] into a plain float so the kernel
        # can bake it as a closure constant. logits_soft_cap == 0.0 disables
        # soft-capping (kernel takes the same path as upstream).
        self.logits_soft_cap: float = 0.0 if logits_soft_cap is None else float(logits_soft_cap)

        # Always compiled: eager index_copy_ rejects an int32 index and falls
        # back to CPU with an int64 one.
        self._reshape_fn = torch.compile(_reshape_and_cache_kernel, dynamic=False)

        # Compiled attention loops, keyed by
        # (num_blocks, padded_query_len, tile_kv_heads, tile_q_heads, batched_decode).
        self._attn_fns: dict[tuple, object] = {}

        logger.debug_once(
            "Using SpyreAttentionBackend with a dense paged KV cache and indirect page gather"
        )

    def _resolve_tile_counts(self, block_size: int, padded_query_len: int) -> tuple[int, int]:
        """(tile_kv_heads, tile_q_heads): explicit overrides, else tuned config.

        `tile_kv_heads` is gated on padded_query_len >= KV_HEAD_TILE_THRESHOLD:
        kv_head tiling only helps at long prefill and is measured harmful at
        decode/short prefill (Plan 8), so short lengths keep tile_kv_heads == 1.
        """
        if (
            self._tile_kv_heads_override is not None
            and self._tile_q_heads_override is not None
        ):
            kv_heads, q_heads = self._tile_kv_heads_override, self._tile_q_heads_override
        else:
            cfg = _get_attn_tile_config(
                self.head_size, self.num_kv_heads, self.num_queries_per_kv, block_size
            )
            kv_heads = (
                self._tile_kv_heads_override
                if self._tile_kv_heads_override is not None
                else cfg["tile_kv_heads"]
            )
            q_heads = (
                self._tile_q_heads_override
                if self._tile_q_heads_override is not None
                else cfg["tile_q_heads"]
            )
        if padded_query_len < KV_HEAD_TILE_THRESHOLD:
            kv_heads = 1
        return kv_heads, q_heads

    def _get_attn_fn(self, num_blocks: int, padded_query_len: int, block_size: int,
                     batched_decode: bool = False):
        # self.alibi_slopes and self.logits_soft_cap are fixed per instance, so
        # has_alibi and logits_soft_cap don't need to be part of the cache key.
        # Tile counts vary with block_size (config is shape-keyed) and
        # padded_query_len (the kv_head threshold), so they are part of the key.
        # batched_decode selects the #612 pre-gathered batched kernel variant
        # (alibi/soft-cap/tiling off), so it is a distinct cache entry.
        if batched_decode:
            tile_kv_heads, tile_q_heads = 1, 1
        else:
            tile_kv_heads, tile_q_heads = self._resolve_tile_counts(block_size, padded_query_len)
        key = (num_blocks, padded_query_len, tile_kv_heads, tile_q_heads, batched_decode)
        if key not in self._attn_fns:
            self._attn_fns[key] = _maybe_compile(
                _create_compilable_page_attn(
                    num_blocks,
                    padded_query_len,
                    self.num_heads,
                    self.head_size,
                    has_alibi=(self.alibi_slopes is not None) and not batched_decode,
                    logits_soft_cap=0.0 if batched_decode else self.logits_soft_cap,
                    tile_kv_heads=tile_kv_heads,
                    tile_q_heads=tile_q_heads,
                    batched_decode=batched_decode,
                ),
                self._compile_attn,
            )
        return self._attn_fns[key]

    def _ragged_decode_eligible(self, attn_metadata: "SpyreAttentionMetadata") -> bool:
        """Group-by-bucket decode fast path applies iff the builder produced a
        schedule (decode-only, no sliding window, num_seqs within lattice) AND
        this layer has neither ALiBi nor a logits soft-cap (the batched decode
        kernel omits both)."""
        if attn_metadata.decode_schedule is None:
            return False
        if self.alibi_slopes is not None:
            return False
        return self.logits_soft_cap == 0.0

    # `kv_cache` widens the base's `torch.Tensor` to `SpyrePagedKVCache`,
    # which `TorchSpyreModelRunner.initialize_kv_cache_tensors` allocates
    # and `bind_kv_cache` smuggles through a dict typed `dict[str, Tensor]`.
    # The matching pair of overrides preserves the runtime contract; ty
    # cannot see the co-evolution.
    @_record_function("spyre_attn::forward")
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output

        k_pages, v_pages = kv_cache
        _target_device = k_pages.device
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Only the first layer of a step pays for the device mirror.
        if attn_metadata.page_index_tables is None:
            table_cpu = attn_metadata.page_index_table_cpu
            assert table_cpu is not None
            attn_metadata.page_index_tables = [
                convert(table_cpu[s].contiguous(), device=_target_device)
                for s in range(table_cpu.shape[0])
            ]
        if attn_metadata.slot_mapping_device is None:
            attn_metadata.slot_mapping_device = convert(
                attn_metadata.slot_mapping[:num_actual_tokens], device=_target_device
            )
        if attn_metadata.attention_mask_tiles_device is None:
            tiles_cpu = attn_metadata.attention_mask_tiles
            assert tiles_cpu is not None, (
                "attention_mask_tiles must be precomputed by the metadata builder"
            )
            attn_metadata.attention_mask_tiles_device = [
                [convert(t, device=_target_device) for t in seq_tiles] for seq_tiles in tiles_cpu
            ]

        # Mirror the ragged-decode schedule's host tensors to device once per
        # step (first layer only). Layer-invariant: every layer reuses these.
        sched = attn_metadata.decode_schedule
        if sched is not None and sched.query_row_ids_sorted_dev is None:
            sched.query_row_ids_sorted_dev = convert(
                sched.query_row_ids_sorted_cpu.contiguous(), device=_target_device
            )
            for grp in sched.groups:
                grp.block_ids_dev = convert(grp.block_ids_cpu.contiguous(), device=_target_device)
                assert grp.mask_cpu is not None
                grp.mask_dev = convert(grp.mask_cpu.contiguous(), device=_target_device)

        # Step 1: Reshape and cache — scatter new tokens into their slots
        self._reshape_and_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata.slot_mapping_device,
        )

        # Step 2: Online softmax attention over pages (varlen)
        output = self._online_softmax_attention(
            query[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata,
            output,
            _target_device,
        )

        return output

    @_record_function("spyre_attn::reshape_and_cache")
    def _reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter new K/V tokens into their cache slots.

        key, value: [num_tokens, num_kv_heads, head_size] on the pages' device,
            strided last-dim views of the fused QKV output
        k_pages, v_pages: [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: [num_tokens] on the pages' device
        """
        # A source on the wrong device falls back to CPU silently, without raising.
        assert key.device.type == k_pages.device.type, (
            f"reshape_and_cache source is on {key.device.type}, pages on {k_pages.device.type}"
        )

        # Valid because a view keeps the slot-outermost device layout.
        slots = (-1, k_pages.shape[2], k_pages.shape[3])
        self._reshape_fn(key, value, k_pages.view(slots), v_pages.view(slots), slot_mapping)

    def _run_ragged_decode(
        self,
        query_dev: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode driver (#612 + #468, group-by-block-bucket): "jump off the train".

        Runs one #612 batched call per block-bucket group over the group's own
        block count. Each sequence is computed to completion in a single call —
        no cross-group carry. Total KV-step × request work tracks the SUM of
        contexts (Σ_groups group_size × group_blocks), not batch × max.

        The single-group case (all seqs in the same block bucket) is exactly the
        #612 bucketed decode: one batched bmm over the whole padded batch.
        """
        sched = attn_metadata.decode_schedule
        assert sched is not None
        num_seqs = attn_metadata.num_seqs
        kv = self.num_kv_heads
        qpk = self.num_queries_per_kv
        num_heads = self.num_heads
        d = self.head_size
        block_size = attn_metadata.block_size

        assert sched.query_row_ids_sorted_dev is not None
        # Query rows in sorted (ascending-blocks) order: [num_seqs, num_heads, D].
        q_sorted = query_dev.index_select(0, sched.query_row_ids_sorted_dev)

        # Per sorted slot, the finalized [num_heads, D] output. Kept on CPU: the
        # finalize reshape merges the kv stick axis into the head axis
        # ([count*kv,qpk,1,d] -> [count,kv*qpk,d]), which is a lossy relayout on
        # device but exact once materialized on CPU. Unsorted and written back to
        # the device `output` in one convert at the end.
        out_sorted = torch.empty(num_seqs, num_heads, d, dtype=output.dtype, device="cpu")

        for grp in sched.groups:
            slot = grp.slot_start
            members = grp.num_members
            b_seqs = grp.seqs_bucket
            b_blocks = grp.blocks_bucket

            # Gather this group's queries, pad to b_seqs, pack to [b_seqs*kv,qpk,1,d].
            # Preallocate + prefix-copy instead of F.pad (F.pad routes through a
            # copy_from_d2d offset view that fails to compile on Spyre here).
            q_seg = q_sorted[slot : slot + members]
            if members < b_seqs:
                q_full = torch.zeros(b_seqs, num_heads, d, dtype=q_seg.dtype, device=q_seg.device)
                q_full[:members] = q_seg
                q_seg = q_full
            q_packed = q_seg.reshape(b_seqs, kv, qpk, d).reshape(b_seqs * kv, qpk, 1, d).contiguous()

            # Pre-gather K/V/mask per block (the #612 batched-decode kernel inputs).
            assert grp.block_ids_dev is not None and grp.mask_dev is not None
            k_list, v_list, mask_list = self._pack_segment_kv(
                k_pages, v_pages, grp.block_ids_dev, grp.mask_dev, b_seqs, b_blocks, kv, block_size, d
            )

            # One #612 batched call for the whole group. result: [b_seqs*kv,qpk,1,d].
            kernel = self._get_attn_fn(b_blocks, 1, block_size, batched_decode=True)
            result = kernel(q_packed, k_list, v_list, None, mask_list, self.scale)

            # Normalize is folded into the kernel (returns attn already divided).
            # Move to CPU (exact) then merge kv into the head axis and scatter into
            # the sorted output slots for this group's real members.
            res_cpu = result[: members * kv].to("cpu").reshape(members, kv * qpk, d)
            out_sorted[slot : slot + members] = res_cpu.to(out_sorted.dtype)

        # Unsort on CPU (output[i] = out_sorted[inv_perm[i]]), then write the whole
        # decode result into the device `output` in one convert.
        assert sched.inv_perm_cpu is not None
        unsorted = out_sorted.index_select(0, sched.inv_perm_cpu)
        output[:num_seqs] = convert(unsorted, device=output.device)
        return output

    def _pack_segment_kv(
        self, k_pages, v_pages, block_ids_dev, mask_dev, b_seqs, b_blocks, kv, block_size, d
    ):
        """Gather + pack a group's K/V pages and mask into per-block lists.

        Mirrors #612's packing: one index_select for the whole (b_seqs, b_blocks)
        grid, folded to per-block [b_seqs*kv, 1, block_size, d] tensors so the
        kernel matmul stays 4-D (lower_bmm handles up to 4 dims).
        """
        k_gath = k_pages.index_select(0, block_ids_dev)
        v_gath = v_pages.index_select(0, block_ids_dev)
        k_by_block = (
            k_gath.reshape(b_seqs, b_blocks, block_size, kv, d).permute(1, 0, 3, 2, 4).contiguous()
        )
        v_by_block = (
            v_gath.reshape(b_seqs, b_blocks, block_size, kv, d).permute(1, 0, 3, 2, 4).contiguous()
        )
        k_list = [k_by_block[i].reshape(b_seqs * kv, 1, block_size, d).clone() for i in range(b_blocks)]
        v_list = [v_by_block[i].reshape(b_seqs * kv, 1, block_size, d).clone() for i in range(b_blocks)]
        # mask_dev: [b_seqs, b_blocks, block_size] -> per-block [b_seqs*kv,1,1,block_size].
        mask_by_block = mask_dev.permute(1, 0, 2).contiguous()
        mask_list = [
            mask_by_block[i]
            .unsqueeze(1)
            .expand(b_seqs, kv, block_size)
            .reshape(b_seqs * kv, 1, 1, block_size)
            .clone()
            for i in range(b_blocks)
        ]
        return k_list, v_list, mask_list

    @_record_function("spyre_attn::online_softmax")
    def _online_softmax_attention(
        self,
        query_dev: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        _target_device: torch.device,
    ) -> torch.Tensor:
        """FlashAttention-style online softmax iterating over KV pages (varlen).

        Handles multiple sequences using query_start_loc for the varlen layout.
        k_pages/v_pages are dense [num_blocks, block_size, num_kv_heads,
        head_size] tensors on Spyre; each iteration gathers one page with a
        one-element int32 device index, then feeds it to bmm without slicing.

        Writes results directly into the caller's output buffer in-place.

        Query is assembled on device into the padded 4D tensor
        [num_kv_heads, num_queries_per_kv, aligned_max_query_len, head_size]
        the kernel expects.

        Args:
            query_dev: Query on the target device, [num_tokens, num_heads, D].
        """
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        num_queries_per_kv = self.num_queries_per_kv
        block_size = attn_metadata.block_size

        num_seqs = attn_metadata.num_seqs
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        mask_tiles_all = attn_metadata.attention_mask_tiles_device
        active_block_indices_all = attn_metadata.active_block_indices
        aligned_max_query_len = attn_metadata.aligned_max_query_len
        page_index_tables = attn_metadata.page_index_tables
        assert mask_tiles_all is not None, (
            "attention_mask_tiles_device must be mirrored by forward()"
        )
        assert page_index_tables is not None, "page_index_tables must be mirrored by forward()"

        # Ragged-decode fast path (#612 + #468): batch the whole decode step into
        # segmented bmm calls. Default-on when eligible; the per-seq loop below is
        # the fallback for small-N / prefill / sliding-window / alibi / soft-cap.
        if self._ragged_decode_eligible(attn_metadata):
            return self._run_ragged_decode(query_dev, k_pages, v_pages, attn_metadata, output)

        for seq_idx in range(num_seqs):
            # Most-naive implementation: no parallelization
            # over sequences or GQA optimization
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            if query_len == 1:
                # Decode: the single real token goes at row 0 of the padded
                # buffer; the trailing padded rows are masked out downstream.
                q_dev = query_dev.unbind(dim=0)[q_start].reshape(
                    num_kv_heads, num_queries_per_kv, 1, head_size
                )
                if aligned_max_query_len > 1:
                    q_dev = torch.nn.functional.pad(q_dev, (0, 0, 0, aligned_max_query_len - 1))
            else:
                q_seq = query_dev[q_start:q_end]

                # Pad query to global aligned_max_query_len (uniform for all seqs)
                if aligned_max_query_len > query_len:
                    q_seq = torch.nn.functional.pad(
                        q_seq,
                        (0, 0, 0, 0, 0, aligned_max_query_len - query_len),
                        mode="constant",
                        value=0.0,
                    )

                # Reshape: [padded_query_len, num_heads, head_size]
                #   → [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
                q = q_seq.unsqueeze(0).transpose(1, 2).contiguous()
                q_dev = q.reshape(
                    num_kv_heads, num_queries_per_kv, aligned_max_query_len, head_size
                )

            num_blocks_needed = (kv_len + block_size - 1) // block_size

            # Restrict to active (non-fully-masked) blocks when sliding window
            # is set. When active_block_indices_all is None (no sliding), all
            # blocks are active in their natural order.
            if active_block_indices_all is not None:
                active_bs = active_block_indices_all[seq_idx]
            else:
                active_bs = list(range(num_blocks_needed))

            if len(active_bs) == 0:
                # Every KV position is outside every query's window. Attention
                # over the empty set is undefined; write zeros.
                output[q_start:q_end] = 0.0
                continue

            page_index_table = page_index_tables[seq_idx]
            # mask_tiles_all[seq_idx] is indexed by position within active_bs.
            mask_tiles = mask_tiles_all[seq_idx][: len(active_bs)]

            # ALiBi bias tiles: slope[h] * (kv_pos - context_len), one per block.
            #
            # The full ALiBi form is slope[h] * (kv_pos - (context_len + q_rel)),
            # which varies over both query and KV positions. The (context_len + q_rel)
            # term is a per-query-row constant, and softmax is invariant under adding
            # any per-row constant to its input (numerator and denominator both pick
            # up the same exp() factor). We therefore drop it and keep only the
            # kv-dependent term — the softmax output is bit-identical to the full
            # form, and each tile stays 1D over KV (block_size floats per head)
            # instead of 2D (aligned_max_query_len * block_size).
            #
            # Matches vllm/v1/attention/ops/triton_attention_helpers.py::apply_alibi_to_score
            # (alibi_offset = seq_offset - context_len) — the production Triton path.
            #
            # Per-tile shape: [num_kv_heads, num_queries_per_kv, 1, block_size].
            alibi_bias_tiles: list[torch.Tensor] | None = None
            if self.alibi_slopes is not None:
                context_len = kv_len - query_len
                alibi_bias_tiles = []
                for b in active_bs:
                    kv_pos = torch.arange(
                        b * block_size,
                        (b + 1) * block_size,
                        dtype=torch.float16,
                    )
                    rel = (kv_pos - context_len).view(1, 1, 1, block_size)
                    bias = self.alibi_slopes * rel
                    alibi_bias_tiles.append(convert(bias, device=_target_device))

            # Run attention on target device
            tile_kv_heads, tile_q_heads = self._resolve_tile_counts(
                block_size, aligned_max_query_len
            )
            if tile_kv_heads > 1 or tile_q_heads > 1:
                # Name the kernel inputs so the head-dim tile hints can bind
                # (PR #3674: naming only intermediates is insufficient).
                _name_attn_inputs(
                    q_dev,
                    k_pages,
                    v_pages,
                    mask_tiles,
                    alibi_bias_tiles,
                    num_kv_heads,
                    num_queries_per_kv,
                    block_size,
                    head_size,
                )
            attn_fn = self._get_attn_fn(len(active_bs), aligned_max_query_len, block_size)
            result = attn_fn(
                q_dev,
                k_pages,
                v_pages,
                page_index_table,
                mask_tiles,
                self.scale,
                alibi_bias_tiles=alibi_bias_tiles,
            )

            assert result.dtype == output.dtype
            # Kernel returns [num_kv_heads, num_queries_per_kv, padded_query_len,
            # head_size]; fold the head dims and move the query axis out to match
            # output's [num_tokens, num_heads, head_size]. .contiguous() is
            # load-bearing under coarse tiling (the accumulator's device layout
            # carries a tiled kv_head dim).
            result = result.reshape(1, self.num_heads, aligned_max_query_len, head_size)
            result = result.transpose(1, 2).contiguous()
            output[q_start:q_end] = result[0, :query_len, :, :]

        return output
