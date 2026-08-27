# Plan 11: "jump-off-the-train" for decode-only batches (issue #468)

Status: PLAN (not started). Builds directly on **PR #612** (`feat(attn): add
bucketed decode fast path`, jvlunteren, open, "In progress") — the concrete first
step of #403. #612 lands the batched-`bmm` decode kernel this plan extends; #468
is the ragged-context optimization on top of #612's pad-to-bucket batching.
Companion to `10-qpk-lq-tiling.md` (head-axis tiling) and the mechanism notes in
`06-kv-page-residency.md` / `08-tiling-crossover-long-prefill.md`.

## What PR #612 already builds (the base to extend)

#612 replaces the per-seq decode loop with a single 4D batched `bmm` over all
sequences (same FlashAttention online-softmax, applied per block across the
batched leading dim). Measured ~4x at N=32 (42.0 vs 10.4 tok/s, granite-3.3-8b).
Key design points that #468 must slot into:

- **(num_seqs, num_blocks) bucket lattice**, Dynamo-unrolled, one compiled kernel
  per bucket: `NUM_SEQS_BUCKETS=(8,16,32)`, `NUM_BLOCKS_BUCKETS=(1,2,4,8,16)`.
  Every decode batch is padded UP to the nearest bucket in BOTH dims.
- Gated to: pure decode, `num_seqs >= 8`, no ALiBi / soft-cap / sliding-window.
  Everything else falls back to the existing per-seq loop.
- **This is exactly the pad-to-max that #468 targets.** #612's own stated con:
  "Bucketization can introduce padding overhead, wasting up to ~50% of FLOPs in
  the worst-case alignment scenario." The `num_blocks` bucket pads EVERY sequence
  in the batch up to the longest sequence's block count — the ragged-context waste
  from the issue. `num_seqs` bucketing is orthogonal padding (batch-dim rounding).

So #468 = make #612's `num_blocks` (KV) loop RAGGED: drop each sequence from the
batched bmm once the loop passes its own block count, instead of running all
sequences for the bucket's full `num_blocks`.

### Reviewer constraints on #612 (must respect in any follow-on)

From the PR review (tdoublep, bringlein) — these bind #468's implementation:

- **No new env vars.** Bucketed decode should be the DEFAULT (drop
  `SPYRE_BUCKETED_DECODE`). Bucket sets should default sensibly (powers of two up
  to `max-num-seqs` for seqs, up to `max-model-len` for blocks); make configurable
  via engine args, not env vars, if at all.
- **Reuse `bisect.bisect_left`** for bucket lookup (the outer graph recorder
  already uses it) rather than a hand-rolled `_bucket_up`.
- **Hot-path materialization is suspect**: reviewers flagged the per-block
  `k/v .reshape(...).clone()` and `mask .expand(...).reshape(...).clone()` lists
  with "could be very slow" / "does this happen at every layer?". #468 adds
  per-segment work on top, so it MUST NOT multiply this per-layer host cost —
  precompute segment membership/masks once in `build()` (host, per step), not
  per layer.
- **Reduce duplication**: bringlein asked to fold the batched kernel into the
  normal specialized online-softmax via `if constant:` branches rather than a
  separate kernel. #468 should extend that unified kernel, not fork a third.

## The problem (from #468)

Under Spyre's "restricted SPMD" model the only lever for more parallelism is
batched matmul (`torch.bmm`, #403): stack the batch's per-request work into one
matmul with a batch dim. For a **decode-only batch** every request contributes
exactly one query token, but the requests have **different context lengths**
(kv_len). A single batched matmul over the KV axis must pick ONE kv loop length,
so the naive batched kernel pads every request to the batch's max context. With a
mixed batch (say most requests ~100 tokens, one ~2000) that is up to ~20x wasted
compute on the short requests.

"Jump off the train": process the whole batch together for the KV range they all
share, and **drop each request from the batched matmul once the KV loop passes
its context end**, shrinking the batch dim as the loop advances. Total KV-step ×
request work then tracks the SUM of context lengths, not batch × max.

    batch (sorted by kv_len ascending):  r0=128  r1=384  r2=2048
    kv loop position ─────────────────────────────────────────►
    [0,128)     all 3 in the batched matmul     (bmm batch = 3)
    [128,384)   r0 dropped                       (bmm batch = 2)
    [384,2048)  r1 dropped                       (bmm batch = 1)

## Why the padding waste is real (post-#612)

Before #612, decode looped per sequence (`_online_softmax_attention:1345`), each
walking only its own `ceil(kv_len/block_size)` blocks — O(sum of contexts) but
batch=1 parallelism per matmul. #612 fixes the parallelism by batching all
sequences into one bmm, but the batched matmul has ONE `num_blocks` loop length,
so it pads every sequence up to the bucket's block count (the longest seq, rounded
up). At high context skew (e.g. 15x128-token + 1x2048-token requests) the short
requests run ~16x more KV steps than they need. #468 removes that: the batched
matmul's `num_blocks` loop becomes segmented, and a sequence leaves the batch once
its blocks are exhausted.

## Load-bearing constraints (mostly settled by #612; verify the rest)

#612 already demonstrates the key facts empirically, so these are firmer than in
the pre-#612 draft. Items still marked VERIFY are where the knowledgebase MCP was
unavailable in this worktree.

- **A1. A compiled kernel bucket is a fixed `(num_seqs, num_blocks)`.** #612
  proves this: it Dynamo-unrolls one kernel per `(num_seqs, num_blocks)` lattice
  point. So a "segment" (a run of KV blocks over a constant surviving sub-batch)
  is just an existing #612 bucket `(surviving_seqs, segment_blocks)` — jump-off
  reuses #612's compiled kernels, it does not need a new kernel key.
- **A2. The schedule is known at forward entry** (settled). All kv_lens are known
  in `build()`; the sequence of (segment_blocks, surviving_members) is pure host
  arithmetic. The issue explicitly relies on this ("schedule known at the start …
  plan the combination of recorded graph buckets ahead of time"). No device-side
  data-dependent control flow.
- **A3. Bucket count stays bounded BY REUSING #612's lattice.** Each segment snaps
  its `num_blocks` to a `NUM_BLOCKS_BUCKETS` rung and its surviving count to a
  `NUM_SEQS_BUCKETS` rung — so segments draw from the SAME finite lattice #612
  already compiles. No new compile-count axis; worst case a batch touches a few
  lattice points instead of one. (Naive per-distinct-context bucketing WOULD
  explode — the lattice snap is what avoids it.)
- **A4 (VERIFY). Cross-segment online-softmax carry is expressible.** Each segment
  emits partial (max,sum,acc) for its survivors; the next segment must resume from
  it. Confirm the carry can live in host-held device tensors handed into the next
  bucket call without a per-segment recompile, and that shrinking the batch dim
  between segments is just "call a smaller bucket" (A1), not an in-graph reshape.

## Design

### Relationship to #612: segment across its existing num_blocks lattice

The cleanest framing (minimal new machinery, respects reviewer "reduce
duplication"): jump-off-the-train is a HOST-SIDE DRIVER that calls #612's existing
bucketed kernel once per segment over the shrinking sub-batch, instead of once
over the full-padded batch.

    #612 today:   one call, bucket (num_seqs=32, num_blocks=16), ALL seqs padded to 16
    #468 driver:  sort by blocks; call sequence of #612 buckets over survivors:
                    seg A: (bucket_seqs>=32, blocks=1..b0)  all 32 seqs
                    seg B: (bucket_seqs>=k1, blocks=b0..b1) survivors after b0
                    seg C: (bucket_seqs>=k2, blocks=b1..b2) ...
                  carrying online-softmax (max,sum,acc) across segments.

Each per-segment call is an already-compiled #612 lattice point, so no new kernel.

### Variant choice (issue's two options)

- **(a) context loop outer, interrupt anytime** — fewer interruptions, may refetch
  q/k/v heads per segment.
- **(b) head loop outer** — avoids refetch, but restarts the jump-schedule inside
  every head group (far more interruptions).

**Choose (a) context-loop-outer**, matching #612 (its bmm batches the leading
`num_seqs*kv_head` dim and loops blocks inside). Plan-10 rationale: the head axis
is what `work_division` splits across ~32 cores and head-major restructuring is
where the coarse-tile/stick machinery is most fragile; keep heads as #612's inner
batched structure and make the ragged batch the outer segmentation. Segment
boundaries are few (≤ number of `NUM_BLOCKS_BUCKETS` rungs the batch spans).

### Segment schedule (host-side, in `build()`)

Decode-only branch, given per-request `kv_len` (→ `blocks = ceil(kv_len/block)`):

1. **Sort** requests by block count ascending; keep the permutation to unsort.
2. **Segment at `NUM_BLOCKS_BUCKETS` rungs** (reuse #612's `bisect.bisect_left` on
   its lattice, per reviewer): segment k spans blocks (rung_{k-1}, rung_k], its
   members are all requests with blocks >= rung_{k-1}+1. Surviving count per
   segment snaps UP to a `NUM_SEQS_BUCKETS` rung.
3. Emit `decode_segments: list[(seqs_bucket, blocks_bucket, member_ids, block_range)]`.
4. Store in `SpyreAttentionMetadata` with the sort permutation. Build per-segment
   page-index / mask views ONCE here (host, per step) — NOT per layer (directly
   addresses the #612 reviewer worry that `.clone()/.expand()` lists run every
   layer). Segment membership + masks are layer-invariant for a step; only the
   q/k/v data changes per layer, so precompute the index/mask structure in build()
   and let each layer's forward reuse it.

Pure host arithmetic (A2); unit-testable off device.

### Batched-kernel driver (extends #612's fast path)

In the unified decode fast path (folded into the specialized online-softmax per
bringlein's request, guarded by `if decode-batched:`), when preconditions hold
(pure decode, num_seqs >= min bucket, no ALiBi/soft-cap/sliding-window):

- For each segment, gather the surviving sub-batch's q `[seqs_bucket*kv_head, qpk,
  1, d]` and the segment's KV block range, and call the #612 bucket
  `(seqs_bucket, blocks_bucket)`.
- **Carry** per-request online-softmax (max,sum,acc) across segments: a dropped
  request finalizes its output at its last segment; survivors resume with carried
  state. Standard flash rescale at segment boundaries (`new_max`, `rescale =
  exp(old_max-new_max)`, `acc = acc*rescale + p@v`) — same math #612 already runs
  per block, now also applied per segment. No new op (A4 pending confirmation).
- **Unsort** and scatter each finalized `[num_heads, d]` into `output`.

### Bucketing / compile management (A3)

Reuse #612's `(NUM_SEQS_BUCKETS, NUM_BLOCKS_BUCKETS)` lattice verbatim — segments
snap into it, so NO new compiled variants beyond what #612 already emits. Per the
#612 review, replace the env-var bucket config with sensible power-of-two defaults
(seqs up to `max-num-seqs`, blocks up to `max-model-len`) and `bisect` lookup; do
that in #612, and #468 inherits it.

## Implementation steps

1. **Land on #612.** Rebase onto #612's merged kernel; read its final API
   (batched q shape, page-gather batching, `(num_seqs,num_blocks)` bucket lookup,
   whether it folded into the specialized online-softmax per bringlein). Adjust
   the field/call names below to match. Do NOT reimplement batched bmm.
2. **Host schedule** in `build()`: decode-only detect, sort+permutation, segment
   at `NUM_BLOCKS_BUCKETS` rungs, per-segment membership + `NUM_SEQS_BUCKETS` snap.
   Pure Python; unit-testable off-device. Precompute per-segment page-index/mask
   views here (once per step), NOT per layer.
3. **Metadata fields**: `decode_segments`, `sort_perm`, per-segment page-index /
   mask views. Prefill + mixed-batch + small-N decode keep #612's fallback.
   current per-seq loop).
4. **Segment driver** in `_online_softmax_attention`: per-segment batched call +
   cross-segment online-softmax carry + unsort scatter to `output`.
5. **Bucket policy**: start with exact buckets; add batch-dim ladder + masked
   padding only if compile count is too high.
6. **Tuner / bench**: extend `examples/tune_attn_tiling.py` (now multi-seq capable
   after plan-10's `--batch-size`) with a `--decode-context-spread` mode that
   builds a decode-only batch with a skewed kv_len distribution, to measure
   jump-off-the-train vs pad-to-max. Device-time via the AIUPTI profiler
   (`LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:...`; see plan-10's rebuild note —
   plain `uv run` reverts the profiler-linked wheel, use the venv python directly
   or `uv run --no-sync`).

## Measurement plan (pod-only; single accelerator, never concurrent)

Decode-only batches, fixed batch size (e.g. 16, 32), varying context skew:

- **Uniform contexts** (all equal): jump-off-the-train must match pad-to-max
  (no drops) — correctness + no regression.
- **Skewed** (e.g. 15×128 + 1×2048): expect device time ≈ Σ contexts, i.e.
  ≈ pad-to-max × (mean/max) — a large win at high skew. Report device time vs the
  naive pad-to-max batched kernel AND vs the current per-seq loop.
- **Compile count**: log distinct compiled buckets per config; confirm bounded.
- Gate correctness against a CPU reference for every skew (reuse the tuner's
  outlier-fraction-vs-untiled-baseline gate from plan-10; pure allclose on the
  fp16 ref drifts with context length).

## Does coarse tiling (plan-10) help here? No — and why is informative

- Plan-10 `kv_head`/`qpk` coarse tiling is gated OFF for decode
  (`KV_HEAD_TILE_THRESHOLD=1024` on `padded_query_len`) and was MEASURED harmful
  at short query lengths (Plan 8: +20.8% decode t4). Its win comes from shrinking
  the QUERY-axis transients (`scores/probs/output` = `[kv_head, qpk, lq, blk]`);
  at decode `lq=1`, so those transients are already tiny — nothing to shrink — and
  the both-pages HBM hoist just adds a round-trip. The win was scheduling on big
  prefill transients, not memory-op reduction (mem-op share stayed ~0.3-1%).
- More fundamentally, tiling and this feature push OPPOSITE directions on the same
  axis: tiling SUBDIVIDES an op too big for the scratchpad; decode's problem
  (#612/#468) is each op is too SMALL (one query token/request), so you BATCH
  (combine requests) for parallelism. Jump-off-the-train is batch-shrinking, not
  tiling.
- What plan-10 DOES contribute to #468 is the finding, not the perf knob:
  `kv_head` is a native page axis that slices cleanly against the slot-major cache
  layout, whereas the query axes hit the "no mechanism to resolve stick
  incompatibility" wall (broadcast-expand fused into the page gather). #612's
  batched decode parallel dimension is `num_seqs × kv_head`; the same clean-slice
  property is what makes its batched page-gather feasible, and confirms the
  batch/seq axis (a real tensor dim on q and the matmul, NOT a page dim) does not
  trip that wall (#612 runs correctly). Verify #468's segmented variant likewise.

## Decision rule

- If skewed decode device time tracks Σ contexts (not batch×max) AND compile count
  stays bounded AND uniform-context is a no-op vs pad-to-max → ship, gated to
  decode-only batches with context skew above a threshold (below it, pad-to-max is
  simpler and the segment overhead is not worth it).
- If segment-switch overhead (bucket switches per forward × per-layer × per-step)
  dominates the saved compute at realistic batch sizes → keep pad-to-max; document
  the crossover. The issue's own hedge (schedule known ahead → plan bucket combos)
  is the mitigation; measure it.

## Risks / caveats

- **A4 (carry across segments) is the main unknown.** If the online-softmax
  (max,sum,acc) carry cannot be threaded between #612 bucket calls without a
  recompile — or if resuming a smaller bucket from a larger bucket's partial state
  needs an in-graph reshape — the clean "reuse #612's lattice" story breaks.
  Confirm first (knowledgebase MCP was unavailable here; read #612's kernel + a
  generated-code check).
- **Segment-switch overhead** is the primary perf risk: each segment is a separate
  kernel call, ×per-layer ×per-step. If that dominates the FLOPs saved at realistic
  batch sizes / skews, pad-to-bucket (#612 as-is) wins. #612 already shows
  bucketization can waste ≤50% FLOPs, so the addressable win is real — but the
  driver overhead must stay well under it. Measure the crossover.
- **Per-layer host cost** (the #612 reviewer worry): the `.clone()/.expand()`
  index/mask materialization must be precomputed once per step in `build()`, not
  rebuilt per segment per layer. #468 adds segments, multiplying this cost if done
  naively — keep it layer-invariant.
- **Ordering vs vLLM scheduler**: the sort permutation must round-trip exactly
  (unsort before writing `output`) and must not disturb slot_mapping / KV scatter
  (keyed by the scheduler's request order). Keep the sort internal to the driver.
- **Interaction with plan-10 head tiling**: head tiling is prefill-only (gated
  `padded_query_len >= 1024`), so it does not fire on decode; disjoint features.
- **Scope**: decode-only, num_seqs >= min bucket, no ALiBi/soft-cap/sliding-window
  — same envelope as #612; everything else keeps #612's per-seq fallback. Mixed
  prefill+decode out of scope.

## File pointers (re-confirm against #612's landed code)

- Kernel + driver: `spyre_inference/v1/attention/backends/spyre_attn.py`
  (#612's bucketed decode fast path → add the segment driver;
  `SpyreAttentionMetadataBuilder.build` → segment schedule;
  `SpyreAttentionMetadata` → new segment fields). #612 introduces
  `NUM_SEQS_BUCKETS` / `NUM_BLOCKS_BUCKETS` and the `(num_seqs,num_blocks)` bucket
  lookup the segments reuse.
- Alignment constants: `KV_LENGTH_ALIGNMENT=256`, `QUERY_CHUNK_SIZE=32`.
- Tuner/bench: `examples/tune_attn_tiling.py` (multi-seq `--batch-size` from
  plan-10; add a decode context-skew mode). Device time via AIUPTI profiler
  (`LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:...`; plan-10 rebuild note).
- Prior art: `.plans/10-qpk-lq-tiling.md` (head tiling, tuner, AIUPTI rebuild),
  `06-kv-page-residency.md`, `08-tiling-crossover-long-prefill.md`.
- Dependencies: **PR #612** (bucketed decode fast path — the concrete base),
  issues #403 / #401.
