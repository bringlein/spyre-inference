# Spyre attention micro-benchmark

Measures the Spyre paged-attention kernel with the torch profiler as the
instrument. Spyre has no CUDA-graph equivalent for excluding host overhead, so
device time attributed to a `record_function` span is the noise-free signal.

> **Notation:** `bs=64` / `bs=128` in this file mean **`block_size`**, not batch size.
> Batch size (`num_reqs`) is 1 in every shape here. The only place `bs` means batch size
> is the run-directory name `granite33_8b_bs1`.

## Run

```bash
SPYRE_ATTN_PROFILING=1 .venv/bin/python3 examples/microbench/spyre_attn_microbench.py \
    --config examples/microbench/configs/granite33_8b_bs1.json
```

`SPYRE_ATTN_PROFILING=1` is **required**: `_record_function` is a no-op decorator
without it (`spyre_inference/v1/attention/backends/spyre_attn.py:47`), so the
spans would not exist and there would be nothing to attribute device time to.
The runner sets it in `os.environ` before importing `spyre_attn` — the module
reads `_ATTN_PROFILING` at import time — and aborts if it did not take effect.

Never run two Spyre commands at once; the device is single-tenant (CLAUDE.md).
A concurrent run fails with `RAS::VFIO::DeviceOpenFail`.

### Which scope to measure

```bash
--span online_softmax   # _online_softmax_attention (default)
--span forward          # the whole SpyreAttentionImpl.forward path
--span reshape_and_cache
```

`forward` encloses the other two, so it is attributed differently: every kernel
starting inside the `forward` window is summed, rather than the innermost-span
rule used for leaf spans.

Use `--span forward` for the *magnitude* of total attention cost; do not read the
`forward - online_softmax` delta itself as a `reshape_and_cache` measurement at
long shapes — profile `--span reshape_and_cache` directly instead.

### Prerequisite: AIUPTI

Device events only appear if torch-spyre was built with `USE_SPYRE_PROFILER=1`
(`pyproject.toml:97`). Verify:

```bash
ldd .venv/lib/python3.12/site-packages/torch_spyre/_C.so | grep libaiupti
```

The runner's startup guard aborts when the probe profile has no device events,
rather than reporting plausible-looking zeros.

## Input modes

Both lower onto the same `(query_lens, seq_lens)` measurement path.

**Request list** — explicit per-request lengths, for reproducing exact shape
buckets:

```json
"capture_batches": [
  {"name": "prefill_512", "query_lens": [512], "seq_lens": [512]},
  {"name": "decode_ctx512", "query_lens": [1], "seq_lens": [512]}
]
```

`query_lens_rle` / `seq_lens_rle` accept `[[value, count], ...]` for wide batches.

**Cartesian grid** — `batch_size × sequence_length × decode_share ×
prompt_pattern`, ported from the GPU framework:

```json
"grid": {
  "batch_sizes": [1, 4],
  "sequence_lengths": [512, 2048],
  "decode_shares": [0.0, 0.5, 1.0],
  "partial_prefill_share": 0.0,
  "prompt_patterns": [[1.0], [1.0, 0.6, 0.3]]
}
```

`block_sizes: [64, 128]` sweeps the block size as an extra axis.

## Output

Tab-separated, written incrementally after every measurement (so a crash keeps
what completed) plus a `_final.csv`. Column names match the GPU CSVs so the
plotting notebook works unchanged.

`ms`/`min_ms`/`max_ms` are **device time, not wall clock** — median, min and max
over `--iterations` separate profiled windows. `benchmark_mode` is
`BenchmarkMode.SPYRE_PROFILER` to mark this; the numbers are not comparable to
GPU wall-clock rows despite the shared column names.

Spyre-specific columns: `device_time_memory_us` and `memory_share_pct`
(memcpy/memset/restickify share), `cpu_time_ms` (the per-sequence Python loop is
real cost), `fallback_clean`, `num_outliers`, `span`, `kv_layout`.

### Why one forward per profile window

The AIUPTI backend has a hardcoded pool of 5 trace buffers and stops capturing
when it fills (`docs/user_guide/kineto_profiling.md` §4.5). A single window
around 10 iterations truncates the device timeline. Separate windows also give a
real per-iteration distribution instead of only a mean.

## Reading the output

Normalize by `num_kv_blocks_iterated` before concluding anything about scaling.
Raw µs can suggest a knee that vanishes once you divide by pages iterated.

Where a row's leaf span has high spread, prefer `min_ms` — attribution leakage
only ever adds time.

## Device memory

The KV cache is `num_blocks * block_size * num_kv_heads * head_size * 2 B` per
tensor, so holding `num_blocks` fixed while doubling `block_size` doubles the
footprint, and allocations can fail:

```
RAS::FLEXALLOCATOR::OutOfMemory  FreeSpaceBytes=28275200 RequestedBytes=37000832
```

Device memory is not fully returned between configs
(`docs/user_guide/kineto_profiling.md` §4.3) even though the runner does
`del inputs; gc.collect()`, so it accumulates across a sweep. Affected rows are
recorded with `error` set and empty `ms`, so they are excluded from plots.

Guidance:

- Pin `num_blocks` constant across any runs you intend to compare — the notebook's
  `check_num_blocks()` errors if one `EXPERIMENTS` entry mixes values.
- Order decode captures before prefill (or run them in a separate process) to
  avoid the fragmentation cascade at its cause.
- A failed allocation strands memory for the rest of the process and degrades the
  *timings* of subsequent rows, so re-run affected shapes in a fresh process. The
  notebook's `flag_post_error()` marks every row following an errored row.

## Attribution

Two Kineto limitations shape this, both confirmed on hardware:

1. Device time is **not** propagated to `record_function` parents — every
   `spyre_attn::*` span reports `self_device_time_total == 0.0`, so
   `key_averages()` cannot be used to scope a span.
2. AIUPTI populates no correlation ids (`corr=None`, `linked=0`), so there is no
   CPU↔device linkage to follow.

Interval overlap is therefore the only mechanism. Each device kernel is
attributed to the innermost span containing the kernel's **start** timestamp.
Start-based rather than containment-based because dispatch is async: a kernel can
start inside a span and end well after the span closes; strict containment would
credit its time to nothing.

This scoping matters: `reshape_and_cache` dwarfs `online_softmax` at some shapes,
so summing the whole trace — as `scripts/tune_attn_tiling.py` does — would be
dominated by the KV write.

## KV page layout

`--kv-layout` selects how the KV pages reach the device. This matters more than it
looks: `_reshape_and_cache` views the pages as `[-1, num_kv_heads, head_size]` and
relies on the slot-outermost device layout to make that view free
(`spyre_inference/v1/attention/backends/spyre_attn.py:967`).

**`plain`** (default) — plain `.to(device)` on a host-populated cache. Matches
`tests/test_spyre_attn.py`, which passes on hardware at these shapes. Correct,
and what the reported numbers use.

**`slot_major_devfill`** — an *attempt* at production fidelity that **does not
work**. Allocates a zeroed slot-major cache on device as the worker does
(`spyre_inference/v1/worker/spyre_model_runner.py:742`), then scatters KV history
into it on device via the kernel's own `index_copy_` (`spyre_attn.py:130`).
`convert()` does not produce the slot-major layout the destination view expects,
so the mismatch simply moves from the cache to the source tensor. **Unresolved** —
do not use for reported numbers.

**`slot_major`** — pins the worker's layout on an already-populated *host*
tensor. **Numerically wrong**; kept only to reproduce the finding. Because the
host tensor was laid out block-major, the kernel's `[-1, H, D]` view reinterprets
bytes that were never written in slot order. The worker gets away with the same
layout only because its host tensor is *zeroed* before transfer — zeros are
identical under any permutation — and the real contents arrive later via
`_reshape_and_cache` on device. A layout mismatch here shows up as a *win*, which
is why the correctness gate runs before every timing.

## Correctness gate

Every configuration is checked against a CPU reference before timing, with the
same semantics as `tests/test_spyre_attn.py`: relative tolerance
`atol + rtol*|expected|` (0.3/0.2) and up to `max_outliers` (default 5) fp16
stragglers. `allclose_pass` and `max_abs_diff` are CSV columns, so a
fast-but-wrong configuration is visible in the plots rather than silently
plotted as a win. Failures do not stop the sweep unless `--stop-on-failure`.

`fallback_clean` records whether a `FallbackWarning` fired — torch-spyre
silently routes unsupported ops to CPU, which would otherwise be plotted as a
Spyre result.

## Notes

- dtype is float16 only; `platform.py` raises otherwise, regardless of what the
  model config declares (granite-3.3-8b declares bfloat16).
- `block_size` 64 is what you get in practice: vLLM's default is 16, and
  `platform.py:352` rounds it up to the next multiple of 64.
- Compiled and eager variants need separate runs — the compilation mode is fixed
  for the process.
- The kernel specializes per `(num_blocks, aligned_max_query_len)`, so a sweep
  legitimately triggers many recompiles; the dynamo recompile limit is raised to
  4096. Warmup runs before the profiled windows so no compile lands inside a
  measured window.
- **Compile cost dominates sweep wall-clock at long shapes.** Dynamo unrolls the
  page loop, so the graph grows with `num_blocks`, and the Spyre backend compiler
  (`dxp_standalone`, forked per bucket) scales accordingly: a few seconds at
  `seqlen=32`, but ~4+ minutes of multi-threaded compile for the `prefill_3000`
  bucket alone. The parent blocks in `do_wait` while that runs — a sweep that
  looks hung at a long shape is usually compiling. Check with
  `ps --ppid <pid> -o time,args` and look for `dxp_standalone` accumulating CPU.
  Compiled artifacts are cached under `/tmp/torchinductor_ngl/inductor-spyre/`,
  so a re-run of the same shapes is far faster.
- The process ends with `os._exit(0)`: `TimestampCalibrator` aborts in its
  destructor at teardown (`kineto_profiling.md` §4.4). stdout is flushed first.
