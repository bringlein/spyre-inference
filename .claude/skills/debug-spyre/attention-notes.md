# Attention-specific debugging notes

Companion to `SKILL.md`. Read this when the failing surface is the Spyre attention backend specifically. The generic workflow (cluster → HTML log → hypothesis queue → cluster validation) in `SKILL.md` still applies; this file just adds attention-flavored content to the hypothesis queue and pipeline-bisection steps.

## Files that matter for attention

- `spyre_inference/v1/attention/backends/spyre_attn.py` — the backend, `SpyreAttentionImpl`, `SpyreAttentionMetadataBuilder`. The module docstring enumerates every torch-spyre limitation the backend routes around.
- `tests/attention/test_spyre_attn.py` — builds real metadata via `SpyreAttentionMetadataBuilder`, calls `SpyreAttentionImpl.forward` on a Spyre device, compares against a CPU reference (`ref_attn`). Tolerances are loose (`atol=0.3, rtol=5.0` for prefill; `atol=0.2, rtol=0.2` for decode) — if mismatch ratios are near 100 % with differences > 1.0, suspect a **structural** bug, not fp16 noise.

## Attention-specific limitations

In addition to the generic table in `SKILL.md`:

| Limitation | Workaround |
|---|---|
| KV alignment to bucketed length | `SpyreAttnBucketer` kv buckets: powers of two from `block_size` to `max_model_len`, consumed as a padded block count, so the same compiled kernel is reused as KV grows |
| Query length must be bucketed | `SpyreAttnBucketer` query buckets: `[1] + multiples of min(512, max_num_batched_tokens)`; queries are padded up to a bucket |
| `head_size` must be a multiple of 64 | `SpyreAttentionBackend.supports_head_size` enforces this (128-byte stick / 2 bytes for fp16) |
| MHA head config | Untested — the attention tests only exercise GQA (`_run_spyre_attn_test` hardcodes 32 q / 8 kv heads, head_size 128). MHA has not been verified. |
| `num_seqs=1` only | The forward path assumes a single sequence at a time |

**Verified working (2026-07-17, gemma-3-1b bring-up):** MQA (`num_kv_heads=1`) and
`head_size=256` both compile and match the CPU reference on Spyre — prefill *and* decode, including
with sliding-window attention. This contradicts an earlier note that "MHA/MQA fail to compile";
that claim was stale. Confirmed by temporarily pointing `_run_spyre_attn_test` at gemma-3-1b's
config (4 q / 1 kv heads, head_size 256) and running `test_spyre_attn_core` +
`test_spyre_attn_sliding_window` on `device_spyre` — all passed. These configs are still not part of
the committed parametrizations, so add coverage rather than assuming it exists.

## Useful test selectors (orientation only — not for `-k`)

These appear in current parametrize IDs. They're for reading collected node IDs; `-k` will not parse the parens/equals/commas.

- `decode(q=1,kv=256)` / `decode(q=1,kv=1024)`
- `prefill(q=32,kv=256)` / `prefill(q=64,kv=512)` / `prefill(q=100,kv=512)`
- `GQA` (the committed tests hardcode GQA 32/8; there are no MHA/MQA parametrizations)
- `head_size(128)` / `head_size(256)`
- `num_blocks(2048)` / `num_blocks(32768)`

## Reading LX residency: `lx_pinning: ... → lx` is a DEFAULT, not a verdict

When a hypothesis is "this buffer spills to HBM", do not trust the planner's debug log. With
`SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` (note: `TORCH_LOGS='+spyre.inductor...'`
crashes torch — it rejects the non-module name) the allocator prints one `lx_pinning:` line per
op, and `→ lx` looks like "resident". It is not:

- `_log_lx_pinning` (`torch_spyre/_inductor/scratchpad/allocator.py:1430`) prints
  `reasons.get(op.name, "lx")` and only runs at DEBUG (`:1433`). Anything missing from the
  dict is labelled `lx`.
- `CoOptimizingAllocator._get_spill_reasons` (`:2588`) returns `solver.spill_reasons`
  verbatim; the base class (`:477`) also synthesizes a `no room on scratchpad` entry for
  every buffer with `address is None`. This is **not** a coverage loss for `cpsat` or
  `simulated_annealing`: both write `_SOLVER_CHOSE_SPILL` for every spilled buffer
  (`sa_cooptimizer.py:837`, `ilp_solver_ortools.py:1292`), i.e. the same bucket under a
  different string.
- It **is** a coverage loss under `ExhaustiveSearchSolver` (`ALLOW_EXHAUSTIVE_SEARCH=1`, or
  `cpsat` without ortools — `_make_cpsat_solver` silently degrades to greedy, `:3748`): it
  copies a placement-only solver's `spill_reasons` (`exhaustive_search.py:220`), and
  `record_exclusions` (`plan_solver.py:903`) records only declared/capacity exclusions, so
  buffers that merely did not fit show up as `lx`.
- After a `SolveError`, `scratchpad_planning` falls back to a base
  `ScratchpadAllocator(GreedyLayoutSolver)` (`:3926`), so the `no room on scratchpad`
  strings come back.
- Independently of spill reasons, `_log_lx_pinning` walks `graph.operations` while `reasons`
  is keyed by solver-buffer name, and `_build_bound_buffers` skips buffers with no lifetime
  (`:982`). Those ops are reported `lx` despite never having been offered to the solver.

So `→ lx` is only trustworthy on the native co-opt solvers, and even there only for ops that
reached the solver. Read residency from the allocation instead — `LifetimeBoundBuffer.address`,
where `None` means HBM:

```python
from torch_spyre._inductor.scratchpad import allocator as alloc

orig = alloc.ScratchpadAllocator._push_allocation

def push(self, graph, buffers, accepted):
    for b in buffers:
        where = "LX " if b.address is not None else "HBM"
        print(f"{where} {b.name:50} addr={b.address} {b.size / 1024:.1f} KiB")
    return orig(self, graph, buffers, accepted)

alloc.ScratchpadAllocator._push_allocation = push
```

Sizes printed there are **logical**; LX is per-core private memory, so divide by the core
count before comparing against capacity. The printed address deltas between two same-sized
buffers confirm the per-core figure. Take the capacity itself from `_lx_planning_size()`
(`torch_spyre/_inductor/scratchpad/allocator.py`) rather than from the physical LX size or
the tracker constant: the frontend reserves `1 - DXP_LX_FRAC_AVAIL` of the tracker capacity
for the backend, so the solver's limit is materially smaller than either.

This mattered concretely: it is what made an experiment aimed at giving the batched decode
kernel LX-resident page gathers look like it had found the shipped kernel *not* resident, when
both paths were resident all along, and the whole premise was void.

## Attention-specific hypotheses to add to the queue

When the failing surface is attention, in addition to the generic starter set, also consider:

- **Device KV cache not seeded from the passed-in `kv_cache`** (historical context lost). `SpyreAttentionImpl._init_device_cache` creates a fresh **zeroed** on-device cache; it does NOT seed from the `kv_cache` argument. The test fixture writes historical KV into the CPU `kv_cache`. For `decode(q=1, kv=N)` cases, the only KV values the device ever sees are the new token scattered in — historical tokens live on CPU and are never transferred. This is the canonical structural-bug hypothesis to check first when decode tests fail with near-100% mismatch.
- **Attention mask off-by-one / wrong padded shape / wrong causal boundary.** `_build_attention_mask` pads to `padded_query_len` and `aligned_max_seq_len`. Off-by-one in causal/padding masking produces near-100 % mismatches that look like "random output." Print the mask for a small case and verify masked positions are `-65504` (fp16 -inf proxy), unmasked are `0`, and the causal boundary matches `context_lens + q_pos`.
- **Scatter selectors wrong** (row/col one-hot mismatched with `slot_mapping`). A diff > 0 at the scatter step means the one-hot selectors are wrong.
- **Gather selector picks physical blocks in wrong order** → compact KV order mismatch with mask. `_gather_from_device_cache` does `bmm(sel_mask, cache)` then reshapes to `[num_kv_heads, aligned_max_seq_len, head_size]`. If the physical block order in `sel_mask` is wrong, the gathered tokens are in the wrong order and the mask no longer matches — silent disaster.
- **A shape that escapes the `SpyreAttnBucketer` kv/query buckets** dispatches to an untested kernel path (correctness, not just recompilation cost).

## Pipeline bisection for `forward()`

`SpyreAttentionImpl.forward` has five clean stages: scatter → gather → reshape → attention → slice. Rerun each stage on CPU against a reference and compare:

```python
# Rough scaffold
impl._init_device_cache(num_blocks, block_size)
impl._scatter_to_device_cache(...); k_dev = impl._k_cache_dev
compact_k, compact_v = impl._gather_from_device_cache(...)
# Pull to CPU and diff vs expected layout
```

A diff > 0 at the scatter step means the one-hot selectors are wrong. A diff only after attention means the `_attn_4d` path (matmul / softmax / mask add) is the issue.

## Per-row magnitude dump for broadcast/dispatch bugs

When you see huge values (near fp16 max, `~60000+`) in the attention output, the bug is usually in a specific broadcast or dispatch axis of a Spyre kernel rather than the math. Right after `self._attn_4d(q_spyre, k, v, self.scale, mask)` in `_compute_attention`, print:

```python
osp = output_spyre.to("cpu")
per_row_max = osp[:, :, 0, :].reshape(-1, osp.shape[-1]).abs().max(dim=-1).values
# Row layout: [kv_head, q_per_kv] in row-major order (0→NUM_KV_HEADS*QPK-1)
for i, m in enumerate(per_row_max.tolist()):
    kv_h, qpk = divmod(i, output_spyre.shape[1])
    mark = "  <<< OVERFLOW" if m > 100 else ""
    print(f"  kv_h={kv_h} qpk={qpk}  max|out|={m:.4f}{mark}")
```

Look for **periodic patterns**: if every Nth row is huge and the rest are sane (e.g. every odd `q_per_kv`), that's near-certain evidence of a broadcast bug along that axis in either the matmul or softmax kernel. A workaround is usually `.expand(-1, N, -1, -1).contiguous()` on the singleton dim of `k`/`v`/`mask` before the call — diagnostic (if expanding makes the overflow go away, the broadcast path is broken) even when it isn't the final fix.

## Attention-specific gotcha: `_maybe_compile` does wrap `_attn_4d` under pytest

The platform docs/comments suggest "`torch.compile` is globally off in this repo," which is only true when `cfg.mode == CompilationMode.NONE` (i.e. `0`). In the pytest `default_vllm_config` fixture, `CompilationConfig(custom_ops=["all"])` leaves `cfg.mode = None` (Python `None`, not the `NONE=0` enum). `_maybe_compile`'s first `if` therefore falls through and `cfg.backend == "inductor"` (default) fails the second `if` too, so `_attn_4d` ends up wrapped with `torch.compile(..., dynamic=False)`. If you're debugging a kernel-looking bug and trying to reproduce it in a plain script with no `torch.compile`, you may be comparing apples to oranges.

## CPU-vs-Spyre standalone repro (attention-flavored)

For attention specifically, a useful standalone-repro recipe under `logs/<slug>/repro_cpu.py`:

```python
# Pseudocode — instantiate SpyreAttentionImpl, then:
impl._target_device = torch.device("cpu")
# also do the same in SpyreAttentionMetadataBuilder
# then run forward() on the same inputs the failing parametrization uses
```

The pytest fixture `requires_spyre` will still skip the real test on CPU-only hosts, so don't rely on editing `_target_device` and rerunning pytest — do it in the standalone script instead. If the CPU variant passes, the bug is in torch-spyre's realization of the operation; if it still fails on CPU, the bug is in our own logic.
