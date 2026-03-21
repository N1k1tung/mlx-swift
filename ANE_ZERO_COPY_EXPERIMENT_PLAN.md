# ANE Zero-Copy Experiment Plan (No Input Memcpy from Array Memory)

Goal: reduce/remove host-side memcpy from `ensure_input_surface_attachment` and
`wrap_array_to_surface`-style paths in ANE private runtime for array-backed inputs.

This plan is scoped to **private runtime dispatch** and is compatible with the current
implementation in `ANE_MEMORY_LAYER.md` and `private_runtime.mm`.

## 1) Baseline: what must be true today

Before any changes:

- `Runtime::dispatch` goes through `private_runtime::dispatch`.
- Inputs without a usable attachment are converted via `ensure_input_surface_attachment`:
  - create IOSurface using array memory
  - install attachment on `in`
- Outputs are bound via `bind_output_surface_to_array`, already avoiding a separate
  output host copy in dispatch path.

Success hypothesis:

- Inputs that are already attached can be reused immediately (current fast path already does this).
- New zero-copy work should target the **first-time attach path** only.

## 2) Experiment matrix

Each experiment must be run in this order:

1. Measure current behavior precisely.
2. Introduce one experiment.
3. Measure and compare with baseline.
4. Decide keep / revert / iterate.

### 2.1 Current-state measurements

- Baseline command run set:
  - dispatch throughput test for small/medium/large tensors.
  - repeated dispatch reuse test with the same input tensors.
  - mixed-contiguity test (contiguous vs non-contiguous/offset/sliced views).
- Collect:
  - `input_copy_bytes`, `input_copy_ns`, `dispatch_ns`.
  - end-to-end latency and output correctness (hash/ulp checks).
  - per-primitive coverage matrix (elementwise, matmul-like, softmax-like, large tensor).

### 2.2 Experiment A: input attachment precondition widening (safety only)

Purpose: prove current attachment restrictions are the gate, not just memcpy itself.

- Verify current guard set in `ensure_input_surface_attachment`:
  - `row_contiguous` only
  - no explicit `offset` check today
  - `data` pointer non-null
  - attach `nbytes >= required_nbytes`
- Add diagnostic logs for reject reasons:
  - `input-not-row-contiguous`
  - `input-data-null`
  - `input-surface-create-failed`
  - `input-surface-base-null`
  - `input-surface-attach-failed`

Expected:
- We may discover many cases blocked before memcpy even starts (important for next experiments).

### 2.3 Experiment B: expose surface-alias candidates from array buffers

Purpose: attempt a path where input memory is already suitable for ANE without copy.

For each `array::Data` buffer type:

- CPU-backed `CommonAllocator` buffer
- Metal `MTL::Buffer` (`MetalAllocator::make_buffer` compatibility)
- Existing `ANE`-already attached outputs

Attempt strategy:

- Extend attachment install logic to support a **non-copying candidate** branch
  in `ensure_input_surface_attachment`.
- If array memory is already represented as a construct that can be wrapped by ANE
  transport, create/use `SurfaceHandle` without memcpy.
- Fall back to memcpy path on failure.

Implementation checkpoints:

- Keep existing memcpy path unchanged; only add a guarded new branch.
- Tag each successful non-copy attach in debug/profile counters:
  - `input_zero_copy_attach_hits`
  - `input_zero_copy_attach_fail_reasons`

Acceptance for this experiment:

- Any non-copy attach must preserve output correctness and run without surface lock errors.
- No regression in memory safety or object lifetime.

### 2.4 Experiment C: skip pre-copy for already pinned and sized arrays

Purpose: maximize reuse in first-touch style workflows.

- For arrays that are already attached but undersized, test growth strategy:
  - reallocate and rebind attachment
  - or force one-time memcpy + resize.
- Compare whether preserving a larger attached surface across calls beats create-per-call.

Acceptance:

- Reduced `input_copy_bytes` and fewer `create_surface` calls after first use.

### 2.5 Experiment D: output alias verification under no-copy input mode

Purpose: ensure zero-copy input mode does not disturb output reuse semantics.

- Repeat experiments with output counts >1 and different tensor sizes.
- Validate output surfaces remain attached, pooled, and not stale across calls.

Acceptance:

- No increase in output binding failures.
- No output correctness drift vs baseline checksum checks.

## 3) Instrumentation to add for all experiments

Add profiling counters and temporary debug logs in `private_runtime.mm`:

- `zero_copy_attach_attempts`
- `zero_copy_attach_success`
- `zero_copy_attach_fallback_memcpy`
- `zero_copy_attach_fallback_reasons` (string bucket counters if available)
- `attach_lifecycle_releases_with_pool`

Keep `profile_mode` counters human-readable and add CSV-style log lines for:

- input bytes
- memcpy durations
- attachment decision path per input index.

## 4) Correctness safety rails (non-negotiable)

Any candidate non-copy path must pass:

- shape/dtype match checks already enforced by MIL compile path.
- `is_available` / synchronization behavior already handled by dispatch staging.
- explicit offset/layout compatibility:
  - prefer requiring `offset == 0` or strictly verify wrapped range covers `nbytes`.
  - reject slices / views that would alias partial storage unless safe.
- deterministic fallback:
  - if any uncertainty exists, execute existing memcpy branch and continue.

## 5) Experiment pass/fail criteria

Pass if:

- `input_copy_bytes` decreases materially on repeated runs
- no increase in dispatch failures with same correctness suite
- latency improves or stays flat for copy-dominated workloads

Fail/revert conditions:

- increase in dispatch failures (`dispatch_failures`)
- intermittent wrong outputs
- crashes / surface lifetime exceptions in address/surface release paths

## 6) Risk register

- ANE runtime contract may still require IOSurface-backed storage regardless of CPU/GPU memory sharing.
- API used to alias buffers may not be stable (private API risk).
- Unified memory hardware does not imply ANE can read arbitrary raw pointers directly.
- Lifetimes of shared buffer wrapper objects may create dangling handles without strict attachment ownership.

Mitigation:

- keep existing memcpy path as fallback
- add strict fallback by default
- guard rollout with `MLX_ANE_ENABLE_IOSURFACE`-style env switch for staged rollout

## 7) Implementation order recommended

1. Add diagnostic counters and baseline measurements.
2. Implement Experiment A.
3. Implement Experiment B behind opt-in env flag.
4. If safe, expand to automatic mode and retire opt-out.
5. If no stable safe zero-copy, keep fast attachment reuse and document blocker.

## 8) Files likely touched

- `Source/Cmlx/mlx/mlx/backend/ane/private_runtime.mm`
- `Source/Cmlx/mlx/mlx/backend/ane/private_runtime.h`
- `ANE_MEMORY_LAYER.md` (optional refresh after validated changes)

