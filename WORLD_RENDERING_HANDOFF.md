# VIctory rendering development

## Objective

Correct and stable Direct3D 12 rendering of GPU-driven
3D scenes using authorised and synthetic test workloads.

## Hardware

Intel i9-14900KF
NVIDIA RTX 3080 Ti 12 GB
32 GB DDR5

## Development baseline

Original repository: D:\GTA 6\VIctory-Hybrid-20261124-142430
Baseline commit: d67a2328cd98755885a2eab78f6acc1655a71121
Development worktree: D:\GTA 6\VIctory-Rendering-20261124-171243
Branch: render-parity-20261124-171243
Reports: D:\GTA 6\VIctory-Render-Reports\20261124-171243

Preserve existing fixes for:

- Wave64 shader translation and numerical validation.
- Native mesh and pixel raster tests.
- Overlapping shadow allocation lookup.
- Dynamic descriptor resource retirement.
- Bindless resource rebinding after allocation changes.
- Sustained bindless streaming tests.

## Patch candidates

Issue #747: adaptive descriptor capacity.
Layout commit: 3e3b8c5841213fbd316a0875c06c5f85cb5200c2
Guard commit: cc449ca941abea558dc56271c10f7ce0459ab66a

These must be integrated together.

## Remaining objectives

1. Correct descriptor allocation and bounds.
2. Validate structured-buffer stride handling.
3. Validate indirect draw and mesh execution.
4. Verify resource state transitions and GPU lifetime.
5. Verify geometry, depth and G-buffer outputs.
6. Verify lighting and final composition.
7. Optimise CPU/GPU transfers after correctness.

Do not skip unsupported operations and describe that
as a rendering fix.

Do not reclaim descriptors while the GPU can read them.
Do not silently discard resubmitted command lists.

Keep regression tests, commits and actual test logs.

## Status

Integration started.
No new rendering improvement verified yet.

## Integration blocked

Patch exit code: 1

Conflicting files:
crates/victory-gpu/src/descriptor/host.rs
crates/victory-shader/src/emit/lower.rs

Resolve these conflicts while preserving both the
existing memory fixes and the new descriptor guards.

Do not build a partially integrated patch.
The original development worktree remains unchanged.

## Integration results

GPU test exit: 101
Shader test exit: 101
Bindless stress exit: -1
Diagnostic build exit: -1

An exit code of -1 means the stage was not run.

Review test logs for internally skipped tests and
confirm physical NVIDIA execution.

A successful build does not establish visual parity.
Actual rendering output must be independently verified.

## Shader integration checkpoint

Resolved lower.rs conflict by combining existing numeric-select
fallback with adaptive heap-bound placement.

Both lowering passes receive numeric_selects and bounds_at.

Compilation and GPU execution remain unverified.
Outstanding conflict: descriptor/host.rs.

## Descriptor conflict resolution

All six descriptor/host.rs conflict regions resolved.

Preserved:
- Bindless tracking and revision counters.
- Resource lifetime and allocation replacement tests.
- Existing streaming stress test.

Integrated:
- Guard-slot exclusion from dynamic allocation.
- Typed null guard initialisation.
- Descriptor exhaustion counters.
- Adaptive heap layout regression tests.

No unresolved Git index entries.
Compilation validation follows.

### merged-compile-check

Exit code: 101
Log: D:\GTA 6\VIctory-Render-Reports\20261124-171243\merged-compile-check.log

fixed-compile-check exit=0; log=D:\GTA 6\VIctory-Render-Reports\20261124-171243\fixed-compile-check.log

fixed-shader-tests exit=0; log=D:\GTA 6\VIctory-Render-Reports\20261124-171243\fixed-shader-tests.log

fixed-gpu-tests exit=0; log=D:\GTA 6\VIctory-Render-Reports\20261124-171243\fixed-gpu-tests.log

fixed-streaming-stress exit=0; log=D:\GTA 6\VIctory-Render-Reports\20261124-171243\fixed-streaming-stress.log

fixed-diagnostic-build exit=0; log=D:\GTA 6\VIctory-Render-Reports\20261124-171243\fixed-diagnostic-build.log

## Integration checkpoint

Migrated three older Dispatch initializers to the new
descriptor layout and populated heap-bound constants.

All validation commands passed.
Actual visual parity remains unverified.

Next: validate GPU-driven geometry, indirect draws,
depth, materials and final composition.
