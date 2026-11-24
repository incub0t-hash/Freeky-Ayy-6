# Compute shader writes through a spilled root UAV

The BVH builder report counts 35 compute programs used by the title. All
35 failed pipeline creation in the reported run. After the root-argument
spill-table patch, 10 of the 35 still encountered a root UAV or typed root
view refusal. The report counted about 690 drops per frame across four
builder programs in that run. [BVH builder report (#614)](https://github.com/Marvisak/VIctory/issues/614)

The integration uses the root-UAV path from #1077 and its spill-table
prerequisites. Its author reported that one identified world compute
program disappeared from the drop summary on a larger integration
branch. The public results leave the program's exact BVH role open and
do not isolate the effect of the root-UAV patch. [Root-UAV integration (#1077)](https://github.com/Marvisak/VIctory/pull/1077)

The regression builds one synthetic compute program. It loads an
address from the root-argument spill table, constructs a raw UAV
descriptor, and stores a dword at byte offset 4. The D3D12 validation
host binds the UAV through the same kind of single-view descriptor
table as the runtime. Readback checks the exact written dword and
checks that a separate heap UAV remains untouched.

Run the focused test on Windows with a D3D12 Shader Model 6.6 adapter.

```powershell
cargo test -p victory-shader --locked compute_writes_through_spilled_root_uav -- --nocapture
```

The dispatch and readback passed on the local hardware adapter and on
WARP. The test covers shader lowering, root signature creation, pipeline
creation, dispatch and readback. The title's command-list binding,
shadow-buffer write-back, frame drop count and visual output still
need an in-game comparison.
