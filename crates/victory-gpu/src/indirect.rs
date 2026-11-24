//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, bounded multidimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Preserve valid three-dimensional mesh dispatches.
        // Retain the conservative per-dispatch work limit.
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));

        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }

        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    // Saturating atomic reservation. Rejected work must not wrap
    // the counter and accidentally permit more excessive work.
    uint before;
    budget.InterlockedAdd(0, 0, before);

    [allow_uav_condition]
    while (true) {
        if (before >= BUDGET) {
            args.Store(at, 0);
            return;
        }

        uint remaining = BUDGET - before;
        uint next = before + min(cost, remaining);
        uint actual;

        budget.InterlockedCompareExchange(
            0, before, next, actual
        );

        if (actual == before) {
            if (cost > remaining) {
                args.Store(at, 0);
            }
            break;
        }

        before = actual;
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }
}


#[cfg(test)]
mod clamp_readback_tests {
    use super::*;

    // Execute the actual argument-clamping shader on the host GPU.
    // No game executable, assets or excessive draws are needed.
    fn run_clamp(
        kind: u32,
        input: &[[u32; 4]],
    ) -> (Vec<[u32; 4]>, u32) {
        let host = host().expect("Physical GPU required");
        let clamp = clamp().expect("Argument clamp required");

        let (args, cpu) = host
            .shared_buffer((input.len() * 16) as u64)
            .expect("Argument buffer");

        let (budget, budget_cpu) = host
            .shared_buffer(256)
            .expect("Budget buffer");

        let list = host.new_list().unwrap();

        let constants = [
            kind,
            16,
            input.len() as u32,
            0,
        ];

        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                cpu.cast::<[u32; 4]>(),
                input.len(),
            );

            budget_cpu.cast::<u32>().write_volatile(0);

            list.list.ResourceBarrier(&[
                transition(
                    &args,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                transition(
                    &budget,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);

            list.list.SetComputeRootSignature(&clamp.root);
            list.list.SetPipelineState(&clamp.pso);

            list.list.SetComputeRoot32BitConstants(
                0,
                4,
                constants.as_ptr().cast(),
                0,
            );

            list.list.SetComputeRootUnorderedAccessView(
                1,
                args.GetGPUVirtualAddress(),
            );

            list.list.SetComputeRootUnorderedAccessView(
                2,
                budget.GetGPUVirtualAddress(),
            );

            list.list.Dispatch(
                (input.len() as u32).div_ceil(64),
                1,
                1,
            );

            list.list.ResourceBarrier(&[
                uav_barrier(),
                transition(
                    &args,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                transition(
                    &budget,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
            ]);
        }

        list.close();
        host.execute(&[&list]);

        let output = (0..input.len())
            .map(|i| unsafe {
                cpu.cast::<[u32; 4]>()
                    .add(i)
                    .read_volatile()
            })
            .collect();

        let spent = unsafe {
            budget_cpu.cast::<u32>().read_volatile()
        };

        list.reset();

        assert!(
            unsafe { host.device.GetDeviceRemovedReason() }.is_ok()
        );

        (output, spent)
    }

    #[test]
    fn preserves_multidimensional_mesh_dispatches() {
        let input = [
            [4, 3, 2, 0x12345678],
            [7, 1, 1, 0x87654321],
            [0, 8, 9, 0xABCDEF01],
            [65535, 65535, 2, 0x12344321],
        ];

        let (output, spent) = run_clamp(3, &input);

        assert_eq!(output[0], input[0]);
        assert_eq!(output[1], input[1]);
        assert_eq!(output[2], input[2]);

        assert_eq!(
            output[3],
            [0, 0, 0, input[3][3]]
        );

        assert_eq!(spent, (24 + 7) * 128);

        println!(
            "VERIFIED: multidimensional mesh arguments preserved"
        );
    }

    #[test]
    fn indirect_work_budget_never_wraps() {
        // Each record costs 2^26 units. The budget permits two.
        // There are enough rejected records to overflow the old
        // 32-bit accumulating counter.
        let input: Vec<[u32; 4]> = (0..96)
            .map(|i| [1 << 24, 4, i, 0x12345678])
            .collect();

        let (output, spent) = run_clamp(0, &input);

        assert_eq!(spent, 1 << 27);

        let accepted = output.iter()
            .filter(|r| r[0] != 0)
            .count();

        assert_eq!(accepted, 2);

        for (before, after) in input.iter().zip(&output) {
            assert_eq!(&before[1..], &after[1..]);
            assert!(after[0] == 0 || after[0] == before[0]);
        }

        println!(
            "VERIFIED: 2 accepted, 94 rejected, budget saturated"
        );
    }
}
