//! Four data-dependent geometry heap indices of a sampled native batch culler.

use super::*;
use crate::host::{HostList, host, transition};

pub(crate) const BYTES: usize = 16;
pub(crate) fn supported(hash: &str) -> bool {
    matches!(hash, "3b471c122be79b3e" | "fc5e66f8f873d3bb")
}

/// Input addresses come from the same retained buffer object. The culler reads
/// T77[0] as its byte offset in the raw scene SRV. Each group uses one 24-byte
/// record, whose low 24 bits select its next descriptor.
#[derive(Clone)]
pub(crate) struct Inputs {
    pub(crate) resource: ID3D12Resource,
    pub(crate) param_offset: u64,
    pub(crate) scene_offset: u64,
    pub(crate) scene_len: u64,
}

const SOURCE: &str = r#"
cbuffer Window : register(b0) { uint argsAt; uint paramAt; uint sceneAt; uint sceneLen; };
RWByteAddressBuffer guest : register(u0);
RWByteAddressBuffer result : register(u1);
[RootSignature("RootConstants(num32BitConstants=4,b0), UAV(u0), UAV(u1)")]
[numthreads(4,1,1)] void main(uint id:SV_GroupIndex) {
    uint3 dims = guest.Load3(argsAt);
    uint value = 0xffffffff;
    if (id < dims.x && dims.y != 0 && dims.z != 0) {
        uint start = guest.Load(paramAt);
        uint delta = id * 24;
        value = 0xfffffffe;
        if ((start & 3) == 0 && start <= sceneLen && sceneLen - start >= delta + 4)
            value = guest.Load(sceneAt + start + delta) & 0x00ffffff;
    }
    result.Store(id*4,value);
}
"#;

/// Called after the indirect buffer has returned to UAV. The helper never
/// writes guest memory. Its private readback contains only the four indices.
pub(crate) fn record(
    list: &HostList,
    inputs: &Inputs,
    argument: (&ID3D12Resource, u64),
    destination: &ID3D12Resource,
    output_offset: u64,
) -> bool {
    let (resource, arg) = argument;
    if resource != &inputs.resource {
        return false;
    }
    // SAFETY: these are retained buffer resources.
    let d = unsafe { resource.GetDesc() };
    let out = unsafe { destination.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        || !arg.is_multiple_of(4)
        || destination == resource
        || out.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        || !output_offset.is_multiple_of(4)
        || output_offset
            .checked_add(BYTES as u64)
            .is_none_or(|end| end > out.Width)
        || !inputs.param_offset.is_multiple_of(4)
        || !inputs.scene_offset.is_multiple_of(4)
        || arg
            .checked_add(12)
            .is_none_or(|v| v > d.Width || v > u64::from(u32::MAX))
        || inputs
            .param_offset
            .checked_add(4)
            .is_none_or(|v| v > d.Width || v > u64::from(u32::MAX))
        || inputs
            .scene_offset
            .checked_add(inputs.scene_len)
            .is_none_or(|v| v > d.Width || v > u64::from(u32::MAX))
    {
        return false;
    }
    static PASS: OnceLock<Option<crate::indirect::ComputePass>> = OnceLock::new();
    let Some(pass) = PASS.get_or_init(|| crate::indirect::compile_pass(SOURCE).ok()) else {
        return false;
    };
    let Some((tmp, _)) = host().and_then(|h| h.shared_buffer(BYTES as u64)) else {
        return false;
    };
    let constants = [
        arg as u32,
        inputs.param_offset as u32,
        inputs.scene_offset as u32,
        inputs.scene_len as u32,
    ];
    // SAFETY: validated input window, private UAV and readback. Pipeline/root
    // state is invalidated by the caller before any further title commands.
    unsafe {
        list.list.SetComputeRootSignature(&pass.root);
        list.list.SetPipelineState(&pass.pso);
        list.list
            .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        list.list
            .SetComputeRootUnorderedAccessView(1, resource.GetGPUVirtualAddress());
        list.list
            .SetComputeRootUnorderedAccessView(2, tmp.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
        list.list.ResourceBarrier(&[transition(
            &tmp,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.list
            .CopyBufferRegion(destination, output_offset, &tmp, 0, BYTES as u64);
    }
    list.keep(&tmp);
    true
}
