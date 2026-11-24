//! Small, opt-in samples of actual shader invocations for a named world pass.
use super::*;
use crate::{host::HostList, pipeline::host::HostPipeline};
use std::collections::HashMap;
use std::sync::{Mutex, atomic::AtomicBool};

pub(crate) struct Query {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ty: D3D12_QUERY_TYPE,
    hash: String,
    command: &'static str,
    reported: AtomicBool,
}

pub(crate) fn watched() -> &'static [String] {
    static WATCH: OnceLock<Vec<String>> = OnceLock::new();
    WATCH.get_or_init(|| {
        std::env::var("VICTORY_CAPTURE_STAGES")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    })
}

impl Query {
    pub(crate) fn begin(
        list: &HostList,
        pipeline: &HostPipeline,
        command: &'static str,
    ) -> Option<Self> {
        let watch = watched();
        if watch.is_empty() {
            return None;
        }
        let hash = &pipeline
            .stages
            .iter()
            .find(|s| watch.contains(&s.shader.hash))?
            .shader
            .hash;
        // Sample the first recording, then progressively later commands. Never retain
        // one readback per dispatch in a game that dispatches thousands per frame.
        static COUNTS: OnceLock<Mutex<HashMap<(String, &'static str), u32>>> = OnceLock::new();
        let mut counts = COUNTS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let count = counts.entry((hash.clone(), command)).or_default();
        *count = count.saturating_add(1);
        if *count > 4096 || !count.is_power_of_two() {
            return None;
        }
        drop(counts);

        let device = &crate::host::host()?.device;
        static EXTENDED: OnceLock<bool> = OnceLock::new();
        let extended = *EXTENDED.get_or_init(|| {
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            // SAFETY: writable OPTIONS9 output with its documented size.
            let supported = unsafe {
                device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut opts).cast(),
                        size_of_val(&opts) as u32,
                    )
                    .is_ok()
            };
            supported && opts.MeshShaderPipelineStatsSupported.as_bool()
        });
        let (heap_type, ty) = if extended {
            (
                D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1,
                D3D12_QUERY_TYPE_PIPELINE_STATISTICS1,
            )
        } else {
            (
                D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
                D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            )
        };
        let mut heap: Option<ID3D12QueryHeap> = None;
        let mut buffer: Option<ID3D12Resource> = None;
        let props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>() as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: the query belongs to this recording. All objects survive list execution.
        unsafe {
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: heap_type,
                        Count: 1,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .ok()?;
            device
                .CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .ok()?;
            let heap = heap?;
            let readback = buffer?;
            list.list.BeginQuery(&heap, ty, 0);
            Some(Self {
                heap,
                readback,
                ty,
                hash: hash.clone(),
                command,
                reported: AtomicBool::new(false),
            })
        }
    }

    pub(crate) fn end(self, list: &HostList) {
        // SAFETY: the matching begin preceded exactly one command on this list.
        unsafe {
            list.list.EndQuery(&self.heap, self.ty, 0);
            list.list
                .ResolveQueryData(&self.heap, self.ty, 0, 1, &self.readback, 0);
        }
        list.stage_queries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(self);
    }

    pub(crate) fn report(&self) {
        if self.reported.swap(true, Ordering::Relaxed) {
            return;
        }
        let extended = self.ty == D3D12_QUERY_TYPE_PIPELINE_STATISTICS1;
        let size = if extended {
            size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>()
        } else {
            size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>()
        };
        let mut data = std::ptr::null_mut();
        let mut stats = D3D12_QUERY_DATA_PIPELINE_STATISTICS1::default();
        // SAFETY: Host::execute verified the submission fence. The base structure is
        // the documented prefix of STATISTICS1. Copy only bytes resolved by the GPU.
        unsafe {
            if self
                .readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: size,
                    }),
                    Some(&mut data),
                )
                .is_err()
            {
                return;
            }
            std::ptr::copy_nonoverlapping(data.cast::<u8>(), (&raw mut stats).cast::<u8>(), size);
            self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        tracing::info!(hash = %self.hash, command = self.command, cs = stats.CSInvocations,
            ms = ?extended.then_some(stats.MSInvocations), primitives = ?extended.then_some(stats.MSPrimitives),
            ps = stats.PSInvocations, "GPU shader invocation readback");
    }
}
