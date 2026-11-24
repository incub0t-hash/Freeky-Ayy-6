//! Small, opt-in samples of actual shader invocations for a named world pass.
use super::*;
use crate::{host::HostList, pipeline::host::HostPipeline};
use std::collections::HashMap;
use std::sync::{Mutex, atomic::AtomicBool};

const SAMPLE_RECORDS: u32 = 4;
const SNAPSHOT_WORDS: usize = 3 * SAMPLE_RECORDS as usize + 1;
const STATS_BYTES: usize = size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>();

#[derive(Default)]
struct SampleBudget {
    recordings: u32,
    late_samples: u8,
    last_late_frame: Option<u64>,
}

impl SampleBudget {
    fn next(&mut self, frame: u64, scene_ready: bool) -> Option<u32> {
        self.recordings = self.recordings.saturating_add(1);
        let early = self.recordings <= 4096 && self.recordings.is_power_of_two();
        let late = scene_ready && self.late_samples < 6
            && self.last_late_frame.is_none_or(|last| frame.saturating_sub(last) >= 8);
        if late {
            self.last_late_frame = Some(frame);
            self.late_samples += 1;
        }
        (early || late).then_some(self.recordings)
    }
}

#[derive(Debug)]
struct Indirect {
    kind: u32,
    stride: u32,
    max: u32,
    argument: (u64, u64), // resource GPU base, byte offset
    count: Option<(u64, u64)>,
    shared_resource: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::host;

    #[test]
    fn samples_compiled_scene_even_after_busy_loading_frames() {
        let mut budget = SampleBudget::default();
        assert_eq!((0..5000).filter(|_| budget.next(5, false).is_some()).count(), 13);
        assert_eq!(budget.next(60, true), Some(5001));
        for frame in 60..68 {
            assert_eq!(budget.next(frame, true), None);
        }
        for frame in [68, 76, 84, 92, 100] {
            assert!(budget.next(frame, true).is_some());
        }
        assert_eq!(budget.next(108, true), None);
    }

    #[test]
    fn indirect_readback_preserves_shared_and_separate_counts() {
        let host = host().expect("host GPU");
        for shared in [true, false] {
            let list = host.new_list().unwrap();
            let (args, mapped) = host.shared_buffer(32).unwrap();
            let separate = (!shared).then(|| host.shared_buffer(8).unwrap());
            let (counter, count_ptr) = separate.as_ref().map_or((&args, mapped), |(r, p)| (r, *p));
            let mut query = Query::allocate("indirect-test", "indirect", 1).unwrap();
            // SAFETY: each mapped shared buffer covers its offset and explicit record sizes.
            unsafe {
                count_ptr.add(4).cast::<u32>().write(1);
                let records = [90_000u32, 1, 1, 90_001, 1, 1];
                std::ptr::copy_nonoverlapping(records.as_ptr(), mapped.add(8).cast(), 6);
                list.buffer_access(&args, D3D12_RESOURCE_STATE_GENERIC_READ);
                if !shared { list.buffer_access(counter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); }
                // Four records would overrun this 32-byte range.
                query.indirect_before(&list, (&args, 8), Some((counter, 4)), (10, 12, 4));
                assert!(query.indirect.is_none());
                query.indirect_before(&list, (&args, 8), Some((counter, 4)), (10, 12, 2));
                assert_eq!(list.buffer_state(&args), D3D12_RESOURCE_STATE_GENERIC_READ);
                if !shared {
                    assert_eq!(list.buffer_state(counter), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
                assert_eq!(query.indirect.as_ref().unwrap().shared_resource, shared);
                assert!(crate::indirect::bound(
                    &list,
                    &args,
                    args.GetGPUVirtualAddress() + 8,
                    (10, 12, 2),
                    Some((counter, counter.GetGPUVirtualAddress() + 4))
                ));
                query.snapshot(&list, (&args, 8), Some((counter, 4)), true);
                crate::indirect::done(&list, &args, Some(counter));
                list.close();
                assert_eq!(host.log_messages(), None, "recorded state transitions");
                host.execute(&[&list]);
                host.device.GetDeviceRemovedReason().unwrap();
                let mut p = std::ptr::null_mut();
                query.readback.Map(0, None, Some(&mut p)).unwrap();
                let start = p.cast::<u8>().add(STATS_BYTES).cast::<u32>();
                for (index, expected) in [(0, &records), (1, &[65_535, 1, 1, 90_001, 1, 1])] {
                    let sample = start.add(index * SNAPSHOT_WORDS);
                    assert_eq!(std::slice::from_raw_parts(sample, 6), expected);
                    assert_eq!(*sample.add(SNAPSHOT_WORDS - 1), 1);
                }
                query.readback.Unmap(0, Some(&D3D12_RANGE::default()));
                assert_eq!(host.log_messages(), None, "submitted state transitions");
            }
        }
    }
}

pub(crate) struct Query {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ty: D3D12_QUERY_TYPE,
    hash: String,
    command: &'static str,
    recording: u32,
    presented_at_record: u64,
    indirect: Option<Indirect>,
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
        let query = Self::prepare(pipeline, command)?;
        query.start(list);
        Some(query)
    }

    pub(crate) fn prepare(pipeline: &HostPipeline, command: &'static str) -> Option<Self> {
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
        static COUNTS: OnceLock<Mutex<HashMap<(String, &'static str), SampleBudget>>> = OnceLock::new();
        let mut counts = COUNTS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let frame = crate::queue::presented();
        let scene_ready = frame.saturating_sub(FIRST_BIND_PRESENT.load(Ordering::Relaxed)) >= 32
            && crate::pipeline::making() == 0;
        let recording = counts.entry((hash.clone(), command)).or_default().next(frame, scene_ready)?;
        drop(counts);

        let mut query = Self::allocate(hash, command, recording)?;
        query.presented_at_record = frame;
        Some(query)
    }

    fn allocate(hash: &str, command: &'static str, recording: u32) -> Option<Self> {
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
            Width: (STATS_BYTES
                + if command == "indirect" {
                    8 * SNAPSHOT_WORDS
                } else {
                    0
                }) as u64,
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
            Some(Self {
                heap,
                readback,
                ty,
                hash: hash.to_owned(),
                command,
                recording,
                presented_at_record: 0,
                indirect: None,
                reported: AtomicBool::new(false),
            })
        }
    }

    pub(crate) fn start(&self, list: &HostList) {
        // SAFETY: called immediately before the sampled guest command, after the clamp.
        unsafe { list.list.BeginQuery(&self.heap, self.ty, 0) };
    }

    pub(crate) fn indirect_before(
        &mut self,
        list: &HostList,
        args: (&ID3D12Resource, u64),
        count: Option<(&ID3D12Resource, u64)>,
        (kind, stride, max): (u32, u32, u32),
    ) {
        if !matches!(kind, 2 | 10) || stride < 12 || !stride.is_multiple_of(4) || max == 0 {
            return;
        }
        let fits = |(r, off): (&ID3D12Resource, u64), bytes: u64| {
            // SAFETY: references come from the validated live shadow ranges.
            let d = unsafe { r.GetDesc() };
            d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                && off.is_multiple_of(4)
                && off.checked_add(bytes).is_some_and(|end| end <= d.Width)
        };
        if !fits(args, u64::from(stride) * u64::from(max)) || count.is_some_and(|c| !fits(c, 4)) {
            return;
        }
        // SAFETY: GetGPUVirtualAddress is defined for both validated buffer resources.
        let address = |(r, off): (&ID3D12Resource, u64)| (unsafe { r.GetGPUVirtualAddress() }, off);
        self.indirect = Some(Indirect {
            kind,
            stride,
            max,
            argument: address(args),
            count: count.map(address),
            shared_resource: count.is_some_and(|c| c.0 == args.0),
        });
        // The clamp or binding can still decline execution after this copy. The
        // command list must retain the destination even if the query is dropped.
        list.keep(&self.readback);
        self.snapshot(list, args, count, false);
    }

    pub(crate) fn snapshot(
        &self,
        list: &HostList,
        (args, offset): (&ID3D12Resource, u64),
        count: Option<(&ID3D12Resource, u64)>,
        after: bool,
    ) {
        let Some(info) = &self.indirect else { return };
        let buffers: Vec<_> = std::iter::once(args)
            .chain(count.filter(|c| c.0 != args).map(|c| c.0))
            .filter_map(|buffer| {
                let before = list.buffer_state(buffer);
                (before.0 & D3D12_RESOURCE_STATE_COPY_SOURCE.0 == 0).then_some((buffer, before))
            })
            .collect();
        for (buffer, _) in &buffers {
            list.buffer_access(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        }
        let start = (STATS_BYTES + usize::from(after) * SNAPSHOT_WORDS * 4) as u64;
        // SAFETY: indirect_before checked both complete source ranges. All sampled
        // records and the optional count have disjoint destinations within this snapshot.
        unsafe {
            for i in 0..u64::from(info.max.min(SAMPLE_RECORDS)) {
                list.list.CopyBufferRegion(
                    &self.readback,
                    start + i * 12,
                    args,
                    offset + i * u64::from(info.stride),
                    12,
                );
            }
            if let Some((counter, at)) = count {
                list.list.CopyBufferRegion(
                    &self.readback,
                    start + 12 * u64::from(SAMPLE_RECORDS),
                    counter,
                    at,
                    4,
                );
            }
        }
        for (buffer, before) in buffers {
            list.buffer_access(buffer, before);
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
        let mut snapshots = [[0u32; SNAPSHOT_WORDS]; 2];
        // SAFETY: Host::execute verified the submission fence. The base structure is
        // the documented prefix of STATISTICS1. Copy only bytes resolved by the GPU.
        unsafe {
            if self
                .readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: if self.indirect.is_some() {
                            STATS_BYTES + size_of_val(&snapshots)
                        } else {
                            size
                        },
                    }),
                    Some(&mut data),
                )
                .is_err()
            {
                return;
            }
            std::ptr::copy_nonoverlapping(data.cast::<u8>(), (&raw mut stats).cast::<u8>(), size);
            if let Some(info) = &self.indirect {
                let n = info.max.min(SAMPLE_RECORDS) as usize * 3;
                for (i, sample) in snapshots.iter_mut().enumerate() {
                    let src = data
                        .cast::<u8>()
                        .add(STATS_BYTES)
                        .cast::<u32>()
                        .add(i * SNAPSHOT_WORDS);
                    std::ptr::copy_nonoverlapping(src, sample.as_mut_ptr(), n);
                    if info.count.is_some() {
                        sample[SNAPSHOT_WORDS - 1] = *src.add(SNAPSHOT_WORDS - 1);
                    }
                }
            }
            self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        if let Some(info) = &self.indirect {
            let n = info.max.min(SAMPLE_RECORDS) as usize * 3;
            let words = |s: &[u32; SNAPSHOT_WORDS]| {
                s[..n]
                    .chunks_exact(3)
                    .map(|v| [v[0], v[1], v[2]])
                    .collect::<Vec<_>>()
            };
            tracing::info!(hash = %self.hash, recording = self.recording, kind = info.kind, stride = info.stride,
                max = info.max, argument = ?info.argument, count = ?info.count, shared_resource = info.shared_resource,
                count_before = ?info.count.map(|_| snapshots[0][SNAPSHOT_WORDS-1]),
                count_after = ?info.count.map(|_| snapshots[1][SNAPSHOT_WORDS-1]),
                before = ?words(&snapshots[0]), after = ?words(&snapshots[1]), presented_at_record = self.presented_at_record, "GPU indirect argument readback");
        }
        tracing::info!(hash = %self.hash, recording = self.recording, command = self.command, cs = stats.CSInvocations,
            ms = ?extended.then_some(stats.MSInvocations), primitives = ?extended.then_some(stats.MSPrimitives),
            ps = stats.PSInvocations, presented_at_record = self.presented_at_record, "GPU shader invocation readback");
    }
}
