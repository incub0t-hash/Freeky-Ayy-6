//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

#[cfg(feature = "diagnostics")]
mod packed_producer;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<Option<u32>>),
    /// A root view's GPU address.
    View(u64),
}

#[cfg(test)]
mod argument_stride_tests {
    #[test]
    fn sparse_root_updates_distinguish_unset_stride_from_zero() {
        let mut state=super::State::default();
        state.set_constants(true,1,&[16],1);
        assert_eq!(state.compute_constant(1,0),None);
        state.set_constants(false,1,&[32],0);
        assert_eq!(state.compute_constant(1,0),None);
        state.set_constants(true,1,&[0],0);
        assert_eq!(state.compute_constant(1,0),Some(0));
        assert_eq!(state.compute_constant(1,1),Some(16));
        state.set_view(true,1,0);
        assert_eq!(state.compute_constant(1,0),None);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// Raw SRVs owned by this list, keyed by view heap, shadow generation and byte window.
    raw_srv_views: std::collections::HashMap<(usize, Option<(usize, u64, u64)>), u32>,
    #[cfg(feature = "diagnostics")]
    root_srv_bindings: std::collections::HashMap<(bool, u32), (u32, Option<(u64, u64)>)>,
    #[cfg(feature = "diagnostics")]
    sample_cull: bool,
    #[cfg(feature = "diagnostics")]
    cull_inputs: Option<crate::capture::cull_indices::Inputs>,
    /// Exact live-buffer generation, byte address and size of each staged CBV.
    direct_cbvs: std::collections::HashMap<(usize, usize, u64, u32), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// A retained allocation's host address and extent distinguish guest allocations
    /// recreated at the same guest address during command-list recording.
    root_uav_slots: std::collections::HashMap<(usize, u64, u32), u32>,
}

fn root_uav_elements(offset: u64, bytes: u64) -> Option<u32> {
    let n = u32::try_from(bytes / 4).ok()?;
    (offset.is_multiple_of(16) && n != 0).then_some(n)
}

#[cfg(test)]
#[test]
fn root_uav_respects_allocation_length_and_raw_alignment() {
    assert_eq!(root_uav_elements(16, 4 * (1 << 20) + 3), Some(1 << 20));
    assert_eq!(root_uav_elements(16, 3), None);
    assert_eq!(root_uav_elements(4, 20), None);
    assert_eq!(root_uav_elements(0, (u64::from(u32::MAX) + 1) * 4), None);
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// Root constant buffers require 256-byte alignment and have a 64 KiB limit.
fn root_cbv(list: &HostList, address: u64) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow.left(address).min(1 << 16);
    let gpu = shadow.gpu(address);
    if gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        list.buffer_access(&shadow.buffer, D3D12_RESOURCE_STATE_GENERIC_READ);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, None);
            }
            for (word, &value) in c[offset as usize..end].iter_mut().zip(values) {
                *word = Some(value);
            }
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            {
                let selected = crate::capture::scene_depth(texture);
                crate::capture::note_target(record.resource);
                // The DSV can select this very frame for capture, after its RTVs were bound.
                if selected {
                    for &target in &self.targets {
                        crate::capture::note_target(target as This);
                    }
                }
            }
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().flatten().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.buffer_access(&shadow.buffer, D3D12_RESOURCE_STATE_GENERIC_READ);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.buffer_access(&shadow.buffer, D3D12_RESOURCE_STATE_GENERIC_READ);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        self.bind_for_dispatch(list, p, compute, None)
    }

    fn bind_for_dispatch(&mut self, list: &HostList, p: &HostPipeline, compute: bool, _groups: Option<[u32; 3]>) -> bool {
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, compute, &t.host.scalar_raw_srvs);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return false;
            }
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None if compute && t.host.direct_raw_indices.contains(&u.index) => {
                            let Some(index) = self.direct_raw_view(list, self.compute_constant(u.slot, u.dword)) else {
                                return false;
                            };
                            index
                        }
                        None if compute && t.host.direct_scalar_indices.contains(&u.index) => {
                            let selected = self.compute_constant(u.slot, u.dword);
                            let Some(index) = self.direct_scalar_view(list, selected) else {
                                return false;
                            };
                            index
                        }
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                #[cfg(feature = "diagnostics")]
                if compute {
                    if self.sample_cull && crate::capture::cull_indices::supported(&stage.shader.hash) {
                        self.cull_inputs = self.cull_inputs_for(&values);
                    }
                    self.trace_compute_bindings(&stage.shader.hash, &t.used, &made, &values, &t.host);
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for &(slot, param) in &stage.root_cbvs {
                let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
                    once!(slot, "a shader reads a root CBV the title did not set");
                    continue;
                };
                let Some(gpu) = root_cbv(list, *address) else { continue };
                // SAFETY: an open list and a live, aligned constant buffer.
                unsafe {
                    if compute {
                        l.SetComputeRootConstantBufferView(param, gpu);
                    } else {
                        l.SetGraphicsRootConstantBufferView(param, gpu);
                    }
                }
            }
            for &(slot, param) in &stage.root_srvs {
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => None,
                };
                if !self.root_srv_table(list, address, param, compute) {
                    return false;
                }
            }
            for &(slot, param) in &stage.root_uavs {
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav_table(list, address, param, compute) {
                    return false;
                }
            }
            #[cfg(feature = "diagnostics")]
            if let Some(groups) = _groups
                && !groups.contains(&0) && p.stages.len() == 1 && packed_producer::enabled()
            {
                self.trace_packed_producers(&stage.shader.hash, &t.used, &made, groups);
            }
        }
        true
    }

    /// Public #493: bind a one-view table so out-of-range vertex fetches return zero.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
    ) -> bool {
        let window = address.and_then(crate::descriptor::host::root_raw_window);
        #[cfg(feature = "diagnostics")]
        let range = window.as_ref().map(|w| w.gpu_range());
        let Some((_slot, handle)) = self.raw_view_slot(list, window) else { return false };
        #[cfg(feature = "diagnostics")]
        self.root_srv_bindings.insert((compute, param), (_slot, range));
        // SAFETY: the slot belongs to the heap bound on this open command list.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Use the complete dword range of the named allocation, starting at the root address.
    /// A root descriptor has no independent length. Unsupported bindings drop the command.
    fn root_uav_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
    ) -> bool {
        let Some((heap, host)) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find_map(|h| h.host().filter(|x| !x.is_sampler()).map(|x| (h, x)))
        else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let Some(address) = address else { return false };
        let backing = if address == 0 {
            None
        } else {
            let Some(s) = crate::shadow::find_named(address, 4) else {
                once!(address, "root UAV has no registered allocation covering a dword");
                return false;
            };
            let offset = s.offset(address);
            let Some(elements) = root_uav_elements(offset, s.left(address)) else {
                once!(address, offset, "root UAV range cannot be represented by a raw view");
                return false;
            };
            Some((s, offset, elements))
        };
        let (gpu, elements) = backing.as_ref().map_or((0, 0), |(s, _, n)| (s.gpu(address), *n));
        let key = (std::ptr::from_ref(heap) as usize, gpu, elements);
        let index = match self.root_uav_slots.get(&key) {
            Some(&index) => index,
            None => {
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_uav_slots.insert(key, index);
                match &backing {
                    Some((s, offset, n)) => host.raw_uav_at(index, &s.buffer, *offset, *n),
                    None => host.null_raw_uav_at(index),
                }
                index
            }
        };
        if let Some((s, _, n)) = &backing {
            let len = u64::from(*n) * 4;
            s.refresh(list, address, len);
            list.buffer_access(&s.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            s.gpu_writes(list, address, len);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Check split-wave record strides from exactly the title root arguments
    /// that bind() will copy. Missing constants cannot certify a dispatch.
    fn permits_compute(&self, p: &HostPipeline) -> bool {
        p.stages.iter().all(|stage| {
            stage.translated().host.permits_dispatch(|slot,dword| self.compute_constant(slot,dword))
        })
    }

    fn compute_constant(&self, slot: u8, dword: u32) -> Option<u32> {
        let Root::Constants(words) = self.compute.get(usize::from(slot))? else { return None };
        words.get(dword as usize).copied().flatten()
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.permits_compute(p) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("dispatch: argument record stride is missing or is not a multiple of 16".into());
            return;
        }
        if !self.bind_for_dispatch(list, p, true, Some([x, y, z])) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        let query = crate::capture::stage_query::Query::begin(list, p, "dispatch");
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        #[cfg(feature = "diagnostics")]
        if let Some(query) = query { query.end(list) }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, param) in &stage.root_srvs {
                let guest = self.roots_ref(compute).get(usize::from(slot));
                let actual = self.root_srv_bindings.get(&(compute, param));
                let _ = write!(text, " root SRV {slot} guest {guest:#x?} descriptor/range {actual:#x?}");
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    #[cfg(feature = "diagnostics")]
                    let query = crate::pipeline::host_pipeline(self.pipeline as This)
                        .and_then(|p| crate::capture::stage_query::Query::begin(list, p, "mesh"));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    #[cfg(feature = "diagnostics")]
                    if let Some(query) = query { query.end(list) }
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    /// Report the actual root values, remapped heap slot and retained resource for
    /// a few dispatches of each compute stage selected for invocation capture.
    #[cfg(feature = "diagnostics")]
    fn trace_compute_bindings(&self, hash: &str, used: &[UsedResource], made: &[(u8, u32)], values: &[u32], shader: &victory_shader::emit::HostShader) {
        use std::{collections::HashMap, sync::{Mutex, OnceLock}};
        if !crate::capture::stage_query::watched().iter().any(|wanted| wanted == hash) { return; }
        static COUNTS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
        let recording = {
            let mut counts = COUNTS.get_or_init(Mutex::default).lock().unwrap_or_else(|e| e.into_inner());
            let n = counts.entry(hash.to_owned()).or_default();
            *n = n.saturating_add(1);
            if *n > 64 || !n.is_power_of_two() { return; }
            *n
        };
        tracing::info!(hash, recording, ?values, "sampled compute root dwords");
        for u in shader.user_data.iter().filter(|u| {
            shader.direct_raw_indices.contains(&u.index) || shader.direct_scalar_indices.contains(&u.index)
        }) {
            let selected = self.compute_constant(u.slot, u.dword);
            let Some(&encoded) = values.get(u.index as usize) else { continue };
            if shader.direct_scalar_indices.contains(&u.index) && encoded & victory_shader::emit::SCALAR_RAW_INDEX == 0 {
                let range = self.direct_cbvs.iter()
                    .find(|((h, _, _, _), slot)| *h == self.heaps[0] && **slot == encoded)
                    .map(|((_, _, address, size), _)| (*address, *size));
                tracing::info!(hash, recording, user_data=u.index, ?selected, alias=encoded, ?range,
                    "sampled private direct constant buffer");
                continue;
            }
            let alias = encoded & !victory_shader::emit::SCALAR_RAW_INDEX;
            let Some((&(_, window), _)) = self.raw_srv_views.iter()
                .find(|((h,_), v)| *h == self.heaps[0] && **v == alias) else {continue};
            let range = window.map(|(_,a,n)|(a,n));
            let gpu = crate::descriptor::heap(self.heaps[0] as This).host()
                .and_then(|h| h.retained_resource(alias)).flatten()
                .and_then(|r| range.and_then(|(a,n)| crate::shadow::bound_range(&r,a,n))).map(|x| x.1);
            tracing::info!(hash, recording, user_data=u.index, ?selected, alias, ?range, ?gpu,
                "sampled private direct heap buffer");
        }
        for &heap in self.heaps.iter().filter(|&&heap| heap != 0) {
            crate::descriptor::heap(heap as This).trace_direct_views(hash, recording);
        }
        for u in used.iter().filter(|u| u.root == RootType::Table) {
            let Some(&Root::Table(table)) = self.compute.get(usize::from(u.slot)) else {
                tracing::info!(hash, recording, slot = u.slot, "sampled compute table missing");
                continue;
            };
            let guest = table.checked_add(u64::from(u.table_offset_dwords) * 4);
            let heap = self.heap_of(table);
            let view = heap.and_then(|h| guest.filter(|&at| h.index(at).is_some()))
                .and_then(crate::descriptor::view);
            let fresh = made.iter().find(|&&(slot, _)| slot == u.slot).map(|&(_, bytes)| bytes / 32);
            let index = fresh.or_else(|| heap.and_then(|h| h.index(table)))
                .and_then(|base| base.checked_add(u.table_offset_dwords / 8));
            // Fresh constant views use the command list's upload resource; SRV/UAV
            // creation instead retains the actual backing object in the heap slot.
            let retained = (u.kind != ResourceType::Cbv).then(|| heap.and_then(|h| h.host())
                .and_then(|host| index.and_then(|i| host.retained_resource(i)))).flatten();
            let backed = retained.as_ref().map(Option::is_some);
            let range = view.as_ref().and_then(crate::descriptor::host::buffer_range);
            let shadow = retained.flatten().and_then(|backing| range.and_then(|(at, n)|
                crate::shadow::bound_range(&backing, at, n)));
            tracing::info!(hash, recording, slot = u.slot, offset = u.table_offset_dwords,
                kind = ?u.kind, stride = u.stride, ?index, ?backed, ?range,
                guest_buffer = ?shadow.map(|pair| pair.0), gpu = ?shadow.map(|pair| pair.1), ?view,
                "sampled compute table descriptor");
        }
    }

    fn direct_raw_view(&mut self, list: &HostList, selected: Option<u32>) -> Option<u32> {
        let heap_id = self.heaps[0];
        if heap_id == 0 {
            return None;
        }
        let heap = crate::descriptor::heap(heap_id as This);
        let window = Self::selected_view(heap, selected)
            .as_ref()
            .and_then(crate::descriptor::host::direct_raw_window);
        self.raw_view_slot(list, window).map(|(slot, _)| slot)
    }

    fn selected_view(heap: &crate::descriptor::Heap, selected: Option<u32>) -> Option<View> {
        selected.and_then(|value| {
            // The native shader shifts a 32-bit index left by five, then
            // addresses a 32-byte view at the resulting byte offset.
            let index = value & (u32::MAX >> 5);
            let at = heap.start.checked_add(32 * u64::from(index))?;
            if heap.index(at) != Some(index) {
                return None;
            }
            crate::descriptor::view(at)
        })
    }

    fn direct_scalar_view(&mut self, list: &HostList, selected: Option<u32>) -> Option<u32> {
        let heap_id = self.heaps[0];
        if heap_id == 0 {
            return None;
        }
        let heap = crate::descriptor::heap(heap_id as This);
        let host = heap.host()?;
        let view = Self::selected_view(heap, selected);
        if let Some(View::Constant(address, size)) = view {
            if size == 0 || size > 65536 || !size.is_multiple_of(16) {
                return None;
            }
            let len = u64::from(size);
            let shadow = crate::shadow::find_named(address, 1);
            if let Some(s) = &shadow {
                if len > s.left(address) {
                    return None;
                }
                // A native CBV may begin/end on a 16-byte boundary. A D3D12
                // CBV rounds its extent up to 256 bytes. For a per-dword
                // scalar read preserve the original range using a raw view.
                if !s.gpu(address).is_multiple_of(256) || !len.is_multiple_of(256) {
                    let window = crate::descriptor::host::named_raw_window(s.clone(), address, len)?;
                    return self
                        .raw_view_slot(list, Some(window))
                        .map(|(slot, _)| slot | victory_shader::emit::SCALAR_RAW_INDEX);
                }
            }
            if let Some(s) = &shadow {
                list.buffer_access(&s.buffer, D3D12_RESOURCE_STATE_GENERIC_READ);
            }
            let generation = shadow
                .as_ref()
                .map_or(0, |s| std::sync::Arc::as_ptr(s) as usize);
            let key = (heap_id, generation, address, size);
            if let Some(&slot) = self.direct_cbvs.get(&key) {
                return Some(slot);
            }
            let gpu = if let Some(s) = shadow {
                let gpu = s.gpu(address);
                crate::shadow::refresh_named(list, &s, address, u64::from(size));
                list.keep(&s.buffer);
                gpu
            } else {
                if !crate::texture::committed(address as *const u8, size as usize) {
                    return None;
                }
                let padded = size.next_multiple_of(256);
                let (gpu, to) = list.scratch(u64::from(padded))?;
                // SAFETY: scratch contains padded bytes. Clear the tail so
                // host CBV alignment cannot expose adjacent guest data.
                unsafe {
                    to.add(size as usize)
                        .write_bytes(0, (padded - size) as usize);
                }
                list.defer_fill(to, address as *const u8, size as usize);
                gpu
            };
            let slot = host.dynamic(1)?;
            host.constant_at(slot, gpu, size);
            self.dynamic.push((heap_id, slot, 1));
            self.direct_cbvs.insert(key, slot);
            return Some(slot);
        }
        let window = match view.as_ref() {
            Some(view) => Some(crate::descriptor::host::direct_scalar_window(view)?),
            None => None,
        };
        self.raw_view_slot(list, window)
            .map(|(slot, _)| slot | victory_shader::emit::SCALAR_RAW_INDEX)
    }

    fn raw_view_slot(
        &mut self,
        list: &HostList,
        window: Option<crate::descriptor::host::ScalarWindow>,
    ) -> Option<(u32, D3D12_GPU_DESCRIPTOR_HANDLE)> {
        let heap_id = self.heaps[0];
        if heap_id == 0 { return None; }
        let host = crate::descriptor::heap(heap_id as This).host()?;
        let key = (heap_id, window.as_ref().map(|w| w.identity()));
        if let Some(&slot) = self.raw_srv_views.get(&key) {
            if let Some(window) = &window { window.read_access(list); }
            return Some((slot, host.gpu_handle(slot)));
        }
        let slot = host.dynamic(1)?;
        if !window.as_ref().is_some_and(|w| w.bind(list, host, slot)) {
            host.null_scalar_buffer(slot);
        }
        self.dynamic.push((heap_id, slot, 1));
        self.raw_srv_views.insert(key, slot);
        Some((slot, host.gpu_handle(slot)))
    }

    /// These two compiled programs have T77 at table dword 8 and the private
    /// raw heap SRV at user dword 6. Verify both host descriptors share the
    /// same resource and neither declared output UAV can alter the records.
    #[cfg(feature = "diagnostics")]
    fn cull_inputs_for(&self, values: &[u32]) -> Option<crate::capture::cull_indices::Inputs> {
        if self.compute_constant(1, 1) != Some(17) {
            return None;
        }
        let &Root::Table(table) = self.compute.first()? else { return None };
        let host = self.heap_of(table)?.host()?;
        let base = *values.get(4)? / 32;
        let private = *values.get(6)?;
        let scene = host.retained_resource(private)??;
        let table_srv = host.retained_resource(base.checked_add(1)?)??;
        if table_srv != scene {
            return None;
        }
        for i in [3, 4] {
            if host.retained_resource(base.checked_add(i)?)?? == scene {
                return None;
            }
        }
        let ((_, window), _) = self.raw_srv_views.iter()
            .find(|((heap, _), alias)| *heap == self.heaps[0] && **alias == private)?;
        let (_, start, scene_len) = (*window)?;
        let (_, (_, scene_offset)) = crate::shadow::bound_range(&scene, start, scene_len)?;
        let view = crate::descriptor::view(table.checked_add(32)?)?;
        let (param, n) = crate::descriptor::host::buffer_range(&view)?;
        if n != 456 {
            return None;
        }
        let (_, (_, param_offset)) = crate::shadow::bound_range(&table_srv, param, n)?;
        Some(crate::capture::cull_indices::Inputs { resource: scene, param_offset, scene_offset, scene_len })
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
        scalar_raw_srvs: &[(u8, u32)],
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let Some(end) = table.checked_add(u64::from(u.table_offset_dwords) * 4) else {
                self.table_failed = true;
                return Vec::new();
            };
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            let raw = scalar_raw_srvs.contains(&(u.slot, u.table_offset_dwords))
                && crate::descriptor::view(end)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::scalar_structured_view);
            if index >= host.count() || raw {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // Validate the full descriptor range before reading its records.
            let last = table.checked_add(u64::from(n - 1) * 32 + 31);
            if last.and_then(|at| heap.index(at)).is_none() {
                self.table_failed = true;
                continue;
            }
            let mut windows = Vec::new();
            for u in entries.iter().filter(|u| {
                scalar_raw_srvs.contains(&(u.slot, u.table_offset_dwords))
            }) {
                let k = u.table_offset_dwords / 8;
                let at = table + u64::from(k) * 32;
                if let Some(view) = crate::descriptor::view(at)
                    && crate::descriptor::host::scalar_structured_view(&view)
                {
                    windows.push((k, crate::descriptor::host::scalar_window(&view, u.stride)));
                }
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for (k, window) in &windows {
                    (k, window.as_ref().map(|w| w.generation())).hash(&mut h);
                }
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let k = u.table_offset_dwords / 8;
                    if let Some((_, window)) = windows.iter().find(|(at, _)| *at == k) {
                        if let Some(w) = window {
                            w.refresh(list);
                        }
                        continue;
                    }
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                if let Some((_, window)) = windows.iter().find(|(at, _)| *at == k) {
                    if !window.as_ref().is_some_and(|w| w.bind(list, host, index)) {
                        host.null_scalar_buffer(index);
                    }
                    continue;
                }
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        let rects = ready && self.topology == RECTANGLES &&
            !crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangle_stage);
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        // A missing or incompatible pipeline cannot consume a single record.
        // In particular, do not modify its argument buffer or spend the
        // list's work budget, which later runnable commands also use.
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        if (kind == DISPATCH) != p.compute || (kind == MESH) != p.mesh {
            why("pipeline kind does not match the command signature");
            return false;
        }
        if kind==DISPATCH && !self.permits_compute(p) {
            why("argument record stride is missing or is not a multiple of 16");
            return false;
        }
        if self.topology == RECTANGLES && (kind == INDEXED || (kind == DRAW && !p.rectangle_stage)) {
            why("rectangle-list indirect draws need vertex expansion");
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        let count_view = counter.as_ref().map(|(s, at)| (&s.buffer, s.gpu(*at)));
        let count_buffer = count_view.map(|(buffer, _)| buffer);
        #[cfg(feature = "diagnostics")]
        let mut query = crate::capture::stage_query::Query::prepare(p, "indirect");
        #[cfg(feature = "diagnostics")]
        let sample_args = (&args.buffer, args.offset(arguments));
        #[cfg(feature = "diagnostics")]
        let sample_count = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
        #[cfg(feature = "diagnostics")]
        if let Some(q) = &mut query { q.indirect_before(list, sample_args, sample_count, (kind, stride, max)); }
        // Bounded first: the clamp binds a compute pipeline of its own.
        if !crate::indirect::bound(
            list,
            &args.buffer,
            args.gpu(arguments),
            (kind, stride, max),
            count_view,
        ) {
            why("no argument clamp or unaligned indirect arguments/count");
            return false;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        #[cfg(feature = "diagnostics")]
        {
            self.sample_cull = query.as_ref().is_some_and(|q| q.wants_cull());
        }
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        #[cfg(feature = "diagnostics")]
        let cull = {
            self.sample_cull = false;
            self.cull_inputs.take()
        };
        if list.indirect_write_conflict(&args.buffer)
            || count_buffer.is_some_and(|buffer| list.indirect_write_conflict(buffer))
        {
            crate::indirect::done(list, &args.buffer, count_buffer);
            why("indirect input also bound for unordered writes on this command");
            return false;
        }
        if !ran {
            crate::indirect::done(list, &args.buffer, count_buffer);
            why("pipeline exists but its resources could not be bound");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        #[cfg(feature = "diagnostics")]
        if let Some(q) = &query {
            q.snapshot(list, sample_args, sample_count, true);
            q.start(list);
        }
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &args.buffer,
                args.offset(arguments),
                count_buffer,
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &args.buffer, count_buffer);
        #[cfg(feature = "diagnostics")]
        if let Some(query) = query
            && query.end_with_cull(list, cull.as_ref().map(|i| (i, sample_args))) {
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            list.buffer_access(&shadow.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            shadow.gpu_writes(list, address, len);
        } else {
            list.buffer_access(&shadow.buffer, D3D12_RESOURCE_STATE_GENERIC_READ);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
                && !(placed.is_some() && crate::resource::copy_packed_view(list, *resource as This))
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(all(test, feature = "diagnostics"))]
#[path = "indirect_scene_tests.rs"]
mod indirect_scene_tests;
