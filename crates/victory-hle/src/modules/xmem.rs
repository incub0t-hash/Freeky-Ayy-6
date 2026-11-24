//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section {
    handle: HANDLE,
    reserved: bool,
    /// The section's whole range, mapped the first time one of its pages is
    /// written and kept until the section goes; `None` once the host has
    /// refused that mapping. See [`Section::persistent_ptr`].
    persistent: OnceLock<Option<MappedBase>>,
    /// The section's size in bytes.
    size: usize,
    /// One bit per 64 KiB page, set once committed through the persistent
    /// view. A saved call, not a guarantee: `commit` is idempotent, so the
    /// DirectStorage workers may write this concurrently and a lost race
    /// costs one redundant call. It records this view's own commits only;
    /// the temporary path's are not recorded, by design.
    committed: Box<[AtomicU64]>,
    /// Whether a warn has been said for a write that fell back to a
    /// temporary view, so a run with the switch on can tell which sections
    /// are really persistent.
    fallback_noted: OnceLock<()>,
}

impl Section {
    fn new(size: usize, directory: Option<&std::path::Path>) -> std::io::Result<Self> {
        use std::os::windows::fs::OpenOptionsExt;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_DELETE_ON_CLOSE;

        // A writable data-file mapping can back the same physical pages at
        // several addresses without charging their size to system commit.
        // Create-new and delete-on-close give each section its own lifetime,
        // even if the process exits while aliases still exist.
        let file = if let Some(dir) = directory {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("victory-pages-{}-{id}.tmp", std::process::id()));
            let file = std::fs::OpenOptions::new().read(true).write(true).create_new(true)
                .custom_flags(FILE_FLAG_DELETE_ON_CLOSE).open(path)?;
            file.set_len(size as u64)?;
            Some(file)
        } else {
            None
        };
        let reserved = file.is_none() && !on_wine();
        let flags = if file.is_some() { 0 } else if reserved { SEC_RESERVE } else { SEC_COMMIT };
        // SAFETY: the file stays open through section creation. A successful
        // section holds a reference to the file until its last view is gone.
        let handle = unsafe {
            CreateFileMappingW(file.as_ref().map_or(INVALID_HANDLE_VALUE, AsRawHandle::as_raw_handle),
                std::ptr::null(), PAGE_READWRITE | flags, (size as u64 >> 32) as u32,
                size as u32, std::ptr::null())
        };
        if handle.is_null() { return Err(std::io::Error::last_os_error()); }
        Ok(Self {
            handle,
            reserved,
            persistent: OnceLock::new(),
            fallback_noted: OnceLock::new(),
            size,
            committed: (0..(size / PAGE).div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
        })
    }
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    /// Says once per section that a write fell back to a temporary view.
    /// Without it a run with the switch on cannot tell a section that
    /// persisted from one that did not, which is the evidence the switch
    /// becoming the default is supposed to rest on.
    fn note_fallback(&self) {
        if self.fallback_noted.set(()).is_ok() {
            tracing::warn!(
                size = self.size,
                "a commit through the persistent view failed; this section's pages keep a temporary view per write"
            );
        }
    }

    /// The section's whole range, mapped read-write once and kept. A write
    /// through it is a store; a view per write is a map, a commit and an
    /// unmap, and every unmap flushes every processor's TLB.
    ///
    /// `None` when the host refuses the mapping, which the title's largest
    /// allocation can provoke: it asks for 6.1 GiB at once, and this maps
    /// all of it. Callers then keep the temporary view per write.
    fn persistent_ptr(&self) -> Option<*mut u8> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: our section, mapped read-write; `MappedBase` owns the view
        // and unmaps it when the section goes.
        let mapped = self.persistent.get_or_init(|| {
            let at = unsafe { MapViewOfFile(self.handle, FILE_MAP_WRITE, 0, 0, 0) };
            if at.Value.is_null() {
                tracing::warn!(
                    size = self.size,
                    error = %std::io::Error::last_os_error(),
                    "mapping a physical-page section whole failed; its pages keep a temporary view per write"
                );
                return None;
            }
            Some(MappedBase(at.Value.cast()))
        });
        mapped.as_ref().map(|base| base.0)
    }

    /// Whether all of pages `[start, start + count)` are committed in the
    /// persistent view.
    fn committed_all(&self, start: usize, count: usize) -> bool {
        let Some(end) = start.checked_add(count) else {
            return false;
        };
        (start..end).all(|page| {
            (self.committed[page / 64].load(Ordering::Relaxed) >> (page % 64)) & 1 != 0
        })
    }

    /// Records pages `[start, start + count)` as committed in the persistent
    /// view. A lost race only costs a call, so relaxed is enough.
    fn mark_committed(&self, start: usize, count: usize) {
        for page in start..start + count {
            self.committed[page / 64].fetch_or(1u64 << (page % 64), Ordering::Relaxed);
        }
    }

    /// Commits pages `[start_page, start_page + count)` of the persistent
    /// view at `base`. A page this view committed before costs no call at
    /// all, which is the point. False when the range leaves the section or
    /// the host refuses.
    ///
    /// The bounds check is a real guard, not a saved call: it is what keeps
    /// `committed_all`'s page arithmetic inside `committed`, and a
    /// zero-length range past the end is refused by it alone (pinned by
    /// `ensure_committed_refuses_ranges_outside_the_section`).
    ///
    /// The `committed_all` early return is the one surviving mutation: a
    /// saved call, not a guard, since `commit` over a committed page
    /// succeeds and changes nothing. Probe run 2026-11-25 — with the branch
    /// forced off, `cargo test -p victory-hle xmem::` still passes, so no
    /// test can see it, and its effect is measured by a run
    /// (`VICTORY_XMEM_PERSISTENT_VIEW=1`) rather than asserted by one. That
    /// choice is deliberate and recorded here.
    fn ensure_committed(&self, base: *mut u8, start_page: usize, count: usize) -> bool {
        let Some(end_page) = start_page.checked_add(count) else {
            return false;
        };
        if !end_page
            .checked_mul(PAGE)
            .is_some_and(|end| end <= self.size)
        {
            return false;
        }
        if self.committed_all(start_page, count) {
            return true;
        }
        // One commit for the whole range: pages committed before are left
        // as they are. A host at its commit limit is reported here and, if
        // the caller then retries on the temporary path, reported again by
        // that path's own `commit` -- two crash-log lines for one failure,
        // which is asked for rather than avoided: the retry is what gives
        // the caller the same answer it would have had before.
        // SAFETY: `[start_page, end_page)` lies inside the section's view.
        let at = unsafe { base.add(start_page * PAGE) };
        if self.reserved && !commit(at as usize, count * PAGE) {
            return false;
        }
        self.mark_committed(start_page, count);
        true
    }
}

/// The section's whole range, mapped once and unmapped when the section
/// goes rather than after every write.
struct MappedBase(*mut u8);

// SAFETY: a mapping may be written from any thread; its writers write bytes
// of their own, the rule a title's own views of the same pages follow.
unsafe impl Send for MappedBase {}
// SAFETY: as above.
unsafe impl Sync for MappedBase {}

impl Drop for MappedBase {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view. Closing the section's handle first is safe: a
        // view keeps the section object alive on its own.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.0.cast(),
            });
        }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    let directory = if cfg!(feature = "diagnostics") {
        std::env::var_os("VICTORY_PHYSICAL_BACKING_DIR").filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
    } else { None };
    let section = match Section::new(size, directory.as_deref()) {
        Ok(section) => Arc::new(section),
        Err(error) => {
            tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
            // SAFETY: checked non-null.
            unsafe { count.write(0) };
            return 0;
        }
    };
    if cfg!(feature = "diagnostics") && size >= 64 * MIB {
        tracing::info!(bytes = size, file_backed = directory.is_some(), "guest physical section allocated");
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // The in-place GPU path accepts committed page-file sections. A temporary
    // file mapping uses the renderer's normal copying path.
    victory_gpu::title_memory_section(section.handle as usize, size as u64, directory.is_none() && on_wine());
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        physical.pages.remove(&unsafe { *pages.add(i) });
    }
    1
}

/// Parses `VICTORY_XMEM_PERSISTENT_VIEW`: off unless the value says yes, so
/// a run that sets nothing keeps the view per write. Opt-in while the
/// persistent path settles, because it maps each section whole and keeps
/// the mapping: 6.1 GiB of address space for the title's largest allocation.
pub(crate) fn parse_persistent_views(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let Some(text) = value.to_str() else {
        return false;
    };
    let trimmed = text.trim();
    trimmed == "1"
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("on")
}

/// Whether pages are written through one persistent view per section
/// (`VICTORY_XMEM_PERSISTENT_VIEW=1`) instead of a view mapped, committed
/// and unmapped for every write.
pub(crate) fn persistent_views() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_persistent_views(std::env::var_os("VICTORY_XMEM_PERSISTENT_VIEW").as_deref())
    })
}

/// How a write reaches its pages: the process default ([`persistent_views`]),
/// or one forced. Tests force either, since the environment is read once per
/// process and cannot be set from a test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mapping {
    Default,
    Persistent,
    Temporary,
}

impl Mapping {
    /// The view `n` slices of `slice` on are written through.
    fn view(self, slice: &Slice, n: usize) -> Option<View> {
        let mapping = match self {
            Mapping::Default if persistent_views() => Mapping::Persistent,
            Mapping::Default => Mapping::Temporary,
            forced => forced,
        };
        match mapping {
            // The temporary path is the one every run before the persistent
            // one used, so a host refusing either half of the persistent
            // path lands there rather than failing a write.
            Mapping::Persistent => View::persistent(slice, n).or_else(|| View::temporary(slice, n)),
            // `Default` was resolved above: both are the old path.
            Mapping::Temporary | Mapping::Default => View::temporary(slice, n),
        }
    }
}

/// A view of our own on consecutive slices of one section, committed and
/// writable.
enum View {
    /// Mapped for this write alone, unmapped when dropped.
    Temporary {
        at: MEMORY_MAPPED_VIEW_ADDRESS,
        len: usize,
    },
    /// The section's persistent view, at `slice`'s pages. Nothing is unmapped
    /// when this drops: the view belongs to the section and goes with it.
    Persistent {
        /// Keeps the section, and so the view, alive while the bytes are used.
        _section: Arc<Section>,
        ptr: *mut u8,
        len: usize,
    },
}

impl View {
    /// Maps `n` slices from `slice` on for this write alone.
    fn temporary(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View::Temporary { at, len };
        (!slice.section.reserved || commit(at.Value as usize, len)).then_some(view)
    }

    /// The section's persistent view over `n` slices from `slice` on, those
    /// pages committed; `None` when the host refuses either half, which
    /// leaves the caller on [`View::temporary`].
    fn persistent(slice: &Slice, n: usize) -> Option<View> {
        let base = slice.section.persistent_ptr()?;
        let len = n.checked_mul(PAGE)?;
        if !slice.section.ensure_committed(base, slice.index, n) {
            slice.section.note_fallback();
            return None;
        }
        // SAFETY: `ensure_committed` proved `[slice.index, slice.index + n)`
        // lies inside the section, so this is inside the section's view.
        let ptr = unsafe { base.add(slice.index * PAGE) };
        // Said once: whether the switch is really in effect is otherwise
        // invisible in a run, and the switch becoming the default rests on a
        // run that showed it working.
        static IN_USE: OnceLock<()> = OnceLock::new();
        if IN_USE.set(()).is_ok() {
            tracing::info!(
                "physical pages are written through one persistent view per section (VICTORY_XMEM_PERSISTENT_VIEW=1)"
            );
        }
        Some(View::Persistent {
            _section: slice.section.clone(),
            ptr,
            len,
        })
    }

    fn bytes(&mut self) -> &mut [u8] {
        let (ptr, len) = match self {
            View::Temporary { at, len } => (at.Value.cast::<u8>(), *len),
            View::Persistent { ptr, len, .. } => (*ptr, *len),
        };
        // SAFETY: the view holds `len` committed read-write bytes. For a
        // temporary view those bytes have an address of their own, so no
        // other borrow of them exists. For the persistent variant the
        // section's one mapping is shared, and keeping two live `&mut [u8]`
        // at one base apart is this module's job: callers hand out one run
        // per write in flight, and two writes in flight in one section never
        // cover the same page, because the title writes each destination
        // alone (`dstorage`). That disjointness is a precondition of this
        // variant, not a convention: overlapping runs would be undefined
        // behaviour here, where two separate virtual views made them merely
        // a race.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // Only a temporary view is ours to take away.
        if let View::Temporary { at, .. } = self {
            use windows_sys::Win32::System::Memory::UnmapViewOfFile;
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(*at) };
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    page_runs(&physical(), pages)
}

fn page_runs(physical: &Physical, pages: &[usize]) -> Option<Vec<(Slice, usize)>> {
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page; with
/// `VICTORY_XMEM_PERSISTENT_VIEW` a run costs no view at all, and without
/// it each view is a map, a commit and an unmap (which flushes every core's
/// TLB). False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    write_physical_with(Mapping::Default, pages, offset, data)
}

/// [`write_physical`] through the views `mapping` names, so tests reach
/// either path without the environment.
fn write_physical_with(mapping: Mapping, pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = mapping.view(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    page_window_with(Mapping::Default, pages, offset, len)
}

/// [`page_window`] through the views `mapping` names, so tests reach either
/// path without the environment.
fn page_window_with(
    mapping: Mapping,
    pages: &[usize],
    offset: usize,
    len: usize,
) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: mapping.view(slice, *n)?,
        within: offset % PAGE,
        len,
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let Some(len) = n.checked_mul(PAGE) else {
        return std::ptr::null_mut();
    };
    let Some(end) = base.checked_add(len) else {
        return std::ptr::null_mut();
    };
    if len == 0 {
        return address;
    }
    if base == 0 || !base.is_multiple_of(PAGE) {
        return std::ptr::null_mut();
    }
    let mut physical = physical();
    let runs = if pages.is_null() {
        Vec::new()
    } else {
        // SAFETY: the title passes n page numbers. Resolve them before changing
        // the destination, which can itself contain the list of page numbers.
        let pages = unsafe { std::slice::from_raw_parts(pages, n) };
        let Some(runs) = page_runs(&physical, pages) else {
            return std::ptr::null_mut();
        };
        runs
    };
    if replace_views(&mut physical, base, end, runs) {
        address
    } else {
        std::ptr::null_mut()
    }
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    if len == 0 {
        return true;
    }
    if to % PAGE != 0 || from % PAGE != 0 {
        return false;
    }
    let Some(mapped_len) = len.div_ceil(PAGE).checked_mul(PAGE) else {
        return false;
    };
    let Some(end) = to.checked_add(mapped_len) else {
        return false;
    };
    if from.checked_add(mapped_len).is_none() {
        return false;
    }
    let mut physical = physical();
    // Snapshot all source slices first, including when the two ranges overlap.
    let mut runs = Vec::new();
    if from != 0 {
        for k in (0..mapped_len).step_by(PAGE) {
            let at = from + k;
            let Some((&base, (size, slice))) = physical.views.range(..=at).next_back() else {
                return false;
            };
            if at - base >= *size {
                return false;
            }
            runs.push((
                Slice {
                    section: slice.section.clone(),
                    index: slice.index + (at - base) / PAGE,
                },
                1,
            ));
        }
    }
    replace_views(&mut physical, to, end, runs)
}

/// Maps a run of a section in the requested virtual range. Windows can replace
/// one exact placeholder per view. Splitting an earlier view leaves separate
/// placeholders on unmap, even when all its physical pages become consecutive again.
fn map(physical: &mut Physical, base: usize, len: usize, slice: &Slice) -> bool {
    let mut offset = 0;
    while offset < len {
        let at = base + offset;
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries the address without dereferencing it.
        let queried =
            unsafe { VirtualQueryEx(GetCurrentProcess(), at as _, &mut info, size_of_val(&info)) };
        if queried == 0
            || info.State != MEM_RESERVE
            || info.Type != windows_sys::Win32::System::Memory::MEM_PRIVATE
        {
            return false;
        }
        let Some(region_end) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            return false;
        };
        let Some(remaining) = region_end.checked_sub(at) else {
            return false;
        };
        let bytes = (len - offset).min(remaining);
        if bytes == 0 || !bytes.is_multiple_of(PAGE) {
            return false;
        }
        if at != info.AllocationBase as usize || bytes != remaining {
            // SAFETY: isolate the requested portion of this placeholder.
            if unsafe { VirtualFree(at as _, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) } == 0 {
                return false;
            }
        }
        let index = slice.index + offset / PAGE;
        // SAFETY: replace exactly this placeholder, using a slice of the live section.
        let view = unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                at as _,
                (index * PAGE) as u64,
                bytes,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        if view.Value.is_null() {
            tracing::warn!(address = at, bytes, error = %std::io::Error::last_os_error(), "cannot map physical view");
            return false;
        }
        // The OS now owns this view, including if commit subsequently fails.
        // The caller's recovery can unmap every piece that actually succeeded.
        physical.views.insert(
            at,
            (
                bytes,
                Slice {
                    section: slice.section.clone(),
                    index,
                },
            ),
        );
        victory_gpu::title_memory_remapped(at, bytes);
        if slice.section.reserved && !commit(at, bytes) {
            return false;
        }
        victory_gpu::title_memory_view(
            at,
            bytes,
            slice.section.handle as usize,
            (index * PAGE) as u64,
        );
        offset += bytes;
    }
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Windows removes an intersecting view in its entirety. The two outer
/// views may extend beyond the request. Interior views are already covered.
/// The caller holds the physical map lock until all surviving ends are back.
fn changing_views(physical: &Physical, base: usize, end: usize) -> (usize, usize) {
    if base == end {
        return (base, 0);
    }
    let first = physical.views.range(..=base).next_back()
        .filter(|(at, (len, _))| **at + *len > base)
        .map_or(base, |(&at, _)| at);
    let last = physical.views.range(..end).next_back()
        .map_or(end, |(&at, (len, _))| end.max(at + len));
    (first, last - first)
}

/// Stop advertising a view only after Windows actually returned it to a placeholder.
fn remove_view(physical: &mut Physical, at: usize) -> bool {
    // Drop any GPU heap opened directly on these pages before asking Windows
    // to unmap them. Keep the physical view record until the call succeeds.
    let (len, _) = physical.views.get(&at).expect("mapped view");
    victory_gpu::title_memory_freeing(at, *len);
    // SAFETY: entries in physical.views are exact host mapping bases.
    let success = unsafe {
        UnmapViewOfFile2(
            GetCurrentProcess(),
            MEMORY_MAPPED_VIEW_ADDRESS { Value: at as _ },
            MEM_PRESERVE_PLACEHOLDER,
        )
    };
    if success == 0 {
        tracing::warn!(address = at, error = %std::io::Error::last_os_error(), "cannot unmap physical view");
        return false;
    }
    let (len, _) = physical.views.remove(&at).expect("mapped view");
    victory_gpu::title_memory_remapped(at, len);
    true
}

/// Replace an interval while excluding readers of the full views Windows must
/// remove. On failure the previous views are restored before releasing that
/// guard. The map always describes the views accepted by Windows, including
/// if the host also refuses a recovery operation.
fn replace_views(
    physical: &mut Physical,
    base: usize,
    end: usize,
    runs: Vec<(Slice, usize)>,
) -> bool {
    let (first, size) = changing_views(physical, base, end);
    let limit = first + size;
    let _changing = victory_gpu::title_memory_remapping_range(first, size);
    let previous: Vec<_> = physical
        .views
        .range(first..limit)
        .map(|(&at, (bytes, slice))| (at, *bytes, slice.clone()))
        .collect();
    let restore = |physical: &mut Physical, views: &[(usize, usize, Slice)]| {
        for (at, bytes, slice) in views {
            if !map(physical, *at, *bytes, slice) {
                tracing::error!(
                    address = *at,
                    bytes = *bytes,
                    "cannot fully restore previous physical view"
                );
            }
        }
    };
    for (i, (at, _, _)) in previous.iter().enumerate() {
        if !remove_view(physical, *at) {
            restore(physical, &previous[..i]);
            return false;
        }
    }
    let apply = |physical: &mut Physical| {
        if first < base && !map(physical, first, base - first, &previous[0].2) {
            return false;
        }
        let mut at = base;
        for (slice, pages) in runs {
            let bytes = pages * PAGE;
            if !map(physical, at, bytes, &slice) {
                return false;
            }
            at += bytes;
        }
        if limit > end {
            let (start, _, slice) = previous.last().expect("intersecting final view");
            let tail = Slice {
                section: slice.section.clone(),
                index: slice.index + (end - start) / PAGE,
            };
            return map(physical, end, limit - end, &tail);
        }
        true
    };
    if apply(physical) {
        return true;
    }
    let partial: Vec<_> = physical
        .views
        .range(first..limit)
        .map(|(&at, _)| at)
        .collect();
    for at in partial {
        if !remove_view(physical, at) {
            tracing::error!(
                address = at,
                "cannot remove replacement view during recovery"
            );
        }
    }
    restore(physical, &previous);
    false
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporary_file_pages_share_writes_and_release_the_file_after_unmapping() {
        let dir = std::env::temp_dir().join(format!("victory-section-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        {
            let slice = Slice { section: Arc::new(Section::new(3 * PAGE, Some(&dir)).unwrap()), index: 1 };
            assert!(!slice.section.reserved);
            let mut first = View::persistent(&slice, 1).unwrap();
            let mut second = View::temporary(&slice, 1).unwrap();
            drop(slice);
            assert!(first.bytes().iter().all(|&value| value == 0));
            first.bytes()[PAGE - 1] = 81;
            assert_eq!(second.bytes()[PAGE - 1], 81);
            drop(first);
            second.bytes()[0] = 15;
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(dir).unwrap();
    }

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // Removing the middle page also removes and reinstalls both
        // neighbours before Windows can map any of these addresses again.
        let start = a as usize;
        assert_eq!(changing_views(&physical(), start + PAGE, start + 2 * PAGE), (start, 3 * PAGE));
        assert_eq!(changing_views(&physical(), start + PAGE, start + PAGE), (start + PAGE, 0));

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A guarded reader queries the first page throughout split and whole-range
    /// replacements. The check remains safe even if a replacement fails.
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        const COUNT: usize = 129;
        let mut pages = [0usize; COUNT];
        let mut n = COUNT;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            COUNT * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, COUNT, pages.as_ptr()), a);
        let first = a as usize;
        let middle = a.wrapping_byte_add(64 * PAGE);
        let last = a.wrapping_byte_add((COUNT - 1) * PAGE);
        let replace = || {
            XMemMapPhysicalPages(middle, 1, pages[64..].as_ptr()) == middle
                && XMemMapPhysicalPages(a, COUNT, pages.as_ptr()) == a
        };
        // Exercise the split and later whole request before starting any reader.
        assert!(replace());
        for (i, address) in [(0, a), (64, middle), (COUNT - 1, last)] {
            assert_eq!(query(address).host.State, MEM_COMMIT);
            let state = physical();
            let (&at, (bytes, slice)) = state.views.range(..=address.addr()).next_back().unwrap();
            assert!(address.addr() - at < *bytes);
            assert_eq!(slice.index + (address.addr() - at) / PAGE, i);
            assert!(Arc::ptr_eq(&slice.section, &state.pages[&pages[i]].section));
        }
        std::thread::scope(|scope| {
            let (ready, started) = std::sync::mpsc::sync_channel(0);
            let (writer, stop) = std::sync::mpsc::channel::<()>();
            let reader = scope.spawn(move || {
                let observed = || {
                    let _reading = victory_gpu::title_memory_reading_range(first, 1);
                    query(first as *const c_void).host.State
                };
                ready.send(observed()).unwrap();
                while matches!(stop.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)) {
                    if observed() != MEM_COMMIT {
                        return false;
                    }
                }
                true
            });
            let initial = started.recv().unwrap();
            let restored = (0..256).all(|_| replace());
            // Disconnecting also wakes the reader when the writer unwinds. No
            // thread dereferences the region, even if the host refuses a map.
            drop(writer);
            let committed = reader.join().unwrap();
            assert_eq!(initial, MEM_COMMIT);
            assert!(restored, "cannot restore mapped physical pages");
            assert!(committed, "reader observed an incomplete mapped interval");
        });
        assert_eq!(XMemMapPhysicalPages(a, COUNT, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(COUNT, pages.as_ptr()), 1);
    }

    #[test]
    fn rejected_replacement_restores_prior_views_and_leaves_other_allocations_intact() {
        let old = allocate(1);
        let new = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, old.as_ptr()), a);
        let other = a.wrapping_byte_add(PAGE);
        // An independent private allocation occupies the second placeholder.
        // SAFETY: our test owns that exact 64-KiB placeholder.
        let private = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                other,
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(private, other);
        assert!(XMemMapPhysicalPages(a, 2, new.as_ptr()).is_null());
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(
            query(other).host.Type,
            windows_sys::Win32::System::Memory::MEM_PRIVATE
        );
        {
            let state = physical();
            let (len, restored) = &state.views[&a.addr()];
            assert_eq!(*len, PAGE);
            assert!(Arc::ptr_eq(
                &restored.section,
                &state.pages[&old[0]].section
            ));
            assert!(!state.views.contains_key(&other.addr()));
        }
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        // SAFETY: private and a are the bases of our distinct host allocations.
        assert_ne!(unsafe { VirtualFree(private, 0, MEM_RELEASE) }, 0);
        assert_ne!(unsafe { VirtualFree(a, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(old.len(), old.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(new.len(), new.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// What a page number is: its section and index in it.
    fn slice_of(page: usize) -> Slice {
        physical().pages.get(&page).cloned().expect("known page")
    }

    /// The bytes of one page, through a view of our own. The temporary path
    /// explicitly, so the tests that existed before the persistent one are
    /// its negative control whatever the environment says.
    fn page_bytes(page: usize) -> Vec<u8> {
        Mapping::Temporary
            .view(&slice_of(page), 1)
            .expect("mapped")
            .bytes()
            .to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// The switch is off unless the environment says yes, so a run that sets
    /// nothing keeps the view per write — which makes every test above the
    /// negative control for the persistent path.
    ///
    /// Mutation run 2026-11-25: returning `true` for `None` or for `"0"`
    /// fails this test and nothing else.
    #[test]
    fn parse_persistent_views_accepts_only_an_explicit_yes() {
        use std::ffi::{OsStr, OsString};
        use std::os::windows::ffi::OsStringExt;
        for yes in ["1", "true", "TRUE", "on", "On", " yes "] {
            assert!(parse_persistent_views(Some(OsStr::new(yes))), "{yes:?}");
        }
        for no in ["", " ", "0", "false", "no", "off", "2", "yes!", "on!"] {
            assert!(!parse_persistent_views(Some(OsStr::new(no))), "{no:?}");
        }
        // Unset is off: the rollout default.
        assert!(!parse_persistent_views(None));
        // Not text at all (a lone surrogate is not UTF-16): off, never a
        // guess, which is the direction that keeps a run on the proven path.
        let lone_surrogate = OsString::from_wide(&[0xd800]);
        assert!(!parse_persistent_views(Some(lone_surrogate.as_os_str())));
    }

    /// A write through the section's one persistent view lands in its pages,
    /// commits each page once, and unmaps nothing when it finishes.
    ///
    /// Mutation runs 2026-11-25, both restored: (a) `mark_committed` made a
    /// no-op fails the `committed_all` assertions below, and nothing else;
    /// (b) the `commit` call removed makes the write fault, which is what
    /// proves the commit is load-bearing rather than decorative.
    #[test]
    fn persistent_views_write_and_commit_once() {
        let pages = allocate(3);
        let section = slice_of(pages[0]).section.clone();
        // Nothing is mapped whole, and no page committed through it, until a
        // write on the persistent path asks.
        assert!(section.persistent.get().is_none());
        assert!(!section.committed_all(0, 3));

        // Two pages from the second half of the first.
        let data: Vec<u8> = (0..PAGE + 8).map(|i| (i * 11 % 251) as u8).collect();
        let offset = PAGE - 8;
        assert!(write_physical_with(Mapping::Persistent, &pages, offset, &data));

        // The persistent path ran, not its fallback: only it maps the
        // section whole and commits through that mapping.
        assert!(
            section.persistent.get().is_some_and(Option::is_some),
            "the section was not mapped whole, so the write fell back to a temporary view"
        );
        let base = section.persistent_ptr().expect("mapped whole");
        assert!(section.committed_all(0, 2), "both written pages committed");
        assert!(!section.committed_all(2, 1), "the unwritten page did not");

        // The bytes are in the section's pages, read through a view of our
        // own: the same bytes a temporary view would have written.
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert_eq!(seen[offset..offset + data.len()], data[..]);
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the write");
        assert!(
            seen[offset + data.len()..].iter().all(|&b| b == 0),
            "after the write"
        );

        // A second write to the same pages commits nothing again and leaves
        // the section's mapping the one it mapped first; a view that
        // unmapped it here would fault on the write above instead.
        assert!(write_physical_with(Mapping::Persistent, &pages, offset, &data));
        assert_eq!(section.persistent_ptr(), Some(base), "the first mapping");
        assert!(section.committed_all(0, 2));

        // Unmapped pages and a bad offset are still refused on this path.
        assert!(!write_physical_with(Mapping::Persistent, &[pages[0], 0x10], 0, &data[..PAGE + 1]));
        assert!(!write_physical_with(Mapping::Persistent, &pages, usize::MAX, &data[..2]));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// A window on the persistent view decodes in place as a temporary one
    /// does, and leaves the section's mapping in place when it drops: the
    /// next window is the same memory, still holding the last one's bytes.
    #[test]
    fn persistent_page_windows_share_one_mapping() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        {
            // Bytes [3, PAGE + 3): the third of the second page on.
            let mut window =
                page_window_with(Mapping::Persistent, &pages, 3, PAGE).expect("one run");
            assert_eq!(window.bytes().len(), PAGE);
            window.bytes().fill(0xa5);
        }
        let base = section.persistent_ptr().expect("mapped whole");
        // The window really went through the section's mapping: only that
        // path commits pages, and a fallback to a temporary view would leave
        // both bits clear while every byte assertion below still held.
        assert!(
            section.committed_all(0, 2),
            "the window fell back to a temporary view"
        );
        // The window dropped, and nothing unmapped: the next window is the
        // same map, and it shows what the first wrote.
        let mut window = page_window_with(Mapping::Persistent, &pages, 0, 4).expect("one run");
        assert_eq!(&window.bytes()[..4], &[0, 0, 0, 0xa5], "the first window's bytes");
        window.bytes()[0] = 0x11;
        drop(window);
        assert_eq!(section.persistent_ptr(), Some(base), "the first mapping");
        assert_eq!(page_bytes(pages[0])[0], 0x11, "the byte just written");
        let first = page_bytes(pages[0]);
        assert!(first[1..3].iter().all(|&b| b == 0));
        assert!(first[3..].iter().all(|&b| b == 0xa5));
        let second = page_bytes(pages[1]);
        assert_eq!(&second[..3], &[0xa5, 0xa5, 0xa5]);
        assert!(second[3..].iter().all(|&b| b == 0));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The temporary path is what every run before this one used, and it
    /// stays exactly that: writes land, and the section is never mapped
    /// whole and never committed through that mapping.
    #[test]
    fn temporary_views_never_map_the_section_whole() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        let data: Vec<u8> = (0..2 * PAGE).map(|i| (i % 253) as u8).collect();
        assert!(write_physical_with(Mapping::Temporary, &pages, 0, &data));
        assert!(
            section.persistent.get().is_none(),
            "the temporary path mapped the section whole"
        );
        assert!(!section.committed_all(0, 2), "it committed through that map");
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert_eq!(seen, data);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The commit guard refuses a range that leaves the section and counts
    /// that overflow, which `runs` never produces but arithmetic can.
    #[test]
    fn ensure_committed_refuses_ranges_outside_the_section() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        let base = section.persistent_ptr().expect("mapped whole");
        assert!(section.ensure_committed(base, 0, 2));
        assert!(section.committed_all(0, 2));
        // All committed: the second call makes none, and says so.
        assert!(section.ensure_committed(base, 0, 2));
        // Past the section, and the two ways adding can overflow.
        assert!(!section.ensure_committed(base, 2, 1));
        assert!(!section.ensure_committed(base, 0, 3));
        assert!(!section.ensure_committed(base, usize::MAX, 1));
        assert!(!section.ensure_committed(base, 0, usize::MAX));
        assert!(!section.ensure_committed(base, usize::MAX / 2, usize::MAX / 2));
        // A range of no pages past the end: refused by the size comparison
        // alone. Delete that comparison and `committed_all` is vacuously
        // true here and the call succeeds, so this pins the clause the host
        // would otherwise supply the answer for.
        assert!(!section.ensure_committed(base, 3, 0));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}
