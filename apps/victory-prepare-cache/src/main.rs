use anyhow::{Result, ensure};
use std::{collections::BTreeSet, env, fs, os::windows::io::AsRawHandle, path::{Path, PathBuf}};
use windows_sys::Win32::{Foundation::CloseHandle, System::Memory::*};
include!(concat!(env!("OUT_DIR"), "/include.rs"));

struct ReadMap(MEMORY_MAPPED_VIEW_ADDRESS, usize);

impl ReadMap {
    fn open(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)?;
        let n = usize::try_from(file.metadata()?.len())?;
        ensure!(n > 0 && n <= isize::MAX as usize, "unsupported input size");
        // SAFETY: read-only file, live handle and a read-only view mapping the complete file.
        unsafe {
            let handle = CreateFileMappingW(file.as_raw_handle(), std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null());
            ensure!(!handle.is_null(), "mapping: {}", std::io::Error::last_os_error());
            let view = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
            let error = std::io::Error::last_os_error();
            CloseHandle(handle);
            ensure!(!view.Value.is_null(), "mapping view: {error}");
            Ok(Self(view, n))
        }
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: this object keeps the read-only mapping alive until Drop.
        unsafe { std::slice::from_raw_parts(self.0.Value.cast(), self.1) }
    }
}
impl Drop for ReadMap {
    fn drop(&mut self) {
        // SAFETY: this is the view returned to this instance.
        unsafe { UnmapViewOfFile(self.0); }
    }
}



fn configure_host() -> Result<(u32, bool)> {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: matches the renderer's default adapter and feature level.
    unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)? };
    let device = device.ok_or_else(|| anyhow::anyhow!("No Direct3D 12 device"))?;
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: a correctly sized feature-query output.
    unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut options).cast(), size_of_val(&options) as u32)? };
    ensure!(options.WaveLaneCountMax > 0, "The adapter does not expose wave operations");
    victory_shader::emit::set_host_max_wave(options.WaveLaneCountMax);
    // SAFETY: the same probe used by the renderer, released immediately.
    let heap: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1_000_001,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        })
    };
    let capped = heap.is_err();
    if capped { victory_shader::emit::set_view_heap_limit(999999); }
    Ok((options.WaveLaneCountMax, capped))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    ensure!(args.len() >= 2, "probe | seed renderer cache witness source...");
    if args[1] == "probe" {
        println!("{:?}", configure_host()?);
        return Ok(());
    }
    ensure!(args.len() >= 6, "seed renderer cache witness source...");
    println!("host translation configuration: {:?}", configure_host()?);
    host_cache::use_shader_cache(PathBuf::from(&args[3]));
    let raw=fs::read_to_string(&args[4])?;
    let requested:BTreeSet<_>=raw.lines().map(|l| l.split_once('\t').unwrap().0).collect();
    let mut seen=BTreeSet::new();
    let mut passed=0u64;
    let mut record=|data:&[u8]| -> Result<()> {
        let hash=blake3::hash(data).to_hex();
        let id=&hash.as_str()[..16];
        if !requested.contains(id) || !seen.insert(id.to_string()){return Ok(())}
        match host_cache::compiled(data) {
            Ok(bytes)=>{ensure!(host_cache::compiled(data).as_ref()==Ok(&bytes)); passed+=1;},
            Err(reason)=>eprintln!("refused {id} {reason}"),
        }
        Ok(())
    };
    for path in &args[5..] {
        let map = ReadMap::open(Path::new(path))?;
        let raw = map.bytes();
        for r in victory_game::sgd5::records(raw) {
            if let Some(data) = r.unpack() { record(&data)?; }
        }
        for p in memchr::memmem::find_iter(raw,b"DXBC") {
            if let Some(n) = raw.get(p+24..p+28).and_then(|v| v.try_into().ok()).map(u32::from_le_bytes)
                && (32..=64*1024*1024).contains(&n)
                && let Some(data) = raw.get(p..p+n as usize) { record(data)?; }
        }
        eprintln!("finished native archive {path}");
    }
    let (hits,misses)=host_cache::stats();
    ensure!(hits>=passed && seen.len()==requested.len());
    println!("found={} verified={passed} hits={hits} misses={misses}", seen.len());
    Ok(())
}
