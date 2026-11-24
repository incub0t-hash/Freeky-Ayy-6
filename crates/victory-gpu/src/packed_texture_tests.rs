use super::*;
use crate::host::{Host, HostList};
use windows::Win32::Graphics::Direct3D12::*;

fn image(address: usize, desc: &ResourceDesc) -> This {
    let mut image = std::ptr::null();
    assert_eq!(
        create_at_address(
            std::ptr::null(),
            address as u64,
            desc,
            0,
            std::ptr::null(),
            &RESOURCE_IIDS[0],
            &mut image
        ),
        S_OK
    );
    image
}

fn layers() -> ResourceDesc {
    ResourceDesc {
        dimension: 3,
        width: 9,
        height: 5,
        depth_or_array_size: 2,
        mip_levels: 3,
        format: 42,
        flags: 4,
        alignment: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
    }
}

fn sample_desc(src: ResourceDesc) -> ResourceDesc {
    ResourceDesc {
        format: 67,
        flags: 0,
        ..src
    }
}

#[test]
fn packed_view_requires_one_live_writer_and_matching_shape_and_tiles() {
    let Some(_) = crate::host::host() else { return };
    let backing = vec![0u8; 65536];
    let address = backing.as_ptr() as usize;
    let desc = layers();
    let src = image(address, &desc);
    let sample = sample_desc(desc);
    let view = image(address, &sample);
    assert!(packed_writer(view).is_some());
    for alternate in [
        ResourceDesc { width: 8, ..sample },
        ResourceDesc {
            mip_levels: 2,
            ..sample
        },
        ResourceDesc {
            depth_or_array_size: 1,
            ..sample
        },
        ResourceDesc {
            layout: 0,
            ..sample
        },
        ResourceDesc {
            format: 41,
            ..sample
        },
    ] {
        let wrong = image(address, &alternate);
        assert!(packed_writer(wrong).is_none());
        resource_release(wrong);
    }
    let beside = image(address + 4, &sample);
    assert!(packed_writer(beside).is_none());
    resource_release(beside);
    for start in [address - 4, address, address + 4] {
        let overlap = image(start, &desc);
        assert!(packed_writer(view).is_none(), "overlap at {start:#x}");
        resource_release(overlap);
    }
    assert!(packed_writer(view).is_some());
    resource_release(src);
    assert!(packed_writer(view).is_none());
    resource_release(view);
    let wrapping = image(usize::MAX - 3, &desc);
    let out_of_bounds = image(usize::MAX - 3, &sample);
    assert!(packed_writer(out_of_bounds).is_none());
    resource_release(wrapping);
    resource_release(out_of_bounds);
}

fn location(image: &ID3D12Resource, sub: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed resource pointer, live through execute.
        pResource: unsafe { std::mem::transmute_copy(image) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: sub,
        },
    }
}

fn footprint(
    buffer: &ID3D12Resource,
    fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as in location.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: fp,
        },
    }
}

fn footprints(
    host: &Host,
    image: This,
    count: u32,
) -> (Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, u64) {
    let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut bytes = 0;
    // SAFETY: the output has one entry per requested subresource.
    unsafe {
        host.device.GetCopyableFootprints(
            &host_resource(image).unwrap().GetDesc(),
            0,
            count,
            0,
            Some(fps.as_mut_ptr()),
            None,
            None,
            Some(&mut bytes),
        );
    }
    (fps, bytes)
}

fn each_word(
    fps: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    pass: u32,
    mut visit: impl FnMut(usize, u32),
) {
    for (sub, fp) in fps.iter().enumerate() {
        let f = fp.Footprint;
        for z in 0..f.Depth {
            for y in 0..f.Height {
                for x in 0..f.Width {
                    let at =
                        fp.Offset as usize + ((z * f.Height + y) * f.RowPitch + 4 * x) as usize;
                    let word = ((13 + pass) << 27)
                        | (sub as u32 + 1) << 18
                        | (z + y + 1) << 9
                        | (x + pass);
                    visit(at, word);
                }
            }
        }
    }
}

fn read_buffer(host: &Host, desc: &D3D12_RESOURCE_DESC) -> ID3D12Resource {
    let mut back = None;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    // SAFETY: a host buffer description previously allocated with this size.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )
            .unwrap()
    }
    back.unwrap()
}

fn upload(
    list: &HostList,
    src: This,
    staging: &ID3D12Resource,
    fps: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
) {
    crate::list::ensure_state(list, src, 0x400);
    for (sub, &fp) in fps.iter().enumerate() {
        let to = location(host_resource(src).unwrap(), sub as u32);
        let from = footprint(staging, fp);
        // SAFETY: valid per-subresource footprints of the live upload buffer.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) }
    }
}

fn verify_shape(desc: ResourceDesc) {
    let Some(host) = crate::host::host() else {
        return;
    };
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    // SAFETY: query the adapter actually backing this device.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        adapter.GetDesc1().unwrap()
    };
    let end = adapter
        .Description
        .iter()
        .position(|c| *c == 0)
        .unwrap_or(adapter.Description.len());
    eprintln!(
        "D3D12 adapter: {}",
        String::from_utf16_lossy(&adapter.Description[..end])
    );
    let backing = vec![0u8; 65536];
    let address = backing.as_ptr() as usize;
    let src = image(address, &desc);
    let temp_memory = vec![0u8; 65536];
    let temp = image(temp_memory.as_ptr() as usize, &desc);
    let dst = image(address, &sample_desc(desc));
    let count = u32::from(desc.mip_levels)
        * if desc.dimension == TEXTURE_3D {
            1
        } else {
            u32::from(desc.depth_or_array_size)
        };
    let (src_fps, bytes) = footprints(host, src, count);
    let (dst_fps, _) = footprints(host, dst, count);
    let producer = host.new_list().unwrap();
    let consumer = host.new_list().unwrap();
    let mut readbacks = vec![];
    for pass in 0..2 {
        let (staging, mapped) = host.upload_buffer(bytes).unwrap();
        each_word(&src_fps, pass, |at, word| {
            assert!(at + 4 <= bytes as usize);
            // SAFETY: every address is within this mapped upload buffer.
            unsafe { mapped.add(at).cast::<u32>().write_unaligned(word) };
        });
        let list = if pass == 0 { &producer } else { &consumer };
        list.keep(&staging);
        upload(list, temp, &staging, &src_fps);
        assert!(crate::list::copy_host_texture(&consumer, src, temp));
        assert!(copy_packed_view(&consumer, dst));
        crate::list::ensure_state(&consumer, dst, 0xc0);
        assert!(copy_packed_view(&consumer, dst));
        crate::list::ensure_state(&consumer, dst, 0x800);
        // SAFETY: the staging buffer has the exact readback size.
        let back = read_buffer(host, &unsafe { staging.GetDesc() });
        for (sub, &fp) in dst_fps.iter().enumerate() {
            let to = footprint(&back, fp);
            let from = location(host_resource(dst).unwrap(), sub as u32);
            // SAFETY: per-subresource footprints returned for the destination.
            unsafe { consumer.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) }
        }
        readbacks.push(back);
    }
    producer.close();
    consumer.close();
    resource_release(src);
    resource_release(dst);
    resource_release(temp);
    host.execute(&[&producer, &consumer]);
    let mut words = 0;
    for (pass, back) in readbacks.iter().enumerate() {
        let mut mapped = std::ptr::null_mut();
        // SAFETY: submission has completed; all offsets lie in the footprint.
        unsafe { back.Map(0, None, Some(&mut mapped)).unwrap() };
        each_word(&dst_fps, pass as u32, |at, word| {
            assert!(at + 4 <= bytes as usize);
            let got = unsafe { mapped.cast::<u8>().add(at).cast::<u32>().read_unaligned() };
            assert_eq!(got, word, "pass {pass}, offset {at}");
            words += 1;
        });
        unsafe { back.Unmap(0, Some(&D3D12_RANGE::default())) };
    }
    eprintln!("R32_UINT to RGB9E5: {count} subresources, {words} packed texels across two writes");
}

#[test]
fn packed_view_cpu_refresh_wins_after_an_intervening_guest_write() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let mut backing = vec![0x5au8; 65536];
    let update = vec![0x2du8; backing.len()];
    let temp_memory = vec![0u8; backing.len()];
    let desc = ResourceDesc {
        mip_levels: 1,
        depth_or_array_size: 1,
        ..layers()
    };
    let src = image(backing.as_ptr() as usize, &desc);
    let view = image(backing.as_ptr() as usize, &sample_desc(desc));
    let temp = image(temp_memory.as_ptr() as usize, &desc);
    let completed = host.new_list().unwrap();
    let recorded = host.new_list().unwrap();
    let list = host.new_list().unwrap();
    assert!(
        !copy_packed_view(&list, view),
        "allocated but never written"
    );
    assert!(crate::list::copy_host_texture(&completed, src, temp));
    completed.close();
    host.execute(&[&completed]);
    assert!(crate::list::copy_host_texture(&recorded, src, temp));
    assert!(
        !copy_packed_view(&list, view),
        "neither an older submission nor an unsubmitted list owns this read"
    );
    crate::list::ensure_state(&list, src, 0x8);
    assert!(
        !copy_packed_view(&list, view),
        "a UAV transition alone writes nothing"
    );
    let dst = host_resource(view).unwrap();
    crate::texture::upload(&list, view, dst);
    assert!(crate::list::copy_host_texture(&list, src, temp));
    assert!(copy_packed_view(&list, view));
    list.defer_copy(backing.as_mut_ptr(), update.as_ptr(), update.len());
    assert!(!copy_packed_view(&list, view));
    crate::texture::upload(&list, view, dst);
    crate::list::ensure_state(&list, view, 0x800);
    let (fps, bytes) = footprints(host, view, 1);
    let (shape, _) = host.upload_buffer(bytes).unwrap();
    // SAFETY: upload and readback buffers have the same size.
    let back = read_buffer(host, &unsafe { shape.GetDesc() });
    let (to, from) = (footprint(&back, fps[0]), location(dst, 0));
    unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    list.close();
    host.execute(&[&list]);
    let mut mapped = std::ptr::null_mut();
    unsafe { back.Map(0, None, Some(&mut mapped)).unwrap() };
    each_word(&fps, 0, |at, _| {
        let got = unsafe { mapped.cast::<u8>().add(at).cast::<u32>().read_unaligned() };
        assert_eq!(got, 0x2d2d2d2d, "final CPU update at {at}");
    });
    unsafe { back.Unmap(0, Some(&D3D12_RANGE::default())) };
    resource_release(src);
    resource_release(temp);
    resource_release(view);
}

#[test]
fn packed_view_copies_all_array_layers_and_mips_on_the_gpu() {
    verify_shape(layers())
}

#[test]
fn packed_view_copies_all_volume_slices_and_mips_on_the_gpu() {
    verify_shape(ResourceDesc {
        dimension: TEXTURE_3D,
        depth_or_array_size: 4,
        ..layers()
    })
}
