use crate::host::CardHash;

const PAGE: usize = 65536;

#[derive(Clone, Copy)]
struct Views {
    now: u64,
    hashes: u64,
    flags: u64,
}

impl Views {
    fn bind(self, list: &ID3D12GraphicsCommandList, kernel: &Clamp) {
        // SAFETY: these are the live views in the card kernel's root signature.
        unsafe {
            list.SetComputeRootSignature(&kernel.root);
            list.SetPipelineState(&kernel.pso);
            list.SetComputeRootShaderResourceView(1, self.now);
            list.SetComputeRootUnorderedAccessView(2, self.hashes);
            list.SetComputeRootUnorderedAccessView(3, self.flags);
        }
    }
}

fn dispatch(list: &ID3D12GraphicsCommandList, mode: CardHash, constants: [u32; 6]) {
    // SAFETY: caller bound the card kernel and enough source, signature and result storage.
    unsafe {
        list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        list.Dispatch(mode.groups(constants[3]), 1, 1);
    }
}

fn check_pieces(harness: &mut WarpHarness, skew: usize) {
    let mode = CardHash::Cooperative;
    let kernel = compile_cs(&harness.device, mode.source()).unwrap();
    let size = PAGE * 2 + 977;
    let pieces = (skew + size).div_ceil(PAGE);
    let content: Vec<u8> = (0..size).map(|i| (i * 29 + i / 113) as u8).collect();
    let mut now = content.clone();
    let mut before = content.clone();
    let (_input, input, gpu) = harness.shared_buffer(size.next_multiple_of(16) as u64);
    let (_signature, signatures, hashes) =
        harness.shared_buffer(mode.signature_bytes(pieces as u64));
    let (_flags, mapped, flags) = harness.shared_buffer((pieces as u64 + 4) * 4);
    let views = Views {
        now: gpu,
        hashes,
        flags,
    };
    let mut verify = |now: &[u8], first: usize, count: usize, operation: u32| {
        let mut expected = vec![7; pieces + 4];
        for piece in first..first + count {
            let lo = (piece * PAGE).saturating_sub(skew);
            let hi = ((piece + 1) * PAGE - skew).min(size);
            if operation != 0 {
                expected[2 + piece - first] = u32::from(before[lo..hi] != now[lo..hi]);
            }
            if operation != 1 {
                before[lo..hi].copy_from_slice(&now[lo..hi]);
            }
        }
        // SAFETY: the preceding submission is complete. Both mappings hold
        // the indicated lengths, and the comparison below follows a fence.
        unsafe {
            std::ptr::copy_nonoverlapping(now.as_ptr(), input, now.len());
            for i in 0..expected.len() {
                mapped.cast::<u32>().add(i).write_volatile(7);
            }
        }
        views.bind(&harness.list, &kernel);
        dispatch(
            &harness.list,
            mode,
            [
                first as u32,
                skew as u32,
                size as u32,
                count as u32,
                8,
                operation,
            ],
        );
        harness.submit_and_wait();
        // SAFETY: the fenced result buffer remains mapped.
        let actual: Vec<_> = (0..expected.len())
            .map(|i| unsafe { mapped.cast::<u32>().add(i).read_volatile() })
            .collect();
        assert_eq!(
            actual, expected,
            "skew={skew} first={first} pieces={count} operation={operation}"
        );
    };
    verify(&now, 0, pieces, 0);
    verify(&now, 0, pieces, 1);
    // The two bytes on each side of an unaligned piece boundary must
    // affect only their own piece. Include both allocation edges.
    let boundary = PAGE - skew;
    // A change inside lane 13's stripe must replace exactly its 128-bit
    // signature. Each lane retains its own history across relay passes.
    let read_signatures = || {
        // SAFETY: the preceding submission is fenced and all pieces have
        // 64 signatures, each initialized by mode zero.
        unsafe { std::slice::from_raw_parts(signatures.cast::<[u32; 4]>(), pieces * 64).to_vec() }
    };
    let previous = read_signatures();
    let byte = (boundary & !15) + (64 + 13) * 16;
    now[byte] ^= 0x10;
    verify(&now, 1, 1, 2);
    let current = read_signatures();
    let changed: Vec<_> = previous
        .iter()
        .zip(&current)
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect();
    assert_eq!(changed, [64 + 13]);
    now[byte] = content[byte];
    verify(&now, 1, 1, 2);
    assert_eq!(previous, read_signatures());
    for byte in [0, boundary - 1, boundary, boundary + 1, size - 1] {
        now[byte] ^= 0x80;
        verify(&now, 0, pieces, 1);
        verify(&now, 0, pieces, 2);
        verify(&now, 0, pieces, 2);
        now[byte] = content[byte];
        verify(&now, 0, pieces, 2);
    }
    // An isolated change in every lane must be observable. Vary which of
    // the sixteen source bytes changes, and repeat with a nonzero first piece.
    for lane in 0..64 {
        let byte = boundary.next_multiple_of(16) + (64 + lane) * 16 + lane % 16;
        now[byte] ^= 0x20;
        verify(&now, 1, 1, 1);
        now[byte] = content[byte];
    }
    // Change an entire piece and restore it. Mode 2 advances the baseline
    // in either direction. Mode 1 leaves that baseline untouched.
    now[boundary..boundary + PAGE].fill(0xad);
    verify(&now, 1, 1, 2);
    verify(&content, 1, 1, 1);
    verify(&content, 1, 1, 2);
    verify(&content, 1, 1, 2);
}

#[test]
fn cooperative_card_hash_matches_bytewise_changes_on_partial_pieces() {
    let Some(mut harness) = WarpHarness::new() else {
        return;
    };
    for skew in [0, 27, PAGE - 13] {
        check_pieces(&mut harness, skew);
    }
}

struct Workload {
    size: u32,
    skew: u32,
    runs: Vec<(u32, u32)>,
}

impl Workload {
    fn from_line(line: &str) -> Option<Self> {
        let mut fields = line.split_once("CARD_HASH_SPANS=v1,")?.1.trim().split(',');
        let skew: u32 = fields.next()?.parse().ok()?;
        let size: u32 = fields.next()?.parse().ok()?;
        if skew >= PAGE as u32 || !(1..=128 << 20).contains(&size) {
            return None;
        }
        let pieces = (skew + size).div_ceil(PAGE as u32);
        let mut runs = Vec::new();
        for field in fields {
            let (first, count) = field.split_once(':')?;
            let first: u32 = first.parse().ok()?;
            let count: u32 = count.parse().ok()?;
            if count == 0 || first.checked_add(count)? > pieces || runs.len() >= 2048 {
                return None;
            }
            runs.push((first, count));
        }
        (!runs.is_empty()).then_some(Self { size, skew, runs })
    }

    fn measured_frame_counts() -> Self {
        // The first visible-world run reported 509 rehash dispatches covering
        // 1388 pieces. The individual positions here are synthetic; a capture
        // supplied to the timing test replays its actual positions instead.
        let mut runs: Vec<_> = (0..509).map(|i| ((i * 691) % 1016, 1)).collect();
        let mut seed = 1664525u32;
        for _ in 509..1388 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            runs[seed as usize % 509].1 += 1;
        }
        Self {
            size: 64 << 20,
            skew: 0,
            runs,
        }
    }
}

#[test]
fn card_hash_replay_checks_every_captured_span() {
    assert_eq!(
        Workload::from_line("info CARD_HASH_SPANS=v1,4,65537,1:1")
            .unwrap()
            .runs,
        [(1, 1)]
    );
    for text in [
        "v1,0,0,0:1",
        "v1,65536,65537,0:1",
        "v1,0,65536,1:1",
        "v1,0,65536,0:0",
        "v1,4,65537,4294967295:2",
        "v1,4,65537,1:1,2:1",
    ] {
        assert!(Workload::from_line(&format!("CARD_HASH_SPANS={text}")).is_none());
    }
}

/// GPU timestamp queries exclude recording and fence-wait time. Use actual
/// device-local source and signature resources, as the production path does.
#[test]
#[ignore = "requires an explicitly selected hardware adapter, runs both hashing kernels"]
fn time_card_hash_kernels_on_sparse_gpu_spans() {
    let wanted = std::env::var("VICTORY_ADAPTER").expect("select the hardware by name");
    let host = crate::host::host().expect("hardware D3D12 device");
    // SAFETY: read the identity of the same adapter the device uses.
    let adapter_name = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        assert_eq!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        String::from_utf16_lossy(
            &desc.Description[..desc.Description.iter().position(|&c| c == 0).unwrap_or(128)],
        )
    };
    assert!(
        adapter_name
            .to_ascii_lowercase()
            .contains(&wanted.to_ascii_lowercase()),
        "{adapter_name}"
    );
    let mut harness = WarpHarness::from_device(host.device.clone()).unwrap();
    check_pieces(&mut harness, PAGE - 13);
    let supplied = std::env::var_os("VICTORY_TEST_CARD_HASH_SPANS");
    let workload = match &supplied {
        Some(path) => std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter_map(Workload::from_line)
            .max_by_key(|w| w.runs.len())
            .expect("no valid captured card hash spans"),
        None => Workload::measured_frame_counts(),
    };
    let pieces = (workload.size + workload.skew).div_ceil(PAGE as u32);
    let covered: u32 = workload.runs.iter().map(|r| r.1).sum();
    let size = u64::from(workload.size).next_multiple_of(16);
    let (staging, ptr) = host.upload_buffer(size).unwrap();
    let now = harness.default_uav(size);
    let modes = [CardHash::Serial, CardHash::Cooperative];
    let kernels = modes.map(|m| compile_cs(&harness.device, m.source()).unwrap());
    let signatures = modes.map(|m| harness.default_uav(m.signature_bytes(pieces.into())));
    let (_flags, _, flags) = harness.shared_buffer(u64::from(covered) * 4);
    let readback = host.read_back_buffer(16).unwrap();
    let mut query = None;
    // SAFETY: upload and destination are sized for the complete padded
    // source; GPU command recording and readback below are fenced.
    unsafe {
        let words = std::slice::from_raw_parts_mut(ptr.cast::<u32>(), size as usize / 4);
        for (i, word) in words.iter_mut().enumerate() {
            *word = (i as u32).wrapping_mul(0x9e3779b1) ^ ((i >> 5) as u32);
        }
        harness.list.CopyBufferRegion(&now, 0, &staging, 0, size);
        harness.submit_and_wait();
        harness
            .device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut query,
            )
            .unwrap();
    }
    let query: ID3D12QueryHeap = query.unwrap();
    // SAFETY: these buffers and query heap outlive all submissions. A fence
    // precedes each timestamp read, and the two signatures never alias.
    unsafe {
        let freq = harness.queue.GetTimestampFrequency().unwrap() as f64;
        let views = signatures.each_ref().map(|s| Views {
            now: now.GetGPUVirtualAddress(),
            hashes: s.GetGPUVirtualAddress(),
            flags,
        });
        for i in 0..2 {
            views[i].bind(&harness.list, &kernels[i]);
            dispatch(
                &harness.list,
                modes[i],
                [0, workload.skew, workload.size, pieces, 0, 0],
            );
            harness.submit_and_wait();
        }
        for operation in [0, 2] {
            let mut times = [Vec::new(), Vec::new()];
            for trial in 0..7 {
                for i in [trial % 2, 1 - trial % 2] {
                    views[i].bind(&harness.list, &kernels[i]);
                    harness.list.EndQuery(&query, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    let mut out_at = 0;
                    for &(first, count) in &workload.runs {
                        dispatch(
                            &harness.list,
                            modes[i],
                            [
                                first,
                                workload.skew,
                                workload.size,
                                count,
                                out_at,
                                operation,
                            ],
                        );
                        out_at += count * 4;
                    }
                    harness.list.EndQuery(&query, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    harness.list.ResolveQueryData(
                        &query,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        0,
                        2,
                        &readback,
                        0,
                    );
                    harness.submit_and_wait();
                    let mut mapping = std::ptr::null_mut();
                    readback.Map(0, None, Some(&mut mapping)).unwrap();
                    let pair = std::slice::from_raw_parts(mapping.cast::<u64>(), 2);
                    let ms = (pair[1] - pair[0]) as f64 * 1000.0 / freq;
                    readback.Unmap(0, None);
                    if trial > 0 {
                        times[i].push(ms);
                    }
                }
            }
            for time in &mut times {
                time.sort_by(f64::total_cmp);
            }
            let median = times.map(|time| (time[2] + time[3]) / 2.0);
            eprintln!(
                "adapter={adapter_name} captured={} mode={operation} dispatches={} pieces={covered} serial_ms={:.3} cooperative_ms={:.3} speedup={:.2}",
                supplied.is_some(),
                workload.runs.len(),
                median[0],
                median[1],
                median[0] / median[1]
            );
        }
    }
}
