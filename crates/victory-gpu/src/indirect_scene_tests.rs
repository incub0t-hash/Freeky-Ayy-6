use super::host_signature;
use crate::host::{HostList, host, transition};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

// Each batch is one dispatch of the production argument clamp. Batches share
// the list budget and execute in order, as they would between title draws.
fn read_clamped(list: &HostList, kind: u32, batches: &[&[[u32; 4]]]) -> Vec<[u32; 4]> {
    let host = host().unwrap();
    let records: Vec<_> = batches
        .iter()
        .flat_map(|batch| batch.iter().copied())
        .collect();
    let bytes = std::mem::size_of_val(records.as_slice());
    // A valid dword-aligned indirect offset that is not a raw UAV base.
    let header = 0x1122_3344_5566_7788u64;
    let (args, mapped) = host.shared_buffer((8 + bytes) as u64).unwrap();
    // SAFETY: our buffer has room for the input records, stays live during
    // recording and submission, and is read only after the host fence.
    unsafe {
        mapped.cast::<u64>().write(header);
        std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), mapped.add(8), bytes);
        list.list.ResourceBarrier(&[transition(
            &args,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        let mut offset = 8;
        for batch in batches {
            assert!(crate::indirect::bound(
                list,
                &args,
                args.GetGPUVirtualAddress() + offset,
                (kind, 16, batch.len() as u32),
                None,
            ));
            crate::indirect::done(list, &args, None);
            offset += std::mem::size_of_val(*batch) as u64;
        }
        list.close();
        assert_eq!(host.log_messages(), None, "clamp recording validation");
        host.execute(&[list]);
        host.device.GetDeviceRemovedReason().unwrap();
        assert_eq!(mapped.cast::<u64>().read_volatile(), header);
        std::slice::from_raw_parts(mapped.add(8).cast::<[u32; 4]>(), records.len()).to_vec()
    }
}

#[test]
fn unavailable_pipeline_leaves_budget_for_runnable_draws() {
    let list = host().unwrap().new_list().unwrap();
    let skipped = [6u32, 1, 0, 0];
    let address = skipped.as_ptr() as usize;
    crate::shadow::register(address, address as u64, 16);
    let mut state = super::State::default();
    assert!(!state.execute_indirect(&list, (0, 16, 1), address as u64, None));
    // Two records cost exactly the list budget. Charging even the six
    // vertices above would zero one of them.
    let draws = [[1 << 24, 4, 0, 0]; 2];
    let output = read_clamped(&list, 0, &[&draws]);
    crate::shadow::unregister(address, address as u64);
    assert_eq!(output, draws);
}

#[test]
fn rejected_draws_do_not_wrap_or_consume_the_remaining_budget() {
    let list = host().unwrap().new_list().unwrap();
    let small = [[1, 1, 2, 3]];
    let large = [[1 << 24, 4, 17, 19]];
    let overflow = [large[0]; 96];
    let output = read_clamped(&list, 0, &[&small, &large, &overflow, &small]);
    assert_eq!(output[0], small[0]);
    assert_eq!(output[1], large[0]);
    assert!(output[2..98].iter().all(|record| *record == [0, 4, 17, 19]));
    assert_eq!(output[98], small[0]);
}

#[test]
fn preserves_multidimensional_mesh_dispatches() {
    let list = host().unwrap().new_list().unwrap();
    let input = [
        [4, 3, 2, 17],
        [7, 1, 1, 18],
        [0, 8, 9, 19],
        [65535, 65535, 2, 20],
    ];
    let output = read_clamped(&list, 10, &[&input]);
    assert_eq!(&output[..3], &input[..3]);
    assert_eq!(output[3], [0, 0, 0, 20]);
}

fn compile(source: &str, target: PCSTR) -> Vec<u8> {
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;

    unsafe {
        let result = D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("indirect_scene"),
            None,
            None,
            s!("main"),
            target,
            0,
            0,
            &mut blob,
            Some(&mut errors),
        );

        if let Err(e) = result {
            if let Some(errors) = errors {
                let bytes = std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                );
                panic!("{e}: {}", String::from_utf8_lossy(bytes));
            }
            panic!("{e}");
        }

        let blob = blob.unwrap();

        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    }
}

fn shader(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

#[test]
fn scalar_scene_record_reads_gpu_and_cpu_updates_through_its_exact_raw_window() {
    scene_raw_window(false);
}

#[test]
fn root_selected_direct_srv_reads_cpu_and_gpu_writes_and_preserves_slot_lifetime() {
    scene_raw_window(true);
}

fn scene_raw_window(direct: bool) {
    use super::{ResourceType, RootType, UsedResource};
    use crate::descriptor::{self, View, host::scalar_window};
    let host = host().expect("GPU required");
    let mut words = vec![0u32; 512];
    for (i, w) in words.iter_mut().enumerate() {
        *w = 0x12340000 + i as u32;
    }
    let base = words.as_ptr() as u64;
    let owner = base as usize;
    let address = base + 16;
    let bytes = 456;
    let view = |address: u64, width: u32, elements: u32| {
        let mut resource = [0; 14];
        (resource[0], resource[4]) = (1, width);
        let mut desc = [0; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, 5768, elements, bytes);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, resource, desc)
    };
    let src = view(address, bytes, 1);
    assert!(scalar_window(&src, bytes).is_none());
    crate::shadow::register(owner, base, (words.len() * 4) as u64);
    let s = crate::shadow::find_named(address, bytes.into()).unwrap();
    assert!(scalar_window(&src, bytes).is_some());
    assert!(scalar_window(&view(address + 4, bytes, 1), bytes).is_none());
    assert!(scalar_window(&view(address, bytes, 2), bytes).is_none());
    assert!(scalar_window(&view(address, bytes - 4, 1), bytes).is_none());
    assert!(scalar_window(&view(base + 2000, bytes, 1), bytes).is_none());
    // A shader-visible heap with the same 8-dword table offset the culler uses.
    // Include the extended guest description's optional placement fields.
    let extended_desc = [0u32, 32, 1, 0, 0, 0, 0, 0];
    let null = std::ptr::null();
    let mut guest_heap = null;
    assert_eq!(
        descriptor::create_heap(
            null,
            extended_desc.as_ptr().cast(),
            &crate::resource::PAGEABLE_IIDS[0],
            &mut guest_heap
        ),
        crate::object::S_OK
    );
    let guest = descriptor::heap(guest_heap);
    let View::Placed(false, resource, desc) = &src else {
        unreachable!()
    };
    descriptor::create_placed_srv(null, resource, desc, guest.start + 32);
    let mut direct_desc = *desc;
    direct_desc[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
    direct_desc[6] = bytes / 4;
    direct_desc[7] = 0;
    direct_desc[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
    let direct_handle = guest.start + 17 * 32;
    descriptor::create_placed_srv(null, resource, &direct_desc, direct_handle);
    let cbv_address = [base as u32, (base >> 32) as u32, 256, 0];
    descriptor::create_cbv(null, &cbv_address, guest.start + 12 * 32);
    let small_cbv = [address as u32, (address >> 32) as u32, 32, 0];
    descriptor::create_cbv(null, &small_cbv, guest.start + 13 * 32);
    let mut small_raw = direct_desc;
    small_raw[6] = 8;
    descriptor::create_placed_srv(null, resource, &small_raw, guest.start + 14 * 32);
    let heap = guest.host().unwrap();
    let u = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 8,
        lower_bound: 77,
        stride: bytes,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };

    let producer = compile(
        r#"
        RWByteAddressBuffer target : register(u0);
        [RootSignature("UAV(u0)")][numthreads(1,1,1)]
        void main() { target.Store(0,0x789abcde); }
    "#,
        s!("cs_5_1"),
    );
    let consumer = compile(
        r#"
        ByteAddressBuffer scene : register(t0);
        RWByteAddressBuffer result : register(u0);
        [RootSignature("DescriptorTable(SRV(t0)), UAV(u0)")][numthreads(1,1,1)]
        void main() { [loop] for (uint i=0;i<115;i++) result.Store(i*4,scene.Load(i*4)); }
    "#,
        s!("cs_5_1"),
    );
    unsafe {
        let dev = &host.device;
        let pipeline = |code: &[u8]| {
            let root: ID3D12RootSignature = dev.CreateRootSignature(0, code).unwrap();
            let pso: ID3D12PipelineState = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: shader(code),
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let (writer_root, writer) = pipeline(&producer);
        let (read_root, read) = pipeline(&consumer);
        let (out, mapped) = host.shared_buffer(115 * 4).unwrap();
        for pass in 0..3 {
            let list = host.new_list().unwrap();
            if pass == 0 {
                s.refresh(&list, address, bytes.into());
                s.gpu_writes(&list, address, bytes.into());
                list.list.SetComputeRootSignature(&writer_root);
                list.list.SetPipelineState(&writer);
                list.list
                    .SetComputeRootUnorderedAccessView(0, s.gpu(address));
                list.list.Dispatch(1, 1, 1);
                list.list.ResourceBarrier(&[transition(
                    &s.buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            let mut state = super::State::default();
            state.set_heaps(&[guest_heap]);
            state.set_table(true, 0, guest.start);
            let slot = if direct {
                let slot = state.direct_raw_view(&list, Some(17)).unwrap();
                let tag = victory_shader::emit::SCALAR_RAW_INDEX;
                assert_eq!(state.direct_scalar_view(&list, Some(17)), Some(slot | tag));
                let cbv = state.direct_scalar_view(&list, Some(12)).unwrap();
                assert_eq!(cbv & tag, 0);
                assert_eq!(state.direct_scalar_view(&list, Some(12)), Some(cbv));
                let small = state.direct_raw_view(&list, Some(14)).unwrap();
                assert_eq!(state.direct_scalar_view(&list, Some(13)), Some(small | tag));
                assert_ne!(small, slot);
                assert_ne!(state.direct_scalar_view(&list, Some(1)).unwrap() & tag, 0);
                assert_eq!(state.direct_raw_view(&list, Some(17)), Some(slot));
                let empty = state.direct_raw_view(&list, Some(32)).unwrap();
                assert_eq!(state.direct_scalar_view(&list, None), Some(empty | tag));
                assert_eq!(state.direct_raw_view(&list, None), Some(empty));
                assert_ne!(slot, empty);
                // Recycle the selected guest slot before submission. The
                // recorded command's private alias must retain the old view.
                let mut moved = direct_desc;
                moved[10] = moved[10].wrapping_add(16);
                descriptor::create_placed_srv(null, resource, &moved, direct_handle);
                assert_ne!(state.direct_raw_view(&list, Some(17)), Some(slot));
                descriptor::create_placed_srv(null, resource, &direct_desc, direct_handle);
                slot
            } else {
                if pass == 0 { assert!(state.tables(&list, &[u], true, &[]).is_empty()); }
                let made = state.tables(&list, &[u], true, &[(0, 8)]);
                assert_eq!(made.len(), 1);
                assert_eq!(made[0].0, 0);
                assert_eq!(state.tables(&list, &[u], true, &[(0, 8)]), made);
                made[0].1 / 32 + 1
            };
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&read_root);
            list.list.SetPipelineState(&read);
            let mut handle = heap.heap.GetGPUDescriptorHandleForHeapStart();
            handle.ptr += u64::from(slot)
                * u64::from(
                    dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                );
            list.list.SetComputeRootDescriptorTable(0, handle);
            list.list
                .SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.list.Dispatch(1, 1, 1);
            list.close();
            if pass == 1 {
                for i in 4..118 {
                    words[i] = 0xd00d0000 + i as u32;
                }
            } else if pass == 2 {
                // Reusing the same guest address before submission must not
                // refresh the retained GPU resource with the replacement's bytes.
                crate::shadow::unregister(owner, base);
                crate::shadow::register(owner, base, (words.len() * 4) as u64);
                let replacement = crate::shadow::find_named(address, bytes.into()).unwrap();
                assert!(!std::sync::Arc::ptr_eq(&s, &replacement));
                words[4..118].fill(0xbad0cafe);
            }
            host.execute(&[&list]);
            let got = std::slice::from_raw_parts(mapped.cast::<u32>(), 115);
            for i in 0..114 {
                assert_eq!(
                    got[i],
                    if pass == 0 && i == 0 {
                        0x789abcde
                    } else if pass == 0 {
                        0x12340004 + i as u32
                    } else {
                        0xd00d0004 + i as u32
                    },
                    "pass {pass}, dword {i}"
                );
            }
            assert_eq!(got[114], 0, "the next record is outside this SRV");
            assert_eq!(host.log_messages(), None, "D3D12 validation");
        }
    }
    crate::shadow::unregister(owner, base);
    assert!(scalar_window(&src, bytes).is_none());
    crate::shadow::register(owner, base, 64);
    assert!(scalar_window(&src, bytes).is_none());
    crate::shadow::unregister(owner, base);
    crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(guest_heap, 16)(
        guest_heap,
    );
    eprintln!(
        "Read 114 in-range dwords and one out-of-range dword after both GPU and CPU updates through the production descriptor table"
    );
}

#[test]
fn gpu_generated_draws_reach_depth_tested_framebuffer() {
    indirect_framebuffer(None);
}

#[test]
fn gpu_generated_count_selects_draws_in_separate_or_shared_buffer() {
    indirect_framebuffer(Some(CountLocation::Separate));
    indirect_framebuffer(Some(CountLocation::ArgumentBuffer));
}

#[derive(Clone, Copy)]
enum CountLocation {
    Separate,
    ArgumentBuffer,
}

fn indirect_framebuffer(location: Option<CountLocation>) {
    let host = host().expect("GPU required");
    let dev = &host.device;
    // Record which device actually rendered the fixture.
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    use windows::core::Interface;
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).unwrap();
        let desc = adapter.GetDesc1().unwrap();
        let end = desc
            .Description
            .iter()
            .position(|u| *u == 0)
            .unwrap_or(desc.Description.len());
        eprintln!(
            "D3D12 framebuffer adapter: {}; debug queue available: {}",
            String::from_utf16_lossy(&desc.Description[..end]),
            dev.cast::<ID3D12InfoQueue>().is_ok()
        );
    }

    // Generate the draw arguments on the GPU.
    let cs = compile(
        r#"
        RWByteAddressBuffer args : register(u0);
        RWByteAddressBuffer countBuffer : register(u1);
        cbuffer Count : register(b0) { uint countOffset; uint counted; };
        [RootSignature("UAV(u0), UAV(u1), RootConstants(num32BitConstants=2, b0)")]
        [numthreads(4,1,1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint vertices = id.x == 3 ? 0 : 6;
            // A disabled record beyond the optional count must stay intact.
            uint instances = id.x == 3 && counted ? 0xffffffffu : 1;
            args.Store4(id.x * 16,
                uint4(vertices, instances, id.x * 6, 0));
            if (id.x == 0) countBuffer.Store(countOffset, 2);
        }
    "#,
        s!("cs_5_1"),
    );

    let vs = compile(
        r#"
        struct Output {
            float4 position : SV_Position;
            float4 color : COLOR0;
        };

        [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
        Output main(float3 position : POSITION,
                    float4 color : COLOR0) {
            Output o;
            o.position = float4(position, 1);
            o.color = color;
            return o;
        }
    "#,
        s!("vs_5_1"),
    );

    let ps = compile(
        r#"
        struct Input {
            float4 position : SV_Position;
            float4 color : COLOR0;
        };

        float4 main(Input input) : SV_Target {
            return input.color;
        }
    "#,
        s!("ps_5_1"),
    );

    let mut vertices: Vec<[f32; 7]> = Vec::new();

    let mut quad = |x0, x1, y0, y1, z, c: [f32; 4]| {
        for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x0, y1), (x1, y0), (x1, y1)] {
            vertices.push([x, y, z, c[0], c[1], c[2], c[3]]);
        }
    };

    // Near red surface.
    quad(-0.875, -0.125, -0.5, 0.5, 0.2, [1., 0., 0., 1.]);

    // Far blue surface: must fail the depth test.
    quad(-0.875, -0.125, -0.5, 0.5, 0.8, [0., 0., 1., 1.]);

    // Visible green surface.
    quad(0.125, 0.875, -0.5, 0.5, 0.5, [0., 1., 0., 1.]);

    // Yellow surface: indirect vertex count must be zero.
    quad(-0.25, 0.25, 0.625, 0.875, 0.1, [1., 1., 0., 1.]);

    let list = host.new_list().unwrap();

    let (args, args_cpu) = host.shared_buffer(80).unwrap();
    let separate = matches!(location, Some(CountLocation::Separate))
        .then(|| host.shared_buffer(16).unwrap().0);
    let counter = separate.as_ref().unwrap_or(&args);
    let count_offset = if separate.is_some() { 4 } else { 68 };

    let vb_bytes = vertices.len() * 28;
    let (vb, vb_cpu) = host.shared_buffer(vb_bytes as u64).unwrap();

    unsafe {
        std::ptr::write_bytes(args_cpu, 0, 64);

        std::ptr::copy_nonoverlapping(vertices.as_ptr().cast::<u8>(), vb_cpu, vb_bytes);

        let compute_root: ID3D12RootSignature = dev.CreateRootSignature(0, &cs).unwrap();

        let compute: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&compute_root),
                CS: shader(&cs),
                ..Default::default()
            })
            .unwrap();

        let graphics_root: ID3D12RootSignature = dev.CreateRootSignature(0, &vs).unwrap();

        let inputs = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("COLOR"),
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
        ];

        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&graphics_root),
            VS: shader(&vs),
            PS: shader(&ps),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: inputs.as_ptr(),
                NumElements: inputs.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };

        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // Explicit valid Direct3D defaults, rather than zeroed enums.
        let blend = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };

        desc.BlendState.RenderTarget = [blend; 8];

        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };

        desc.DepthStencilState.FrontFace = stencil;
        desc.DepthStencilState.BackFace = stencil;
        desc.DepthStencilState.StencilReadMask = 0xff;
        desc.DepthStencilState.StencilWriteMask = 0xff;

        desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

        eprintln!("Creating validated graphics pipeline");

        let graphics: ID3D12PipelineState = match dev.CreateGraphicsPipelineState(&desc) {
            Ok(pso) => pso,
            Err(e) => {
                dump_pso_messages(dev);
                panic!("Graphics pipeline creation failed: {e}");
            }
        };

        eprintln!("Graphics pipeline created successfully");

        let make_texture = |format, flags, state| {
            let mut texture: Option<ID3D12Resource> = None;

            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 64,
                    Height: 64,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut texture,
            )
            .unwrap();

            texture.unwrap()
        };

        let color = make_texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );

        let depth = make_texture(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );

        let rtv = host.rtv(&color, None);
        let dsv = host.dsv(&depth, None);

        // GPU-generated arguments.
        list.list.ResourceBarrier(&[
            transition(
                &args,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            transition(
                &vb,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            ),
        ]);

        list.list.SetComputeRootSignature(&compute_root);
        list.list.SetPipelineState(&compute);
        if separate.is_some() {
            list.list.ResourceBarrier(&[transition(
                counter,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        list.list
            .SetComputeRootUnorderedAccessView(1, counter.GetGPUVirtualAddress());
        let constants = [count_offset as u32, u32::from(location.is_some())];
        list.list
            .SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);

        list.list
            .SetComputeRootUnorderedAccessView(0, args.GetGPUVirtualAddress());

        list.list.Dispatch(1, 1, 1);

        // Exercise the actual VIctory argument clamp.
        assert!(crate::indirect::bound(
            &list,
            &args,
            args.GetGPUVirtualAddress(),
            (0, 16, 4),
            location.map(|_| (counter, counter.GetGPUVirtualAddress() + count_offset)),
        ));

        list.list.SetGraphicsRootSignature(&graphics_root);
        list.list.SetPipelineState(&graphics);

        list.list
            .ClearRenderTargetView(rtv, &[0., 0., 0., 1.], None);

        list.list
            .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);

        list.list
            .OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));

        list.list.RSSetViewports(&[D3D12_VIEWPORT {
            Width: 64.,
            Height: 64.,
            MaxDepth: 1.,
            ..Default::default()
        }]);

        list.list.RSSetScissorRects(&[RECT {
            left: 0,
            top: 0,
            right: 64,
            bottom: 64,
        }]);

        list.list
            .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        list.list.IASetVertexBuffers(
            0,
            Some(&[D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: vb.GetGPUVirtualAddress(),
                SizeInBytes: vb_bytes as u32,
                StrideInBytes: 28,
            }]),
        );

        let signature = host_signature(0, 16).expect("Indirect draw signature");

        list.list.ExecuteIndirect(
            &signature,
            4,
            &args,
            0,
            location.map(|_| counter),
            location.map_or(0, |_| count_offset),
        );

        crate::indirect::done(&list, &args, location.map(|_| counter));

        list.list.ResourceBarrier(&[transition(
            &color,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);

        let (readback, footprint) =
            crate::capture::read_back(dev, &list.list, &color).expect("Framebuffer readback");

        list.close();
        assert_eq!(
            host.log_messages(),
            None,
            "framebuffer recording validation"
        );
        host.execute(&[&list]);

        assert!(dev.GetDeviceRemovedReason().is_ok());

        let generated = std::slice::from_raw_parts(args_cpu.cast::<[u32; 4]>(), 4);

        assert_eq!(
            generated,
            &[
                [6, 1, 0, 0],
                [6, 1, 6, 0],
                [6, 1, 12, 0],
                [0, location.map_or(1, |_| u32::MAX), 18, 0],
            ]
        );

        let mut mapped = std::ptr::null_mut();

        readback.Map(0, None, Some(&mut mapped)).unwrap();

        let pitch = footprint.Footprint.RowPitch as usize;

        let pixels = std::slice::from_raw_parts(
            mapped.cast::<u8>().add(footprint.Offset as usize),
            pitch * 64,
        );

        let mut red = 0;
        let mut green = 0;
        let mut black = 0;
        let mut incorrect = 0;

        for y in 0..64usize {
            for x in 0..64usize {
                let at = y * pitch + x * 4;

                let expected: [u8; 4] = if (16..48).contains(&y) && (4..28).contains(&x) {
                    red += 1;
                    [255, 0, 0, 255]
                } else if location.is_none() && (16..48).contains(&y) && (36..60).contains(&x) {
                    green += 1;
                    [0, 255, 0, 255]
                } else {
                    black += 1;
                    [0, 0, 0, 255]
                };

                if pixels[at..at + 4] != expected {
                    incorrect += 1;
                }
            }
        }

        // Save the actual GPU output, even if verification fails.
        let frame = match location {
            None => 1,
            Some(CountLocation::Separate) => 2,
            Some(CountLocation::ArgumentBuffer) => 3,
        };
        crate::capture::save(frame, 64, 64, pitch, false, pixels);

        readback.Unmap(0, None);
        list.reset();

        assert_eq!(incorrect, 0, "Incorrect framebuffer pixels");

        assert_eq!(
            (red, green, black),
            if location.is_some() {
                (768, 0, 3328)
            } else {
                (768, 768, 2560)
            }
        );

        println!(
            "VERIFIED GPU SCENE: 4096 exact pixels; \
             red={red}, green={green}, background={black}"
        );
    }
}

fn dump_pso_messages(device: &ID3D12Device) {
    use windows::core::Interface;

    let Ok(queue) = device.cast::<ID3D12InfoQueue>() else {
        eprintln!(
            "D3D12 InfoQueue unavailable. \
             Windows Graphics Tools may be missing."
        );
        return;
    };

    unsafe {
        let count = queue.GetNumStoredMessages();
        eprintln!("D3D12 validation messages: {count}");

        for i in 0..count {
            let mut size = 0usize;

            if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                continue;
            }

            let mut buffer = vec![0u64; size.div_ceil(8)];
            let ptr = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();

            if queue.GetMessage(i, Some(ptr), &mut size).is_ok() {
                let msg = &*ptr;

                if !msg.pDescription.is_null() {
                    let text =
                        std::slice::from_raw_parts(msg.pDescription, msg.DescriptionByteLength);

                    eprintln!(
                        "D3D12 ID={} severity={}: {}",
                        msg.ID.0,
                        msg.Severity.0,
                        String::from_utf8_lossy(text).trim_end_matches('\0')
                    );
                }
            }
        }
    }
}
