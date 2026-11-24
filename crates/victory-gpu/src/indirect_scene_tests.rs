use super::host_signature;
use crate::host::{host, transition};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{s, PCSTR};

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

        std::slice::from_raw_parts(
            blob.GetBufferPointer().cast::<u8>(),
            blob.GetBufferSize(),
        ).to_vec()
    }
}

fn shader(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

#[test]
fn gpu_generated_draws_reach_depth_tested_framebuffer() {
    let host = host().expect("GPU required");
    let dev = &host.device;

    // Generate the draw arguments on the GPU.
    let cs = compile(r#"
        RWByteAddressBuffer args : register(u0);

        [RootSignature("UAV(u0)")]
        [numthreads(4,1,1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint vertices = id.x == 3 ? 0 : 6;
            args.Store4(id.x * 16,
                uint4(vertices, 1, id.x * 6, 0));
        }
    "#, s!("cs_5_1"));

    let vs = compile(r#"
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
    "#, s!("vs_5_1"));

    let ps = compile(r#"
        struct Input {
            float4 position : SV_Position;
            float4 color : COLOR0;
        };

        float4 main(Input input) : SV_Target {
            return input.color;
        }
    "#, s!("ps_5_1"));

    let mut vertices: Vec<[f32; 7]> = Vec::new();

    let mut quad = |x0, x1, y0, y1, z, c: [f32; 4]| {
        for (x,y) in [
            (x0,y0), (x1,y0), (x0,y1),
            (x0,y1), (x1,y0), (x1,y1),
        ] {
            vertices.push([x,y,z,c[0],c[1],c[2],c[3]]);
        }
    };

    // Near red surface.
    quad(-0.875,-0.125,-0.5,0.5,0.2,[1.,0.,0.,1.]);

    // Far blue surface: must fail the depth test.
    quad(-0.875,-0.125,-0.5,0.5,0.8,[0.,0.,1.,1.]);

    // Visible green surface.
    quad(0.125,0.875,-0.5,0.5,0.5,[0.,1.,0.,1.]);

    // Yellow surface: indirect vertex count must be zero.
    quad(-0.25,0.25,0.625,0.875,0.1,[1.,1.,0.,1.]);

    let list = host.new_list().unwrap();

    let (args, args_cpu) = host.shared_buffer(64).unwrap();

    let vb_bytes = vertices.len() * 28;
    let (vb, vb_cpu) = host.shared_buffer(vb_bytes as u64).unwrap();

    unsafe {
        std::ptr::write_bytes(args_cpu, 0, 64);

        std::ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            vb_cpu,
            vb_bytes,
        );

        let compute_root: ID3D12RootSignature =
            dev.CreateRootSignature(0, &cs).unwrap();

        let compute: ID3D12PipelineState =
            dev.CreateComputePipelineState(
                &D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(
                        &compute_root
                    ),
                    CS: shader(&cs),
                    ..Default::default()
                }
            ).unwrap();

        let graphics_root: ID3D12RootSignature =
            dev.CreateRootSignature(0, &vs).unwrap();

        let inputs = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlotClass:
                    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("COLOR"),
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                AlignedByteOffset: 12,
                InputSlotClass:
                    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
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
            PrimitiveTopologyType:
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
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
            RenderTargetWriteMask:
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
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

        desc.RasterizerState.ConservativeRaster =
            D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

        eprintln!("Creating validated graphics pipeline");

        let graphics: ID3D12PipelineState =
            match dev.CreateGraphicsPipelineState(&desc) {
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
            ).unwrap();

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

        list.list.SetComputeRootUnorderedAccessView(
            0, args.GetGPUVirtualAddress()
        );

        list.list.Dispatch(1,1,1);

        // Exercise the actual VIctory argument clamp.
        assert!(crate::indirect::bound(
            &list,
            &args,
            args.GetGPUVirtualAddress(),
            (0,16,4),
        ));

        list.list.SetGraphicsRootSignature(&graphics_root);
        list.list.SetPipelineState(&graphics);

        list.list.ClearRenderTargetView(
            rtv, &[0.,0.,0.,1.], None
        );

        list.list.ClearDepthStencilView(
            dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None
        );

        list.list.OMSetRenderTargets(
            1, Some(&rtv), false, Some(&dsv)
        );

        list.list.RSSetViewports(&[D3D12_VIEWPORT {
            Width: 64.,
            Height: 64.,
            MaxDepth: 1.,
            ..Default::default()
        }]);

        list.list.RSSetScissorRects(&[RECT {
            left: 0, top: 0, right: 64, bottom: 64,
        }]);

        list.list.IASetPrimitiveTopology(
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
        );

        list.list.IASetVertexBuffers(
            0,
            Some(&[D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: vb.GetGPUVirtualAddress(),
                SizeInBytes: vb_bytes as u32,
                StrideInBytes: 28,
            }]),
        );

        let signature = host_signature(0,16)
            .expect("Indirect draw signature");

        list.list.ExecuteIndirect(
            &signature,
            4,
            &args,
            0,
            None::<&ID3D12Resource>,
            0,
        );

        crate::indirect::done(&list, &args);

        list.list.ResourceBarrier(&[
            transition(
                &color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
        ]);

        let (readback, footprint) =
            crate::capture::read_back(
                dev, &list.list, &color
            ).expect("Framebuffer readback");

        list.close();
        host.execute(&[&list]);

        assert!(dev.GetDeviceRemovedReason().is_ok());

        let generated = std::slice::from_raw_parts(
            args_cpu.cast::<[u32;4]>(), 4
        );

        assert_eq!(generated, &[
            [6,1,0,0],
            [6,1,6,0],
            [6,1,12,0],
            [0,1,18,0],
        ]);

        let mut mapped = std::ptr::null_mut();

        readback.Map(0,None,Some(&mut mapped)).unwrap();

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
                let at = y*pitch+x*4;

                let expected: [u8;4] =
                    if (16..48).contains(&y)
                        && (4..28).contains(&x) {
                        red += 1;
                        [255,0,0,255]
                    } else if (16..48).contains(&y)
                        && (36..60).contains(&x) {
                        green += 1;
                        [0,255,0,255]
                    } else {
                        black += 1;
                        [0,0,0,255]
                    };

                if pixels[at..at+4] != expected {
                    incorrect += 1;
                }
            }
        }

        // Save the actual GPU output, even if verification fails.
        crate::capture::save(1,64,64,pitch,false,pixels);

        readback.Unmap(0,None);
        list.reset();

        assert_eq!(incorrect,0,"Incorrect framebuffer pixels");

        assert_eq!((red,green,black),(768,768,2560));

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

            if queue.GetMessage(i, None, &mut size).is_err()
                || size == 0
            {
                continue;
            }

            let mut buffer = vec![0u64; size.div_ceil(8)];
            let ptr = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();

            if queue.GetMessage(i, Some(ptr), &mut size).is_ok() {
                let msg = &*ptr;

                if !msg.pDescription.is_null() {
                    let text = std::slice::from_raw_parts(
                        msg.pDescription,
                        msg.DescriptionByteLength,
                    );

                    eprintln!(
                        "D3D12 ID={} severity={}: {}",
                        msg.ID.0,
                        msg.Severity.0,
                        String::from_utf8_lossy(text)
                            .trim_end_matches('\0')
                    );
                }
            }
        }
    }
}
