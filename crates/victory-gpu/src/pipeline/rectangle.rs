//! Geometry stages for vertex-number rectangle lists, adapted from VIctory
//! PR #1020 and the rectangle winding rules in fxceless/victory-gta6.

/// The geometry stage completing rectangles for a vertex stage with output
/// signature as in `vs` (a translated shader's container): three vertices
/// in, the rectangle they span out as a strip. `None` if the signature has
/// no position, an element our source cannot name, or the compiler lays the
/// elements out in other registers than the vertex stage's.
///
/// The common corner shares x with one vertex and y with the other. Two
/// independent triangles preserve the input winding and provoking vertex.
pub(super) fn rect_gs(vs: &[u8], ps: &[u8]) -> Option<Vec<u8>> {
    static MADE: std::sync::Mutex<
        Option<std::collections::HashMap<Vec<Element>, Option<Vec<u8>>>>,
    > = std::sync::Mutex::new(None);
    let outputs = signature(vs, b"OSG1")?;
    if !ps.is_empty()
        && signature(ps, b"ISG1")?.iter().any(|e| {
            let name = e.name.to_ascii_uppercase();
            name.starts_with("SV_")
                && !matches!(
                    name.as_str(),
                    "SV_POSITION" | "SV_ISFRONTFACE" | "SV_SAMPLEINDEX" | "SV_COVERAGE"
                )
        })
    {
        return None;
    }
    let mut fields = String::new();
    let mut fourth = String::new();
    let mut position = None;
    for (n, e) in outputs.iter().enumerate() {
        let comps = e.mask.count_ones();
        // Elements start at x and are contiguous in what we emit.
        if comps == 0 || u32::from(e.mask) != (1u32 << comps) - 1 {
            return None;
        }
        let ty = match e.kind {
            1 => "uint",
            2 => "int",
            3 => "float",
            _ => return None,
        };
        if !e
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return None;
        }
        let system = e.name.to_ascii_uppercase().starts_with("SV_");
        if system && !e.name.eq_ignore_ascii_case("SV_Position") {
            return None;
        }
        let semantic = if system && e.index == 0 {
            e.name.clone()
        } else {
            format!("{}{}", e.name, e.index)
        };
        let interp = if e.kind == 3 { "" } else { "nointerpolation " };
        fields.push_str(&format!("    {interp}{ty}{comps} e{n} : {semantic};\n"));
        if e.kind == 3 {
            fourth.push_str(&format!(
                "    d.e{n} = i[s].e{n} + i[t].e{n} - i[r].e{n};\n"
            ));
        } else {
            fourth.push_str(&format!("    d.e{n} = i[0].e{n};\n"));
        }
        if e.name.eq_ignore_ascii_case("SV_Position") && comps == 4 {
            position = Some(n);
        }
    }
    let p = position?;
    let source = format!(
        "struct V {{\n{fields}}};\n\
         [maxvertexcount(6)]\n\
         void main(triangle V i[3], inout TriangleStream<V> o) {{\n\
         \x20   float2 a = i[0].e{p}.xy, b = i[1].e{p}.xy, c = i[2].e{p}.xy;\n\
         \x20   bool plain = i[0].e{p}.w > 0 && i[0].e{p}.w == i[1].e{p}.w && i[0].e{p}.w == i[2].e{p}.w;\n\
         \x20   bool corner0 = (a.x == b.x && a.y == c.y) || (a.x == c.x && a.y == b.y);\n\
         \x20   bool corner1 = (b.x == a.x && b.y == c.y) || (b.x == c.x && b.y == a.y);\n\
         \x20   bool corner2 = (c.x == a.x && c.y == b.y) || (c.x == b.x && c.y == a.y);\n\
         \x20   if (!plain || !(corner0 || corner1 || corner2) || all(a == b) || all(a == c) || all(b == c)) {{\n\
         \x20       o.Append(i[0]); o.Append(i[1]); o.Append(i[2]); return;\n\
         \x20   }}\n\
         \x20   uint r = corner0 ? 0 : (corner1 ? 1 : 2);\n\
         \x20   uint s = (r + 1) % 3, t = (r + 2) % 3;\n\
         \x20   V d;\n{fourth}\
         \x20   o.Append(i[0]); o.Append(i[1]); if (corner0) o.Append(d); else o.Append(i[2]);\n\
         \x20   o.RestartStrip(); o.Append(i[0]);\n\
         \x20   if (corner1) {{ o.Append(i[2]); o.Append(d); }}\n\
         \x20   else {{ o.Append(d); if (corner0) o.Append(i[2]); else o.Append(i[1]); }}\n\
         }}\n"
    );
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(std::collections::HashMap::new);
    if let Some(done) = made.get(&outputs) {
        return done.clone();
    }
    // The host takes no pipeline mixing shader models below 6 with ours,
    // so the stage is compiled to DXIL.
    let bytes = match dxc_compile(&source, "gs_6_0") {
        Ok(bytes) => {
            // The stage must hand the pixel stage what the vertex stage did,
            // in the same registers.
            let layout_matches = |part| {
                signature(&bytes, part).is_some_and(|o| {
                    o.len() == outputs.len()
                        && o.iter().zip(&outputs).all(|(x, y)| {
                            (x.register, x.mask, x.index, x.kind)
                                == (y.register, y.mask, y.index, y.kind)
                                && x.name.eq_ignore_ascii_case(&y.name)
                        })
                })
            };
            let same = layout_matches(b"ISG1") && layout_matches(b"OSG1");
            if !same {
                tracing::warn!(
                    "rectangle geometry stage: registers differ from the vertex stage's"
                );
            }
            same.then_some(bytes)
        }
        Err(e) => {
            tracing::warn!(error = %e, "rectangle geometry stage not compiled");
            None
        }
    };
    if bytes.is_some() {
        tracing::info!(elements = outputs.len(), "a rectangle geometry stage made");
    }
    made.insert(outputs, bytes.clone());
    bytes
}

/// One element of a shader container's signature part.
#[derive(Eq, PartialEq, Hash)]
struct Element {
    name: String,
    index: u32,
    kind: u32,
    register: u32,
    mask: u8,
}

/// The 32-byte records of an ISG1/OSG1 signature in a DXIL container.
fn signature(code: &[u8], part: &[u8; 4]) -> Option<Vec<Element>> {
    let word = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
    };
    let container = victory_shader::Container::parse(code).ok()?;
    let d = container.part(victory_shader::FourCC(*part))?.data;
    let count = word(d, 0)?;
    let records = word(d, 4)?;
    let mut out = Vec::new();
    for k in 0..count {
        let e = records + k * 32 + 4;
        let name_at = word(d, e)?;
        let end = d.get(name_at..)?.iter().position(|&b| b == 0)?;
        out.push(Element {
            name: String::from_utf8_lossy(&d[name_at..name_at + end]).into_owned(),
            index: word(d, e + 4)? as u32,
            kind: word(d, e + 12)? as u32,
            register: word(d, e + 16)? as u32,
            mask: *d.get(e + 20)?,
        });
    }
    Some(out)
}

/// Compiles HLSL `source` (entry `main`) for `target` with the Windows
/// SDK's `dxcompiler.dll` (`VICTORY_DXC` names another), signed by the
/// `dxil.dll` beside it.
///
/// shortcut: needs the Windows SDK on the machine; without it rectangle
/// lists drawn by vertex number stay unexpanded.
fn dxc_compile(source: &str, target: &str) -> Result<Vec<u8>, String> {
    use windows::Win32::Graphics::Direct3D::Dxc::{
        CLSID_DxcCompiler, DXC_CP_UTF8, DxcBuffer, IDxcBlob, IDxcBlobEncoding, IDxcCompiler3,
        IDxcResult,
    };
    use windows::core::{GUID, HRESULT, Interface, PCWSTR};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    type Create =
        unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut std::ffi::c_void) -> HRESULT;
    static CREATE: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
    let create = CREATE
        .get_or_init(|| {
            let path = match std::env::var_os("VICTORY_DXC") {
                Some(p) => std::path::PathBuf::from(p),
                None => {
                    let kits =
                        std::path::PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
                    let mut versions: Vec<std::path::PathBuf> = std::fs::read_dir(&kits)
                        .map_err(|e| format!("no Windows SDK: {e}"))?
                        .filter_map(Result::ok)
                        .map(|e| e.path().join(r"x64\dxcompiler.dll"))
                        .filter(|p| p.exists())
                        .collect();
                    versions.sort();
                    versions
                        .pop()
                        .ok_or("no dxcompiler.dll in the Windows SDK")?
                }
            };
            let wide = |p: &std::path::Path| -> Vec<u16> {
                p.as_os_str()
                    .to_string_lossy()
                    .encode_utf16()
                    .chain([0])
                    .collect()
            };
            // The signer first, by path: the compiler then finds it loaded.
            let signer = wide(&path.with_file_name("dxil.dll"));
            let compiler = wide(&path);
            // SAFETY: plain Win32 calls with NUL-terminated names.
            unsafe {
                LoadLibraryW(signer.as_ptr());
                let module = LoadLibraryW(compiler.as_ptr());
                if module.is_null() {
                    return Err(format!("cannot load {}", path.display()));
                }
                GetProcAddress(module, c"DxcCreateInstance".as_ptr().cast())
                    .map(|f| f as usize)
                    .ok_or_else(|| "dxcompiler.dll lacks DxcCreateInstance".to_string())
            }
        })
        .clone()?;
    // SAFETY: the export has this signature.
    let create: Create = unsafe { std::mem::transmute(create) };
    let mut raw = std::ptr::null_mut();
    // SAFETY: a valid class and interface; the out pointer is a local.
    unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw) }
        .ok()
        .map_err(|e| e.to_string())?;
    // SAFETY: the object `create` returned, owned from here.
    let compiler = unsafe { IDxcCompiler3::from_raw(raw) };
    let buffer = DxcBuffer {
        Ptr: source.as_ptr().cast(),
        Size: source.len(),
        Encoding: DXC_CP_UTF8.0,
    };
    let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain([0]).collect() };
    let words = [wide("-E"), wide("main"), wide("-T"), wide(target)];
    let args: Vec<PCWSTR> = words.iter().map(|w| PCWSTR(w.as_ptr())).collect();
    // SAFETY: the buffer and arguments live through the call.
    let result: IDxcResult =
        unsafe { compiler.Compile(&buffer, Some(&args), None) }.map_err(|e| e.to_string())?;
    // SAFETY: a live result.
    let status = unsafe { result.GetStatus() }.map_err(|e| e.to_string())?;
    if status.is_err() {
        // SAFETY: a live result; the blob's own bytes.
        let text = unsafe {
            result.GetErrorBuffer().ok().map(|b: IDxcBlobEncoding| {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .into_owned()
            })
        };
        return Err(format!("{status:?}: {}", text.unwrap_or_default()));
    }
    // SAFETY: a live result; the blob's own bytes.
    unsafe {
        let blob: IDxcBlob = result.GetResult().map_err(|e| e.to_string())?;
        Ok(
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec(),
        )
    }
}

#[cfg(test)]
mod rect_tests {
    use super::*;
    use crate::host::host;
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
    };
    use windows::core::Interface;

    /// Vertex stages placing a rectangle's corners by vertex number, as
    /// the title's full-screen and tiled passes do. In the first the second
    /// vertex sits between the other two (top left, bottom left, bottom
    /// right); in the second the first vertex does. The stage does not
    /// depend on which.
    const VS: [&str; 2] = [
        "float4 main(uint id : SV_VertexID) : SV_Position {
            float x = (id % 3) == 2 ? 1.0 : -1.0;
            float y = (id % 3) == 0 ? 1.0 : -1.0;
            return float4(x, y, 0.5, 1.0);
        }",
        "float4 main(uint id : SV_VertexID) : SV_Position {
            float x = (id % 3) == 2 ? 1.0 : -1.0;
            float y = (id % 3) == 1 ? -1.0 : 1.0;
            return float4(x, y, 0.5, 1.0);
        }",
    ];
    const PS: &str = "float4 main() : SV_Target { return float4(1, 1, 1, 1); }";
    const SIDE: u32 = 8;

    /// The signature parser reads what the compiler wrote.
    #[test]
    fn signature_of_a_compiled_vertex_stage() {
        let Ok(vs) = dxc_compile(VS[0], "vs_6_0") else {
            return; // no compiler on this machine
        };
        let outputs = signature(&vs, b"OSG1").unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].name.eq_ignore_ascii_case("SV_Position"));
        assert_eq!(
            (outputs[0].register, outputs[0].mask, outputs[0].kind),
            (0, 0xf, 3)
        );
        let inputs = signature(&vs, b"ISG1").unwrap();
        assert!(inputs[0].name.eq_ignore_ascii_case("SV_VertexID"));
    }

    /// Three vertices placed by number draw the whole rectangle through the
    /// geometry stage, and only the triangle they span without it.
    #[test]
    fn rectangles_placed_by_vertex_number_are_completed() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        for source in VS {
            let (Ok(vs), Ok(ps)) = (dxc_compile(source, "vs_6_0"), dxc_compile(PS, "ps_6_0"))
            else {
                return; // no compiler on this machine
            };
            let gs = rect_gs(&vs, &ps).expect("a geometry stage for a position-only vertex stage");
            let with = covered(
                &host.device,
                &vs,
                Some(&gs),
                &ps,
                D3D12_CULL_MODE_NONE,
                false,
            );
            let without = covered(&host.device, &vs, None, &ps, D3D12_CULL_MODE_NONE, false);
            eprintln!("rectangle coverage: {with}/64; triangle coverage: {without}/64");
            assert_eq!(with, SIDE * SIDE, "the rectangle covers the target");
            assert!(
                without < SIDE * SIDE && without > 0,
                "three vertices alone span a triangle: {without} pixels"
            );
        }
    }

    #[test]
    fn indirect_rectangle_preserves_first_vertex_flat_attributes_and_winding() {
        let Some(host) = host() else { return };
        let vs = dxc_compile(r#"
            struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; nointerpolation uint id : TEXCOORD1; };
            V main(uint vertex : SV_VertexID, uint first : SV_StartVertexLocation) {
                uint id = vertex + first;
                uint corner = id % 3; V v;
                v.uv = float2(corner == 2 ? 1 : -1, corner == 0 ? 1 : -1);
                v.p = float4(v.uv, 0.5, 1);
                v.id = id; return v;
            }"#, "vs_6_8").expect("compile vertex-number test");
        let ps = dxc_compile(r#"
            float4 main(float4 p:SV_Position, float2 uv:TEXCOORD0, nointerpolation uint id:TEXCOORD1) : SV_Target {
                float2 ndc = float2(p.x / 4 - 1, 1 - p.y / 4);
                return float4(id == 3 && all(abs(uv - ndc) < 0.001) ? 1 : 0, 0, 0, 1);
            }"#, "ps_6_0").expect("compile pixel reference");
        let gs = rect_gs(&vs, &ps).expect("link float and flat integer outputs");
        let mut visible_faces = 0;
        for cull in [D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_BACK] {
            let triangle = covered(&host.device, &vs, None, &ps, cull, true);
            let rectangle = covered(&host.device, &vs, Some(&gs), &ps, cull, true);
            let visible = triangle != 0;
            assert_eq!(rectangle, if visible { SIDE * SIDE } else { 0 });
            visible_faces += u32::from(visible);
            eprintln!(
                "indirect first=3, cull={}: rectangle={rectangle}, triangle={triangle}",
                cull.0
            );
        }
        assert_eq!(visible_faces, 1);
    }

    /// How many pixels of a `SIDE`-square target one draw of three vertices
    /// covers with these stages.
    fn covered(
        device: &ID3D12Device,
        vs: &[u8],
        gs: Option<&[u8]>,
        ps: &[u8],
        cull: D3D12_CULL_MODE,
        indirect: bool,
    ) -> u32 {
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        // SAFETY: plain D3D12 calls on objects made here; every pointer
        // names live local data.
        unsafe {
            let mut blob = None;
            D3D12SerializeVersionedRootSignature(
                &D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                    Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                    Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                        Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1::default(),
                    },
                },
                &mut blob,
                None,
            )
            .unwrap();
            let blob: ID3DBlob = blob.unwrap();
            let root: ID3D12RootSignature = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    ),
                )
                .unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(vs),
                GS: gs.map_or(D3D12_SHADER_BYTECODE::default(), code),
                PS: code(ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: cull,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                DSVFormat: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();

            let texture = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(SIDE),
                Height: SIDE,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut target: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &texture,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .unwrap();
            let target = target.unwrap();
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0;
            device.GetCopyableFootprints(
                &texture,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
            let mut readback: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
            let readback = readback.unwrap();
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let view = views.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(&target, None, view);

            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso)
                .unwrap();
            list.SetGraphicsRootSignature(&root);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: SIDE as i32,
                bottom: SIDE as i32,
            }]);
            list.OMSetRenderTargets(1, Some(&view), false, None);
            list.ClearRenderTargetView(view, &[0.0; 4], None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let command = indirect.then(|| {
                let (buffer, mapped) = host().unwrap().shared_buffer(16).unwrap();
                mapped.cast::<[u32; 4]>().write([3, 1, 3, 0]);
                let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                    ..Default::default()
                };
                let mut signature: Option<ID3D12CommandSignature> = None;
                device
                    .CreateCommandSignature(
                        &D3D12_COMMAND_SIGNATURE_DESC {
                            ByteStride: 16,
                            NumArgumentDescs: 1,
                            pArgumentDescs: &arg,
                            NodeMask: 0,
                        },
                        None::<&ID3D12RootSignature>,
                        &mut signature,
                    )
                    .unwrap();
                (buffer, signature.unwrap())
            });
            if let Some((args, signature)) = &command {
                list.ExecuteIndirect(signature, 1, args, 0, None::<&ID3D12Resource>, 0);
            } else {
                list.DrawInstanced(3, 1, 0, 0);
            }
            list.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyTextureRegion(
                &D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: footprint,
                    },
                },
                0,
                0,
                0,
                &D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&target),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                },
                None,
            );
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let started = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(started.elapsed().as_secs() < 30, "the draw did not finish");
                std::thread::yield_now();
            }
            let mut at = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut at)).unwrap();
            let pitch = footprint.Footprint.RowPitch as usize;
            let mut covered = 0;
            for y in 0..SIDE as usize {
                for x in 0..SIDE as usize {
                    if *at.cast::<u8>().add(y * pitch + x * 4) == 255 {
                        covered += 1;
                    }
                }
            }
            readback.Unmap(0, None);
            covered
        }
    }
}
