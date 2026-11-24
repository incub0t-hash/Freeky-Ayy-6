use super::*;
use crate::emit::{Container, FourCC, Target, container, emit};

/// The console returns I,J as the contribution of vertices one and two.
/// Independently interpolate a per-vertex basis over a triangle whose clip
/// W varies, then compare every pixel to an export of the raw I,J registers.
#[test]
#[ignore = "requires a hardware adapter with raw barycentrics, DXC and the validator"]
fn raw_center_weights_match_interpolated_vertex_basis() {
    let path = crate::emit::validator::Validator::default_path().unwrap();
    let mut validator = crate::emit::validator::Validator::load(Some(&path)).unwrap();
    let pc = path.with_file_name("dxc.exe");
    let dir = std::env::temp_dir().join(format!("victory-bary-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let output = container::signature(&[container::SigElement {
        name: "SV_Target".into(), index: 0, kind: 16, comp: 9, interp: 0,
        row: Some(0), col: 0, cols: 4, usage: 15, half: false,
    }], false);
    let dxbc = crate::dxbc::build(&[(FourCC::OSG1, &output)]);
    let c = Container::parse(&dxbc).unwrap();
    let native = crate::NativeProgram {
        stage: crate::HwStage::Ps, wave64: false, vgprs: 2, layout: vec![], used: vec![],
        io: crate::native::StageIo::Ps {
            inputs: vec![], input_addr: 2, db_shader_control: 0, patches: vec![],
        },
        code: [0xf800_180fu32, 0x0100_0100, 0xbf81_0000]
            .iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let ir = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let blocks = crate::structure::structure(&ir).unwrap();
    let resources = crate::resource::track(&ir, &native.layout, native.stage, Some(&blocks)).unwrap();
    let io = crate::io::raise(&ir, &native).unwrap();
    let mut ps = emit(&c, &native, &ir, &blocks, &io, &resources, Target::ANY).unwrap();
    validator.validate(&mut ps.dxil).unwrap();

    let source = "[RootSignature(RS)] float4 ps(V i) : SV_Target { return i.uv.xyxy; }";
    let reference = compile(&pc, &dir, source, "ps", "ps_6_6");
    let vs = compile(&pc, &dir, source, "vs", "vs_6_6");
    let host = Host::new(false).unwrap();
    // SAFETY: the PC compiler encoded the root signature in the reference shader.
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &reference) }.unwrap();
    let vertices = [
        -1., -1., 0.5, 1., 0., 0.,
        -2.,  6., 1.0, 2., 1., 0.,
        12., -4., 2.0, 4., 0., 1.,
    ];
    let render = |shader: &[u8]| {
        let pso = host.pipeline_with(&root, &vs, shader).unwrap();
        host.draw(&root, &pso, &Draw {
            roots: vec![(0, Root::Constants(vec![0; 4]))],
            textures: &[], samplers: &[], vertices: &vertices, stride: 24,
            size: (32, 32), mesh_groups: 0,
        }).unwrap()
    };
    let expected = render(&reference);
    let actual = render(&ps.dxil);
    assert_eq!((actual.len(), expected.len()), (1024, 1024));
    assert!(expected.iter().any(|p| p[0] > 0.25 && p[1] > 0.01));
    for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!((a - b).abs() < 0.00001, "{a} vs {b}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
