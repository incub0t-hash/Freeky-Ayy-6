//! Gathers at an explicit level (`image_gather4_l`, `image_gather4_l_o`),
//! which DXIL's gather (level 0 only) cannot express: hand-assembled GFX10
//! compute programs lowered to DXIL, validated by `dxil.dll` when present,
//! and run on the host GPU and WARP over a texture with a chain of levels,
//! each lane's four values checked against the console's footprint.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Level 0's side; the chain halves it down to 1x1.
const SIDE: u32 = 8;
const LEVELS: u32 = 4;

/// The program's start: the views' table in s0 (the texture at entry 0,
/// a raw UAV at entry 1), the samplers' in s1; lane l takes u = (l & 7) /
/// 8 + 0.03, v = 1 - u and level l >> 3 (v2, v3, v4), and its
/// result's byte offset 16 l (v16).
const PROLOGUE: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0302,
    0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
    0xF408_0203,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408_0A02,
    0xFA00_0020, // s_load_dwordx4 s[40:43], s[4:5], 0x20
    0x3602_0087, // v_and_b32 v1, 7, v0
    0x7E04_0D01, // v_cvt_f32_u32 v2, v1
    0x1004_04FF,
    0x3E00_0000, // v_mul_f32 v2, 0.125, v2
    0x0604_04FF,
    0x3CF5_C28F, // v_add_f32 v2, 0.03, v2
    0x0806_04F2, // v_sub_f32 v3, 1.0, v2
    0x2C02_0083, // v_lshrrev_b32 v1, 3, v0
    0x7E08_0D01, // v_cvt_f32_u32 v4, v1
    0x3420_0084, // v_lshlrev_b32 v16, 4, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// The program's end: the four gathered values to the UAV.
const EPILOGUE: &[u32] = &[
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE078_1000,
    0x800A_0C10, // buffer_store_dwordx4 v[12:15], v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// `image_gather4_l v[12:15], v[2:4], s[12:19], s[8:11] dmask:0x1 dim:2D`.
const GATHER4_L: &[u32] = &[0xF110_0108, 0x0043_0C02];

/// The texel offset word (x +1, y -1) in v1, then
/// `image_gather4_l_o v[12:15], v[1:4], s[12:19], s[8:11] dmask:0x1 dim:2D`.
const GATHER4_L_O: &[u32] = &[
    0x7E02_02FF,
    0x0000_3F01, // v_mov_b32 v1, 0x3f01
    0xF150_0108,
    0x0043_0C01,
];

/// Lowers a wave32 compute program of 32 threads with `body` between the
/// prologue and the epilogue.
fn build(body: &[u32], sampler: Option<[u32; 4]>) -> Result<HostShader, String> {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    let mut words = PROLOGUE.to_vec();
    if let Some(sampler) = sampler {
        // Overwrite s[8:11] with the four literal sampler words.
        for (i, word) in sampler.into_iter().enumerate() {
            words.extend([0xbe88_03ff + ((i as u32) << 16), word]);
        }
    }
    words.extend_from_slice(body);
    words.extend_from_slice(EPILOGUE);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Uav, 8, 4, 0),
            range(ResourceType::Sampler, 0, 4, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let container = crate::Container::parse(&bytes).unwrap();
    let mut shader = super::emit(&container, &native, &p, &st, &io, &res, super::Target::ANY)
        .map_err(|e| e.to_string())?;
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil)
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// Texel (x, y) of level `level`: its red channel names all three.
fn texel(level: u32, x: u32, y: u32) -> f32 {
    (level * 100 + y * 10 + x) as f32
}

/// The levels' texels in turn, level 0 first.
fn chain() -> Vec<[f32; 4]> {
    (0..LEVELS)
        .flat_map(|level| {
            let side = (SIDE >> level).max(1);
            (0..side * side).map(move |at| [texel(level, at % side, at / side), 0.0, 0.0, 1.0])
        })
        .collect()
}

/// What the console's gather returns to lane `lane` with texel offset
/// `offset`: the 2x2 texels around the lane's coordinates at the nearest
/// level (within the chain), clamped to the edges, in gather order.
fn expected(lane: u32, base: u32, max_lod: u32, wrap: bool, offset: (i32, i32)) -> [f32; 4] {
    let u = (lane & 7) as f32 * 0.125 + 0.03;
    let v = 1.0 - u;
    let level = (base + (lane >> 3).min(max_lod)).min(LEVELS - 1);
    let side = (SIDE >> level).max(1) as i32;
    let first = |c: f32, o: i32| (c * side as f32 - 0.5).floor() as i32 + o;
    let (i0, j0) = (first(u, offset.0), first(v, offset.1));
    let coord = |x: i32| if wrap { x.rem_euclid(side) } else { x.clamp(0, side - 1) } as u32;
    let at = |i: i32, j: i32| texel(level, coord(i), coord(j));
    [
        at(i0, j0 + 1),
        at(i0 + 1, j0 + 1),
        at(i0 + 1, j0),
        at(i0, j0),
    ]
}

/// Runs `shader` over the chain with a point, clamping sampler on the
/// hardware GPU and on WARP (each skipped without a device): per one that
/// ran, each lane's four values.
fn run(shader: &HostShader, base: u32) -> Vec<(&'static str, Vec<[f32; 4]>)> {
    let texels = chain();
    let views = [
        View::Mips { width: SIDE, height: SIDE, levels: LEVELS, base, texels: &texels },
        View::RawUav(32 * 4),
    ];
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // Both tables start at heap index 0.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let lanes = words[0]
            .chunks(4)
            .map(|c| [0, 1, 2, 3].map(|k| f32::from_bits(c[k])))
            .collect();
        out.push((label, lanes));
    }
    out
}

#[test]
fn gather_reads_selected_mip_with_sampler_and_view_bounds() {
    for (base, max_lod, wrap, body, offset) in [
        (0, 3, false, GATHER4_L, (0, 0)),
        (1, 3, true, GATHER4_L_O, (1, -1)),
        (0, 1, false, GATHER4_L_O, (1, -1)),
        (1, 0, false, GATHER4_L, (0, 0)),
    ] {
        let address = if wrap { 0 } else { 2 | (2 << 3) };
        let sampler = [address, max_lod << 20, 1 << 26, 0];
        let shader = build(body, Some(sampler)).unwrap();
        let want: Vec<_> = (0..32).map(|l| expected(l, base, max_lod, wrap, offset)).collect();
        for (label, got) in run(&shader, base) {
            assert_eq!(got, want, "{label}, base {base}, max_lod {max_lod}, wrap {wrap}");
        }
    }
}

#[test]
fn gather_refuses_unknown_sampler_and_fractional_mip_rules() {
    let reason = |body: &[u32], sampler| build(body, sampler).err().expect("expected refusal");
    assert!(reason(GATHER4_L, None).contains("runtime sampler"));
    assert!(reason(GATHER4_L, Some([0, 3 << 20, 2 << 26, 0])).contains("sampler state"));
    let mut fractional = vec![0x1008_08ff, 1.2f32.to_bits()]; // v_mul_f32 v4, 1.2, v4
    fractional.extend_from_slice(GATHER4_L);
    assert!(reason(&fractional, Some([0, 3 << 20, 1 << 26, 0])).contains("integral LOD"));
}
