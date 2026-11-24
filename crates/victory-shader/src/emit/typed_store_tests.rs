use super::host::{Dispatch, Host, Root, View};
use crate::native::{ResourceType, RootParam, RootType, UsedResource};

#[test]
fn mixed_float_and_uint_texture_stores_preserve_packed_words() {
    let table = RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    };
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = [
        Host::new(false).expect("hardware D3D12"),
        Host::new(true).expect("WARP D3D12"),
    ];
    for first in [0x0000_0001u32, 0x7fa1_2345, 0xfc12_3456, 0xffff_ffff] {
        let pixels = [first, 0x3f00_0000, 0x4000_0000, 0x4040_0000];
        let mut words = vec![
            0xbe84_0300,
            0xbe85_0380, // table pointer into s4:5
            0xf40c_0302,
            0xfa00_0000, // first texture into s12:19
            0xf40c_0502,
            0xfa00_0020, // next texture into s20:27
            0x7e00_0280,
            0x7e02_0280, // (x, y) = (0, 0)
        ];
        for (i, value) in pixels.into_iter().enumerate() {
            words.extend([0x7e04_02ff + (i as u32) * 0x20000, value]);
        }
        // Query the first UAV, then convert its width to the green channel.
        words.extend([0xf038_1108, 0x0003_0600, 0x7e06_0d06]);
        words.extend([
            0xf020_1f08,
            0x0003_0200,
            0xf020_1f08,
            0x0005_0200,
            0xbf81_0000,
        ]);
        let mut native = crate::NativeProgram {
            stage: crate::HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: Vec::new(),
            used: Vec::new(),
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 0,
                threads: [1; 3],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        native.layout = vec![table];
        // The second live descriptor uses the extra mask word, as a ten-output scattering pass does.
        native.used = [0, 16, 24, 32, 40, 8]
            .map(|offset| UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: offset,
                lower_bound: offset / 8,
                stride: 4,
                dwords: 8,
                user_data: 0,
                slot: 0,
            })
            .to_vec();
        let c = crate::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![],
        };
        let ir = crate::ir::build(&native.code, false, native.vgprs).unwrap();
        let st = crate::structure::structure(&ir).unwrap();
        let res = crate::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&ir, &native).unwrap();
        let mut shader = super::emit(&c, &native, &ir, &st, &io, &res).unwrap();
        if let Ok(mut validator) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            validator.validate(&mut shader.dxil).unwrap();
        }
        let extra = shader.user_data.iter().find(|u| u.root == RootType::Other(super::USER_UINT_TEXTURES)).unwrap();
        assert_eq!((extra.slot, extra.dword), (0, 1));
        for host in &hosts {
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            for (mask, order) in [
                (1, [View::UintTextureUav(1, 1), View::TextureUav(1, 1)]),
                (2, [View::TextureUav(1, 1), View::UintTextureUav(1, 1)]),
            ] {
                let [a, b] = order;
                let views = [View::RawUav(1), a, b];
                let mut values = vec![0; 4.max(extra.index as usize + 1)];
                values[0] = 32 | u32::from(mask == 1);
                values[extra.index as usize] = u32::from(mask == 2);
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(values))],
                            views: &views,
                            samplers: &[],
                            groups: [1; 3],
                        },
                    )
                    .unwrap();
                let integer = if mask == 1 { 1 } else { 2 };
                let floating = 3 - integer;
                assert_eq!(out[integer], [first]);
                // Floating values may quiet an sNaN or flush a subnormal on store.
                assert_eq!(
                    &out[floating][1..],
                    &[1.0f32.to_bits(), pixels[2], pixels[3]]
                );
            }
        }
    }
}
