use super::host::{Dispatch, Host, Root, View};
use crate::native::{ComputeIo, RootParam, RootType, StageIo};

// s0 supplies the UAV table, s1 the stride, s2 the group ID, v0 the lane.
fn program(last_word: u32) -> crate::NativeProgram {
    let words = [
        0xbe84_0300,
        0xbe85_0380, // s[4:5] = table
        0xf408_0202,
        0xfa00_0000, // load descriptor into s[8:11]
        0xd746_0006,
        0x0401_0c02, // v6 = (group.x << 6) + thread.x
        0xd569_0007,
        0x0002_0c01, // v7 = stride * v6
        0x7e00_0280,
        0x7e02_0281,
        0x7e04_0281,
        last_word,
        0xe078_1000,
        0x8002_0007, // raw store v[0:3] at byte v7
        0xbf81_0000,
    ];
    let mut p = super::tests::compute(&words, true);
    let param = |slot, root| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: slot,
        slot,
    };
    p.layout = vec![param(0, RootType::Table), param(1, RootType::Constants)];
    p.io = StageIo::Cs(ComputeIo {
        rsrc2: 2 << 1 | 1 << 7,
        threads: [64, 1, 1],
    });
    p
}

#[test]
fn argument_tuple_requires_its_bound_aligned_root_stride() {
    let fixed = super::Target {
        lanes_min: 32,
        lanes_max: 32,
    };
    let p = program(0x7e06_0280);
    let host_stage = super::tests::emit_for(&p, fixed).unwrap();
    let requirement = host_stage.record_stride.unwrap();
    assert_eq!((requirement.slot, requirement.dword), (1, 0));
    for (value, expected) in [
        (None, false),
        (Some(0), true),
        (Some(4), false),
        (Some(12), false),
        (Some(16), true),
        (Some(20), false),
        (Some(32), true),
    ] {
        assert_eq!(
            host_stage.permits_dispatch(|s, d| {
                assert_eq!((s, d), (1, 0));
                value
            }),
            expected
        );
    }
    let varying = program(0x7e06_0306); // v3 = global thread ID
    assert!(
        super::tests::emit_for(&varying, fixed)
            .unwrap()
            .record_stride
            .is_none()
    );
    assert!(
        super::tests::emit_for(&p, super::Target::ANY)
            .unwrap()
            .record_stride
            .is_none()
    );
}

#[test]
#[ignore = "requires a fixed-wave32 hardware adapter"]
fn argument_tuple_initializes_two_groups_without_touching_record_padding() {
    let host = Host::new(false).unwrap();
    assert_eq!(
        host.target(),
        super::Target {
            lanes_min: 32,
            lanes_max: 32
        }
    );
    let mut cs = match std::env::var_os("VICTORY_TEST_RECORD_INIT_CONTAINER") {
        Some(path) => super::translate_for(&std::fs::read(path).unwrap(), host.target()).unwrap(),
        None => super::tests::emit_for(&program(0x7e06_0280), host.target()).unwrap(),
    };
    let root_word = cs.record_stride.unwrap();
    assert_eq!(
        (root_word.slot, root_word.dword, root_word.root),
        (1, 0, RootType::Constants)
    );
    for invalid in [None, Some(4), Some(12), Some(20)] {
        assert!(!cs.permits_dispatch(|_, _| invalid));
    }
    let mut validator =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
            .unwrap();
    validator.validate(&mut cs.dxil).unwrap();
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    for stride in [0, 16, 32] {
        assert!(cs.permits_dispatch(|_, _| Some(stride)));
        let mut data = vec![0; cs.user_data_dwords() as usize];
        data[cs.record_stride.unwrap().index as usize] = stride;
        let size = 128 * 8;
        let views = [View::RawUav(size)];
        let rows = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views: &views,
                    samplers: &[],
                    groups: [2, 1, 1],
                },
            )
            .unwrap();
        let mut expected = vec![0; size as usize];
        for lane in 0..128 {
            let offset = lane * stride as usize / 4;
            expected[offset..offset + 4].copy_from_slice(&[0, 1, 1, 0]);
        }
        assert_eq!(rows[0], expected, "stride {stride}");
    }
}
