//! Fixed table entries whose translated compute accesses all read raw bytes.

use std::collections::{BTreeMap, BTreeSet};

use crate::gfx10::Fields;
use crate::ir::Program;
use crate::resource::{Resources, Role, Site, Source};
use crate::{HwStage, NativeProgram, ResourceType, RootType};

pub(super) fn entries(n: &NativeProgram, p: &Program, res: &Resources) -> Vec<(u8, u32)> {
    if n.stage != HwStage::Cs {
        return Vec::new();
    }
    let scalar: BTreeMap<_, _> = res.scalar.iter().copied().collect();
    let mut raw = BTreeSet::new();
    let mut other = BTreeSet::new();
    let mut dynamic_tables = BTreeSet::new();
    for d in &res.descriptors {
        let inst = &p.blocks[d.site.0].insts[d.site.1];
        let raw_load = d.role == Role::Resource
            && matches!(d.source, Source::Loaded { .. })
            && inst.name().starts_with("s_buffer_load_dword");
        for leaf in d.source.leaves() {
            let (Source::Loaded { load, first } | Source::Fetched { load, first }) = leaf else {
                continue;
            };
            let Some(address) = scalar.get(load) else {
                continue;
            };
            let Some(root) = n.layout.get(address.param) else {
                continue;
            };
            if root.root != RootType::Table {
                continue;
            }
            let byte = address.offset.checked_add(4 * u32::from(*first));
            let Some(byte) = byte.filter(|v| v.is_multiple_of(32) && address.dynamic.is_none())
            else {
                dynamic_tables.insert(root.slot);
                continue;
            };
            let entry = (root.slot, byte / 4);
            let mut matches = n.used.iter().filter(|u| {
                u.root == RootType::Table && u.slot == entry.0 && u.table_offset_dwords == entry.1
            });
            let single_srv = matches.next().is_some_and(|u| {
                u.kind == ResourceType::Srv && u.user_data == root.user_data && u.dwords == 4
            }) && matches.next().is_none();
            if raw_load && single_srv {
                raw.insert(entry);
            } else {
                other.insert(entry);
            }
        }
    }
    raw.retain(|entry| !other.contains(entry) && !dynamic_tables.contains(&entry.0));
    raw.into_iter().collect()
}

/// A compute program can use the same descriptor for a scalar and vector dword
/// load. The vector operation already requires a raw SRV. When both consumers
/// read precisely the same four SGPR values, the scalar load must use it too.
pub(super) fn heap_raw_loads(n: &NativeProgram, p: &Program, res: &Resources) -> BTreeSet<Site> {
    let mut loads = BTreeSet::new();
    if n.stage != HwStage::Cs {
        return loads;
    }
    let scalar: BTreeMap<_, _> = res.scalar.iter().copied().collect();
    let mut raw = BTreeSet::new();
    let mut other = BTreeSet::new();
    let mut candidates = Vec::new();
    for d in &res.descriptors {
        let Source::Loaded { load, first } = d.source else {
            continue;
        };
        let Some(a) = scalar.get(&load) else { continue };
        if !n
            .layout
            .get(a.param)
            .is_some_and(|rp| rp.root == RootType::ResourceHeap)
        {
            continue;
        }
        let inst = &p.blocks[d.site.0].insts[d.site.1];
        let origin = (load, first);
        if d.role == Role::Resource && inst.name().starts_with("s_buffer_load_dword") {
            candidates.push((d.site, origin));
        } else if d.role == Role::Resource
            && inst.name().starts_with("buffer_load_dword")
            && matches!(
                inst.fields,
                Fields::Buffer {
                    idxen: false,
                    format: None,
                    ..
                }
            )
        {
            raw.insert(origin);
        } else {
            other.insert(origin);
        }
    }
    for (site, origin) in candidates {
        if raw.contains(&origin) && !other.contains(&origin) {
            loads.insert(site);
        }
    }
    loads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{
        Target,
        host::{Dispatch, Host, Root, Stride, View},
        tests::{compute_of, emit_for},
    };
    use crate::native::{RootParam, StageIo, UsedResource};

    fn native(root: RootType, vector_read: bool) -> NativeProgram {
        let mut words = vec![
            0xbe82_0300,
            0xbe83_0380, // s2:s3 = s0:0
            0xf408_0101,
            0xfa00_0000, // s[4:7] = input descriptor
            0xf420_0202,
            0xfa00_0004, // s8 = input.Load(4)
            0xbe82_0301, // s2 = s1 (output table)
            0xf408_0301,
            0xfa00_0000, // s[12:15] = output descriptor
        ];
        if vector_read {
            words.extend([0xe030_0000, 0x8001_0100]); // v1 = input.Load(0)
        } else {
            words.push(0x7e02_0280); // v1 = 0
        }
        words.extend([
            0x4a02_0208, // v1 = s8 + v1
            0xe070_0000,
            0x8003_0100, // output.Store(0, v1)
            0xbf81_0000,
        ]);
        let mut n = compute_of(&words, false, 1);
        let StageIo::Cs(ref mut io) = n.io else {
            unreachable!()
        };
        io.rsrc2 = 2 << 1;
        n.layout = [root, RootType::Table]
            .into_iter()
            .enumerate()
            .map(|(i, root)| RootParam {
                root,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: i as u8,
                slot: i as u8,
            })
            .collect();
        n.used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 1,
            stride: 1,
            dwords: 4,
            user_data: 1,
            slot: 1,
        });
        if root == RootType::Table {
            n.used.push(UsedResource {
                kind: ResourceType::Srv,
                root,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 456,
                dwords: 4,
                user_data: 0,
                slot: 0,
            });
        }
        n
    }

    #[test]
    fn scalar_table_entries_require_exclusive_fixed_byte_loads() {
        let fixed = native(RootType::Table, false);
        assert_eq!(
            emit_for(&fixed, Target::ANY).unwrap().scalar_raw_srvs,
            [(0, 0)]
        );
        let mixed = native(RootType::Table, true);
        assert!(
            emit_for(&mixed, Target::ANY)
                .unwrap()
                .scalar_raw_srvs
                .is_empty()
        );
    }

    #[test]
    fn scalar_heap_load_uses_its_vector_consumers_raw_srv() {
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for companion in [false, true] {
            let n = native(RootType::ResourceHeap, companion);
            let shader = emit_for(&n, Target::ANY).unwrap();
            let input = [0xf1ee_0011, 0x2123_4567, 0];
            let input_view = if companion {
                View::Buffer {
                    data: &input,
                    stride: Stride::Raw,
                }
            } else {
                View::Cbv(&input)
            };
            let views = [input_view, View::RawUav(1)];
            let mut ud = vec![0; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                assert_eq!(u.slot, 1);
                ud[u.index as usize] = 32;
            }
            for warp in [false, true] {
                let host = Host::new(warp).unwrap();
                let (root, pso) = host.compute_pipeline(&shader).unwrap();
                let result = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud.clone()))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap();
                let expected = input[1].wrapping_add(if companion { input[0] } else { 0 });
                assert_eq!(
                    result[0],
                    [expected],
                    "warp={warp} raw companion={companion}"
                );
            }
        }
    }
}
