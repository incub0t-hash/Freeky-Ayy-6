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

/// A root constant can be rebound to a command-list-local host slot only if
/// every use of its value is the guest view index multiplied by 32 and every
/// use of that result loads the identical four-dword raw buffer descriptor.
pub(super) fn direct_indices(n: &NativeProgram, p: &Program, res: &Resources) -> Vec<u32> {
    let raw = heap_raw_loads(n, p, res);
    if raw.is_empty() {
        return Vec::new();
    }
    let mut raw_loads = BTreeSet::new();
    let mut excluded = BTreeSet::new();
    for d in &res.descriptors {
        let inst = &p.blocks[d.site.0].insts[d.site.1];
        let eligible = d.role == Role::Resource
            && matches!(d.source, Source::Loaded { first: 0, .. })
            && (raw.contains(&d.site)
                || (inst.name().starts_with("buffer_load_dword")
                    && matches!(
                        inst.fields,
                        Fields::Buffer {
                            idxen: false,
                            format: None,
                            ..
                        }
                    )));
        for leaf in d.source.leaves() {
            if let Source::Loaded { load, .. } = leaf {
                if eligible {
                    raw_loads.insert(*load);
                } else {
                    excluded.insert(*load);
                }
            }
        }
    }
    selections(n, p, res, &raw_loads, &excluded)
        .into_keys()
        .collect()
}

/// Each scalar-only heap load selected by an exclusive root constant, and
/// the original user SGPR value whose five shifted-out bits are available
/// to communicate the concrete host descriptor kind.
pub(super) fn scalar_selections(
    n: &NativeProgram,
    p: &Program,
    res: &Resources,
) -> BTreeMap<Site, (u32, crate::ir::Value)> {
    if n.stage != HwStage::Cs {
        return BTreeMap::new();
    }
    let mut consumers = BTreeMap::<Site, Vec<Site>>::new();
    let mut excluded = BTreeSet::new();
    for d in &res.descriptors {
        let i = &p.blocks[d.site.0].insts[d.site.1];
        if let Source::Loaded { load, first: 0 } = d.source
            && d.role == Role::Resource
            && i.name().starts_with("s_buffer_load_dword")
        {
            consumers.entry(load).or_default().push(d.site);
        } else {
            for leaf in d.source.leaves() {
                if let Source::Loaded { load, .. } = leaf {
                    excluded.insert(*load);
                }
            }
        }
    }
    // Replacing a descriptor preserves its accessible bytes. It cannot
    // preserve arbitrary arithmetic on the descriptor words themselves.
    let mut owner = BTreeMap::new();
    for &load in consumers.keys() {
        for &(_, v) in &p.blocks[load.0].insts[load.1].writes {
            owner.insert(v, load);
        }
    }
    for (b, blk) in p.blocks.iter().enumerate() {
        for val in blk.phis.iter().flat_map(|phi| &phi.args) {
            if let Some(load) = owner.get(val) {
                excluded.insert(*load);
            }
        }
        for (k, i) in blk.insts.iter().enumerate() {
            for &(reg, v) in &i.reads {
                let Some(&load) = owner.get(&v) else { continue };
                let direct_operand = matches!(&i.fields,Fields::Smem{base:crate::gfx10::Operand::Scalar{code,dwords:4},soffset,..}
                    if matches!(reg,crate::ir::Reg::S(r) if (u16::from(*code)..u16::from(*code)+4).contains(&u16::from(r)))
                    && !matches!(soffset,crate::gfx10::Operand::Scalar{code,..} if reg==crate::ir::Reg::S(*code)));
                if !direct_operand || !consumers[&load].contains(&(b, k)) {
                    excluded.insert(load);
                }
            }
        }
    }
    let captures = selections(n, p, res, &consumers.keys().copied().collect(), &excluded);
    let mut result = BTreeMap::new();
    for (index, (v, loads)) in captures {
        for load in loads {
            for &site in &consumers[&load] {
                result.insert(site, (index, v));
            }
        }
    }
    result
}

fn selections(
    n: &NativeProgram,
    p: &Program,
    res: &Resources,
    allowed: &BTreeSet<Site>,
    excluded: &BTreeSet<Site>,
) -> BTreeMap<u32, (crate::ir::Value, BTreeSet<Site>)> {
    use crate::gfx10::Operand;
    use crate::ir::{Reg, Value};
    let mut uses = BTreeMap::<Value, Vec<Option<Site>>>::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            for v in &phi.args {
                uses.entry(*v).or_default().push(None);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.reads {
                uses.entry(v).or_default().push(Some((b, i)));
            }
        }
    }
    let addresses: BTreeMap<_, _> = res.scalar.iter().copied().collect();
    let mut indices = BTreeMap::new();
    for &(value, param, dword) in &res.params {
        let root = &n.layout[param];
        if root.root != RootType::Constants {
            continue;
        }
        let Some(uses_of_value) = uses.get(&value) else {
            continue;
        };
        let mut selected = BTreeSet::new();
        let all_selected = uses_of_value.iter().all(|site| {
            let Some((b, i)) = *site else { return false };
            let shift = &p.blocks[b].insts[i];
            let Fields::Sop {
                dst: Some(Operand::Scalar { code: out, .. }),
                src,
                ..
            } = &shift.fields
            else {
                return false;
            };
            let [Operand::Scalar { code: input, .. }, Operand::Const(133)] = src.as_slice() else {
                return false;
            };
            if shift.name() != "s_lshl_b32"
                || shift.read(Reg::S(*input)) != Some(value)
                || shift
                    .write(Reg::Scc)
                    .is_some_and(|scc| uses.contains_key(&scc))
            {
                return false;
            }
            let Some(bytes) = shift.write(Reg::S(*out)) else {
                return false;
            };
            uses.get(&bytes).is_some_and(|reads| {
                reads.iter().all(|site| {
                    let Some(site) = site else { return false };
                    let inst = &p.blocks[site.0].insts[site.1];
                    selected.insert(*site);
                    allowed.contains(site)
                        && !excluded.contains(site)
                        && inst.name() == "s_load_dwordx4"
                        && addresses.get(site).is_some_and(|a| {
                            a.offset == 0
                                && a.dynamic == Some(bytes)
                                && n.layout
                                    .get(a.param)
                                    .is_some_and(|r| r.root == RootType::ResourceHeap)
                        })
                })
            })
        });
        if all_selected && !selected.is_empty() {
            indices.insert(u32::from(root.user_data) + dword, (value, selected));
        }
    }
    indices
}

#[cfg(all(test, windows))]
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
    fn direct_raw_indices_require_exclusive_root_index_use() {
        for (extra, shift, expected) in [
            (false, 0x8f09_8502u32, true),
            (true, 0x8f09_8502u32, false),
            (false, 0x8f09_8402u32, false),
        ] {
            for companion in [false, true] {
                let mut n = native(RootType::ResourceHeap, companion);
                n.layout.push(RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 2,
                    slot: 2,
                });
                let StageIo::Cs(ref mut io) = n.io else {
                    unreachable!()
                };
                io.rsrc2 = 3 << 1;
                n.code[12..16].copy_from_slice(&0x1200_0000u32.to_le_bytes()); // descriptor offset in s9
                let mut prefix = shift.to_le_bytes().to_vec();
                if extra {
                    prefix.extend(0xbe8a_0302u32.to_le_bytes());
                } // s10 = s2 is a second use
                prefix.append(&mut n.code);
                n.code = prefix;
                let shader = emit_for(&n, Target::ANY).unwrap();
                assert_eq!(
                    shader.direct_raw_indices,
                    if expected && companion {
                        vec![2]
                    } else {
                        Vec::new()
                    }
                );
                assert_eq!(
                    shader.direct_scalar_indices,
                    if expected && !companion {
                        vec![2]
                    } else {
                        Vec::new()
                    }
                );
            }
        }
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
