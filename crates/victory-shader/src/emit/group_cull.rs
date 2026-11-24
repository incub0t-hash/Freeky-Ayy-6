//! Wave-independent acyclic compute using one group-wide LDS allocation.
//!
//! Adapt the uniform LDS checkpoint rule from alfycs/VIctory#1 and
//! Marvisak/VIctory#189. A group may contain two guest waves only when
//! every checkpoint is reached by both and no wave-wide result survives.

use super::Target;
use crate::gfx10::{Fields, Operand};
use crate::io::{Input, Io};
use crate::ir::{Exit, Inst, Program, Reg, Value};
use crate::native::{NativeProgram, StageIo};
use crate::resource::Resources;
use crate::structure::{Atom, Bool, LaneExit, Structured};
use std::collections::HashSet;

pub(super) fn candidate(n: &NativeProgram, p: &Program, t: Target) -> bool {
    n.wave64
        && !t.runs(64)
        && t.runs(32)
        && matches!(n.io,StageIo::Cs(cs) if cs.threads==[128,1,1] && cs.lds_bytes()>0)
        && p.blocks
            .iter()
            .enumerate()
            .all(|(b, blk)| blk.preds.iter().all(|&q| q < b))
        && p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_readfirstlane_b32")
}

/// Skipping a straight-line vector body when EXEC is empty is optional.
/// Its LDS operations retain EXEC. Only dead scalar temporaries may change
/// on a skipped wave. Rebuild the original SSA/EXEC analysis after removing
/// the skip so the masked LDS instructions sit in a converged block.
pub(super) fn predicated_code(n: &NativeProgram, p: &Program) -> Option<Vec<u8>> {
    let mut offsets = Vec::new();
    for (head, b) in p.blocks.iter().enumerate() {
        let Exit::Branch {
            taken: join,
            not_taken: body,
        } = b.exit
        else {
            continue;
        };
        let arm = &p.blocks[body];
        let Some(branch) = b.insts.last() else {
            continue;
        };
        if branch.name() != "s_cbranch_execz"
            || arm.preds != [head]
            || arm.exit != Exit::Jump(join)
            || !arm
                .insts
                .iter()
                .any(|i| matches!(i.fields, Fields::Ds { .. }))
        {
            continue;
        }
        let safe = arm.insts.iter().all(|i| match &i.fields {
            Fields::Valu(v) => {
                v.sdwa.is_none()
                    && v.dpp.is_none()
                    && plain_vector(i)
                    && !i.name().starts_with("v_cmpx")
                    && i.name() != "v_readfirstlane_b32"
            }
            Fields::Sop { .. } => i.name() == "s_add_u32",
            Fields::Sopp { .. } => matches!(i.name(), "s_waitcnt" | "s_nop"),
            Fields::Ds { gds: false, .. } => matches!(
                i.name(),
                "ds_read_b32"
                    | "ds_read2_b32"
                    | "ds_read2st64_b32"
                    | "ds_write_b32"
                    | "ds_write2_b32"
                    | "ds_write2st64_b32"
            ),
            _ => false,
        });
        if !safe
            || arm
                .insts
                .iter()
                .flat_map(|i| &i.writes)
                .any(|r| matches!(r.0, Reg::S(126 | 127)))
        {
            continue;
        }
        let scalars: HashSet<_> = arm
            .insts
            .iter()
            .flat_map(|i| &i.writes)
            .filter(|(r, _)| !matches!(r, Reg::V(_)))
            .map(|r| r.1)
            .collect();
        if p.blocks
            .iter()
            .enumerate()
            .filter(|(b, _)| *b != body)
            .any(|(_, block)| {
                block
                    .insts
                    .iter()
                    .flat_map(|i| &i.reads)
                    .any(|r| scalars.contains(&r.1))
                    || block
                        .phis
                        .iter()
                        .flat_map(|p| &p.args)
                        .any(|v| scalars.contains(v))
            })
        {
            continue;
        }
        offsets.push(branch.machine.offset);
    }
    if offsets.is_empty() {
        return None;
    }
    let mut code = n.patched_code();
    for offset in offsets {
        let word = code.get_mut(offset..offset + 4)?;
        if u32::from_le_bytes((*word).try_into().ok()?) >> 16 != 0xbf88 {
            return None;
        }
        word.copy_from_slice(&0xbf800000u32.to_le_bytes()); // s_nop 0
    }
    Some(code)
}

fn plain_vector(i: &Inst) -> bool {
    matches!(
        i.name(),
        "v_add_f32"
            | "v_add_nc_u32"
            | "v_add3_u32"
            | "v_and_b32"
            | "v_bfe_u32"
            | "v_ceil_f32"
            | "v_cmp_eq_u32"
            | "v_cmp_ge_f32"
            | "v_cmp_gt_f32"
            | "v_cmp_gt_u32"
            | "v_cmp_lt_f32"
            | "v_cmp_ne_u32"
            | "v_cmpx_gt_u32"
            | "v_cndmask_b32"
            | "v_cvt_f32_i32"
            | "v_cvt_f32_u32"
            | "v_cvt_i32_f32"
            | "v_exp_f32"
            | "v_floor_f32"
            | "v_fma_f32"
            | "v_fmac_f32"
            | "v_log_f32"
            | "v_lshl_add_u32"
            | "v_lshl_or_b32"
            | "v_lshlrev_b32"
            | "v_lshrrev_b32"
            | "v_max_f32"
            | "v_min_f32"
            | "v_min_u32"
            | "v_min3_f32"
            | "v_mov_b32"
            | "v_mul_legacy_f32"
            | "v_mul_lo_u32"
            | "v_mul_u32_u24"
            | "v_perm_b32"
            | "v_rcp_f32"
            | "v_readfirstlane_b32"
            | "v_sub_f32"
    )
}

#[derive(Default)]
struct Uniform {
    data: HashSet<Value>,
    lane: HashSet<Value>,
}
impl Uniform {
    fn predicate(&self, st: &Structured, g: Bool) -> bool {
        st.bools.support(g).iter().all(|a| match a {
            Atom::Uniform(v) => self.data.contains(v),
            Atom::Lane(v) => self.lane.contains(v),
            _ => false,
        })
    }
    fn operand(&self, i: &Inst, o: &Operand) -> bool {
        match *o {
            Operand::Const(128..=208 | 240..=248)
            | Operand::Literal(_)
            | Operand::Scalar { code: 125, .. } => true,
            Operand::Scalar { code, dwords } => (0..dwords).all(|n| {
                i.read(Reg::S(code + n))
                    .is_some_and(|v| self.data.contains(&v))
            }),
            Operand::Vector { reg, dwords } => (0..dwords).all(|n| {
                i.read(Reg::V(reg + n))
                    .is_some_and(|v| self.data.contains(&v))
            }),
            _ => false,
        }
    }
}

pub(super) fn prove(p: &Program, st: &Structured, res: &Resources, io: &Io) -> bool {
    let local_x = io
        .inputs
        .iter()
        .find(|x| x.1 == Input::ThreadIdInGroup(0))
        .map(|x| x.0);
    let mut u = Uniform::default();
    u.data.extend(res.params.iter().map(|x| x.0));
    u.data.extend(
        io.inputs
            .iter()
            .filter(|x| matches!(x.1, Input::GroupId(_)))
            .map(|x| x.0),
    );
    let mut uniform_edges = HashSet::new();
    let mut single = Vec::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        let converged = u.predicate(st, st.reach[b]);
        for phi in &blk.phis {
            if blk.preds.iter().all(|q| uniform_edges.contains(q))
                && phi.args.iter().all(|v| u.data.contains(v))
            {
                u.data.insert(phi.value);
            }
        }
        for (k, i) in blk.insts.iter().enumerate() {
            let n = i.name();
            let full = converged && st.guards[b][k].is_none();
            let all = i.reads.iter().all(|r| u.data.contains(&r.1));
            let operand = |o: &Operand| u.operand(i, o);
            let uniform = match &i.fields {
                Fields::Sopp { .. }
                    if matches!(
                        n,
                        "s_nop"
                            | "s_waitcnt"
                            | "s_barrier"
                            | "s_branch"
                            | "s_endpgm"
                            | "s_cbranch_scc0"
                            | "s_cbranch_scc1"
                            | "s_cbranch_execz"
                            | "s_cbranch_vccz"
                            | "s_cbranch_vccnz"
                    ) =>
                {
                    if n == "s_barrier" && !converged {
                        return false;
                    }
                    false
                }
                Fields::Sop { .. }
                    if matches!(
                        n,
                        "s_add_i32"
                            | "s_add_u32"
                            | "s_and_b32"
                            | "s_and_b64"
                            | "s_andn2_b64"
                            | "s_bfe_u32"
                            | "s_cmp_eq_u32"
                            | "s_cmp_lg_u32"
                            | "s_cmp_gt_i32"
                            | "s_cmp_lt_i32"
                            | "s_cselect_b64"
                            | "s_lshl_b32"
                            | "s_lshr_b32"
                            | "s_mov_b32"
                            | "s_mov_b64"
                            | "s_movk_i32"
                            | "s_mul_i32"
                            | "s_or_b32"
                            | "s_or_b64"
                            | "s_sub_i32"
                            | "s_xor_b32"
                    ) =>
                {
                    converged && all
                }
                // Masking instructions are tracked bit by bit by structure.
                Fields::Sop { .. } if n == "s_and_saveexec_b64" => false,
                // These shaders initialize the unused flat scratch register pair.
                // The proof accepts no scratch memory operations or register reads.
                Fields::Sop { imm: Some(imm), .. }
                    if n == "s_setreg_imm32_b32" && matches!(imm & 63, 20 | 21) =>
                {
                    false
                }
                Fields::Smem { .. }
                    if n.starts_with("s_load_") || n.starts_with("s_buffer_load_") =>
                {
                    converged && all
                }
                Fields::Valu(v) if v.dpp.is_none() && plain_vector(i) => {
                    if n == "v_cmp_eq_u32"
                        && v.sdwa.is_none()
                        && v.neg == 0
                        && v.abs == 0
                        && v.opsel == 0
                        && st.execs[b][k] == Some(Bool::TRUE)
                        && let [Operand::Const(128), Operand::Vector { reg, dwords: 1 }] =
                            v.src.as_slice()
                        && local_x.is_some()
                        && i.read(Reg::V(*reg)) == local_x
                        && let Some(Operand::Scalar { code, dwords: 2 }) = v.dst
                        && let Some(lo) = i.write(Reg::S(code))
                        && let Some(mask) = st.mask_bits(lo, i.write(Reg::S(code + 1)))
                    {
                        single.push(mask);
                    }
                    let result = if n == "v_cndmask_b32" {
                        v.src.len() == 3
                            && v.src[..2].iter().all(operand)
                            && matches!(v.src[2],Operand::Scalar{code,dwords:2} if
                                i.read(Reg::S(code)).and_then(|lo| st.mask_bits(lo,i.read(Reg::S(code+1)))).is_some_and(|m|u.predicate(st,m)))
                    } else {
                        v.src.iter().all(operand)
                            && (n != "v_fmac_f32" || v.dst.as_ref().is_some_and(operand))
                    };
                    if n == "v_readfirstlane_b32" && !(full && result) {
                        return false;
                    }
                    full && v.sdwa.is_none_or(|s| {
                        s.dst_sel == 6 || s.dst_unused != 2 || v.dst.as_ref().is_some_and(operand)
                    }) && result
                }
                Fields::Ds {
                    gds: false,
                    addr: Some(addr),
                    ..
                } if converged
                    && matches!(
                        n,
                        "ds_read_b32"
                            | "ds_read2_b32"
                            | "ds_read2st64_b32"
                            | "ds_write_b32"
                            | "ds_write2_b32"
                            | "ds_write2st64_b32"
                            | "ds_or_b32"
                    ) =>
                {
                    // Checkpoints before and after the group of reads prevent group-shared
                    // data from changing while different physical waves load it.
                    full && n.starts_with("ds_read") && operand(addr)
                }
                Fields::Buffer {
                    addr,
                    rsrc,
                    soffset,
                    lds: false,
                    tfe: false,
                    ..
                } if n.starts_with("buffer_load_") => {
                    full && operand(addr) && operand(rsrc) && operand(soffset)
                }
                Fields::Image {
                    addr,
                    rsrc,
                    sampler: Some(sampler),
                    ..
                } if matches!(n, "image_gather4_l" | "image_gather4_l_o") => {
                    full && addr.iter().all(operand) && operand(rsrc) && operand(sampler)
                }
                Fields::Buffer {
                    lds: false,
                    tfe: false,
                    ..
                } if matches!(
                    n,
                    "buffer_atomic_add" | "buffer_store_format_xyzw" | "buffer_store_dword"
                ) =>
                {
                    if !single.iter().any(|&one| {
                        !st.bools.meet(&[
                            (st.reach[b], true),
                            (st.guards[b][k].unwrap_or(Bool::TRUE), true),
                            (one, false),
                        ])
                    }) {
                        return false;
                    }
                    false
                }
                _ => return false,
            };
            if uniform {
                for &(_, v) in &i.writes {
                    if n.starts_with("v_cmp") {
                        u.lane.insert(v);
                    } else {
                        u.data.insert(v);
                    }
                }
            }
        }
        let branch_uniform = match st.exits[b] {
            LaneExit::Branch { when, .. } => u.predicate(st, when),
            LaneExit::Jump(_) | LaneExit::End => true,
            LaneExit::Indirect => return false,
        };
        if !branch_uniform && !empty_exec_skip(p, b) {
            return false;
        }
        if converged && branch_uniform {
            uniform_edges.insert(b);
        }
    }
    true
}

fn empty_exec_skip(p: &Program, b: usize) -> bool {
    let head = &p.blocks[b];
    if !head
        .insts
        .last()
        .is_some_and(|i| i.name() == "s_cbranch_execz")
    {
        return false;
    }
    let Exit::Branch { taken, not_taken } = head.exit else {
        return false;
    };
    let skip = &p.blocks[taken];
    if skip.insts.len() == 1 && skip.insts[0].name() == "s_endpgm" {
        // A lane absent from the final EXEC can finish here only if the tail
        // never reactivates it. The only externally visible writer is lane 0.
        let mut next = not_taken;
        loop {
            let block = &p.blocks[next];
            if block
                .insts
                .iter()
                .flat_map(|i| &i.writes)
                .any(|r| matches!(r.0, Reg::S(126 | 127)))
            {
                return false;
            }
            match block.exit {
                Exit::Jump(q) if q > next => next = q,
                Exit::End => return true,
                _ => return false,
            }
        }
    }
    let body = &p.blocks[not_taken];
    body.preds == [b]
        && body.exit == Exit::Jump(taken)
        && body.insts.iter().all(|i| {
            i.writes.iter().all(|w| matches!(w.0, Reg::V(_)))
                && match &i.fields {
                    Fields::Valu(v) => v.dpp.is_none() && plain_vector(i),
                    Fields::Sopp { .. } => matches!(i.name(), "s_nop" | "s_waitcnt"),
                    Fields::Buffer {
                        lds: false,
                        tfe: false,
                        ..
                    } => i.name().starts_with("buffer_load_"),
                    _ => false,
                }
        })
}

/// Every LDS instruction is now in a block reached uniformly by the group.
/// A read and another read may proceed together (#189 / c047885). A write
/// or atomic must finish before the next access. Existing s_barrier fulfills
/// the pending checkpoint, including at the end of a sequence of reads.
pub(super) fn checkpoints(p: &Program) -> HashSet<(usize, usize)> {
    let mut sites = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate() {
        let mut previous = None;
        let mut last = 0;
        for (k, i) in block.insts.iter().enumerate() {
            if i.name() == "s_barrier" {
                previous = None;
            } else if matches!(i.fields, Fields::Ds { .. }) {
                let write = !i.name().starts_with("ds_read");
                if previous.is_some_and(|prior| prior || write) {
                    sites.insert((b, k));
                }
                previous = Some(write);
                last = k;
            }
        }
        if previous.is_some() {
            sites.insert((b, last + 1));
        }
    }
    sites
}
