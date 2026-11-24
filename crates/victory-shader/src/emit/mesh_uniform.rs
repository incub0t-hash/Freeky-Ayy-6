//! A group-uniform allocation in a mesh program with per-vertex outputs. The console
//! asks for its outputs from thread zero. DXIL needs the same counts in one
//! converged call before any thread can write an output.

use std::collections::{HashMap, HashSet};

use super::lower::Parts;
use crate::gfx10::{Fields, Operand};
use crate::io::Input;
use crate::ir::{EXEC, Exit, Inst, M0, Reg, Value};
use crate::native::{MeshIo, ResourceType, Usage};
use crate::resource::Site;
use crate::structure::{Atom, Bool, LaneExit};

#[derive(Clone, Copy)]
pub(super) struct Allocation {
    pub site: Site,
    pub vertices: Value,
    pub primitives: Value,
}

fn operand(i: &Inst, op: Operand) -> Option<Value> {
    match op {
        Operand::Scalar { code, .. } => i.read(Reg::S(code)),
        Operand::Vector { reg, .. } => i.read(Reg::V(reg)),
        _ => None,
    }
}

fn immediate(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c @ 128..=192) => Some(u32::from(c - 128)),
        Operand::Const(c @ 193..=208) => Some(0u32.wrapping_sub(u32::from(c - 192))),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

fn sources(i: &Inst) -> &[Operand] {
    match &i.fields {
        Fields::Sop { src, .. } => src,
        Fields::Valu(v) => &v.src,
        _ => &[],
    }
}

struct Proof<'a> {
    x: &'a Parts<'a>,
    defs: HashMap<Value, &'a Inst>,
    uniform: HashSet<Value>,
    full_writes: HashSet<Value>,
}

impl Proof<'_> {
    fn predicate(&self, g: Bool) -> bool {
        self.x
            .st
            .bools
            .support(g)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(v) if self.uniform.contains(v)))
    }

    fn group_operand(&self, i: &Inst, op: Operand) -> bool {
        immediate(op).is_some() || operand(i, op).is_some_and(|v| self.uniform.contains(&v))
    }

    fn definition(&self, v: Value, name: &str) -> Option<&Inst> {
        self.defs.get(&v).copied().filter(|i| i.name() == name)
    }

    fn argument(&self, i: &Inst, n: usize) -> Option<Value> {
        operand(i, *sources(i).get(n)?)
    }

    // The native allocation is reached by exactly the first thread. Rebuild
    // the flat ID from the wave in MergedWaveInfo and both halves of mbcnt.
    fn thread_zero(&self, i: &Inst) -> Option<()> {
        let [zero, thread] = sources(i) else {
            return None;
        };
        if i.name() != "v_cmpx_eq_u32" || immediate(*zero) != Some(0) {
            return None;
        }
        let add = self.definition(operand(i, *thread)?, "v_lshl_add_u32")?;
        let [wave, six, lane] = sources(add) else {
            return None;
        };
        if immediate(*six) != Some(6) {
            return None;
        }
        let bfe = self.definition(operand(add, *wave)?, "s_bfe_u32")?;
        let [src, range] = sources(bfe) else {
            return None;
        };
        if immediate(*range) != Some(0x40018)
            || !self
                .x
                .io
                .inputs
                .contains(&(operand(bfe, *src)?, Input::MergedWaveInfo))
        {
            return None;
        }
        let hi = self.definition(operand(add, *lane)?, "v_mbcnt_hi_u32_b32")?;
        let [mask, lo] = sources(hi) else { return None };
        if immediate(*mask) != Some(u32::MAX) {
            return None;
        }
        let lo = self.definition(operand(hi, *lo)?, "v_mbcnt_lo_u32_b32")?;
        let [mask, base] = sources(lo) else {
            return None;
        };
        (immediate(*mask) == Some(u32::MAX)
            && immediate(*base) == Some(0)
            && [lo, hi, add]
                .iter()
                .all(|op| op.writes.iter().all(|w| self.full_writes.contains(&w.1))))
        .then_some(())
    }

    // Unsigned bounds on the few integer operations used to build counts.
    // A selected alternative is bounded only when both alternatives are.
    fn bounds(&self, i: &Inst, op: Operand, depth: u8) -> Option<(u32, u32)> {
        if let Some(x) = immediate(op) {
            return Some((x, x));
        }
        let value = operand(i, op)?;
        let inst = *self.defs.get(&value)?;
        let depth = depth.checked_sub(1)?;
        let arg = |op| self.bounds(inst, op, depth);
        match (inst.name(), sources(inst)) {
            ("v_readfirstlane_b32" | "v_mov_b32" | "s_mov_b32", [src]) => arg(*src),
            ("v_cndmask_b32", [a, b, _]) => {
                let (a, b) = (arg(*a)?, arg(*b)?);
                Some((a.0.min(b.0), a.1.max(b.1)))
            }
            ("s_add_i32" | "s_add_u32", [a, b]) => {
                let (a, b) = (arg(*a)?, arg(*b)?);
                Some((a.0.checked_add(b.0)?, a.1.checked_add(b.1)?))
            }
            ("s_and_b32", [a, b]) => {
                Some((0, arg(*a).into_iter().chain(arg(*b)).map(|r| r.1).min()?))
            }
            ("s_bfe_u32", [_, field]) => {
                let width = (immediate(*field)? >> 16) & 0x7f;
                (width < 32).then(|| (0, (1u32 << width) - 1))
            }
            _ => None,
        }
    }

    fn allocation(&self, site: Site, cap: u32) -> Option<Allocation> {
        let p = self.x.p;
        let (b, k) = site;
        let m0 = p.blocks[b].insts[k].read(Reg::S(M0))?;
        let mov = self.definition(m0, "s_mov_b32")?;
        let packed = self.definition(self.argument(mov, 0)?, "s_or_b32")?;
        let [hi, lo] = sources(packed) else {
            return None;
        };
        let shift = self.definition(operand(packed, *hi)?, "s_lshl_b32")?;
        let [prim, twelve] = sources(shift) else {
            return None;
        };
        let vertex_bounds = self.bounds(packed, *lo, 12)?;
        if immediate(*twelve) != Some(12)
            || vertex_bounds.0 == 0
            || vertex_bounds.1 > cap
            || self.bounds(shift, *prim, 12)?.1 > cap
        {
            return None;
        }
        let vertices = operand(packed, *lo)?;
        let primitives = operand(shift, *prim)?;
        if ![vertices, primitives]
            .iter()
            .all(|v| self.uniform.contains(v))
        {
            return None;
        }
        let &[head] = p.blocks[b].preds.as_slice() else {
            return None;
        };
        let block = &p.blocks[head];
        let [.., compare, branch] = block.insts.as_slice() else {
            return None;
        };
        let Exit::Branch { taken, not_taken } = block.exit else {
            return None;
        };
        if not_taken != b
            || taken == b
            || branch.name() != "s_cbranch_execz"
            || self.x.st.reach[head] != Bool::TRUE
            || self.x.st.execs[head][block.insts.len() - 2] != Some(Bool::TRUE)
        {
            return None;
        }
        self.thread_zero(compare)?;
        // Emitting at the compare requires both counts to be available there,
        // and all output writes and exits to be dominated by that block.
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &b) in order.iter().enumerate() {
            index[b] = n
        }
        let dom = crate::ir::idoms(&p.blocks, &order, &index);
        let dominates = |a: usize, mut b: usize| loop {
            if a == b {
                break true;
            }
            if dom[b] == b || dom[b] == usize::MAX {
                break false;
            }
            b = dom[b];
        };
        for &n in &order {
            let blk = &p.blocks[n];
            if (matches!(blk.exit, Exit::End)
                || blk
                    .insts
                    .iter()
                    .any(|i| matches!(i.fields, Fields::Exp { .. })))
                && (n == head || !dominates(head, n))
            {
                return None;
            }
            for (k, i) in blk.insts.iter().enumerate() {
                if i.writes
                    .iter()
                    .any(|w| [vertices, primitives].contains(&w.1))
                    && (!dominates(n, head) || n == head && k >= block.insts.len() - 2)
                {
                    return None;
                }
            }
        }
        Some(Allocation {
            site: (head, block.insts.len() - 2),
            vertices,
            primitives,
        })
    }
}

// D3D12 allows 32 signature rows, with one interpolation mode per row,
// and 32 KiB of output storage per group. Host attributes are 32 bits
// even where the source packs two half-precision values into a register.
fn vertex_signature(outputs: &[super::container::SigElement], vertices: u32) -> bool {
    outputs
        .iter()
        .all(|e| matches!(e.kind, 0 | 3) && e.row.is_some_and(|r| r < 32))
        && outputs.iter().map(|e| u64::from(e.cols)).sum::<u64>() * 4 * u64::from(vertices) <= 32768
        && outputs.iter().all(|e| {
            outputs
                .iter()
                .all(|f| e.row != f.row || e.interp == f.interp)
        })
}

pub(super) fn prove<'a>(x: &'a Parts<'a>, ms: &MeshIo) -> Option<Allocation> {
    if !x.native.wave64
        || ms.threads != 128
        || ms.max_vertices != 128
        || ms.max_primitives != 128
        || ms.lds_bytes != 0
        || ms.topology != 2
        || !x.prims.is_empty()
        || !vertex_signature(&x.outputs, ms.max_vertices)
        || ms
            .outputs
            .iter()
            .any(|o| o.usage != Usage::Generic || o.param >= 32)
        || x.native.used.iter().any(|r| r.kind == ResourceType::Uav)
    {
        return None;
    }
    let p = x.p;
    let mut proof = Proof {
        x,
        defs: HashMap::new(),
        uniform: x.res.params.iter().map(|p| p.0).collect(),
        full_writes: HashSet::new(),
    };
    for &(v, i) in &x.io.inputs {
        if i == Input::MeshGroup {
            proof.uniform.insert(v);
        }
    }
    let mut alloc = None;
    let mut visited = HashSet::new();
    let mut group_edges = HashSet::new();
    let mut scalar_values = proof.uniform.clone();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        if !block.preds.iter().all(|q| visited.contains(q)) {
            return None;
        }
        visited.insert(b);
        let together = proof.predicate(x.st.reach[b]);
        let same_edge = block.preds.iter().all(|q| group_edges.contains(q));
        for phi in &block.phis {
            // A scalar surviving divergent paths could reach inactive lanes
            // through the console's SGPR file while their host copy differs.
            if !same_edge && !matches!(phi.reg, Reg::V(_)) {
                return None;
            }
            if same_edge && phi.args.iter().all(|v| proof.uniform.contains(v)) {
                proof.uniform.insert(phi.value);
                scalar_values.insert(phi.value);
            }
        }
        for (k, i) in block.insts.iter().enumerate() {
            let name = i.name();
            let full = together && x.st.execs[b][k] == Some(Bool::TRUE);
            let scalar_expression = match &i.fields {
                Fields::Sop { .. } => {
                    matches!(
                        name,
                        "s_mov_b32"
                            | "s_movk_i32"
                            | "s_add_i32"
                            | "s_add_u32"
                            | "s_mul_i32"
                            | "s_and_b32"
                            | "s_or_b32"
                            | "s_bfe_u32"
                            | "s_lshl_b32"
                            | "s_lshr_b32"
                            | "s_cselect_b32"
                    ) || name.starts_with("s_cmp_")
                }
                Fields::Smem { .. } => {
                    name.starts_with("s_load_") || name.starts_with("s_buffer_load_")
                }
                _ => false,
            };
            let uniform = match &i.fields {
                Fields::Exp {
                    target: 12 | 20 | 32..=63,
                    compr: false,
                    ..
                } => false,
                // Only scratch address seed registers. This proof rejects
                // every use of scratch or flat memory.
                Fields::Sop { imm: Some(imm), .. }
                    if name == "s_setreg_imm32_b32" && matches!(imm & 63, 20 | 21) =>
                {
                    false
                }
                Fields::Image {
                    tfe: false,
                    lwe: false,
                    ..
                } if name == "image_load" => false,
                Fields::Sopp { imm } if name == "s_sendmsg" => {
                    if *imm != 9 || alloc.replace((b, k)).is_some() {
                        return None;
                    }
                    false
                }
                Fields::Sopp { .. }
                    if matches!(
                        name,
                        "s_nop"
                            | "s_inst_prefetch"
                            | "s_waitcnt"
                            | "s_barrier"
                            | "s_endpgm"
                            | "s_branch"
                    ) || name.starts_with("s_cbranch_") =>
                {
                    false
                }
                Fields::Sop { .. } if !name.contains("getreg") && !name.contains("setreg") => {
                    together
                        && scalar_expression
                        && i.reads.iter().all(|(_, v)| proof.uniform.contains(v))
                }
                Fields::Smem { .. }
                    if name.starts_with("s_load_") || name.starts_with("s_buffer_load_") =>
                {
                    together && i.reads.iter().all(|(_, v)| proof.uniform.contains(v))
                }
                Fields::Buffer {
                    lds: false,
                    tfe: false,
                    ..
                } if name.starts_with("buffer_load_") || name.starts_with("tbuffer_load_") => {
                    full && i.reads.iter().all(|&(r, v)| {
                        matches!(r, Reg::S(c) if c == EXEC || c == EXEC+1)
                            || proof.uniform.contains(&v)
                    })
                }
                Fields::Valu(v)
                    if v.dpp.is_none() && v.sdwa.is_some() && name == "v_cvt_f32_u32" =>
                {
                    false
                }
                Fields::Valu(v) if v.dpp.is_none() && v.sdwa.is_none() => match name {
                    "v_readfirstlane_b32" => {
                        let group_value =
                            full && v.src.first().is_some_and(|&op| proof.group_operand(i, op));
                        // A scalar converted in this block is broadcast to
                        // every lane in this EXEC. Unchanged EXEC and an
                        // unguarded writer ensure its first lane agrees with
                        // every lane that consumes this scalar.
                        let previous = proof.argument(i, 0).and_then(|v| proof.defs.get(&v));
                        let local = previous.is_some_and(|def| {
                            let Some(j) =
                                block.insts[..k].iter().position(|w| std::ptr::eq(w, *def))
                            else {
                                return false;
                            };
                            def.name() == "v_cvt_u32_f32"
                                && matches!(sources(def), [Operand::Scalar { code: 0..=105, .. }])
                                && proof
                                    .argument(def, 0)
                                    .is_some_and(|v| scalar_values.contains(&v))
                                && x.st.guards[b][j].is_none()
                                && x.st.guards[b][k].is_none()
                                && [Reg::S(EXEC), Reg::S(EXEC + 1)]
                                    .iter()
                                    .all(|&r| i.read(r) == def.read(r))
                        });
                        if !group_value && !local {
                            return None;
                        }
                        group_value
                    }
                    "v_cndmask_b32" => {
                        full && v.src.len() == 3
                            && v.src[..2].iter().all(|&a| proof.group_operand(i, a))
                            && proof
                                .argument(i, 2)
                                .and_then(|m| x.st.masks.get(&m).or_else(|| x.st.others.get(&m)))
                                .is_some_and(|&g| proof.predicate(g))
                    }
                    "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                        if v.src.first().and_then(|&a| immediate(a)) != Some(u32::MAX) {
                            return None;
                        }
                        false
                    }
                    _ if name.contains("lane") => return None,
                    "v_mov_b32" => full && v.src.iter().all(|&a| proof.group_operand(i, a)),
                    _ => false,
                },
                _ => return None,
            };
            // Only the documented wave-index bits of s3 are synthesized.
            if i.reads
                .iter()
                .any(|(_, v)| x.io.inputs.contains(&(*v, Input::MergedWaveInfo)))
                && !(name == "s_bfe_u32"
                    && sources(i).get(1).and_then(|&a| immediate(a)) == Some(0x40018))
            {
                return None;
            }
            let local_scalar =
                scalar_expression && i.reads.iter().all(|(_, v)| scalar_values.contains(v));
            for &(reg, v) in &i.writes {
                proof.defs.insert(v, i);
                if full && x.st.reach[b] == Bool::TRUE {
                    proof.full_writes.insert(v);
                }
                if uniform
                    && !matches!(reg, Reg::S(EXEC))
                    && !x.st.masks.contains_key(&v)
                    && !x.st.highs.contains_key(&v)
                {
                    proof.uniform.insert(v);
                }
                if (local_scalar || proof.uniform.contains(&v))
                    && matches!(reg, Reg::S(0..=105) | Reg::Scc)
                    && !x.st.masks.contains_key(&v)
                    && !x.st.highs.contains_key(&v)
                {
                    scalar_values.insert(v);
                }
            }
        }
        if together
            && match x.st.exits[b] {
                LaneExit::Branch { when, .. } => proof.predicate(when),
                LaneExit::Jump(_) | LaneExit::End => true,
                LaneExit::Indirect => return None,
            }
        {
            group_edges.insert(b);
        }
    }
    proof.allocation(alloc?, ms.threads)
}

#[cfg(test)]
mod tests {
    use super::vertex_signature;
    use crate::emit::container::SigElement;

    #[test]
    fn expanded_attributes_fit_host_output_storage_and_packing() {
        let row = |n| SigElement {
            name: "TEXCOORD".into(),
            index: n,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(n),
            col: 0,
            cols: 4,
            usage: 15,
            half: true,
        };
        let mut outputs: Vec<_> = (0..16).map(row).collect();
        assert!(vertex_signature(&outputs, 128));
        outputs.push(row(16));
        assert!(!vertex_signature(&outputs, 128));
        let mut shared = [row(1), row(1)];
        shared[0].cols = 1;
        shared[1].cols = 1;
        shared[1].col = 1;
        assert!(vertex_signature(&shared, 128));
        shared[1].interp = 1;
        assert!(!vertex_signature(&shared, 128));
    }
}
