//! Prove a gather's clamped, finite LOD is integral on every lane that consumes it.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, Structured};

pub(super) fn integral(
    p: &Program,
    st: &Structured,
    defs: &HashMap<Value, usize>,
    value: Value,
    site: (usize, usize),
    console_flow: bool,
) -> bool {
    // Across loop iterations the same mask SSA atom can refer to different
    // lanes. Only compare masks across a forward-only control-flow graph.
    let forward = p
        .blocks
        .iter()
        .enumerate()
        .all(|(b, block)| block.preds.iter().all(|&pred| pred < b));
    let active = st.execs[site.0][site.1].unwrap_or(Bool::TRUE);
    let mut proof = Proof {
        p,
        st,
        defs,
        active,
        forward,
        console_flow,
        seen: HashMap::new(),
        single_visit: HashMap::new(),
    };
    proof.value(value, 0)
}

struct Proof<'a> {
    p: &'a Program,
    st: &'a Structured,
    defs: &'a HashMap<Value, usize>,
    active: Bool,
    forward: bool,
    console_flow: bool,
    seen: HashMap<Value, bool>,
    single_visit: HashMap<usize, bool>,
}

impl Proof<'_> {
    fn value(&mut self, value: Value, depth: usize) -> bool {
        if depth >= 64 {
            return false;
        }
        if let Some(&known) = self.seen.get(&value) {
            return known;
        }
        self.seen.insert(value, false);
        let Some(&b) = self.defs.get(&value) else {
            return false;
        };
        let block = &self.p.blocks[b];
        let proven = if let Some(phi) = block.phis.iter().find(|p| p.value == value) {
            !phi.args.is_empty() && phi.args.iter().all(|&arg| self.value(arg, depth + 1))
        } else if let Some((i, inst, dest)) =
            block.insts.iter().enumerate().find_map(|(i, inst)| {
                inst.writes
                    .iter()
                    .find(|w| w.1 == value)
                    .map(|&(dest, _)| (i, inst, dest))
            })
        {
            let guard = if self.console_flow {
                self.st.execs[b][i]
            } else {
                self.st.guards[b][i]
            };
            let all_write = guard.is_none_or(|g| {
                g == Bool::TRUE
                    || (!self.st.bools.meet(&[(self.active, true), (g, false)])
                        && self.stable_mask(g))
            });
            (all_write
                || inst
                    .read(dest)
                    .is_some_and(|old| self.value(old, depth + 1)))
                && self.result(inst, depth + 1)
        } else {
            false
        };
        self.seen.insert(value, proven);
        proven
    }

    /// The producer's predicate can be compared to the later consumer's
    /// EXEC when every atom in the former is immutable for this invocation.
    /// An unrelated loop elsewhere does not redefine a value written once.
    fn stable_mask(&mut self, guard: Bool) -> bool {
        if self.forward {
            return true;
        }
        self.st.bools.support(guard).into_iter().all(|atom| {
            let v = match atom {
                Atom::Constant(_) => return true,
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Phi(v)
                | Atom::Quad(v)
                | Atom::Opaque(v) => v,
            };
            let Some(&b) = self.defs.get(&v) else {
                return false;
            };
            *self.single_visit.entry(b).or_insert_with(|| {
                let mut todo = self.p.blocks[b].exit.successors();
                let mut seen = HashSet::new();
                while let Some(succ) = todo.pop() {
                    if succ == b {
                        return false;
                    }
                    if seen.insert(succ) {
                        todo.extend(self.p.blocks[succ].exit.successors());
                    }
                }
                true
            })
        })
    }

    fn result(&mut self, inst: &Inst, depth: usize) -> bool {
        let Fields::Valu(op) = &inst.fields else {
            return false;
        };
        // Partial writes, lane permutations and division by two can change
        // this property. Refuse modified forms outside the ones proved here.
        if op.sdwa.is_some() || op.dpp.is_some() || op.omod != 0 || op.opsel != 0 {
            return false;
        }
        match inst.name() {
            // Non-finite results stay non-finite or become an integral endpoint
            // through the accepted arithmetic and the final finite LOD clamps.
            "v_floor_f32" | "v_ceil_f32" | "v_trunc_f32" | "v_rndne_f32" | "v_cvt_f32_u32"
            | "v_cvt_f32_i32" => true,
            "v_mov_b32" | "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_max_f32"
            | "v_min_f32" => op.src.iter().all(|src| match *src {
                Operand::Vector { reg, dwords: 1 } => {
                    inst.read(Reg::V(reg)).is_some_and(|v| self.value(v, depth))
                }
                Operand::Literal(bits) => integral_bits(bits),
                Operand::Const(code) => crate::resource::inline(code).is_some_and(integral_bits),
                _ => false,
            }),
            _ => false,
        }
    }
}

fn integral_bits(bits: u32) -> bool {
    let x = f32::from_bits(bits);
    !x.is_finite() || x.fract() == 0.0
}
