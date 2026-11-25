//! Read the size fields of a direct-heap descriptor whose kind the shader
//! does not prove. A concrete host texture view supplies the measured bits.
use super::*;

fn reads_only(word: usize, inst: &Inst, reg: Reg) -> bool {
    let Fields::Sop { src, .. } = &inst.fields else {
        return false;
    };
    let [Operand::Scalar { code, dwords: 1 }, constant] = src.as_slice() else {
        return false;
    };
    if reg != Reg::S(*code) {
        return false;
    }
    let n = match constant {
        Operand::Const(n @ 128..=192) => u32::from(*n - 128),
        Operand::Literal(n) => *n,
        _ => return false,
    };
    let mask = match inst.name() {
        "s_lshr_b32" => u32::MAX << (n & 31),
        "s_lshl_b32" => u32::MAX >> (n & 31),
        "s_bfe_u32" => {
            let width = (n >> 16) & 127;
            if width == 0 || width > 32 {
                return false;
            }
            (u32::MAX >> (32 - width)) << (n & 31)
        }
        "s_and_b32" => n,
        _ => return false,
    };
    let known = match word {
        1 => 3 << 30,
        2 => 0x3fff_ffff,
        3 => 0xf000,
        _ => 0,
    };
    mask != 0 && mask & !known == 0
}

/// Only fields the host can reconstruct may affect the pixel. A slot of an
/// unsupported kind kills its lane before it can export or store anything.
fn measured_uses(p: &Program, site: Site, base: u8) -> bool {
    let load = &p.blocks[site.0].insts[site.1];
    let Some(words) = (0..8)
        .map(|k| load.write(Reg::S(base + k)))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let mut read = false;
    for block in &p.blocks {
        if block
            .phis
            .iter()
            .any(|phi| phi.args.iter().any(|v| words.contains(v)))
        {
            return false;
        }
        for inst in &block.insts {
            for &(reg, value) in &inst.reads {
                if let Some(word) = words.iter().position(|v| *v == value) {
                    if !reads_only(word, inst, reg) {
                        return false;
                    }
                    read = true;
                }
            }
        }
    }
    let effect = |i: &Inst| {
        matches!(i.fields, Fields::Exp { .. })
            || matches!(i.fields, Fields::Ds { .. }) && !i.name().starts_with("ds_swizzle")
            || i.name().contains("store")
            || i.name().contains("atomic")
            || i.name() == "s_barrier"
    };
    if !read || p.blocks[site.0].insts[..site.1].iter().any(effect) {
        return false;
    }
    let mut seen = HashSet::new();
    let mut preds = p.blocks[site.0].preds.clone();
    while let Some(b) = preds.pop() {
        if seen.insert(b) {
            if p.blocks[b].insts.iter().any(effect) {
                return false;
            }
            preds.extend_from_slice(&p.blocks[b].preds);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_image_size_requires_exact_native_read_bits_before_effects() {
        let scalar = |op: u32, source: u32, rhs: u32| {
            0x8000_0000 | (op << 23) | (31 << 16) | (rhs << 8) | source
        };
        let check = |prefix: &[u32], ops: &[u32]| {
            let words = [prefix, &[0xf40c_0200, 0xfa00_0000], ops, &[0xbf81_0000]].concat();
            let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let p = crate::ir::build(&bytes, false, 4).unwrap();
            let site = p.blocks[0]
                .insts
                .iter()
                .position(|i| i.name() == "s_load_dwordx8")
                .unwrap();
            measured_uses(&p, (0, site), 8)
        };
        let uses = [
            scalar(0x20, 9, 158),
            scalar(0x1e, 10, 130),
            scalar(0x27, 10, 255),
            16 << 16 | 14,
            scalar(0x27, 11, 255),
            4 << 16 | 12,
        ];
        assert!(check(&[], &uses));
        // d1 bit 29, d2 bit 30, d3 last mip and d0 (address) are unknown.
        for ops in [
            vec![scalar(0x20, 9, 157)],
            vec![scalar(0x1e, 10, 129)],
            vec![scalar(0x27, 11, 255), 8 << 16 | 12],
            vec![scalar(0x20, 8, 158)],
            vec![scalar(0x20, 9, 5)],
        ] {
            assert!(!check(&[], &ops));
        }
        assert!(!check(&[0xbf8a_0000], &uses)); // s_barrier before an unchecked slot
    }
}

impl Lower<'_> {
    /// A direct index from one dword of a fixed table CBV. The host can
    /// take the matching words at submission without scanning the heap.
    fn material_source(&self, site: Site) -> Option<(u8, u32)> {
        let a = self.scalar.get(&site)?;
        if a.offset != 0 {
            return None;
        }
        let def = |v| {
            let b = *self.def_block.get(&v)?;
            let i = self.x.p.blocks[b]
                .insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))?;
            Some(((b, i), &self.x.p.blocks[b].insts[i]))
        };
        let (_, shift) = def(a.dynamic?)?;
        let Fields::Sop { src, .. } = &shift.fields else {
            return None;
        };
        let [Operand::Scalar { code, dwords: 1 }, Operand::Const(133)] = src.as_slice() else {
            return None;
        };
        if shift.name() != "s_lshl_b32" {
            return None;
        }
        let v = shift.read(Reg::S(*code))?;
        let (load, contents) = def(v)?;
        let Fields::Smem {
            data: Some(Operand::Scalar { code: base, .. }),
            soffset: Operand::Scalar { code: NULL, .. },
            offset,
            ..
        } = contents.fields
        else {
            return None;
        };
        if !matches!(
            contents.name(),
            "s_buffer_load_dword" | "s_buffer_load_dwordx2" | "s_buffer_load_dwordx4"
        ) || offset < 0
            || offset % 4 != 0
        {
            return None;
        }
        let k = contents.writes.iter().find_map(|&(r, value)| match r {
            Reg::S(r) if value == v => r.checked_sub(base),
            _ => None,
        })?;
        let cbword = (offset as u32 / 4).checked_add(u32::from(k))?;
        let Source::Loaded {
            load: descriptor,
            first: 0,
        } = *self.descriptors.get(&(load, false))?
        else {
            return None;
        };
        let da = self.scalar.get(descriptor)?;
        let root = self.x.native.layout.get(da.param)?;
        let table = da.offset / 4;
        if root.root != RootType::Table
            || da.dynamic.is_some()
            || da.offset % 32 != 0
            || table > 0xff_ffff
            || cbword > 255
            || !self.x.native.used.iter().any(|u| {
                u.kind == ResourceType::Cbv
                    && u.root == RootType::Table
                    && u.user_data == root.user_data
                    && u.slot == root.slot
                    && u.table_offset_dwords == table
            })
        {
            return None;
        }
        Some((root.slot, (table << 8) | cbword))
    }
    pub(super) fn measured_size(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
    ) -> Result<(), Why> {
        const REFUSED: Why = "a descriptor read only as data";
        let address = self.scalar.get(&site).ok_or(REFUSED)?;
        if !self.s.pixel
            || count != 8
            || !self
                .x
                .native
                .layout
                .get(address.param)
                .is_some_and(|p| p.root == RootType::ResourceHeap)
            || !measured_uses(self.x.p, site, code)
        {
            return Err(REFUSED);
        }
        let slot = self.heap_index(site, 0, false)?;
        let (source_slot, source_dword) = self.material_source(site).ok_or(REFUSED)?;
        let marker = RootType::Other(super::super::USER_IMAGE_SIZE);
        if self
            .user_reads
            .values()
            .any(|u| u.root == marker && (u.slot, u.dword) != (source_slot, source_dword))
        {
            return Err(REFUSED);
        }
        self.host_word(marker, source_slot, source_dword);
        let c = self.cur;
        let h = self.s.root_srv_handle(super::super::SLOT_IMAGE_SIZE);
        let zero = self.s.m.i32(0);
        let two = self.s.m.i32(2);
        let four = self.s.m.i32(4);
        let raw = |l: &mut Self, offset| {
            let u = l.s.m.undef(l.s.t.i32);
            let mask = l.s.m.i8(1);
            let data = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, offset, u, mask, four],
            );
            let f = l.s.m.extract(c, data, 0);
            l.as_int(f)
        };
        let selected = raw(self, zero);
        let matches = self.s.m.cmp(c, pred::EQ, slot, selected);
        let packed = raw(self, four);
        let present = self.s.m.cmp(c, pred::NE, packed, zero);
        let valid = self.s.m.bin(c, Bin::And, matches, present);
        let pass = self.s.m.new_block();
        let abort = self.abort_block();
        self.s.m.terminate(c, Term::CondBr(valid, pass, abort));
        self.cur = pass;
        let ones = self.s.m.i32(u32::MAX);
        let bits = self.s.m.bin(pass, Bin::Xor, packed, ones);
        let k30 = self.s.m.i32(30);
        let k28 = self.s.m.i32(28);
        let k12 = self.s.m.i32(12);
        let k14 = self.s.m.i32(14);
        let mask = self.s.m.i32(0x3fff);
        let d1 = self.s.m.bin(pass, Bin::Shl, bits, k30);
        let w_hi = self.s.m.bin(pass, Bin::And, bits, mask);
        let w_hi = self.s.m.bin(pass, Bin::LShr, w_hi, two);
        let ht = self.s.m.bin(pass, Bin::LShr, bits, k14);
        let ht = self.s.m.bin(pass, Bin::And, ht, mask);
        let ht = self.s.m.bin(pass, Bin::Shl, ht, k14);
        let d2 = self.s.m.bin(pass, Bin::Or, w_hi, ht);
        let mip = self.s.m.bin(pass, Bin::LShr, bits, k28);
        let d3 = self.s.m.bin(pass, Bin::Shl, mip, k12);
        for (word, val) in [(1, d1), (2, d2), (3, d3)] {
            self.write(inst, Reg::S(code + word), val, None);
        }
        Ok(())
    }
}
