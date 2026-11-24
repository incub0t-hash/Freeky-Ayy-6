//! Identify direct dispatches binding the UINT textures sampled later as RGB9E5.
//! A matching UAV and nonzero dispatch establish a candidate, not pixel coverage.

use super::{ResourceType, Root, RootType, State, This, UsedResource, View};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CAPTURE_PACKED_WRITERS").is_some_and(|v| v == "1"))
}

impl State {
    pub(super) fn trace_packed_producers(
        &self,
        hash: &str,
        used: &[UsedResource],
        made: &[(u8, u32)],
        groups: [u32; 3],
    ) {
        let Ok(program) = u64::from_str_radix(hash, 16) else {
            return;
        };
        for u in used
            .iter()
            .filter(|u| u.kind == ResourceType::Uav && u.root == RootType::Table)
        {
            let Some(&Root::Table(table)) = self.compute.get(usize::from(u.slot)) else {
                continue;
            };
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(handle) = table.checked_add(u64::from(u.table_offset_dwords) * 4) else {
                continue;
            };
            let Some(guest_index) = heap.index(handle) else {
                continue;
            };
            let Some(View::Unordered(resource, _, view)) = crate::descriptor::view(handle) else {
                continue;
            };
            let Some((address, desc)) = crate::resource::texture_memory(resource as This) else {
                continue;
            };
            if !desc.is_packed_producer() {
                continue;
            }
            let index = match made.iter().find(|&&(slot, _)| slot == u.slot) {
                Some(&(_, start)) => (start / 32).checked_add(u.table_offset_dwords / 8),
                None => Some(guest_index),
            };
            let retained = index
                .and_then(|i| heap.host()?.retained_resource(i))
                .flatten();
            let matched = retained
                .as_ref()
                .is_some_and(|r| Some(r) == crate::resource::host_resource(resource as This));
            type Key = (u64, u8, u32, usize, [u32; 3], Option<[u32; 10]>, bool);
            static SEEN: OnceLock<Mutex<HashSet<Key>>> = OnceLock::new();
            let mut seen = SEEN
                .get_or_init(Mutex::default)
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if seen.len() >= 512
                || !seen.insert((
                    program,
                    u.slot,
                    u.table_offset_dwords,
                    address as usize,
                    groups,
                    view,
                    matched,
                ))
            {
                continue;
            }
            drop(seen);
            tracing::info!(
                hash,
                ?groups,
                slot = u.slot,
                offset = u.table_offset_dwords,
                ?address,
                ?desc,
                ?view,
                ?index,
                matched,
                presented = crate::queue::presented(),
                "direct dispatch packed texture UAV"
            );
        }
    }
}
