//! Independent buffer preparation; GPU commands and CPU-copy barriers stay ordered.

use super::{Batch, Step};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Action {
    write: bool,
    address: usize,
    bytes: usize,
}

pub(super) fn eligible(step: &Step) -> bool {
    matches!(step, Step::Refresh(..) | Step::GpuWrite(..))
}

fn disjoint(mut spans: Vec<(usize, usize)>) -> bool {
    spans.sort_unstable();
    spans.iter().all(|&(a, b)| a <= b) && spans.windows(2).all(|w| w[0].1 <= w[1].0)
}

fn group<T>(items: impl IntoIterator<Item = (usize, T)>) -> BTreeMap<usize, Vec<T>> {
    let mut groups = BTreeMap::<usize, Vec<T>>::new();
    for (key, item) in items {
        groups.entry(key).or_default().push(item);
    }
    groups
}

/// Returns the consumed prefix, or zero for the unchanged serial path.
pub(super) fn run(steps: &[Step], batch: &mut Batch) -> usize {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_PARALLEL_PREPARE").is_some_and(|v| v == "1")) || super::refresh_once() {
        return 0;
    }
    run_independent(steps, batch)
}

fn run_independent(steps: &[Step], batch: &mut Batch) -> usize {
    let count = steps.iter().take_while(|s| eligible(s)).count();
    if count < 8 {
        return 0;
    }
    let mut copies = BTreeMap::new();
    let mut actions = Vec::with_capacity(count);
    for step in &steps[..count] {
        let (shadow, address, bytes, write) = match step {
            Step::Refresh(s, a, n) => (s, *a, *n, false),
            Step::GpuWrite(s, a, n) => (s, *a, *n, true),
            _ => unreachable!(),
        };
        let key = Arc::as_ptr(shadow) as usize;
        copies.entry(key).or_insert_with(|| shadow.clone());
        actions.push((key, Action { write, address, bytes }));
    }
    if copies.len() < 2 {
        return 0;
    }
    let Some(spans) = copies.values().map(|s| s.preparation_storage()).collect::<Option<Vec<_>>>() else {
        return 0;
    };
    if !disjoint(spans) {
        return 0;
    }
    // A previous segment must finish before this segment starts. Each job
    // then preserves all refresh/snapshot dependencies of one buffer.
    batch.flush();
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| tracing::info!(steps = count, buffers = copies.len(), "independent buffer refresh/snapshot sequences run across memory workers (VICTORY_PARALLEL_PREPARE=1)"));
    let started = std::time::Instant::now();
    let before = super::PREPARING[0].load(super::Ordering::Relaxed);
    let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
    for (key, actions) in group(actions) {
        let shadow = copies.remove(&key).expect("copy collected above");
        jobs.push(Box::new(move || {
            let mut local = Batch::default();
            for action in actions {
                if action.write {
                    local.flush_one(&shadow);
                    let started = std::time::Instant::now();
                    shadow.writing(action.address, action.bytes);
                    #[cfg(feature = "diagnostics")]
                    {
                        use crate::copystat::{C, add};
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, started.elapsed().as_micros() as u64);
                    }
                    let _ = started;
                } else {
                    #[cfg(feature = "diagnostics")]
                    {
                        use crate::copystat::{C, add};
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, action.bytes as u64);
                    }
                    local.add(&shadow, action.address, action.bytes);
                }
            }
            local.flush();
        }));
    }
    crate::shadow::wide_run(jobs);
    let summed = super::PREPARING[0].load(super::Ordering::Relaxed).saturating_sub(before);
    let over = summed.saturating_sub(started.elapsed().as_micros() as u64);
    super::PREPARING[0].fetch_sub(over, super::Ordering::Relaxed);
    crate::framelog::sub(crate::framelog::T::CopyUs, over);
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_storage_stays_serial() {
        assert!(!disjoint(vec![(100, 200), (150, 250)]));
        assert!(!disjoint(vec![(100, 300), (120, 150)]));
        assert!(!disjoint(vec![(200, 100)]));
        assert!(disjoint(vec![(200, 300), (100, 200)]));
    }

    #[test]
    fn each_buffers_refresh_and_snapshot_order_is_preserved() {
        let input = vec![(2, "refresh old"), (1, "refresh"), (2, "snapshot old"), (2, "refresh new"), (1, "snapshot"), (2, "snapshot new")];
        let groups = group(input.clone());
        assert_eq!(groups[&2], ["refresh old", "snapshot old", "refresh new", "snapshot new"]);
        assert_eq!(groups[&1], ["refresh", "snapshot"]);
        assert_eq!(groups.values().map(Vec::len).sum::<usize>(), input.len());
    }

    #[test]
    fn prepared_snapshots_merge_shader_changes_without_losing_cpu_writes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        super::super::host().expect("a host device for real shadow allocations");
        const LEN: usize = 64 << 10;
        let make = || {
            // Kept for the process lifetime: the resource registries can
            // retain the title address beyond this test's local handles.
            let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_COMMIT | MEM_RESERVE | 0x0020_0000, PAGE_READWRITE) } as usize;
            assert_ne!(title, 0);
            unsafe { std::ptr::write_bytes(title as *mut u8, 0x44, LEN) };
            crate::shadow::register(title, title as u64, LEN as u64);
            let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a registered shadow");
            assert!(!shadow.carded(), "small test buffers use CPU-visible copies");
            (title, shadow)
        };
        let (a, sa) = make();
        let (b, sb) = make();
        let single: Vec<_> = (0..8).map(|_| Step::Refresh(sa.clone(), a, LEN)).collect();
        assert_eq!(run_independent(&single, &mut Batch::default()), 0, "one buffer keeps its serial path");
        let steps = vec![
            Step::Refresh(sa.clone(), a, LEN), Step::Refresh(sb.clone(), b, LEN),
            Step::GpuWrite(sa.clone(), a, 16), Step::GpuWrite(sb.clone(), b, 16),
            Step::Refresh(sa.clone(), a + 8, 4), Step::Refresh(sb.clone(), b + 8, 4),
            Step::GpuWrite(sa.clone(), a + 12, 4), Step::GpuWrite(sb.clone(), b + 12, 4),
            Step::Fill(0, 0, 0), // a barrier must not be consumed
        ];
        assert_eq!(run_independent(&steps, &mut Batch::default()), 8);
        // The queued snapshot copies complete before GPU work begins.
        crate::shadow::take_snapshots();
        for (title, shadow) in [(a, sa), (b, sb)] {
            let (data, _) = shadow.preparation_storage().unwrap();
            unsafe {
                assert_eq!(((data + 4) as *const u32).read_volatile(), 0x44444444);
                ((title + 8) as *mut u32).write_volatile(0x12345678);
                // Model completed GPU stores in the mapped shared copy.
                ((data + 4) as *mut u32).write_volatile(0xaabbccdd);
                ((data + 12) as *mut u32).write_volatile(0xccddeeff);
            }
            // Word 8 is inside the declared range but unchanged by the GPU;
            // its later CPU write must survive the entire range's merge.
            shadow.copy_back(title, 16);
            unsafe {
                assert_eq!(((title + 4) as *const u32).read_volatile(), 0xaabbccdd);
                assert_eq!(((title + 8) as *const u32).read_volatile(), 0x12345678);
                assert_eq!(((title + 12) as *const u32).read_volatile(), 0xccddeeff);
            }
        }
    }
}
