//! Submission-time evidence for the two refused PS_PuddleDetect variants.
//! Their native scalar loads use material word 0 for the metadata and words
//! 1 and 2 for the sampled views. They do not prove that those indices match.
use super::{HostList, Ordering, Root, State, This, View};
use crate::descriptor::observation;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64};

const MATERIAL_BYTES: usize = 16;

impl State {
    pub(super) fn observe_puddle(&self, list: &HostList, draw: &'static str) {
        static ON: OnceLock<bool> = OnceLock::new();
        static SAMPLES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_PUDDLE_TRACE").is_some_and(|v| v == "1")) {
            return;
        }
        let Some((which, shader)) = crate::pipeline::puddle_shader(self.pipeline as This) else {
            return;
        };
        let sample = SAMPLES[which].fetch_add(1, Ordering::Relaxed) + 1;
        if sample > 4 && !sample.is_power_of_two() {
            return;
        }
        let frame = crate::frame::current();
        // Embedded b19 metadata in both shaders: table parameter 1, offset
        // 72 dwords. This table can be in title memory outside a visible heap.
        let handle = match self.graphics.get(1) {
            Some(Root::Table(table)) if *table != 0 => table.checked_add(72 * 4),
            _ => None,
        };
        let Some((key, View::Constant(address, size))) = handle.and_then(observation::view) else {
            tracing::info!(
                shader,
                sample,
                frame,
                draw,
                ?handle,
                "puddle material: missing b19 CBV"
            );
            return;
        };
        let heap = observation::bound_heap(self.heaps[0]);
        if size < MATERIAL_BYTES as u32
            || address == 0
            || address.checked_add(MATERIAL_BYTES as u64).is_none()
        {
            tracing::info!(
                shader,
                sample,
                frame,
                draw,
                address,
                size,
                "puddle material: invalid CBV extent"
            );
            return;
        }
        let Some((_, to)) = list.scratch(MATERIAL_BYTES as u64) else {
            tracing::info!(
                shader,
                sample,
                frame,
                draw,
                "puddle material: no upload scratch"
            );
            return;
        };
        let repeats = AtomicU32::new(0);
        observe_fill(list, to, address, move |words| {
            // A previously recorded list may be submitted indefinitely.
            if repeats
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    (n < 3).then_some(n + 1)
                })
                .is_err()
            {
                return;
            }
            let submitted = crate::frame::current();
            let current_key = handle.and_then(observation::view).map(|(k, _)| k);
            tracing::info!(
                shader,
                sample,
                frame,
                submitted,
                draw,
                address,
                size,
                key,
                ?current_key,
                ?words,
                "puddle material"
            );
            if let Some(indices) = words {
                for (role, index) in ["metadata", "sample1", "sample2"].into_iter().zip(indices) {
                    let slot = heap
                        .map(|h| h.slot(index))
                        .unwrap_or_else(|| "no bound view heap".into());
                    tracing::info!(shader, sample, frame, submitted, role, index, %slot, "puddle view");
                }
            }
        });
    }
}

/// Copy through the same deferred Fill used for table CBVs. The callback sees
/// its uploaded words after earlier list steps. Confirm source availability so
/// an unmapped guest page zero-filled by the uploader cannot identify slot 0.
fn observe_fill(
    list: &HostList,
    to: *mut u8,
    address: u64,
    inspect: impl Fn(Option<[u32; 4]>) + Send + 'static,
) {
    list.defer_fill(to, address as *const u8, MATERIAL_BYTES);
    let to = to as usize;
    list.defer_host(move || {
        let mut source = [0u8; MATERIAL_BYTES];
        let mut uploaded = [0u8; MATERIAL_BYTES];
        let valid = crate::host::read_into(address as *const u8, &mut source)
            && crate::host::read_into(to as *const u8, &mut uploaded)
            && uploaded == source;
        inspect(valid.then(|| {
            std::array::from_fn(|i| {
                u32::from_le_bytes(uploaded[i * 4..i * 4 + 4].try_into().unwrap())
            })
        }));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn puddle_material_observes_the_words_copied_to_the_gpu_at_submission() {
        let host = crate::host::host().expect("D3D12 adapter");
        let list = host.new_list().expect("direct list");
        let (upload, destination) = list.upload(host, 256).expect("upload");
        let readback = host.read_back_buffer(16).expect("readback");
        let mut material = [0xDEAD_BEEFu32; 4];
        let source = [13u32, 29, 47, 0xABCD_EF01];
        let seen = Arc::new(Mutex::new(None));
        let copy = Arc::clone(&seen);
        list.defer_copy(material.as_mut_ptr().cast(), source.as_ptr().cast(), 16);
        observe_fill(&list, destination, material.as_ptr() as u64, move |words| {
            *copy.lock().unwrap() = words;
        });
        // SAFETY: upload and readback live through submission. These two heap
        // types are already in GENERIC_READ and COPY_DEST respectively.
        unsafe { list.list.CopyBufferRegion(&readback, 0, &upload, 0, 16) };
        list.close();
        assert!(seen.lock().unwrap().is_none());
        host.execute(&[&list]);
        assert_eq!(*seen.lock().unwrap(), Some(source));
        let mut mapped = std::ptr::null_mut();
        // SAFETY: synchronous execution has completed the copy of four words.
        unsafe {
            readback
                .Map(0, None, Some(&mut mapped))
                .expect("map readback");
            let gpu = mapped.cast::<[u32; 4]>().read_unaligned();
            readback.Unmap(0, None);
            assert_eq!(gpu, source);
        }
    }
}
