use super::State;
use crate::{descriptor, host, object, resource};
use victory_shader::{ResourceType, RootType, UsedResource};

#[test]
fn relocated_typed_uavs_follow_view_offsets_and_recorded_lifetime() {
    let host = host::host().unwrap();
    let list = host.new_list().unwrap();
    let null = std::ptr::null();
    let mut heap = null;
    let heap_desc = [0u32, 32, 1, 0, 0, 0, 0, 0];
    let iid = &resource::PAGEABLE_IIDS[0];
    assert_eq!(
        descriptor::create_heap(null, heap_desc.as_ptr().cast(), iid, &mut heap),
        object::S_OK
    );
    let backing = vec![0u64; 16384];
    let address = backing.as_ptr() as u64;
    #[repr(C, align(8))]
    struct Desc([u32; 14]);
    let mut guest = Desc([3, 0, 0, 0, 1, 0, 1, 0x10001, 42, 1, 0, 1, 4, 0]);
    let mut uint = null;
    let mut float = null;
    assert_eq!(
        resource::create_at_address(
            null,
            address,
            guest.0.as_ptr().cast(),
            0,
            null.cast(),
            iid,
            &mut uint
        ),
        object::S_OK
    );
    guest.0[8] = 2;
    assert_eq!(
        resource::create_at_address(
            null,
            address + 65536,
            guest.0.as_ptr().cast(),
            0,
            null.cast(),
            iid,
            &mut float
        ),
        object::S_OK
    );
    let root = descriptor::heap(heap).start + 64;
    let used = [8, 24].map(|offset| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: offset,
        lower_bound: offset / 8,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 3,
    });
    let mut state = State::default();
    state.set_heaps(&[heap]);
    state.set_table(true, 3, root);
    let bind = |state: &mut State| state.tables(&list, &used, &[], true);
    let mask = |state: &State, at| state.fixed_uint_mask(&used, 3, at, 0);
    descriptor::create_uav(null, float, null, null.cast(), root + 32);
    descriptor::create_uav(null, float, null, null.cast(), root + 96);
    assert!(bind(&mut state).is_empty());
    descriptor::create_uav(null, uint, null, null.cast(), root + 96);
    let first = bind(&mut state)[0].1;
    assert!(first >= 32 * 32 && first.is_multiple_of(32));
    assert_eq!(mask(&state, first), 2);
    assert_eq!(bind(&mut state), [(3, first)]);

    descriptor::create_uav(null, uint, null, null.cast(), root + 32);
    // A real integer resource with an incompatible view becomes a null UAV.
    let incompatible = [2u32, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    descriptor::create_uav(null, uint, null, &incompatible, root + 96);
    let second = bind(&mut state)[0].1;
    assert_ne!(first, second);
    assert_eq!(mask(&state, second), 1);
    assert_eq!(mask(&state, first), 2);

    let mut placed = [0u32; 16];
    placed[0] = 42;
    placed[1] = 4;
    placed[10] = address as u32;
    placed[11] = (address >> 32) as u32;
    guest.0[8] = 42;
    // A placed view past the only mip binds a null UAV and leaves its bit clear.
    let mut past_last_mip = placed;
    past_last_mip[2] = 1;
    descriptor::create_placed_uav(null, &guest.0, &past_last_mip, root + 96);
    let rejected = bind(&mut state)[0].1;
    assert_eq!(mask(&state, rejected), 1);

    // A compatible placed texture UAV reuses the named integer target.
    descriptor::create_placed_uav(null, &guest.0, &placed, root + 96);
    let third = bind(&mut state)[0].1;
    assert_eq!(mask(&state, third), 3);
    assert_eq!(state.typed_table_flags(&used, 3, first, 0), 2);
    assert_eq!(state.typed_table_flags(&used, 3, second, 0), 1);
    // Two heap objects are allowed to cover the very same guest descriptor
    // records. Their dynamic slots belong to different host heaps.
    let start = descriptor::heap(heap).start;
    #[repr(C, align(8))]
    struct PlacedHeap([u32; 8]);
    let extended = PlacedHeap([0, 32, 1, 0, start as u32, (start >> 32) as u32, 32 * 32, 0]);
    let mut alias = null;
    assert_eq!(
        descriptor::create_heap(null, extended.0.as_ptr().cast(), iid, &mut alias),
        object::S_OK
    );
    let other = descriptor::heap(alias).host().unwrap();
    let reservation = other.dynamic(20).unwrap();
    state.set_heaps(&[alias]);
    let aliased = bind(&mut state)[0].1;
    assert_ne!(
        aliased, third,
        "the second host heap must allocate its own table"
    );
    assert_eq!(state.typed_table_flags(&used, 3, aliased, 0), 3);
    state.set_heaps(&[heap]);
    assert_eq!(bind(&mut state), [(3, third)]);
    // The recording still uses the exact first heap after the title drops it.
    assert_eq!(descriptor::release_heap(heap), 1);
    other.release(reservation, 20);
    list.close();
    host.execute(&[&list]);
    state.tables_done();
    assert!(
        state
            .typed_dynamic
            .contains(&(heap as usize, first / 32, 4))
    );
    assert!(
        state
            .typed_dynamic
            .contains(&(heap as usize, second / 32, 4))
    );
    assert!(!state.typed_dynamic.iter().any(|&(_, i, _)| i == third / 32));
    drop(state);
    for obj in [alias, uint, float] {
        object::slot::<extern "system" fn(object::This) -> u32>(obj, 16)(obj);
    }
}
