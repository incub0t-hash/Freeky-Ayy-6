#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // source extent in bytes (logical for the cooperative kernel)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// Stored at upload/initialization or when the relay advances its baseline.
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
uint4 mix_words(uint4 h, uint4 words) {
    h = (h ^ words) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
    return h.yzwx ^ (h >> 15);
}

static const uint4 seed = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);

#ifdef COOPERATIVE
// The load is DWORD aligned; select precisely the guest bytes in [lo, hi).
uint byte_mask(uint at, uint lo, uint hi) {
    uint begin = min(4u, max(at, lo) - at);
    uint end = min(4u, max(at, hi) - at);
    uint left = begin == 4u ? 0u : 0xffffffffu << (begin * 8u);
    uint right = end == 4u ? 0xffffffffu : (1u << (end * 8u)) - 1u;
    return left & right;
}

groupshared uint changed;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    if (group.x >= pieces) return;
    uint piece = first + group.x;
    uint lo = max(piece * 65536u, skew) - skew;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    uint4 h = seed;
    // Adjacent lanes read adjacent 16-byte stripes. A short final load
    // remains in the padded GPU allocation; its unused bytes are masked.
    for (uint at = (lo & ~15u) + lane * 16u; at < hi; at += 64u * 16u) {
        uint4 words = now.Load4(at);
        if (at < lo || at + 16u > hi) {
            words &= uint4(byte_mask(at, lo, hi), byte_mask(at + 4u, lo, hi),
                           byte_mask(at + 8u, lo, hi), byte_mask(at + 12u, lo, hi));
        }
        h = mix_words(h, words);
    }
    uint signature = (piece * 64u + lane) * 16u;
    if (compare != 0u) {
        if (lane == 0u) changed = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (any(h != hashes.Load4(signature))) InterlockedOr(changed, 1u);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) differs.Store(out_at + group.x * 4u, changed);
    }
    if (compare != 1u) hashes.Store4(signature, h);
}
#else
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = seed;
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = mix_words(h, now.Load4(at));
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
#endif
