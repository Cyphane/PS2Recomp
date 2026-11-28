// s3recomp: turbo segments of generated VU1 blocks.
//
// A lean block (see the VU1 program generator) already knows, when it is generated, the cycle at
// which each of its pairs issues. A turbo segment is a run of plain pairs in such a block whose flag
// results are scheduled at generation time too: its FMACs and CLIPs keep their MAC/clip results in
// locals, a MAC/clip read inside the segment takes the result of the latest write that is visible at
// its cycle (four cycles after issue, as in the flag pipeline), and the segment's end publishes the
// results to the core exactly as the flag pipeline would hold them at that cycle. Status results are
// applied once at the end (turbo code runs only in programs that never read the status flag, where
// VU1Fast applies them at issue anyway). The per-pair cycle and commit bookkeeping is gone: only the
// instructions that depend on Q/P commits or issue pipeline entries see the core's cycle.
//
// Values are computed exactly as in VU1Fast::fmacSimdInlT<true> (generated code runs with MXCSR
// denormals-are-zero set), and anything unusual goes to the same exact scalar code, with its flags
// captured (VU1Fast::turboUpperSlow). The VU1 replay benchmark checks the results against the generic
// core bit for bit.
#ifndef PS2_VU1_TURBO_INL_H
#define PS2_VU1_TURBO_INL_H

#include "ps2_vu1_fast_upper_inl.h"

namespace vu1turbo
{
    enum Kind : int { kNone = 0, kAdd, kSub, kMul, kMadd, kMsub };
    enum Src : int { kBc = 0, kQ, kI, kVt, kOp };
    struct FmacInfo
    {
        Kind kind;
        Src src;
    };
    // The same classification as fmacSimdInlT.
    constexpr FmacInfo fmacInfo(uint32_t instr)
    {
        const bool special = (instr & 0x3Cu) == 0x3Cu;
        const uint32_t code = special ? ((instr & 0x3u) | ((instr >> 4) & 0x7Cu)) : (instr & 0x3Fu);
        if (code <= 0x03u) return {kAdd, kBc};
        if (code <= 0x07u) return {kSub, kBc};
        if (code <= 0x0Bu) return {kMadd, kBc};
        if (code <= 0x0Fu) return {kMsub, kBc};
        if (code >= 0x18u && code <= 0x1Bu) return {kMul, kBc};
        switch (code)
        {
        case 0x1Cu: return {kMul, kQ};
        case 0x1Eu: return {kMul, kI};
        case 0x20u: return {kAdd, kQ};
        case 0x21u: return {kMadd, kQ};
        case 0x22u: return {kAdd, kI};
        case 0x23u: return {kMadd, kI};
        case 0x24u: return {kSub, kQ};
        case 0x25u: return {kMsub, kQ};
        case 0x26u: return {kSub, kI};
        case 0x27u: return {kMsub, kI};
        case 0x28u: return {kAdd, kVt};
        case 0x29u: return {kMadd, kVt};
        case 0x2Au: return {kMul, kVt};
        case 0x2Cu: return {kSub, kVt};
        case 0x2Du: return {kMsub, kVt};
        case 0x2Eu: return {special ? kMul : kMsub, kOp};
        default: return {kNone, kVt};
        }
    }
    // FMACs a turbo segment computes itself (OPMULA/OPMSUB with a w destination always take the
    // scalar path in VU1Fast, so they are left out).
    constexpr bool isTurboFmac(uint32_t instr)
    {
        const FmacInfo f = fmacInfo(instr);
        return f.kind != kNone && !(f.src == kOp && (((instr >> 21) & 0xFu) & 1u) != 0u);
    }
    constexpr bool isClip(uint32_t instr)
    {
        return (instr & 0x3Cu) == 0x3Cu && ((instr & 0x3u) | ((instr >> 4) & 0x7Cu)) == 0x1Fu;
    }
    // Upper instructions that set no flags and always finish in miscSimdInl.
    constexpr bool isMiscNoFlags(uint32_t instr)
    {
        const bool special = (instr & 0x3Cu) == 0x3Cu;
        const uint32_t code = special ? ((instr & 0x3u) | ((instr >> 4) & 0x7Cu)) : (instr & 0x3Fu);
        if (!special)
            return (code >= 0x10u && code <= 0x17u) || code == 0x1Du || code == 0x1Fu || code == 0x2Bu || code == 0x2Fu;
        return (code >= 0x10u && code <= 0x17u) || code == 0x1Du;
    }

    struct Sticky
    {
        __m128 zero; // lanes (written ones only) whose result or product was an exact zero
        __m128 sign; // the same for signs (sign bits)
        uint32_t slow; // sticky bits from FMACs that took the exact path
    };
}

// One FMAC of a turbo segment. WantMac: compute its MAC result into mac. WantCur: compute its current
// status bits (Z, S, U, O) into cur. Sticky bits always accumulate into st.
template <uint32_t I, bool WantMac, bool WantCur>
VU1_FORCEINLINE void vu1TurboFmac(VU1Fast &v, vu1turbo::Sticky &st, uint32_t &mac, uint32_t &cur)
{
    using namespace vu1turbo;
    constexpr FmacInfo info = fmacInfo(I);
    static_assert(info.kind != kNone, "not an FMAC");
    constexpr bool special = (I & 0x3Cu) == 0x3Cu;
    constexpr uint32_t dest = (I >> 21) & 0xFu;
    static_assert(dest != 0u, "an FMAC with no destination does nothing");
    static_assert(!(info.src == kOp && (dest & 1u)), "OPMULA/OPMSUB w takes the scalar path");
    constexpr uint32_t moveMask = vu1inl::kRev4[dest];
    constexpr uint32_t fs = (I >> 11) & 0x1Fu, ft = (I >> 16) & 0x1Fu, fd = (I >> 6) & 0x1Fu;
    VU1State &s = v.m_state;

    auto load = [](const float *p) -> __m128 { return _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))); };
    auto nonFinite = [](__m128 x) -> __m128i {
        const __m128i expMask = _mm_set1_epi32(0x7F800000);
        return _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(x), expMask), expMask);
    };
    __m128 vs = load(s.vf[fs]);
    __m128 right;
    if constexpr (info.src == kOp)
    {
        right = load(s.vf[ft]);
        vs = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(3, 0, 2, 1));
        right = _mm_shuffle_ps(right, right, _MM_SHUFFLE(3, 1, 0, 2));
    }
    else if constexpr (info.src == kQ)
        right = _mm_set1_ps(v.normalizeOperand(s.q));
    else if constexpr (info.src == kI)
        right = _mm_set1_ps(v.normalizeOperand(s.i));
    else
    {
        right = load(s.vf[ft]);
        if constexpr (info.src == kBc)
        {
            constexpr int sh = (I & 3u) == 0u ? 0x00 : (I & 3u) == 1u ? 0x55 : (I & 3u) == 2u ? 0xAA : 0xFF;
            right = _mm_shuffle_ps(right, right, sh);
        }
    }

    __m128i unusualOperands = _mm_or_si128(nonFinite(vs), nonFinite(right));
    __m128 result;
    __m128i unusual;
    __m128 zeroLanes;           // exact zero results
    __m128 productZero = _mm_setzero_ps();
    __m128 product = _mm_setzero_ps();
    if constexpr (info.kind == kAdd || info.kind == kSub)
    {
        result = info.kind == kAdd ? _mm_add_ps(vs, right) : _mm_sub_ps(vs, right);
        const __m128i rb = _mm_castps_si128(result);
        const __m128 cancels = info.kind == kAdd ? _mm_cmpeq_ps(vs, _mm_xor_ps(right, _mm_set1_ps(-0.0f))) : _mm_cmpeq_ps(vs, right);
        const __m128i zero = _mm_and_si128(_mm_cmpeq_epi32(_mm_and_si128(rb, _mm_set1_epi32(0x7FFFFFFF)), _mm_setzero_si128()),
                                           _mm_castps_si128(cancels));
        unusual = _mm_andnot_si128(zero, vu1inl::unusual4(rb));
        zeroLanes = _mm_castsi128_ps(zero);
    }
    else
    {
        product = _mm_mul_ps(vs, right);
        productZero = _mm_or_ps(_mm_cmpeq_ps(vs, _mm_setzero_ps()), _mm_cmpeq_ps(right, _mm_setzero_ps()));
        const __m128i productUnusual = _mm_andnot_si128(_mm_castps_si128(productZero), vu1inl::unusual4(_mm_castps_si128(product)));
        if constexpr (info.kind == kMul)
        {
            result = product;
            unusual = productUnusual;
            zeroLanes = productZero;
        }
        else
        {
            const __m128 acc = load(s.acc);
            unusualOperands = _mm_or_si128(unusualOperands, nonFinite(acc));
            result = info.kind == kMadd ? _mm_add_ps(acc, product) : _mm_sub_ps(acc, product);
            const __m128 resultZero = _mm_and_ps(productZero, _mm_cmpeq_ps(acc, _mm_setzero_ps()));
            unusual = _mm_or_si128(_mm_andnot_si128(_mm_castps_si128(resultZero), vu1inl::unusual4(_mm_castps_si128(result))), productUnusual);
            zeroLanes = resultZero;
        }
    }
    unusual = _mm_or_si128(unusual, unusualOperands);
    if (__builtin_expect((static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(unusual))) & moveMask) != 0u, 0))
    {
        v.turboUpperSlow(I);
        if (v.m_capValid)
        {
            st.slow |= (v.m_capStatus & 0xFu) | v.m_capSticky;
            if constexpr (WantMac)
                mac = v.m_capMac;
            if constexpr (WantCur)
                cur = v.m_capStatus & 0xFu;
        }
        return;
    }
    float *dst = special ? s.acc : s.vf[fd];
    const __m128 lm = _mm_castsi128_ps(_mm_set_epi32((dest & 1u) ? -1 : 0, (dest & 2u) ? -1 : 0, (dest & 4u) ? -1 : 0, (dest & 8u) ? -1 : 0));
    if constexpr (dest == 0xFu)
        _mm_storeu_ps(dst, result);
    else
        _mm_storeu_ps(dst, _mm_blendv_ps(_mm_loadu_ps(dst), result, lm));
    st.zero = _mm_or_ps(st.zero, _mm_and_ps(zeroLanes, lm));
    st.sign = _mm_or_ps(st.sign, _mm_and_ps(result, lm));
    if constexpr (info.kind == kMadd || info.kind == kMsub)
    {
        // extra sticky bits: a zero product (Z) and a negative product (S) in a written lane
        st.zero = _mm_or_ps(st.zero, _mm_and_ps(productZero, lm));
        st.sign = _mm_or_ps(st.sign, _mm_and_ps(product, lm));
    }
    if constexpr (WantMac || WantCur)
    {
        const uint32_t zeros = static_cast<uint32_t>(_mm_movemask_ps(zeroLanes)) & moveMask;
        const uint32_t signs = static_cast<uint32_t>(_mm_movemask_ps(result)) & moveMask;
        if constexpr (WantMac)
            mac = static_cast<uint32_t>(vu1inl::kRev4[zeros]) | (static_cast<uint32_t>(vu1inl::kRev4[signs]) << 4);
        if constexpr (WantCur)
            cur = (zeros ? 0x1u : 0u) | (signs ? 0x2u : 0u);
    }
}

// The sticky bits a segment's FMACs set (bits Z, S, U, O of the status flag before the << 6).
VU1_FORCEINLINE uint32_t vu1TurboStickyBits(const vu1turbo::Sticky &st)
{
    return st.slow | (_mm_movemask_ps(st.zero) ? 0x1u : 0u) | (_mm_movemask_ps(st.sign) ? 0x2u : 0u);
}

// CLIP's six judgement bits, as miscSimdInl computes them.
template <uint32_t I>
VU1_FORCEINLINE uint32_t vu1TurboClipBits(VU1Fast &v)
{
    constexpr uint32_t fs = (I >> 11) & 0x1Fu, ft = (I >> 16) & 0x1Fu;
    uint32_t wBits = 0u;
    std::memcpy(&wBits, &v.m_state.vf[ft][3], sizeof(wBits));
    const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
    const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(v.m_state.vf[fs]));
    const __m128i lim = _mm_set1_epi32(limit);
    const uint32_t pos = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(bits, lim))));
    const uint32_t neg = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(
        _mm_cmpgt_epi32(_mm_xor_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u))), lim))));
    return (pos & 1u) | ((neg & 1u) << 1) | ((pos & 2u) << 1) | ((neg & 2u) << 2) | ((pos & 4u) << 2) | ((neg & 4u) << 3);
}

#endif
