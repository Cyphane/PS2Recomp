// Inlinable body of VU1Fast::execUpper. Generated VU1 programs include this so that each
// call receives a constant instruction word and the decode folds away at compile time.
#ifndef PS2_VU1_FAST_UPPER_INL_H
#define PS2_VU1_FAST_UPPER_INL_H

#include "ps2_vu1_fast.h"
#include "ps2_vu1_detail.h"

#include <cmath>
#include <cstring>
#include <limits>

#if defined(_MSC_VER)
#define VU1_FORCEINLINE __forceinline
#else
#define VU1_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace vu1inl
{
    inline int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }
}


VU1_FORCEINLINE uint32_t VU1Fast::fmacProductStickyInl(uint32_t upper, uint8_t dest, uint8_t productFlagsOut[4]) const
{
    uint32_t extraSticky = 0u;
    productFlagsOut[0] = productFlagsOut[1] = productFlagsOut[2] = productFlagsOut[3] = 0u;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & (8u >> component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
            right = normalizeOperand(m_state.q);
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
            right = normalizeOperand(m_state.i);
        else if (op == 0x2Eu)
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        else
            right = normalizeOperand(m_state.vf[ft][component]);

        float product = left * right;
        uint32_t pbits;
        std::memcpy(&pbits, &product, 4);
        const uint32_t pexp = (pbits >> 23) & 0xFFu;
        uint8_t productFlags;
        if (pexp != 0u && pexp != 0xFFu && (pbits & 0x7FFFFFFFu) != 0x7F7FFFFFu)
            productFlags = (pbits & 0x80000000u) ? 0x2u : 0u;
        else
            productFlags = normalizeFmacExactResult(product, static_cast<double>(left) * static_cast<double>(right));
        productFlagsOut[component] = productFlags;
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

VU1_FORCEINLINE bool VU1Fast::fmacExactInl(uint32_t upper, uint32_t component, double &result) const
{
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<double>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const double q = operand(m_state.q);
    const double i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

VU1_FORCEINLINE void VU1Fast::applyFmacDestInl(uint32_t upper, float *dst, float *result, uint8_t dest)
{
    uint8_t productFlags[4];
    const uint32_t sticky = fmacProductStickyInl(upper, dest, productFlags);
    uint8_t laneFlags[4] = {0u, 0u, 0u, 0u};
    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint32_t lane = 8u >> component;
        if ((dest & lane) == 0u)
            continue;
        uint32_t bits;
        std::memcpy(&bits, &result[component], 4);
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent != 0u && exponent != 0xFFu && (bits & 0x7FFFFFFFu) != 0x7F7FFFFFu &&
            (productFlags[component] & 0x8u) == 0u)
            laneFlags[component] = (bits & 0x80000000u) ? 0x2u : 0u;
        else
        {
            double exact = 0.0;
            if (fmacExactInl(upper, component, exact))
                laneFlags[component] = normalizeFmacExactResult(result[component], exact);
            else
                normalizeFmacLaneSlow(component, result, laneFlags);
        }
        const uint32_t flags = laneFlags[component];
        mac |= ((flags & 1u) ? lane : 0u) | ((flags & 2u) ? lane << 4 : 0u) |
               ((flags & 4u) ? lane << 8 : 0u) | ((flags & 8u) ? lane << 12 : 0u);
        status |= flags;
    }
    if (dest == 0u)
        return;
    if (dest & 8u) dst[0] = result[0];
    if (dest & 4u) dst[1] = result[1];
    if (dest & 2u) dst[2] = result[2];
    if (dest & 1u) dst[3] = result[3];
    if (flagsAtIssue(mac, status, sticky))
        return;
    FlagPipelineEntry *entry = pushFlagEntry();
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = sticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}


#if defined(__SSE4_1__)
#include <smmintrin.h>
#define VU1_HAVE_SIMD_FMAC 1

namespace vu1inl
{
    // normalizeOperand on four lanes: zero exponent -> signed zero, max exponent -> signed FLT_MAX.
    VU1_FORCEINLINE __m128i normalize4(__m128i b)
    {
        const __m128i expMask = _mm_set1_epi32(0x7F800000);
        const __m128i sign = _mm_and_si128(b, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        const __m128i e = _mm_and_si128(b, expMask);
        __m128i r = _mm_blendv_epi8(b, sign, _mm_cmpeq_epi32(e, _mm_setzero_si128()));
        return _mm_blendv_epi8(r, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF)), _mm_cmpeq_epi32(e, expMask));
    }
    // Lanes whose value is not a plain normal number below FLT_MAX in magnitude (all-ones where so).
    VU1_FORCEINLINE __m128i unusual4(__m128i b)
    {
        const __m128i expMask = _mm_set1_epi32(0x7F800000);
        const __m128i e = _mm_and_si128(b, expMask);
        const __m128i mag = _mm_and_si128(b, _mm_set1_epi32(0x7FFFFFFF));
        return _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi32(e, _mm_setzero_si128()), _mm_cmpeq_epi32(e, expMask)),
                            _mm_cmpeq_epi32(mag, _mm_set1_epi32(0x7F7FFFFF)));
    }
    // VU lane masks are x=8 .. w=1; SSE movemask bits are x=1 .. w=8.
    constexpr uint8_t kRev4[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
}

// s3recomp: four-lane fast path for the common FMAC operations. It handles an instruction only when
// every written lane gives a normal result (and, for multiply-add/subtract, a normal product), where
// the result flags are just the signs - plus exact zeros for add/subtract, whose exact result is then
// zero too. Otherwise it changes nothing and returns false, and the exact per-lane code runs.
// s3recomp: Daz = generated code, which runs with the MXCSR denormals-are-zero bit set (VU1Fast::run).
// Denormal operands then act as signed zeros in hardware, exactly as normalize4 makes them, so only
// infinities and NaNs need catching: those lanes take the exact scalar path, which normalizes them.
template <bool Daz>
VU1_FORCEINLINE bool VU1Fast::fmacSimdInlT(uint32_t instr)
{
    const bool special = (instr & 0x3Cu) == 0x3Cu;
    const uint32_t code = special ? ((instr & 0x3u) | ((instr >> 4) & 0x7Cu)) : (instr & 0x3Fu);
    enum Kind { kAdd, kSub, kMul, kMadd, kMsub, kNone };
    enum Src { kBc, kQ, kI, kVt, kOp };
    Kind kind = kNone;
    Src src = kVt;
    if (code <= 0x03u) { kind = kAdd; src = kBc; }
    else if (code <= 0x07u) { kind = kSub; src = kBc; }
    else if (code <= 0x0Bu) { kind = kMadd; src = kBc; }
    else if (code <= 0x0Fu) { kind = kMsub; src = kBc; }
    else if (code >= 0x18u && code <= 0x1Bu) { kind = kMul; src = kBc; }
    else switch (code)
    {
    case 0x1Cu: kind = kMul; src = kQ; break;
    case 0x1Eu: kind = kMul; src = kI; break;
    case 0x20u: kind = kAdd; src = kQ; break;
    case 0x21u: kind = kMadd; src = kQ; break;
    case 0x22u: kind = kAdd; src = kI; break;
    case 0x23u: kind = kMadd; src = kI; break;
    case 0x24u: kind = kSub; src = kQ; break;
    case 0x25u: kind = kMsub; src = kQ; break;
    case 0x26u: kind = kSub; src = kI; break;
    case 0x27u: kind = kMsub; src = kI; break;
    case 0x28u: kind = kAdd; src = kVt; break;
    case 0x29u: kind = kMadd; src = kVt; break;
    case 0x2Au: kind = kMul; src = kVt; break;
    case 0x2Cu: kind = kSub; src = kVt; break;
    case 0x2Du: kind = kMsub; src = kVt; break;
    // OPMULA (special) / OPMSUB: the cross-product terms vs.yzx * vt.zxy. Their w result is 0, not
    // a product, so a w destination takes the scalar path.
    case 0x2Eu: kind = special ? kMul : kMsub; src = kOp; break;
    default: break;
    }
    if (kind == kNone)
        return false;
    if (src == kOp && (DEST(instr) & 1u) != 0u)
        return false;

    const uint32_t dest = DEST(instr);
    const uint32_t moveMask = vu1inl::kRev4[dest];
    auto loadOperand = [](const float *p) -> __m128 {
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        return _mm_castsi128_ps(Daz ? b : vu1inl::normalize4(b));
    };
    // lanes holding an infinity or NaN (only possible from loads); Daz code sends them to the exact path
    auto nonFinite = [](__m128 v) -> __m128i {
        const __m128i expMask = _mm_set1_epi32(0x7F800000);
        return _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(v), expMask), expMask);
    };
    __m128 vs = loadOperand(m_state.vf[FS(instr)]);
    __m128 right;
    if (src == kOp)
    {
        right = loadOperand(m_state.vf[FT(instr)]);
        vs = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(3, 0, 2, 1));          // (y, z, x, w)
        right = _mm_shuffle_ps(right, right, _MM_SHUFFLE(3, 1, 0, 2)); // (z, x, y, w)
    }
    else if (src == kQ)
        right = _mm_set1_ps(normalizeOperand(m_state.q));
    else if (src == kI)
        right = _mm_set1_ps(normalizeOperand(m_state.i));
    else
    {
        right = loadOperand(m_state.vf[FT(instr)]);
        if (src == kBc)
        {
            switch (instr & 3u)
            {
            case 0u: right = _mm_shuffle_ps(right, right, 0x00); break;
            case 1u: right = _mm_shuffle_ps(right, right, 0x55); break;
            case 2u: right = _mm_shuffle_ps(right, right, 0xAA); break;
            default: right = _mm_shuffle_ps(right, right, 0xFF); break;
            }
        }
    }

    __m128i unusualOperands = Daz ? _mm_or_si128(nonFinite(vs), nonFinite(right)) : _mm_setzero_si128();
    __m128 result;
    __m128i unusual;
    uint32_t extraSticky = 0u;
    uint32_t zeros = 0u;
    if (kind == kAdd || kind == kSub)
    {
        result = kind == kAdd ? _mm_add_ps(vs, right) : _mm_sub_ps(vs, right);
        const __m128i rb = _mm_castps_si128(result);
        // A zero result is exact only when the operands cancel (x + -x, x - x), whatever the host's
        // flush-to-zero setting; a tiny nonzero difference takes the exact path.
        const __m128 cancels = kind == kAdd ? _mm_cmpeq_ps(vs, _mm_xor_ps(right, _mm_set1_ps(-0.0f))) : _mm_cmpeq_ps(vs, right);
        const __m128i zero = _mm_and_si128(_mm_cmpeq_epi32(_mm_and_si128(rb, _mm_set1_epi32(0x7FFFFFFF)), _mm_setzero_si128()),
                                           _mm_castps_si128(cancels));
        unusual = _mm_andnot_si128(zero, vu1inl::unusual4(rb));
        zeros = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(zero))) & moveMask;
    }
    else
    {
        // A product is an exact zero when an operand is zero: its flags are then Z and its sign.
        const __m128 product = _mm_mul_ps(vs, right);
        const __m128i productZero = _mm_castps_si128(_mm_or_ps(_mm_cmpeq_ps(vs, _mm_setzero_ps()), _mm_cmpeq_ps(right, _mm_setzero_ps())));
        const __m128i productUnusual = _mm_andnot_si128(productZero, vu1inl::unusual4(_mm_castps_si128(product)));
        if (kind == kMul)
        {
            result = product;
            unusual = productUnusual;
            zeros = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(productZero))) & moveMask;
        }
        else
        {
            // acc +- product is an exact zero when both are exact zeros; any other zero result, which
            // may be a rounded product cancelling acc, takes the exact path.
            const __m128 acc = loadOperand(m_state.acc);
            if (Daz)
                unusualOperands = _mm_or_si128(unusualOperands, nonFinite(acc));
            result = kind == kMadd ? _mm_add_ps(acc, product) : _mm_sub_ps(acc, product);
            const __m128i resultZero = _mm_and_si128(productZero, _mm_castps_si128(_mm_cmpeq_ps(acc, _mm_setzero_ps())));
            unusual = _mm_or_si128(_mm_andnot_si128(resultZero, vu1inl::unusual4(_mm_castps_si128(result))), productUnusual);
            zeros = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(resultZero))) & moveMask;
            const uint32_t productZeros = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(productZero))) & moveMask;
            extraSticky = ((static_cast<uint32_t>(_mm_movemask_ps(product)) & moveMask) ? 0x2u : 0u) | (productZeros ? 0x1u : 0u);
        }
    }
    if (Daz)
        unusual = _mm_or_si128(unusual, unusualOperands);
    if (static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(unusual))) & moveMask)
        return false;

    m_currentUpperInstruction = instr;
    if (dest == 0u)
        return true;
    const uint32_t signs = static_cast<uint32_t>(_mm_movemask_ps(result)) & moveMask;
    float *dst = special ? m_state.acc : m_state.vf[FD(instr)];
    const __m128i laneMask = _mm_set_epi32((dest & 1u) ? -1 : 0, (dest & 2u) ? -1 : 0, (dest & 4u) ? -1 : 0, (dest & 8u) ? -1 : 0);
    const __m128 old = _mm_loadu_ps(dst);
    _mm_storeu_ps(dst, _mm_blendv_ps(old, result, _mm_castsi128_ps(laneMask)));

    const uint32_t mac = static_cast<uint32_t>(vu1inl::kRev4[zeros]) | (static_cast<uint32_t>(vu1inl::kRev4[signs]) << 4);
    const uint32_t status = (zeros ? 0x1u : 0u) | (signs ? 0x2u : 0u);
    if (flagsAtIssue(mac, status, extraSticky))
        return true;
    FlagPipelineEntry *entry = pushFlagEntry();
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return true;
    }
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
    return true;
}

// s3recomp: four-lane versions of the common upper instructions that set no MAC/status flags (MAX,
// MINI, ITOF, FTOI, ABS) and of CLIP, with the same results as the scalar code.
VU1_FORCEINLINE bool VU1Fast::miscSimdInl(uint32_t instr)
{
    const bool special = (instr & 0x3Cu) == 0x3Cu;
    const uint32_t code = special ? ((instr & 0x3u) | ((instr >> 4) & 0x7Cu)) : (instr & 0x3Fu);
    const uint32_t dest = DEST(instr);
    const __m128i laneMask = _mm_set_epi32((dest & 1u) ? -1 : 0, (dest & 2u) ? -1 : 0, (dest & 4u) ? -1 : 0, (dest & 8u) ? -1 : 0);
    auto load = [](const float *p) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p)); };
    auto store = [&](float *dst, __m128 value) {
        _mm_storeu_ps(dst, _mm_blendv_ps(_mm_loadu_ps(dst), value, _mm_castsi128_ps(laneMask)));
    };
    if (!special)
    {
        const bool isMax = (code >= 0x10u && code <= 0x13u) || code == 0x1Du || code == 0x2Bu;
        const bool isMin = (code >= 0x14u && code <= 0x17u) || code == 0x1Fu || code == 0x2Fu;
        if (!isMax && !isMin)
            return false;
        m_currentUpperInstruction = instr;
        const __m128 vs = _mm_castsi128_ps(vu1inl::normalize4(load(m_state.vf[FS(instr)])));
        __m128 right;
        if (code == 0x1Du || code == 0x1Fu)
            right = _mm_set1_ps(normalizeOperand(m_state.i));
        else
        {
            right = _mm_castsi128_ps(vu1inl::normalize4(load(m_state.vf[FT(instr)])));
            if (code < 0x18u)
            {
                switch (instr & 3u)
                {
                case 0u: right = _mm_shuffle_ps(right, right, 0x00); break;
                case 1u: right = _mm_shuffle_ps(right, right, 0x55); break;
                case 2u: right = _mm_shuffle_ps(right, right, 0xAA); break;
                default: right = _mm_shuffle_ps(right, right, 0xFF); break;
                }
            }
        }
        // (vs > r) ? vs : r and (vs < r) ? vs : r, which is exactly what maxps/minps compute.
        store(m_state.vf[FD(instr)], isMax ? _mm_max_ps(vs, right) : _mm_min_ps(vs, right));
        return true;
    }
    if (code >= 0x10u && code <= 0x13u) // ITOF0/4/12/15
    {
        m_currentUpperInstruction = instr;
        static constexpr float kScale[4] = {1.0f, 1.0f / 16.0f, 1.0f / 4096.0f, 1.0f / 32768.0f};
        // Scaling by a power of two is exact, the same as the scalar code's division.
        const __m128 value = _mm_mul_ps(_mm_cvtepi32_ps(load(m_state.vf[FS(instr)])), _mm_set1_ps(kScale[code & 3u]));
        store(m_state.vf[FT(instr)], value);
        return true;
    }
    if (code >= 0x14u && code <= 0x17u) // FTOI0/4/12/15
    {
        m_currentUpperInstruction = instr;
        static constexpr float kScale[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
        // The scaled value is exact in float (normal operands, power-of-two scale; an overflow gives
        // infinity), so truncating it gives the scalar code's result, except that values of 2^31 and
        // above must saturate to INT32_MAX instead of cvttps's 0x80000000.
        const __m128 scaled = _mm_mul_ps(_mm_castsi128_ps(vu1inl::normalize4(load(m_state.vf[FS(instr)]))), _mm_set1_ps(kScale[code & 3u]));
        __m128i result = _mm_cvttps_epi32(scaled);
        const __m128 high = _mm_cmpge_ps(scaled, _mm_set1_ps(2147483648.0f));
        result = _mm_blendv_epi8(result, _mm_set1_epi32(0x7FFFFFFF), _mm_castps_si128(high));
        store(m_state.vf[FT(instr)], _mm_castsi128_ps(result));
        return true;
    }
    if (code == 0x1Du) // ABS
    {
        m_currentUpperInstruction = instr;
        const __m128i vs = vu1inl::normalize4(load(m_state.vf[FS(instr)]));
        store(m_state.vf[FT(instr)], _mm_castsi128_ps(_mm_and_si128(vs, _mm_set1_epi32(0x7FFFFFFF))));
        return true;
    }
    if (code == 0x1Fu) // CLIP
    {
        m_currentUpperInstruction = instr;
        uint32_t wBits = 0u;
        std::memcpy(&wBits, &m_state.vf[FT(instr)][3], sizeof(wBits));
        const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
        const __m128i bits = load(m_state.vf[FS(instr)]);
        const __m128i lim = _mm_set1_epi32(limit);
        const uint32_t pos = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(bits, lim))));
        const uint32_t neg = static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(
            _mm_cmpgt_epi32(_mm_xor_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u))), lim))));
        const uint32_t flags = (pos & 1u) | ((neg & 1u) << 1) | ((pos & 2u) << 1) | ((neg & 2u) << 2) | ((pos & 4u) << 2) | ((neg & 4u) << 3);
        queueClip(flags);
        return true;
    }
    return false;
}
#else
template <bool Daz>
VU1_FORCEINLINE bool VU1Fast::fmacSimdInlT(uint32_t)
{
    return false;
}
VU1_FORCEINLINE bool VU1Fast::miscSimdInl(uint32_t)
{
    return false;
}
#endif

// Generated programs inline only the four-lane FMAC fast path; everything else (other upper
// instructions, and FMAC results that need the exact path) is one shared out-of-line copy, which
// keeps the generated code small enough for the instruction cache.
VU1_FORCEINLINE bool VU1Fast::fmacSimdInl(uint32_t instr)
{
    return fmacSimdInlT<false>(instr);
}

VU1_FORCEINLINE void VU1Fast::execUpperInl(uint32_t instr)
{
    if (fmacSimdInl(instr) || miscSimdInl(instr))
        return;
    execUpperScalar(instr);
}

// s3recomp: for generated code only (it runs with denormals-are-zero set, see fmacSimdInlT).
VU1_FORCEINLINE void VU1Fast::execUpperInlDaz(uint32_t instr)
{
    if (fmacSimdInlT<true>(instr) || miscSimdInl(instr))
        return;
    // The exact path (unusual results and operands) runs without denormals-are-zero, as everywhere else.
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr & ~0x0040u);
    execUpperScalar(instr);
    _mm_setcsr(csr);
}

VU1_FORCEINLINE void VU1Fast::execUpperScalarInl(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    if ((instr & 0x3Cu) == 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        if (special == 0x2Fu || special == 0x30u) // NOP
            return;
    }
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    float *vd = m_state.vf[fd];
    float normalizedVs[4];
    float normalizedVt[4];
    float normalizedAcc[4];
    for (uint32_t component = 0; component < 4u; ++component)
    {
        normalizedVs[component] = normalizeOperand(m_state.vf[fs][component]);
        normalizedVt[component] = normalizeOperand(m_state.vf[ft][component]);
        normalizedAcc[component] = normalizeOperand(m_state.acc[component]);
    }
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * bc;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * bc;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + q;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * q;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + i;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * i;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - q;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * q;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - i;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * i;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * vt[c];
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * vt[c];
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = acc[0] - vs[1] * vt[2];
        result[1] = acc[1] - vs[2] * vt[0];
        result[2] = acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDestInl(instr, vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * bc;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * bc;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1inl::vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1inl::vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1inl::vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vu1inl::vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + q;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * q;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + i;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * i;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - q;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * q;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - i;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * i;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * vt[c];
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * vt[c];
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestInl(instr, m_state.acc, result, dest);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}

#endif
