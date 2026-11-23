// SPDX-License-Identifier: GPL-3.0-or-later
// Minimal stand-ins for the PCSX2 infrastructure its IPU code (in IPU/, from PCSX2,
// GPL-3.0+) expects. The IPU is driven synchronously by ps2_ipu.cpp instead of PCSX2's
// event scheduler: the "schedule" hooks below set flags that its pump loop services.
#pragma once

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
#define ARCH_X86 1
#else
#include "sse2neon.h"
#define ARCH_X86 1
#endif

namespace ps2ipu
{
    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using s8 = int8_t;
    using s16 = int16_t;
    using s32 = int32_t;
    using s64 = int64_t;
    using uint = unsigned int;
    using uptr = uintptr_t;

    union alignas(16) u128
    {
        u64 _u64[2];
        u32 _u32[4];
        u16 _u16[8];
        u8 _u8[16];
    };
    using mem128_t = u128;

    inline void CopyQWC(void *dst, const void *src) { std::memcpy(dst, src, 16); }

    using r128 = __m128i;
    inline r128 r128_zero() { return _mm_setzero_si128(); }
    inline r128 r128_load(const void *p) { return _mm_load_si128(reinterpret_cast<const __m128i *>(p)); }
    inline void r128_store(void *p, r128 v) { _mm_store_si128(reinterpret_cast<__m128i *>(p), v); }
    inline r128 r128_from_u32_dup(u32 v) { return _mm_set1_epi32(static_cast<int>(v)); }

    struct NullLog
    {
        template <typename... A> void WriteLn(A &&...) const {}
        template <typename... A> void Warning(A &&...) const {}
        template <typename... A> void Error(A &&...) const {}
    };
    inline const NullLog Console{};
    inline const NullLog DevCon{};

    namespace StringUtil
    {
        inline std::string StdStringFromFormat(const char *fmt, ...)
        {
            char buf[256];
            va_list ap;
            va_start(ap, fmt);
            std::vsnprintf(buf, sizeof(buf), fmt, ap);
            va_end(ap);
            return buf;
        }
    }

    // DMA channel view kept in step with the runtime's D3/D4 registers by ps2_ipu.cpp.
    union tDMA_CHCR
    {
        struct
        {
            u32 DIR : 1;
            u32 _r0 : 1;
            u32 MOD : 2;
            u32 ASP : 2;
            u32 TTE : 1;
            u32 TIE : 1;
            u32 STR : 1;
            u32 _r1 : 7;
            u32 TAG : 16;
        };
        u32 _u32;
    };
    struct DMACh
    {
        tDMA_CHCR chcr;
        u32 madr;
        u32 qwc;
        u32 tadr;
    };
    extern DMACh ipu0ch; // fromIPU (D3)
    extern DMACh ipu1ch; // toIPU (D4)

    struct CpuRegsCompat
    {
        u32 eCycle[32];
        u64 cycle;
        u32 pc;
        u32 interrupt;
    };
    extern CpuRegsCompat cpuRegs;

    enum
    {
        DMAC_FROM_IPU = 3,
        DMAC_TO_IPU = 4,
        IPU_PROCESS = 20,
        INTC_IPU = 8,
    };

    // hooks implemented in ps2_ipu.cpp
    void bridgeScheduleProcess();
    void bridgeScheduleTo();       // toIPU DMA event (IPU1 interrupt)
    void bridgeScheduleFrom();     // fromIPU DMA event (IPU0 interrupt)
    void bridgeScheduleToDma();    // run the toIPU DMA (data requested)
    void hwIntcIrq(int cause);
}

#define CPU_INT(ch, cycles) \
    do { if ((ch) == ::ps2ipu::DMAC_TO_IPU) ::ps2ipu::bridgeScheduleToDma(); \
         else if ((ch) == ::ps2ipu::DMAC_FROM_IPU) ::ps2ipu::bridgeScheduleFrom(); \
         else ::ps2ipu::bridgeScheduleProcess(); } while (0)
#define CPU_SET_DMASTALL(ch, v) ((void)0)

#define __fi
#define __ri
#define __noinline
#define pxAssert(x) ((void)0)
#define pxAssume(x) ((void)0)
#define pxAssertMsg(x, m) ((void)0)
#define jNO_DEFAULT default: break;
#define IsDebugBuild false
#define IPU_LOG(...) ((void)0)
#define DMA_LOG(...) ((void)0)

#define MULTI_ISA_COMPILE_ONCE 1
#define MULTI_ISA_UNSHARED_START
#define MULTI_ISA_UNSHARED_END
#define MULTI_ISA_DEF(...) __VA_ARGS__
