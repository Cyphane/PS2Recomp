#include <unordered_set>
#include <cstdlib>
#include <cstdio>
// s3recomp: registry for code overlays.
//
// Some games load extra code from their data files into RAM at run time and jump into it.
// The launcher recompiles those overlay images ahead of time; each one registers here with
// its address range, a dense table of recompiled entry points and a copy of its code bytes.
// Several overlays may be linked at the same addresses, so a lookup compares the bytes that
// are in guest RAM right now with each candidate's image and picks the one that is loaded.
//
// The descriptor layout is duplicated in the generated registration code (no SDK header).

#include "ps2_runtime.h"

#include <cstring>
#include <mutex>
#include <vector>

struct S3OverlayModuleDesc
{
    const char *name;
    uint32_t codeBase;                        // first byte covered by code[]
    uint32_t codeSize;
    const uint8_t *code;                      // image bytes for [codeBase, codeBase+codeSize)
    uint32_t tableBase;                       // dense entry table, one slot per word
    uint32_t tableSlots;
    PS2Runtime::RecompiledFunction const *table;
};

namespace
{
    std::vector<const S3OverlayModuleDesc *> &modules()
    {
        static std::vector<const S3OverlayModuleDesc *> list;
        return list;
    }

    thread_local const S3OverlayModuleDesc *t_lastHit = nullptr;

    PS2Runtime::RecompiledFunction slotOf(const S3OverlayModuleDesc *m, uint32_t address)
    {
        if (address < m->tableBase)
            return nullptr;
        const uint32_t slot = (address - m->tableBase) >> 2;
        if (slot >= m->tableSlots)
            return nullptr;
        return m->table[slot];
    }

    bool bytesMatch(const S3OverlayModuleDesc *m, const uint8_t *rdram, uint32_t address)
    {
        if (address < m->codeBase || address >= m->codeBase + m->codeSize)
            return false;
        const uint32_t offset = address - m->codeBase;
        const uint32_t len = std::min<uint32_t>(256u, m->codeSize - offset);
        const uint32_t phys = address & (PS2_RAM_SIZE - 1u);
        if (phys + len > PS2_RAM_SIZE)
            return false;
        return std::memcmp(rdram + phys, m->code + offset, len) == 0;
    }
}

void s3_register_overlay(const S3OverlayModuleDesc *module)
{
    if (module)
        modules().push_back(module);
}

namespace
{
    // PS2X_OVL_TRACE=1: print the first call into each overlay function (debug aid).
    void traceFirstCall(const S3OverlayModuleDesc *m, uint32_t address)
    {
        static const bool on = std::getenv("PS2X_OVL_TRACE") != nullptr;
        if (!on)
            return;
        static std::mutex mx;
        static std::unordered_set<uint32_t> seen;
        std::lock_guard<std::mutex> lock(mx);
        if (seen.size() < 20000 && seen.insert(address).second)
            std::fprintf(stderr, "[ovl] %08x %s\n", address, m->name);
    }
}

PS2Runtime::RecompiledFunction s3_find_overlay_function(const uint8_t *rdram, uint32_t address)
{
    if (!rdram || modules().empty() || (address & 3u) != 0u)
        return nullptr;
    if (const S3OverlayModuleDesc *m = t_lastHit)
    {
        if (PS2Runtime::RecompiledFunction fn = slotOf(m, address))
            if (bytesMatch(m, rdram, address))
            {
                traceFirstCall(m, address);
                return fn;
            }
    }
    for (const S3OverlayModuleDesc *m : modules())
    {
        PS2Runtime::RecompiledFunction fn = slotOf(m, address);
        if (fn && bytesMatch(m, rdram, address))
        {
            t_lastHit = m;
            traceFirstCall(m, address);
            return fn;
        }
    }
    return nullptr;
}

size_t s3_overlay_count()
{
    return modules().size();
}
