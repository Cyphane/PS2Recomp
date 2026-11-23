#pragma once

struct R5900Context;
class  PS2Runtime;

#include <cstdint>
#include "ps2_call_list.h"
#include "runtime/ps2_memory.h"
#include "Stubs/Unimplemented.h"

namespace ps2_stubs
{
#define PS2_DECLARE_STUB(name) void name(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    PS2_STUB_LIST(PS2_DECLARE_STUB)
#undef PS2_DECLARE_STUB

    void resetSifState();
}

// s3recomp: optional boot-triage traces (PS2X_TRACE_STUBS=1 / PS2X_TRACE_SYSCALLS=1).
// Polling loops repeat the same few calls every frame, so a call whose (name, ra) matches one of
// the last 8 distinct calls is counted instead of printed; the count is flushed when the pattern breaks.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
inline bool s3_env_flag(const char *name)
{
    const char *v = std::getenv(name);
    return v && v[0] && v[0] != '0';
}
inline bool s3_trace_stubs_enabled()
{
    static const bool enabled = s3_env_flag("PS2X_TRACE_STUBS");
    return enabled;
}
inline void s3_trace_emit(const char *tag, const char *name, uint32_t a0, uint32_t a1, uint32_t ra)
{
    struct Key { const char *tag; const char *name; uint32_t ra; };
    static std::mutex m;
    static Key recent[8] = {};
    static int next = 0;
    static unsigned long long suppressed = 0;
    std::lock_guard<std::mutex> lock(m);
    for (const Key &k : recent)
    {
        if (k.tag == tag && k.ra == ra && k.name && std::strcmp(k.name, name) == 0)
        {
            if ((++suppressed % 100000ull) == 0)
                std::fprintf(stderr, "   ... %llu repeated calls in a loop so far\n", suppressed);
            return;
        }
    }
    if (suppressed)
    {
        std::fprintf(stderr, "   ... (%llu repeated calls suppressed)\n", suppressed);
        suppressed = 0;
    }
    recent[next] = {tag, name, ra};
    next = (next + 1) % 8;
    std::fprintf(stderr, "%s %s a0=0x%x a1=0x%x ra=0x%x\n", tag, name, a0, a1, ra);
}
#define S3_TRACE_STUB(name, ctx)                                                                     \
    do                                                                                               \
    {                                                                                                \
        if (s3_trace_stubs_enabled())                                                                \
            s3_trace_emit("[stub]", name, getRegU32(ctx, 4), getRegU32(ctx, 5), getRegU32(ctx, 31)); \
    } while (0)
