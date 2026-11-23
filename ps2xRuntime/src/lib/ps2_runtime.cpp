#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/gs/gs_frontend.h"
#include "gs/gs_vulkan_backend.h"
#include "vu/ps2_vu1_fast.h"
#include "runtime/ee_scheduler.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <deque>
#include <sstream>

namespace ps2_stubs
{
    void resetSifState();
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));

        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        alignas(16) uint32_t rWords[4]{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rWords), _mm_castps_si128(ctx->vu0_r));
        state.r = 0x3F800000u | (rWords[0] & 0x007FFFFFu);
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.dBitEnabled = (ctx->vu0_fbrst & (1u << 2)) != 0u;
        state.tBitEnabled = (ctx->vu0_fbrst & (1u << 3)) != 0u;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(state.r)));
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & 0xFF00u) | (state.stoppedByD ? (1u << 1) : 0u) | (state.stoppedByT ? (1u << 2) : 0u);
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

// s3recomp debug: PS2X_VU1_CAPTURE=<file>. Creating <file>.go appends the next 400 VU1 calls (MSCAL and
// MSCNT) to <file>: {kind, startPC, top, itop}, VU1State, 16 KiB code, 16 KiB data. Used to replay VU1
// programs outside the game when testing VU1 changes.
namespace s3vucap
{
    inline void maybeCapture(uint32_t kind, uint32_t startPC, uint32_t top, uint32_t itop, const VU1State &state,
                             const uint8_t *code, const uint8_t *data)
    {
        static const char *path = std::getenv("PS2X_VU1_CAPTURE");
        static int left = 0;
        static uint64_t calls = 0;
        if (!path)
            return;
        if (left == 0 && (++calls % 256u) == 0u)
        {
            const std::string go = std::string(path) + ".go";
            if (FILE *t = std::fopen(go.c_str(), "rb"))
            {
                std::fclose(t);
                std::remove(go.c_str());
                left = 400;
            }
        }
        if (left <= 0)
            return;
        --left;
        if (FILE *f = std::fopen(path, "ab"))
        {
            const uint32_t hdr[4] = {kind, startPC, top, itop};
            std::fwrite(hdr, 4, 4, f);
            std::fwrite(&state, sizeof(VU1State), 1, f);
            std::fwrite(code, 1, PS2_VU1_CODE_SIZE, f);
            std::fwrite(data, 1, PS2_VU1_DATA_SIZE, f);
            std::fclose(f);
        }
    }
}

// s3recomp: VU1 calls run on VU1Fast (same results, without the cycle-by-cycle pipeline model) unless
// PS2X_VU1_REF=1 selects the reference interpreter. Both share the interpreter's VU1State.
namespace s3vu1
{
    // Generated (native) VU1 programs are used when the game build includes them, unless
    // PS2X_VU1_COMPILED=0.
    inline VU1Fast &fast(VU1Interpreter &vu)
    {
        static VU1Fast core(vu.state(), VU1Fast::Unit::VU1);
        static const bool configured = [] {
            const char *v = std::getenv("PS2X_VU1_COMPILED");
            core.m_compiledEnabled = !(v && v[0] == '0');
            const size_t n = vu1CompiledRegionCount();
            std::fprintf(stderr, "[vu1] %zu native VU1 programs in this build%s\n", n,
                         n && !core.m_compiledEnabled ? " (disabled by PS2X_VU1_COMPILED=0)" : "");
            return true;
        }();
        (void)configured;
        return core;
    }
    inline bool useReference()
    {
        static const bool ref = [] {
            const char *v = std::getenv("PS2X_VU1_REF");
            return v && v[0] == '1';
        }();
        return ref;
    }
}

namespace s3gfx
{
    extern std::atomic<uint64_t> kickNs, overlapNs, syncs, syncReason[6], workerBusyNs, syncWaitNs, gsRegWaits[0x110], stat[16], stubCalls[8];
    extern void (*g_wakeEe)();
    extern uint32_t (*g_captureFbrst)();
    extern void (*g_applyVpuStat)(uint32_t);
    bool enabled();
    bool jobFbrst(uint32_t &fbrst);
    bool jobSetVpuStat(uint32_t bits);
}

namespace s3prof
{
    void registerThread(const char *name); // s3_profiler.cpp (PS2X_PROFILE=1)
}

// s3recomp: GS on its own thread. GIF packets (from VU1 XGKICKs, VIF1 DIRECT and GIF DMA) are copied
// into a queue and processed in order by a GS thread, so the GS front end and the renderer run
// alongside the game thread instead of inside it. Nothing the game waits on depends on that work:
// DMA completion does not, and whatever does read GS results - privileged GS registers, CSR flags
// (FINISH/SIGNAL), image downloads, the libgraph stubs, buffer flips - first waits for the queue to
// empty (drain). PS2X_GS_THREAD=0 turns it off; it is also off with the experimental graphics worker.
namespace s3gsq
{
    namespace
    {
        std::mutex g_mutex;
        std::condition_variable g_workCv, g_idleCv;
        std::deque<std::vector<uint8_t>> g_queue;
        std::vector<std::vector<uint8_t>> g_pool;
        std::atomic<uint64_t> g_submitted{0}, g_completed{0};
        std::atomic<bool> g_workerSleeping{false};
        std::atomic<bool> g_started{false};
        std::thread::id g_workerId;
        std::function<void(const uint8_t *, uint32_t)> g_process;

        void workerMain()
        {
            ThreadNaming::SetCurrentThreadName("GSThread");
            s3prof::registerThread("gs");
            g_workerId = std::this_thread::get_id();
            for (;;)
            {
                // Spin briefly before sleeping: packets arrive in quick bursts during a frame.
                for (uint32_t n = 0; n < 20000u && g_completed.load(std::memory_order_acquire) == g_submitted.load(std::memory_order_acquire); ++n)
                    if ((n & 255u) == 255u)
                        std::this_thread::yield();
                std::vector<uint8_t> packet;
                {
                    std::unique_lock<std::mutex> lock(g_mutex);
                    if (g_queue.empty())
                    {
                        g_workerSleeping.store(true, std::memory_order_release);
                        g_workCv.wait(lock, [] { return !g_queue.empty(); });
                        g_workerSleeping.store(false, std::memory_order_release);
                    }
                    packet = std::move(g_queue.front());
                    g_queue.pop_front();
                }
                g_process(packet.data(), static_cast<uint32_t>(packet.size()));
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    if (g_pool.size() < 256u)
                        g_pool.push_back(std::move(packet));
                    g_completed.fetch_add(1, std::memory_order_acq_rel);
                }
                g_idleCv.notify_all();
            }
        }
    }

    bool enabledSetting()
    {
        static const bool on = [] {
            const char *v = std::getenv("PS2X_GS_THREAD");
            return !(v && v[0] == '0');
        }();
        return on;
    }

    bool active()
    {
        return enabledSetting() && !s3gfx::enabled();
    }

    void setProcessor(std::function<void(const uint8_t *, uint32_t)> fn)
    {
        g_process = std::move(fn);
    }

    void enqueue(const uint8_t *data, uint32_t size)
    {
        if (!g_started.load(std::memory_order_acquire))
        {
            g_started.store(true, std::memory_order_release);
            std::fprintf(stderr, "[gs] GS thread on (PS2X_GS_THREAD=0 to turn off)\n");
            std::thread(workerMain).detach();
        }
        std::vector<uint8_t> packet;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (!g_pool.empty())
            {
                packet = std::move(g_pool.back());
                g_pool.pop_back();
            }
        }
        packet.assign(data, data + size);
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_queue.push_back(std::move(packet));
            g_submitted.fetch_add(1, std::memory_order_acq_rel);
        }
        if (g_workerSleeping.load(std::memory_order_acquire))
            g_workCv.notify_one();
    }

    bool pending()
    {
        return g_completed.load(std::memory_order_acquire) != g_submitted.load(std::memory_order_acquire);
    }

    // Waits until every queued packet has been processed (no-op on the GS thread itself).
    void drain()
    {
        if (!pending() || std::this_thread::get_id() == g_workerId)
            return;
        for (uint32_t n = 0; n < 4096u && pending(); ++n)
            if ((n & 63u) == 63u)
                std::this_thread::yield();
        if (!pending())
            return;
        std::unique_lock<std::mutex> lock(g_mutex);
        g_idleCv.wait(lock, [] { return !pending(); });
    }
}

// s3recomp: fast-forward, a testing aid for getting through cutscenes. F on the keyboard (or clicking
// both sticks) turns it on and off. While on, the game's clock (vblanks, timers, alarms) runs
// PS2X_FF_SPEED times faster (default 4) and only every PS2X_FF_DRAW_EVERY-th frame (default 4) is
// drawn: the others skip VU1 programs and GS drawing, which is most of the work per frame. Texture
// uploads and GS register writes still happen, so drawn frames have what they need.
namespace s3ff
{
    std::atomic<bool> active{false};          // fast-forward on
    std::atomic<bool> skipDraw{false};        // the frame being built now is not drawn
    std::atomic<bool> shownFrameDrawn{true};  // the frame on screen was drawn (else keep the last picture)
    static uint64_t s_lastDisp = ~0ull;       // EE thread only
    static uint64_t s_frame = 0;
    static std::atomic<uint64_t> s_lastFrameNs{0};

    static uint32_t envCount(const char *name, uint32_t fallback)
    {
        const char *v = std::getenv(name);
        const unsigned long n = v ? std::strtoul(v, nullptr, 10) : 0ul;
        return n >= 1ul && n <= 64ul ? static_cast<uint32_t>(n) : fallback;
    }
    uint32_t speed()
    {
        static const uint32_t s = envCount("PS2X_FF_SPEED", 4u);
        return s;
    }
    static uint32_t drawEvery()
    {
        static const uint32_t n = envCount("PS2X_FF_DRAW_EVERY", 4u);
        return n;
    }
    static uint64_t nowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }

    // Called on the game thread with the current display buffer registers: a change means the game
    // showed a new frame and is about to build the next one.
    void checkFrame(uint64_t disp)
    {
        if (disp == s_lastDisp)
            return;
        s3gsq::drain(); // packets of the finished frame use its own skip setting
        s_lastDisp = disp;
        s_lastFrameNs.store(nowNs(), std::memory_order_relaxed);
        shownFrameDrawn.store(!skipDraw.load(std::memory_order_relaxed), std::memory_order_release);
        ++s_frame;
        const bool on = active.load(std::memory_order_acquire);
        skipDraw.store(on && (s_frame % drawEvery()) != 0u, std::memory_order_release);
    }

    // Called at each vblank: if the game stops flipping buffers (a still screen), draw again so the
    // screen does not stay frozen.
    void onVBlank(uint64_t disp)
    {
        checkFrame(disp);
        if (skipDraw.load(std::memory_order_relaxed) &&
            nowNs() - s_lastFrameNs.load(std::memory_order_relaxed) > 250000000ull)
        {
            skipDraw.store(false, std::memory_order_release);
            shownFrameDrawn.store(true, std::memory_order_release);
        }
    }

    void setActive(bool on)
    {
        active.store(on, std::memory_order_release);
        if (!on)
            skipDraw.store(false, std::memory_order_release);
        std::fprintf(stderr, "[ff] fast-forward %s (x%u, drawing 1 frame in %u)\n", on ? "on" : "off", speed(), drawEvery());
    }
}

// s3recomp: performance counters for the FPS overlay and the periodic [perf] log line.
namespace s3perf
{
    std::atomic<uint64_t> vu1Ns{0};      // VU1 microprogram execution (game thread)
    std::atomic<uint64_t> gsNs{0};       // GIF packet processing incl. GS drawing (game thread)
    std::atomic<uint64_t> gsInVu1Ns{0};  // part of gsNs spent inside VU1 XGKICKs
    std::atomic<uint64_t> frames{0};     // distinct displayed frames (front-buffer flips)
    std::atomic<uint64_t> vu1Cycles{0};       // VU1 cycles run
    std::atomic<uint64_t> vu1NativeCycles{0}; // of which in generated (native) VU1 programs
    std::atomic<uint64_t> iopNs{0};           // IOP emulation run from EE time accounting (game thread)
    std::atomic<uint64_t> eeSwitches{0};      // EE thread switches through the dispatcher
    std::atomic<uint64_t> eeIdleNs{0};        // game thread waiting with no guest thread to run
    thread_local bool inVu1 = false;
    inline uint64_t nowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }
}

static void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    const uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
    const bool needsLatch = !s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick;
    if (needsLatch)
    {
        rt->gs().latchHostPresentationFrame();
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }
    else if (s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    if (displayFbp != s_lastDisplayFbp)
        s3perf::frames.fetch_add(1u, std::memory_order_relaxed);
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    // s3recomp: during fast-forward, frames that were not drawn keep the last drawn picture on screen.
    if (!s3ff::shownFrameDrawn.load(std::memory_order_acquire) && s_hasUploadedFrame)
    {
        outWidth = width;
        outHeight = height;
        return;
    }

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
}

PS2Runtime::PS2Runtime()
{
    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);

    m_eeScheduler = std::make_unique<EeScheduler>(*this);

    // Assign rather than memset: R5900Context's constructor zeroes itself and
    // then applies the COP0 reset values, which a memset here would discard.
    m_cpuContext = R5900Context{};

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    m_cpuContext.vu0_q = 1.0f;
    m_cpuContext.vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_asyncCallbackStackTop = PS2_RAM_SIZE;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModule(std::string_view path, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModule(path, arguments, argumentSize);
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModuleBuffer(guestAddress, arguments, argumentSize);
}

bool PS2Runtime::stopIopModule(int32_t moduleId, int32_t *result)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->stopModule(moduleId, result);
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

bool PS2Runtime::canBindIopRpc(uint32_t sid) const noexcept
{
    return m_iopSubsystem->canBindRpc(sid);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::advanceIopEeCycles(uint64_t eeCycles) noexcept
{
    const uint64_t t0 = s3perf::nowNs();
    m_iopSubsystem->runEeCycles(eeCycles);
    s3perf::iopNs.fetch_add(s3perf::nowNs() - t0, std::memory_order_relaxed);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

uint32_t PS2Runtime::allocateIopMemory(uint32_t size, uint32_t alignment)
{
    return m_iopSubsystem ? m_iopSubsystem->allocateMemory(size, alignment) : 0u;
}

bool PS2Runtime::freeIopMemory(uint32_t address)
{
    return m_iopSubsystem && m_iopSubsystem->freeMemory(address);
}

bool PS2Runtime::readIopMemory(uint32_t address, void *destination, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->readMemory(address, destination, size);
}

bool PS2Runtime::writeIopMemory(uint32_t address, const void *source, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->writeMemory(address, source, size);
}

bool PS2Runtime::zeroIopMemory(uint32_t address, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->zeroMemory(address, size);
}

bool PS2Runtime::isIopMemoryRange(uint32_t address, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->isMemoryRange(address, size);
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    {
        // s3recomp: draw on the GPU with Vulkan unless PS2X_GS_BACKEND=cpu, falling back to the
        // software rasteriser when no suitable Vulkan device is present.
        static bool s_rendererChosen = false;
        if (!s_rendererChosen)
        {
            s_rendererChosen = true;
            const char *choice = std::getenv("PS2X_GS_BACKEND");
            if (choice && (choice[0] == 'c' || choice[0] == 'C'))
            {
                std::fprintf(stderr, "[gs] software renderer (PS2X_GS_BACKEND=cpu)\n");
            }
            else
            {
                std::string why;
                std::unique_ptr<GSRasterBackend> vk = GSVulkanBackend::TryCreate(why);
                if (vk)
                    m_gs.setRasterBackend(std::move(vk));
                else
                    std::fprintf(stderr, "[gs] Vulkan renderer unavailable (%s); using the software renderer\n", why.c_str());
            }
        }
    }
    s3gsq::setProcessor([this](const uint8_t *data, uint32_t size)
                        {
                            const uint64_t t0 = s3perf::nowNs();
                            m_gs.processGIFPacket(data, size);
                            const uint64_t dt = s3perf::nowNs() - t0;
                            s3perf::gsNs.fetch_add(dt, std::memory_order_relaxed);
                            if (s3perf::inVu1)
                                s3perf::gsInVu1Ns.fetch_add(dt, std::memory_order_relaxed); });
    m_gifArbiter.setProcessPacketFn([this](const uint8_t *data, uint32_t size)
                                    {
                                        if (s3gsq::active())
                                        {
                                            s3gsq::enqueue(data, size);
                                            return;
                                        }
                                        const uint64_t t0 = s3perf::nowNs();
                                        m_gs.processGIFPacket(data, size);
                                        const uint64_t dt = s3perf::nowNs() - t0;
                                        s3perf::gsNs.fetch_add(dt, std::memory_order_relaxed);
                                        if (s3perf::inVu1)
                                            s3perf::gsInVu1Ns.fetch_add(dt, std::memory_order_relaxed); });
    m_memory.setGifArbiter(&m_gifArbiter);
    {
        // s3recomp: graphics worker hooks (see s3gfx in ps2_memory.cpp).
        static PS2Runtime *s_gfxRuntime = nullptr;
        s_gfxRuntime = this;
        s3gfx::g_wakeEe = [] {
            if (s_gfxRuntime && s_gfxRuntime->m_eeScheduler)
            {
                EeEvent event{};
                event.type = EeEventType::Dmac;
                s_gfxRuntime->m_eeScheduler->postEvent(event);
            }
        };
        s3gfx::g_captureFbrst = []() -> uint32_t {
            R5900Context *c = s_gfxRuntime && s_gfxRuntime->m_eeScheduler ? s_gfxRuntime->m_eeScheduler->currentContext() : nullptr;
            return c ? c->vu0_fbrst : (s_gfxRuntime ? s_gfxRuntime->m_cpuContext.vu0_fbrst : 0u);
        };
        s3gfx::g_applyVpuStat = [](uint32_t bits) {
            R5900Context *c = s_gfxRuntime && s_gfxRuntime->m_eeScheduler ? s_gfxRuntime->m_eeScheduler->currentContext() : nullptr;
            if (!c && s_gfxRuntime)
                c = &s_gfxRuntime->m_cpuContext;
            if (c)
                c->vu0_vpu_stat = (c->vu0_vpu_stat & ~0x0600u) | bits;
        };
        if (s3gfx::enabled())
            std::fprintf(stderr, "[gfx] VIF1/VU1/GS run on a separate thread (PS2X_GFX_THREAD=1)\n");
    }
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t top, uint32_t itop)
                                 {
                                     // s3recomp: fast-forward skips VU1 programs on frames that are not drawn.
                                     if (s3ff::skipDraw.load(std::memory_order_relaxed))
                                         return;
                                     // s3recomp: on the graphics worker the EE context belongs to another
                                     // thread: FBRST comes from the kick and VPU_STAT is applied at completion.
                                     uint32_t fbrst = 0u;
                                     R5900Context *cpuContext = nullptr;
                                     if (!s3gfx::jobFbrst(fbrst))
                                     {
                                         cpuContext = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
                                         if (!cpuContext)
                                         {
                                             cpuContext = &m_cpuContext;
                                         }
                                         fbrst = cpuContext->vu0_fbrst;
                                     }
                                     m_vu1.state().dBitEnabled =
                                         (fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled =
                                         (fbrst & (1u << 11)) != 0u;
                                     // s3recomp debug: PS2X_VU1_DUMP=<trigger file>; touching it dumps the next 12 MSCALs.
                                     static const char *vuTrigger = std::getenv("PS2X_VU1_DUMP");
                                     static int vuDumpLeft = 0;
                                     static int vuDumpIndex = 0;
                                     static uint64_t vuCalls = 0;
                                     if (vuTrigger && vuDumpLeft == 0 && (++vuCalls % 64u) == 0u)
                                     {
                                         if (FILE *t = std::fopen(vuTrigger, "rb"))
                                         {
                                             std::fclose(t);
                                             std::remove(vuTrigger);
                                             vuDumpLeft = 12;
                                             vuDumpIndex = 0;
                                         }
                                     }
                                     FILE *vuDump = nullptr;
                                     if (vuDumpLeft > 0)
                                     {
                                         const std::string path = std::string(vuTrigger) + "." + std::to_string(vuDumpIndex++) + ".bin";
                                         vuDump = std::fopen(path.c_str(), "wb");
                                         --vuDumpLeft;
                                         if (vuDump)
                                         {
                                             const uint32_t hdr[4] = {startPC, top, itop, 0x31555653u};
                                             std::fwrite(hdr, 4, 4, vuDump);
                                             std::fwrite(&m_vu1.state(), sizeof(VU1State), 1, vuDump);
                                             std::fwrite(m_memory.getVU1Code(), 1, PS2_VU1_CODE_SIZE, vuDump);
                                             std::fwrite(m_memory.getVU1Data(), 1, PS2_VU1_DATA_SIZE, vuDump);
                                         }
                                     }
                                     s3vucap::maybeCapture(0u, startPC, top, itop, m_vu1.state(), m_memory.getVU1Code(), m_memory.getVU1Data());
                                     if (s3vu1::useReference())
                                         m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                       m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                       m_gs, &m_memory, startPC, top, itop, 65536);
                                     else
                                         s3vu1::fast(m_vu1).execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                                    m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                                    m_gs, &m_memory, startPC, top, itop, 65536);
                                     if (vuDump)
                                     {
                                         std::fwrite(&m_vu1.state(), sizeof(VU1State), 1, vuDump);
                                         std::fwrite(m_memory.getVU1Data(), 1, PS2_VU1_DATA_SIZE, vuDump);
                                         std::fclose(vuDump);
                                     }
                                     const uint32_t vpuBits = (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                                              (m_vu1.state().stoppedByT ? 0x0400u : 0u);
                                     if (!s3gfx::jobSetVpuStat(vpuBits))
                                         cpuContext->vu0_vpu_stat = (cpuContext->vu0_vpu_stat & ~0x0600u) | vpuBits; });
    m_memory.setVu1MscntCallback([this](uint32_t top, uint32_t itop)
                                 {
                                     // s3recomp: fast-forward skips VU1 programs on frames that are not drawn.
                                     if (s3ff::skipDraw.load(std::memory_order_relaxed))
                                         return;
                                     // s3recomp: on the graphics worker the EE context belongs to another
                                     // thread: FBRST comes from the kick and VPU_STAT is applied at completion.
                                     uint32_t fbrst = 0u;
                                     R5900Context *cpuContext = nullptr;
                                     if (!s3gfx::jobFbrst(fbrst))
                                     {
                                         cpuContext = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
                                         if (!cpuContext)
                                         {
                                             cpuContext = &m_cpuContext;
                                         }
                                         fbrst = cpuContext->vu0_fbrst;
                                     }
                                     m_vu1.state().dBitEnabled =
                                         (fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled =
                                         (fbrst & (1u << 11)) != 0u;
                                     s3vucap::maybeCapture(1u, m_vu1.state().pc, top, itop, m_vu1.state(), m_memory.getVU1Code(), m_memory.getVU1Data());
                                     if (s3vu1::useReference())
                                         m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                      m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                      m_gs, &m_memory, top, itop, 65536);
                                     else
                                         s3vu1::fast(m_vu1).resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                                   m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                                   m_gs, &m_memory, top, itop, 65536);
                                     const uint32_t vpuBits = (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                                              (m_vu1.state().stoppedByT ? 0x0400u : 0u);
                                     if (!s3gfx::jobSetVpuStat(vpuBits))
                                         cpuContext->vu0_vpu_stat = (cpuContext->vu0_vpu_stat & ~0x0600u) | vpuBits; });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();
    s3vu1::fast(m_vu1).reset();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title);
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
#endif
        SetExitKey(KEY_NULL); // Escape is a pad button (Circle), not "close the game"
        SetTargetFPS(60);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
        m_asyncCallbackStackFloor = std::min(std::max(hardLimit, suggestedHeapBase), PS2_RAM_SIZE);
        m_asyncCallbackStackTop = PS2_RAM_SIZE;
        // s3recomp: the top of RAM is where games put their main thread stack (SetupThread),
        // so interrupt/callback stacks carved from there trampled live stack frames. Use the
        // EE kernel reserve at 0x80000-0x100000 instead, which games never touch, whenever
        // the executable is loaded above it.
        if (moduleBase != std::numeric_limits<uint32_t>::max() && moduleBase >= 0x00100000u)
        {
            m_asyncCallbackStackFloor = 0x00080000u;
            m_asyncCallbackStackTop = 0x00100000u;
        }
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string romError;
    if (!m_romDevice.configure(identity, &romError))
    {
        std::cerr << "[ROM0] failed to configure profile: " << romError << std::endl;
        return false;
    }

    m_iopSubsystem->reset();

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

namespace
{
    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = func;
    return true;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

PS2Runtime::RecompiledFunction s3_find_overlay_function(const uint8_t *rdram, uint32_t address);

bool PS2Runtime::hasFunction(uint32_t address) const
{
    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot) && g_ps2RecompiledFunctionTable[slot] != nullptr)
        return true;
    // s3recomp: code loaded from disc at run time (recompiled overlays).
    return s3_find_overlay_function(const_cast<PS2Memory &>(m_memory).getRDRAM(), address) != nullptr;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);

    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
        if (fn != nullptr)
        {
            return fn;
        }
    }
    if (RecompiledFunction fn = s3_find_overlay_function(m_memory.getRDRAM(), address))
    {
        return fn;
    }

    std::cerr << "Error: No exact recompiled function for guest PC 0x" << std::hex << address
              << " tableBase=0x" << g_ps2RecompiledFunctionTableBase
              << " tableEnd=0x" << g_ps2RecompiledFunctionTableEnd
              << " codeRegion=" << (m_memory.isCodeAddress(address) ? "yes" : "no")
              << " trace=" << formatDispatchHistory()
              << std::dec << std::endl;

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc = ctx->pc;
    const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
    const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
    const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
    const uint32_t a0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0));
    const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0));
    const uint32_t a2 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[6], 0));
    const uint32_t a3 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[7], 0));
    const uint32_t s0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[16], 0));
    const uint32_t s1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[17], 0));
    const uint32_t v0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0));
    const uint32_t v1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0));

    auto readGuestU32At = [rdram](uint32_t addr, uint32_t &out) -> bool
    {
        // TODO this !rdram exist only because of test fix those test later
        if (!rdram || addr > PS2_RAM_SIZE - sizeof(uint32_t))
        {
            out = 0u;
            return false;
        }

        std::memcpy(&out, rdram + addr, sizeof(uint32_t));
        return true;
    };

    auto readGuestU32Offset = [&readGuestU32At](uint32_t base, uint32_t offset, uint32_t &out) -> bool
    {
        if (base > PS2_RAM_SIZE - sizeof(uint32_t) || offset > PS2_RAM_SIZE - sizeof(uint32_t) - base)
        {
            out = 0u;
            return false;
        }

        return readGuestU32At(base + offset, out);
    };

    uint32_t a0Word0 = 0u;
    uint32_t a0Word4 = 0u;
    uint32_t a0Word8 = 0u;
    uint32_t a0WordC = 0u;
    const bool a0Readable =
        readGuestU32Offset(a0, 0x00u, a0Word0) &&
        readGuestU32Offset(a0, 0x04u, a0Word4) &&
        readGuestU32Offset(a0, 0x08u, a0Word8) &&
        readGuestU32Offset(a0, 0x0cu, a0WordC);

    uint32_t s0Word0 = 0u;
    uint32_t s0Word4 = 0u;
    uint32_t s0Word8 = 0u;
    uint32_t s0WordC = 0u;
    const bool s0Readable =
        readGuestU32Offset(s0, 0x00u, s0Word0) &&
        readGuestU32Offset(s0, 0x04u, s0Word4) &&
        readGuestU32Offset(s0, 0x08u, s0Word8) &&
        readGuestU32Offset(s0, 0x0cu, s0WordC);

    uint32_t recordWord0 = 0u;
    uint32_t recordWord4 = 0u;
    uint32_t recordWord8 = 0u;
    uint32_t recordWordC = 0u;
    const bool recordReadable =
        s0Readable && s0Word4 != 0u &&
        readGuestU32Offset(s0Word4, 0x00u, recordWord0) &&
        readGuestU32Offset(s0Word4, 0x04u, recordWord4) &&
        readGuestU32Offset(s0Word4, 0x08u, recordWord8) &&
        readGuestU32Offset(s0Word4, 0x0cu, recordWordC);

    uint32_t vtableSlot0 = 0u;
    uint32_t vtableSlot4 = 0u;
    uint32_t vtableSlot8 = 0u;
    uint32_t vtableSlotC = 0u;
    const bool vtableReadable =
        a0Readable && a0Word0 != 0u &&
        readGuestU32Offset(a0Word0, 0x00u, vtableSlot0) &&
        readGuestU32Offset(a0Word0, 0x04u, vtableSlot4) &&
        readGuestU32Offset(a0Word0, 0x08u, vtableSlot8) &&
        readGuestU32Offset(a0Word0, 0x0cu, vtableSlotC);

    if (firstReport)
    {
        std::ostringstream oss;
        oss << "[guest-branch:missing-target] kind=" << describeGuestBranchKind(kind)
            << " op=" << (debugName ? debugName : "<unknown>")
            << " source=0x" << std::hex << sourcePc
            << " target=0x" << targetPc
            << " pc=0x" << pc
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " a1=0x" << a1
            << " a2=0x" << a2
            << " a3=0x" << a3
            << " s0=0x" << s0
            << " s1=0x" << s1
            << " v0=0x" << v0
            << " v1=0x" << v1
            << " a0Readable=" << (a0Readable ? "yes" : "no")
            << " a0[0]=0x" << a0Word0
            << " a0[4]=0x" << a0Word4
            << " a0[8]=0x" << a0Word8
            << " a0[c]=0x" << a0WordC
            << " s0Readable=" << (s0Readable ? "yes" : "no")
            << " s0[0]=0x" << s0Word0
            << " s0[4]=0x" << s0Word4
            << " s0[8]=0x" << s0Word8
            << " s0[c]=0x" << s0WordC
            << " recordReadable=" << (recordReadable ? "yes" : "no")
            << " record[0]=0x" << recordWord0
            << " record[4]=0x" << recordWord4
            << " record[8]=0x" << recordWord8
            << " record[c]=0x" << recordWordC
            << " vtableReadable=" << (vtableReadable ? "yes" : "no")
            << " vtbl[0]=0x" << vtableSlot0
            << " vtbl[4]=0x" << vtableSlot4
            << " vtbl[8]=0x" << vtableSlot8
            << " vtbl[c]=0x" << vtableSlotC
            << " codeRegion=" << (m_memory.isCodeAddress(targetPc) ? "yes" : "no")
            << " policy=" << static_cast<uint32_t>(policy)
            << " trace=" << formatDispatchHistory()
            << std::dec;

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    // s3recomp: keep a snapshot of EE RAM from the first miss so the module that should hold the
    // target can be identified from the bytes around it (logs/eeram-missing-<target>.bin).
    static std::mutex s_dumpMutex;
    static uint32_t s_dumped[3] = {};
    static uint32_t s_dumpCount = 0;
    bool dumpNow = false;
    if (rdram)
    {
        std::lock_guard<std::mutex> lock(s_dumpMutex);
        const uint32_t page = targetPc & ~0xFFFFu;
        bool seen = false;
        for (uint32_t i = 0; i < s_dumpCount; ++i)
            seen |= s_dumped[i] == page;
        if (!seen && s_dumpCount < 3)
        {
            s_dumped[s_dumpCount++] = page;
            dumpNow = true;
        }
    }
    if (dumpNow)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "logs/eeram-missing-%08x.bin", targetPc);
        if (FILE *f = std::fopen(name, "wb"))
        {
            std::fwrite(rdram, 1, PS2_RAM_SIZE, f);
            std::fclose(f);
            std::cerr << "[guest-branch:missing-target] EE RAM saved to " << name << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

extern thread_local bool t_s3Unwinding; // defined with eeCheckpointDue below

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);

    // Every inter-function transfer is also a deterministic EE safe point.
    // Backward edges inside generated functions use eeCheckpointDue(), while
    // this charge bounds straight-line call chains that have no local loop.
    if (m_eeScheduler && m_eeScheduler->checkpointDue(EeScheduler::kGuestDispatchCycles))
    {
        t_s3Unwinding = true;
        return false;
    }

    if (!isCall)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        ctx->pc = targetPc;
        t_s3Unwinding = true;
        return false;
    }

    if (!hasFunction(targetPc))
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        // s3recomp: a call into overlay code that was not found when the game was built (callbacks
        // are the usual case) is skipped with v0/v1 = 0 instead of ending the game thread. The game
        // keeps running - usually with something missing on screen - and the log names the target.
        const bool s3SkipOverlayCall = targetPc < 0x01600000u && targetPc >= 0x00100000u;
        if ((policy == MissingFunctionPolicy::SkipCallDebug || s3SkipOverlayCall) && isCall)
        {
            static std::atomic<uint32_t> s_skipped{0};
            const uint32_t n = s_skipped.fetch_add(1, std::memory_order_relaxed);
            if (n < 32 || (n & (n - 1)) == 0)
                std::cerr << "[guest-branch:skip] call 0x" << std::hex << sourcePc << " -> 0x" << targetPc
                          << " skipped (not recompiled), count=" << std::dec << (n + 1) << std::endl;
            ctx->r[2] = _mm_setzero_si128();
            ctx->r[3] = _mm_setzero_si128();
            ctx->pc = fallthroughPc;
            return true;
        }
        t_s3Unwinding = true;

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            // if you need the app to keep open to open debug pannel change this to false
            return false;
        }

        return false;
    }

    RecompiledFunction targetFn = lookupFunction(targetPc);
    const uint32_t entryPc = ctx->pc;
    // s3recomp: callee-saved registers (s0-s7, sp, fp) must survive a call. A violation here is
    // logged with both addresses: it pins down register or stack corruption at the call that
    // caused it instead of the crash that follows much later.
    uint32_t saved[10];
    for (int i = 0; i < 8; ++i)
        saved[i] = GPR_U32(ctx, 16 + i);
    saved[8] = GPR_U32(ctx, 29);
    saved[9] = GPR_U32(ctx, 30);
    t_s3Unwinding = false; // any unwind that starts inside this call sets it again
    targetFn(rdram, ctx, this);

    if (isStopRequested() || ctx->pc == 0u || t_s3Unwinding)
    {
        return false;
    }

    if (ctx->pc == entryPc)
    {
        ctx->pc = fallthroughPc; // native stubs leave ctx->pc alone
    }

    if (ctx->pc == fallthroughPc)
    {
        static std::atomic<int> s_reports{0};
        bool bad = GPR_U32(ctx, 29) != saved[8] || GPR_U32(ctx, 30) != saved[9];
        for (int i = 0; i < 8 && !bad; ++i)
            bad = GPR_U32(ctx, 16 + i) != saved[i];
        if (bad && s_reports.fetch_add(1, std::memory_order_relaxed) < 24)
        {
            std::fprintf(stderr, "[abi-check] call 0x%x -> 0x%x (%s) returned with changed callee-saved registers:", sourcePc,
                         targetPc, debugName ? debugName : "?");
            for (int i = 0; i < 8; ++i)
                if (GPR_U32(ctx, 16 + i) != saved[i])
                    std::fprintf(stderr, " s%d %08x->%08x", i, saved[i], GPR_U32(ctx, 16 + i));
            if (GPR_U32(ctx, 29) != saved[8])
                std::fprintf(stderr, " sp %08x->%08x", saved[8], GPR_U32(ctx, 29));
            if (GPR_U32(ctx, 30) != saved[9])
                std::fprintf(stderr, " fp %08x->%08x", saved[9], GPR_U32(ctx, 30));
            std::fprintf(stderr, " ra=%08x\n", GPR_U32(ctx, 31));
        }
    }

    return ctx->pc == fallthroughPc;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, 4096);
    copyVu0StateToContext(m_vu0.state(), ctx);
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    {
        // s3recomp: optional syscall trace for headless boot triage (PS2X_TRACE_SYSCALLS=1)
        static const bool s_traceSyscalls = s3_env_flag("PS2X_TRACE_SYSCALLS");
        if (s_traceSyscalls)
        {
            static thread_local char idName[16];
            std::snprintf(idName, sizeof(idName), "id=0x%x", syscallId);
            // Names must be stable pointers for repeat detection: intern per syscall id.
            static std::mutex s_internMutex;
            static std::map<uint32_t, std::string> s_interned;
            const char *stable = nullptr;
            {
                std::lock_guard<std::mutex> lock(s_internMutex);
                auto &slot = s_interned[syscallId];
                if (slot.empty())
                    slot = idName;
                stable = slot.c_str();
            }
            s3_trace_emit("[sc]", stable, getRegU32(ctx, 4), getRegU32(ctx, 5), getRegU32(ctx, 31));
        }
    }

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    for (uint32_t cause : m_memory.consumeCompletedDmacCauses())
    {
        if (cause & 0x80000000u) // INTC cause queued by a device (e.g. the IPU)
            eeScheduler().dispatchIrq(false, cause & 0xFFu);
        else
            ps2_syscalls::dispatchDmacHandlersForCause(rdram, this, cause);
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u)
    {
        return hardLimit;
    }
    // s3recomp: an explicit limit (from SetupHeap) may reach the main thread's stack.
    return std::min(guestLimit, PS2_RAM_SIZE);
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    try
    {
        m_memory.write32(vaddr, value);
        drainCompletedDmacHandlers(rdram);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    if (m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    if (m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
    drainCompletedDmacHandlers(rdram);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    if (m_eeScheduler)
    {
        m_eeScheduler->requestStop();
    }
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

EeScheduler &PS2Runtime::eeScheduler()
{
    return *m_eeScheduler;
}

const EeScheduler &PS2Runtime::eeScheduler() const
{
    return *m_eeScheduler;
}

void PS2Runtime::postEeEvent(EeEvent event)
{
    m_eeScheduler->postEvent(event);
}

// s3recomp: set while generated code is unwinding back to the EE scheduler (a checkpoint, or a
// jump that leaves the host call chain). dispatchGuestBranch must then pass the unwind on instead
// of guessing from ctx->pc: a checkpoint taken at the entry of a recursive call leaves ctx->pc
// equal to the outer call's target, which looked like "the callee returned" and resumed the
// caller with the callee's registers and stack (the Suikoden III scene-change crashes in the
// object tree walk 0x18BCFE0 -> 0x16CB808 -> 0x18BCFE0).
thread_local bool t_s3Unwinding = false;
void s3ClearUnwinding() noexcept { t_s3Unwinding = false; }

bool PS2Runtime::eeCheckpointDue(uint32_t cycles) noexcept
{
    const bool due = m_eeScheduler->checkpointDue(cycles);
    if (due)
        t_s3Unwinding = true;
    return due;
}

[[noreturn]] void PS2Runtime::eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc)
{
    const uint64_t currentTick = m_eeScheduler->currentVSyncTick();
    const uint64_t waitTicks = std::max<uint64_t>(1u, ticks);
    m_eeScheduler->waitVSync(currentTick + waitTicks - 1u,
                             0,
                             [resumePc](R5900Context &context)
                             {
                                 context.pc = resumePc;
                             });
}

void PS2Runtime::addEeExitHandler(int threadId, uint32_t function, uint32_t argument)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers[threadId].push_back({function, argument});
}

std::vector<PS2Runtime::EeExitHandlerRegistration> PS2Runtime::takeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    auto it = m_eeExitHandlers.find(threadId);
    if (it == m_eeExitHandlers.end())
    {
        return {};
    }
    auto handlers = std::move(it->second);
    m_eeExitHandlers.erase(it);
    return handlers;
}

void PS2Runtime::removeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers.erase(threadId);
}

bool PS2Runtime::findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const
{
    std::lock_guard lock(m_eeKernelStateMutex);
    const auto it = m_eeSyscallOverrides.find(syscallNumber);
    if (it == m_eeSyscallOverrides.end())
    {
        return false;
    }
    handler = it->second;
    return true;
}

void PS2Runtime::setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler)
{
    constexpr uint32_t kTableBase = 0x80011F80u & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
    const int64_t address = static_cast<int64_t>(kTableBase) + offset;

    std::lock_guard lock(m_eeKernelStateMutex);
    if (handler == 0u)
    {
        m_eeSyscallOverrides.erase(syscallNumber);
    }
    else
    {
        m_eeSyscallOverrides[syscallNumber] = handler;
    }
    if (!rdram || address < 0 || address + 4 > kMirrorLimit)
    {
        return;
    }
    const uint32_t guestAddress = static_cast<uint32_t>(address);
    std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
    if (handler == 0u)
    {
        m_eeSyscallMirrorAddresses.erase(guestAddress);
    }
    else
    {
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::initializeEeKernelState(uint8_t *rdram)
{
    if (!rdram)
    {
        return;
    }
    constexpr uint32_t kTableGuestBase = 0x80011F80u;
    constexpr uint32_t kTableBase = kTableGuestBase & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    constexpr uint32_t kProbeBase = 0x000002F0u;

    std::lock_guard lock(m_eeKernelStateMutex);
    for (const uint32_t address : m_eeSyscallMirrorAddresses)
    {
        const uint32_t zero = 0u;
        std::memcpy(rdram + address, &zero, sizeof(zero));
    }
    m_eeSyscallMirrorAddresses.clear();
    const uint32_t high = kTableGuestBase >> 16;
    const uint32_t low = kTableGuestBase & 0xFFFFu;
    std::memcpy(rdram + kProbeBase, &high, sizeof(high));
    std::memcpy(rdram + kProbeBase + 8u, &low, sizeof(low));
    m_eeSyscallMirrorAddresses.insert(kProbeBase);
    m_eeSyscallMirrorAddresses.insert(kProbeBase + 8u);

    for (const auto &[syscallNumber, handler] : m_eeSyscallOverrides)
    {
        const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
        const int64_t address = static_cast<int64_t>(kTableBase) + offset;
        if (address < 0 || address + 4 > kMirrorLimit)
        {
            continue;
        }
        const uint32_t guestAddress = static_cast<uint32_t>(address);
        std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetMpegStubState();
    initializeEeKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    {
        // Smooth scaling to the window by default, closer to how a PS2 looks on a TV;
        // PS2X_SCALE_FILTER=point keeps hard pixels.
        const char *filter = std::getenv("PS2X_SCALE_FILTER");
        SetTextureFilter(frameTex, (filter && (filter[0] == 'p' || filter[0] == 'P')) ? TEXTURE_FILTER_POINT
                                                                                    : TEXTURE_FILTER_BILINEAR);
    }
    UnloadImage(blank);

    std::atomic<bool> gameThreadFinished{false};
    s3prof::registerThread("presenter");

    std::thread gameThread([&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        s3prof::registerThread("game");
        try
        {
            m_eeScheduler->reset(m_memory.getRDRAM(), m_cpuContext);
            m_eeScheduler->run();
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        gameThreadFinished.store(true, std::memory_order_release); });

    uint64_t tick = 0;
    while (!isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire))
    {
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const auto eeSnapshot = m_eeScheduler->snapshot();

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << eeSnapshot.threads.size()
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);

            }
        });
        // s3recomp: PS2X_DUMP_THREADS=1 prints every guest thread's state every ~5 seconds.
        {
            static const bool dumpThreads = s3_env_flag("PS2X_DUMP_THREADS");
            static uint64_t dumpTick = 0;
            if (dumpThreads && (++dumpTick % 300u) == 0u)
            {
                const auto snap = m_eeScheduler->snapshot();
                std::fprintf(stderr, "[threads] t=%llu running=%d\n", static_cast<unsigned long long>(dumpTick), snap.runningThreadId);
                for (const auto &t : snap.threads)
                    std::fprintf(stderr, "[thread] id=%d st=%d wait=%d wid=%d pc=0x%x ra=0x%x sp=0x%x entry=0x%x pri=%d susp=%d inv=%u\n",
                                 t.id, static_cast<int>(t.status), static_cast<int>(t.waitReason), t.waitId, t.pc, t.ra, t.sp,
                                 t.entry, t.currentPriority, t.suspendCount, t.invocationDepth);
                for (const auto &sm : snap.semaphores)
                    if (sm.waiters)
                        std::fprintf(stderr, "[sema] id=%d count=%d max=%d waiters=%u\n", sm.id, sm.count, sm.maxCount, sm.waiters);
                for (const auto &ef : snap.eventFlags)
                    if (ef.waiters)
                        std::fprintf(stderr, "[evf] id=%d bits=0x%x waiters=%u\n", ef.id, ef.bits, ef.waiters);
                // PS2X_WATCH=addr,addr,... prints guest words alongside the thread dump.
                static const char *watch = std::getenv("PS2X_WATCH");
                if (watch && *watch)
                {
                    std::string list(watch);
                    size_t pos = 0;
                    while (pos < list.size())
                    {
                        size_t end = list.find(',', pos);
                        if (end == std::string::npos)
                            end = list.size();
                        const uint32_t addr = static_cast<uint32_t>(std::strtoul(list.substr(pos, end - pos).c_str(), nullptr, 16)) & 0x1FFFFFFCu;
                        if (addr + 4u <= PS2_RAM_SIZE)
                        {
                            uint32_t value = 0;
                            std::memcpy(&value, m_memory.getRDRAM() + addr, 4);
                            std::fprintf(stderr, "[watch] 0x%08x = 0x%08x\n", addr, value);
                        }
                        else if (addr >= 0x10000000u && addr < 0x10010000u)
                        {
                            std::fprintf(stderr, "[watch] io 0x%08x = 0x%08x\n", addr, m_memory.readIORegister(addr));
                        }
                        pos = end + 1;
                    }
                }
            }
        }
        {
            // s3recomp: fast-forward toggle: F, or clicking both sticks together.
            static bool s_chordWasDown = false;
            bool chordDown = false;
            for (int pad = 0; pad < 4; ++pad)
                if (IsGamepadAvailable(pad) && IsGamepadButtonDown(pad, GAMEPAD_BUTTON_LEFT_THUMB) &&
                    IsGamepadButtonDown(pad, GAMEPAD_BUTTON_RIGHT_THUMB))
                    chordDown = true;
            // The key queue also catches a tap shorter than one frame.
            bool fTapped = IsKeyPressed(KEY_F);
            for (int key = GetKeyPressed(); key != 0; key = GetKeyPressed())
                if (key == KEY_F)
                    fTapped = true;
            const bool toggle = fTapped || (chordDown && !s_chordWasDown);
            s_chordWasDown = chordDown;
            if (toggle)
                s3ff::setActive(!s3ff::active.load(std::memory_order_acquire));
        }
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        UploadFrame(frameTex, this, presentWidth, presentHeight);

        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        // A PS2 stretches whatever width it renders (512, 640, ...) across a 4:3 TV picture.
        // PS2X_ASPECT=native shows square pixels instead; PS2X_ASPECT=stretch fills the window.
        static const int aspectMode = [] {
            const char *v = std::getenv("PS2X_ASPECT");
            if (v && (v[0] == 'n' || v[0] == 'N'))
                return 1;
            if (v && (v[0] == 's' || v[0] == 'S'))
                return 2;
            return 0;
        }();
        float dstWidth = screenWidth, dstHeight = screenHeight;
        if (aspectMode == 1)
        {
            const float scale = std::min(screenWidth / srcWidth, screenHeight / srcHeight);
            dstWidth = srcWidth * scale;
            dstHeight = srcHeight * scale;
        }
        else if (aspectMode == 0)
        {
            const float scale = std::min(screenWidth / 4.0f, screenHeight / 3.0f);
            dstWidth = 4.0f * scale;
            dstHeight = 3.0f * scale;
        }
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        DrawTexturePro(frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        {
            // s3recomp: once a second, sample the counters; every 5 s write a [perf] line to the log.
            // PS2X_SHOW_FPS=1 draws them in the corner of the window.
            static const bool showFps = [] {
                const char *v = std::getenv("PS2X_SHOW_FPS");
                return v && v[0] == '1';
            }();
            static uint64_t lastNs = s3perf::nowNs(), lastLogNs = lastNs;
            static uint64_t lastFrames = 0, lastVsync = 0, lastVu = 0, lastGs = 0, lastGsVu = 0, presents = 0;
            static uint64_t logFrames = 0, logVsync = 0, logVu = 0, logGs = 0, logGsVu = 0, logPresents = 0;
            static uint64_t logVuCycles = 0, logVuNative = 0, logIop = 0, logSwitches = 0, logKick = 0, logOverlap = 0, logSyncs = 0;
            static char overlay[160] = "";
            ++presents;
            const uint64_t now = s3perf::nowNs();
            if (now - lastNs >= 1000000000ull)
            {
                const double secs = double(now - lastNs) / 1e9;
                const uint64_t frames = s3perf::frames.load(std::memory_order_relaxed);
                const uint64_t vsync = eeScheduler().currentVSyncTick();
                const uint64_t vu = s3perf::vu1Ns.load(std::memory_order_relaxed);
                const uint64_t gs = s3perf::gsNs.load(std::memory_order_relaxed);
                const uint64_t gsvu = s3perf::gsInVu1Ns.load(std::memory_order_relaxed);
                const double wall = double(now - lastNs);
                const double vuPct = 100.0 * double((vu - lastVu) - (gsvu - lastGsVu)) / wall;
                const double gsPct = 100.0 * double(gs - lastGs) / wall;
                std::snprintf(overlay, sizeof(overlay), "FPS %.1f  vsync %.0f/60  VU1 %.0f%%  GS %.0f%%",
                              double(frames - lastFrames) / secs, double(vsync - lastVsync) / secs, vuPct, gsPct);
                lastNs = now;
                lastFrames = frames; lastVsync = vsync; lastVu = vu; lastGs = gs; lastGsVu = gsvu;
                if (now - lastLogNs >= 5000000000ull)
                {
                    const double ls = double(now - lastLogNs) / 1e9, lw = double(now - lastLogNs);
                    const double lvu = 100.0 * double((vu - logVu) - (gsvu - logGsVu)) / lw;
                    const double lgs = 100.0 * double(gs - logGs) / lw;
                    const uint64_t vuCycles = s3perf::vu1Cycles.load(std::memory_order_relaxed);
                    const uint64_t vuNative = s3perf::vu1NativeCycles.load(std::memory_order_relaxed);
                    const double nativePct = vuCycles > logVuCycles
                                                 ? 100.0 * double(vuNative - logVuNative) / double(vuCycles - logVuCycles)
                                                 : 0.0;
                    const uint64_t iop = s3perf::iopNs.load(std::memory_order_relaxed);
                    const uint64_t switches = s3perf::eeSwitches.load(std::memory_order_relaxed);
                    const double liop = 100.0 * double(iop - logIop) / lw;
                    static uint64_t logIdle = 0;
                    const uint64_t idle = s3perf::eeIdleNs.load(std::memory_order_relaxed);
                    const double lidle = 100.0 * double(idle - logIdle) / lw;
                    logIdle = idle;
                    std::fprintf(stderr, "[perf] fps=%.1f vsync=%.1f/s present=%.1f/s game-thread: vu1=%.0f%% gs=%.0f%% idle=%.0f%% other=%.0f%% (iop %.0f%%) vu1-native=%.0f%% ee-switches=%.0f/s\n",
                                 double(frames - logFrames) / ls, double(vsync - logVsync) / ls, double(presents - logPresents) / ls,
                                 lvu, lgs, lidle, std::max(0.0, 100.0 - lvu - lgs - lidle), liop, nativePct, double(switches - logSwitches) / ls);
                    logIop = iop;
                    logSwitches = switches;
                    {
                        const uint64_t kick = s3gfx::kickNs.load(), overlap = s3gfx::overlapNs.load(), syncs = s3gfx::syncs.load();
                        static uint64_t logBusy = 0, logWait = 0;
                        const uint64_t busy = s3gfx::workerBusyNs.load(), wait = s3gfx::syncWaitNs.load();
                        std::fprintf(stderr, "[gfx] %s dma-processing=%.1fms/s overlappable=%.1fms/s worker-busy=%.0f%% ee-waiting=%.0f%% syncs=%.0f/s reasons(dstat,chan,vifgif,gs,vu1,stub)=%llu,%llu,%llu,%llu,%llu,%llu\n",
                                     s3gfx::enabled() ? "threaded" : "inline",
                                     double(kick - logKick) / 1e6 / ls, double(overlap - logOverlap) / 1e6 / ls,
                                     100.0 * double(busy - logBusy) / lw, 100.0 * double(wait - logWait) / lw, double(syncs - logSyncs) / ls,
                                     (unsigned long long)s3gfx::syncReason[0].load(), (unsigned long long)s3gfx::syncReason[1].load(),
                                     (unsigned long long)s3gfx::syncReason[2].load(), (unsigned long long)s3gfx::syncReason[3].load(),
                                     (unsigned long long)s3gfx::syncReason[4].load(), (unsigned long long)s3gfx::syncReason[5].load());
                        logKick = kick; logOverlap = overlap; logSyncs = syncs; logBusy = busy; logWait = wait;
                        {
                            static uint64_t lastStat[16] = {}, lastStub[8] = {};
                            uint64_t st[16], sb[8];
                            for (int k = 0; k < 16; ++k) st[k] = s3gfx::stat[k].load();
                            for (int k = 0; k < 8; ++k) sb[k] = s3gfx::stubCalls[k].load();
                            auto rate = [&](int k) { return double(st[k] - lastStat[k]) / ls; };
                            auto srate = [&](int k) { return double(sb[k] - lastStub[k]) / ls; };
                            const uint64_t applied = st[2] - lastStat[2];
                            std::fprintf(stderr, "[gfx] per-second: kicks vif1=%.0f gif=%.0f completions=%.0f (event %.0f, sync %.0f, poll %.0f) latency avg=%.2fms max=%.2fms | reads dstat=%.0f chcr=%.0f csr=%.0f | stubs syncpath=%.0f syncv=%.0f swap=%.0f putdisp=%.0f putdraw=%.0f dmasend=%.0f dmasync=%.0f\n",
                                         rate(0), rate(1), rate(2), rate(8), rate(9), rate(10),
                                         applied ? double(st[6] - lastStat[6]) / 1000.0 / double(applied) : 0.0, double(st[7]) / 1000.0,
                                         rate(3), rate(4), rate(5), srate(0), srate(1), srate(2), srate(3), srate(4), srate(5), srate(6));
                            s3gfx::stat[7].store(0);
                            for (int k = 0; k < 16; ++k) lastStat[k] = st[k];
                            for (int k = 0; k < 8; ++k) lastStub[k] = sb[k];
                        }
                        std::string gsWaits;
                        for (uint32_t r = 0; r < 0x110u; ++r)
                            if (const uint64_t n = s3gfx::gsRegWaits[r].load())
                                gsWaits += " " + std::to_string(r << 4) + ":" + std::to_string(n);
                        if (!gsWaits.empty())
                            std::fprintf(stderr, "[gfx] waits by GS register offset:%s\n", gsWaits.c_str());
                    }
                    logVuCycles = vuCycles;
                    logVuNative = vuNative;
                    lastLogNs = now;
                    logFrames = frames; logVsync = vsync; logVu = vu; logGs = gs; logGsVu = gsvu; logPresents = presents;
                }
            }
            if (s3ff::active.load(std::memory_order_relaxed))
            {
                char ffText[48];
                std::snprintf(ffText, sizeof(ffText), "FAST-FORWARD x%u", s3ff::speed());
                const int fontSize = std::max(16, GetScreenHeight() / 40);
                const int w = MeasureText(ffText, fontSize);
                const int x = GetScreenWidth() - w - 18;
                DrawRectangle(x - 6, 6, w + 12, fontSize + 8, Color{0, 0, 0, 170});
                DrawText(ffText, x, 10, fontSize, Color{255, 210, 90, 255});
            }
            if (showFps && overlay[0])
            {
                const int fontSize = std::max(16, GetScreenHeight() / 40);
                const int w = MeasureText(overlay, fontSize);
                DrawRectangle(6, 6, w + 12, fontSize + 8, Color{0, 0, 0, 170});
                DrawText(overlay, 12, 10, fontSize, Color{120, 255, 120, 255});
            }
        }
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();
    if (gameThread.joinable())
    {
        gameThread.join();
    }

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }
    UnloadTexture(frameTex);
    CloseWindow();

    RUNTIME_LOG("[run] exiting loop");
}
