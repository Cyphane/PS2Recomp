// s3recomp: built-in sampling profiler (Windows). With PS2X_PROFILE=1, a background thread samples
// the registered threads (game thread, presenter, graphics worker) about 1000 times a second: it
// suspends each one, walks its stack with the x64 unwind tables, and resumes it. Every 10 seconds it
// writes the busiest functions of each thread to stderr (boot.log), by self time (where the thread
// was) and by inclusive time (the function was anywhere on the stack). Function names come from the
// executable's own COFF symbol table; time in other modules (the graphics driver, Windows waits) is
// shown by module name.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace s3prof
{
    bool enabled()
    {
        static const bool on = [] {
            const char *v = std::getenv("PS2X_PROFILE");
            return v && v[0] == '1';
        }();
        return on;
    }
}

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0602 // GetCurrentThreadStackLimits
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <map>
#include <cxxabi.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace s3prof
{
    namespace
    {
        constexpr int kMaxThreads = 8;
        constexpr int kMaxDepth = 64;
        constexpr DWORD kReportMs = 10000;

        struct ThreadSlot
        {
            HANDLE handle = nullptr;
            const char *name = "";
            ULONG_PTR stackLow = 0;
            ULONG_PTR stackHigh = 0;
        };
        ThreadSlot g_threads[kMaxThreads];
        std::atomic<int> g_threadCount{0};
        std::mutex g_registerMutex;

        struct Symbol
        {
            uintptr_t address;
            std::string name; // mangled
        };
        std::vector<Symbol> g_symbols; // sorted by address
        uintptr_t g_exeBase = 0;
        uintptr_t g_exeEnd = 0;

        // Function ids: index into g_symbols, or kModuleBase + module index for other modules.
        constexpr uint32_t kUnknown = 0xFFFFFFFFu;
        constexpr uint32_t kModuleBase = 0x80000000u;
        std::vector<std::string> g_modules;
        std::unordered_map<uintptr_t, uint32_t> g_moduleIds; // HMODULE -> index

        bool readAt(HANDLE file, uint64_t offset, void *out, size_t size)
        {
            LARGE_INTEGER pos;
            pos.QuadPart = static_cast<LONGLONG>(offset);
            if (!SetFilePointerEx(file, pos, nullptr, FILE_BEGIN))
                return false;
            uint8_t *dst = static_cast<uint8_t *>(out);
            while (size != 0)
            {
                DWORD chunk = static_cast<DWORD>(std::min<size_t>(size, 64u << 20));
                DWORD got = 0;
                if (!ReadFile(file, dst, chunk, &got, nullptr) || got == 0)
                    return false;
                dst += got;
                size -= got;
            }
            return true;
        }

        // Loads function symbols from the executable's COFF symbol table.
        void loadSymbols()
        {
            HMODULE exe = GetModuleHandleW(nullptr);
            g_exeBase = reinterpret_cast<uintptr_t>(exe);
            const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(exe);
            const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(g_exeBase + dos->e_lfanew);
            g_exeEnd = g_exeBase + nt->OptionalHeader.SizeOfImage;
            const IMAGE_SECTION_HEADER *sections = IMAGE_FIRST_SECTION(nt);
            const unsigned sectionCount = nt->FileHeader.NumberOfSections;
            const uint64_t symbolOffset = nt->FileHeader.PointerToSymbolTable;
            const uint64_t symbolCount = nt->FileHeader.NumberOfSymbols;
            if (symbolOffset == 0 || symbolCount == 0)
            {
                std::fprintf(stderr, "[prof] the executable has no symbol table; only module names will be shown\n");
                return;
            }

            wchar_t path[MAX_PATH * 2];
            if (!GetModuleFileNameW(nullptr, path, MAX_PATH * 2))
                return;
            HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (file == INVALID_HANDLE_VALUE)
                return;
            std::vector<uint8_t> table(static_cast<size_t>(symbolCount) * 18u);
            uint32_t stringSize = 0;
            std::vector<char> strings;
            const bool ok = readAt(file, symbolOffset, table.data(), table.size()) &&
                            readAt(file, symbolOffset + table.size(), &stringSize, 4);
            if (ok && stringSize >= 4)
            {
                strings.resize(stringSize);
                std::memcpy(strings.data(), &stringSize, 4);
                if (!readAt(file, symbolOffset + table.size() + 4, strings.data() + 4, stringSize - 4))
                    strings.clear();
            }
            CloseHandle(file);
            if (!ok)
                return;

            for (uint64_t i = 0; i < symbolCount; ++i)
            {
                const uint8_t *s = table.data() + i * 18u;
                uint32_t value;
                int16_t section;
                uint16_t type;
                std::memcpy(&value, s + 8, 4);
                std::memcpy(&section, s + 12, 2);
                std::memcpy(&type, s + 14, 2);
                const uint8_t storage = s[16];
                const uint8_t aux = s[17];
                if (section > 0 && static_cast<unsigned>(section) <= sectionCount &&
                    (storage == IMAGE_SYM_CLASS_EXTERNAL || storage == IMAGE_SYM_CLASS_STATIC) &&
                    (sections[section - 1].Characteristics & IMAGE_SCN_MEM_EXECUTE))
                {
                    std::string name;
                    uint32_t zeros;
                    std::memcpy(&zeros, s, 4);
                    if (zeros == 0)
                    {
                        uint32_t offset;
                        std::memcpy(&offset, s + 4, 4);
                        if (offset < strings.size())
                            name.assign(strings.data() + offset, strnlen(strings.data() + offset, strings.size() - offset));
                    }
                    else
                        name.assign(reinterpret_cast<const char *>(s), strnlen(reinterpret_cast<const char *>(s), 8));
                    const bool isFunction = (type & 0x30u) == 0x20u;
                    if (!name.empty() && (isFunction || (name[0] != '.' && name[0] != '$')))
                        g_symbols.push_back({g_exeBase + sections[section - 1].VirtualAddress + value, std::move(name)});
                }
                i += aux;
            }
            std::sort(g_symbols.begin(), g_symbols.end(), [](const Symbol &a, const Symbol &b) { return a.address < b.address; });
            std::fprintf(stderr, "[prof] %zu function symbols loaded\n", g_symbols.size());
        }

        uint32_t functionId(uintptr_t pc)
        {
            if (pc >= g_exeBase && pc < g_exeEnd && !g_symbols.empty())
            {
                auto it = std::upper_bound(g_symbols.begin(), g_symbols.end(), pc,
                                           [](uintptr_t value, const Symbol &s) { return value < s.address; });
                if (it != g_symbols.begin())
                    return static_cast<uint32_t>((it - g_symbols.begin()) - 1);
            }
            HMODULE module = nullptr;
            if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                    reinterpret_cast<LPCWSTR>(pc), &module) ||
                !module)
                return kUnknown;
            auto found = g_moduleIds.find(reinterpret_cast<uintptr_t>(module));
            if (found != g_moduleIds.end())
                return kModuleBase + found->second;
            char path[MAX_PATH];
            const DWORD n = GetModuleFileNameA(module, path, MAX_PATH);
            std::string name = n ? std::string(path, n) : std::string("?");
            const size_t slash = name.find_last_of("\\/");
            if (slash != std::string::npos)
                name = name.substr(slash + 1);
            const uint32_t id = static_cast<uint32_t>(g_modules.size());
            g_modules.push_back("[" + name + "]");
            g_moduleIds[reinterpret_cast<uintptr_t>(module)] = id;
            return kModuleBase + id;
        }

        std::string functionName(uint32_t id)
        {
            if (id == kUnknown)
                return "[unknown]";
            if (id >= kModuleBase)
                return g_modules[id - kModuleBase];
            const std::string &mangled = g_symbols[id].name;
            int status = 0;
            char *demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
            std::string out = (status == 0 && demangled) ? std::string(demangled) : mangled;
            std::free(demangled);
            if (out.size() > 110)
                out = out.substr(0, 107) + "...";
            return out;
        }

        // Walks the stack of a suspended thread into frames[]; no allocation or locking here.
        int walk(const CONTEXT &start, const ThreadSlot &slot, uintptr_t *frames)
        {
            CONTEXT ctx = start;
            int depth = 0;
            while (depth < kMaxDepth && ctx.Rip != 0)
            {
                frames[depth++] = static_cast<uintptr_t>(ctx.Rip);
                DWORD64 imageBase = 0;
                PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &imageBase, nullptr);
                if (fn)
                {
                    void *handlerData = nullptr;
                    DWORD64 establisher = 0;
                    RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, ctx.Rip, fn, &ctx, &handlerData, &establisher, nullptr);
                }
                else
                {
                    // Leaf function: the return address is at the top of the stack.
                    if (ctx.Rsp < start.Rsp || ctx.Rsp < slot.stackLow || ctx.Rsp + 8 > slot.stackHigh)
                        break;
                    ctx.Rip = *reinterpret_cast<const DWORD64 *>(ctx.Rsp);
                    ctx.Rsp += 8;
                }
                // Unwinding only moves up the stack; anything else means a bad frame. Staying at or
                // above the sampled stack pointer also keeps reads inside committed stack memory.
                if (ctx.Rsp < start.Rsp || ctx.Rsp >= slot.stackHigh)
                    break;
            }
            return depth;
        }

        constexpr int kStackKey = 7; // frames kept per call path, from the leaf up
        using StackKey = std::array<uint32_t, kStackKey>;
        struct Counts
        {
            uint64_t samples = 0;
            std::unordered_map<uint32_t, uint64_t> self, inclusive;
            std::map<StackKey, uint64_t> stacks;
        };

        void report(Counts *counts, int threads, double seconds)
        {
            for (int t = 0; t < threads; ++t)
            {
                Counts &c = counts[t];
                if (c.samples == 0)
                    continue;
                std::fprintf(stderr, "[prof] ---- %s thread, last %.1f s, %llu samples ----\n", g_threads[t].name, seconds,
                             static_cast<unsigned long long>(c.samples));
                auto top = [&](const std::unordered_map<uint32_t, uint64_t> &m, const char *what, size_t limit) {
                    std::vector<std::pair<uint64_t, uint32_t>> v;
                    v.reserve(m.size());
                    for (const auto &[id, n] : m)
                        v.push_back({n, id});
                    std::sort(v.rbegin(), v.rend());
                    std::fprintf(stderr, "[prof] %s:\n", what);
                    for (size_t i = 0; i < v.size() && i < limit; ++i)
                    {
                        const double pct = 100.0 * double(v[i].first) / double(c.samples);
                        if (pct < 0.5)
                            break;
                        std::fprintf(stderr, "[prof] %5.1f%%  %s\n", pct, functionName(v[i].second).c_str());
                    }
                };
                top(c.self, "self (where the thread was)", 30);
                top(c.inclusive, "inclusive (on the stack)", 40);
                {
                    std::vector<std::pair<uint64_t, const StackKey *>> v;
                    for (const auto &[key, n] : c.stacks)
                        v.push_back({n, &key});
                    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
                    std::fprintf(stderr, "[prof] call paths (leaf <- callers):\n");
                    for (size_t i = 0; i < v.size() && i < 30; ++i)
                    {
                        const double pct = 100.0 * double(v[i].first) / double(c.samples);
                        if (pct < 0.7)
                            break;
                        std::string line;
                        for (uint32_t id : *v[i].second)
                        {
                            if (id == kUnknown && !line.empty())
                                break;
                            std::string name = functionName(id);
                            const size_t paren = name.find('(');
                            if (paren != std::string::npos && paren > 0)
                                name = name.substr(0, paren);
                            if (name.size() > 48)
                                name = name.substr(0, 45) + "...";
                            line += line.empty() ? name : " <- " + name;
                        }
                        std::fprintf(stderr, "[prof] %5.1f%%  %s\n", pct, line.c_str());
                    }
                }
                c = Counts{};
            }
            std::fflush(stderr);
        }

        void samplerMain()
        {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            loadSymbols();
            Counts counts[kMaxThreads];
            uintptr_t frames[kMaxDepth];
            uint32_t ids[kMaxDepth];
            DWORD lastReport = GetTickCount();
            for (;;)
            {
                Sleep(1);
                const int threads = g_threadCount.load(std::memory_order_acquire);
                for (int t = 0; t < threads; ++t)
                {
                    const ThreadSlot &slot = g_threads[t];
                    if (SuspendThread(slot.handle) == static_cast<DWORD>(-1))
                        continue;
                    CONTEXT ctx;
                    std::memset(&ctx, 0, sizeof(ctx));
                    ctx.ContextFlags = CONTEXT_FULL;
                    int depth = 0;
                    if (GetThreadContext(slot.handle, &ctx))
                        depth = walk(ctx, slot, frames);
                    ResumeThread(slot.handle);
                    if (depth == 0)
                        continue;
                    Counts &c = counts[t];
                    ++c.samples;
                    StackKey key;
                    key.fill(kUnknown);
                    int keyLen = 0;
                    int unique = 0;
                    for (int d = 0; d < depth; ++d)
                    {
                        // Return addresses point after the call; step back into the calling function.
                        const uint32_t id = functionId(d == 0 ? frames[d] : frames[d] - 1);
                        if (d == 0)
                            ++c.self[id];
                        if (keyLen < kStackKey && (keyLen == 0 || key[keyLen - 1] != id))
                            key[keyLen++] = id;
                        bool seen = false;
                        for (int k = 0; k < unique && !seen; ++k)
                            seen = ids[k] == id;
                        if (!seen)
                        {
                            ids[unique++] = id;
                            ++c.inclusive[id];
                        }
                    }
                    ++c.stacks[key];
                }
                const DWORD now = GetTickCount();
                if (now - lastReport >= kReportMs)
                {
                    report(counts, threads, double(now - lastReport) / 1000.0);
                    lastReport = now;
                }
            }
        }
    }

    void registerThread(const char *name)
    {
        if (!enabled())
            return;
        std::lock_guard<std::mutex> lock(g_registerMutex);
        const int n = g_threadCount.load(std::memory_order_relaxed);
        if (n >= kMaxThreads)
            return;
        ThreadSlot &slot = g_threads[n];
        slot.handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, GetCurrentThreadId());
        if (!slot.handle)
            return;
        slot.name = name;
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        slot.stackLow = low;
        slot.stackHigh = high;
        g_threadCount.store(n + 1, std::memory_order_release);
        static std::once_flag started;
        std::call_once(started, [] {
            std::fprintf(stderr, "[prof] sampling profiler on (PS2X_PROFILE=1)\n");
            std::thread(samplerMain).detach();
        });
    }
}

#else

namespace s3prof
{
    void registerThread(const char *) {}
}

#endif
