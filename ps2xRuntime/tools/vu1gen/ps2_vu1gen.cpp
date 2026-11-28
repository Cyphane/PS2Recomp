// VU1 program generator: finds VU1 microprograms uploaded with VIF MPG commands in a PS2 executable
// (or overlay) and writes them out as C++ that runs on VU1Fast. Each instruction pair becomes
// straight-line code with its timing bookkeeping resolved at generation time.
#define private public
#include <functional>
#include <map>
#include "ps2_vu1_fast.h"
#include "ps2_vu1_turbo_inl.h"
#undef private
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <iterator>

static void emit(std::string &o, const char *fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    if (static_cast<size_t>(n) < sizeof(buf))
    {
        o.append(buf, static_cast<size_t>(n));
        return;
    }
    std::string big(static_cast<size_t>(n) + 1, '\0');
    va_start(ap, fmt);
    std::vsnprintf(big.data(), big.size(), fmt, ap);
    va_end(ap);
    o.append(big.data(), static_cast<size_t>(n));
}

// Rewrites a file only when its contents change, so unchanged programs are not recompiled.
static bool writeIfChanged(const std::filesystem::path &path, const std::string &text)
{
    {
        std::ifstream in(path, std::ios::binary);
        if (in)
        {
            const std::string old((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (old == text)
                return true;
        }
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return static_cast<bool>(out);
}

struct Program { uint32_t addr; std::vector<uint8_t> code; std::string origin; };

static uint32_t rd32(const std::vector<uint8_t> &b, size_t o) { uint32_t v; std::memcpy(&v, &b[o], 4); return v; }

// Scan a file image for MPG uploads; consecutive uploads that continue each other form one program.
static std::vector<Program> findPrograms(const std::vector<uint8_t> &img, size_t startLimit, const std::string &name, VU1Fast &dec)
{
    struct Chunk { size_t off; uint32_t addr; size_t codeOff; uint32_t bytes; };
    std::vector<Chunk> chunks;
    for (size_t i = 0; i + 8 <= img.size(); i += 4)
    {
        const uint32_t w = rd32(img, i);
        if (((w >> 24) & 0x7Fu) != 0x4Au)
            continue;
        const uint32_t num = ((w >> 16) & 0xFFu) ? ((w >> 16) & 0xFFu) : 256u;
        const uint32_t addr = (w & 0xFFFFu) * 8u;
        size_t start = i + 4;
        if (start % 8) start += 4;
        if (start + num * 8u > img.size() || addr + num * 8u > 16384u)
            continue;
        // every pair must decode cleanly
        bool ok = true;
        for (uint32_t p = 0; p < num && ok; ++p)
        {
            const auto d = dec.decodeInstructionPair(img.data() + start, p * 8u);
            ok = !d.upperUsage.reserved && !d.lowerUsage.reserved;
        }
        if (!ok)
            continue;
        chunks.push_back({i, addr, start, num * 8u});
        i = start + num * 8u - 4;
    }
    std::vector<Program> progs;
    for (size_t k = 0; k < chunks.size(); ++k)
    {
        if (chunks[k].off >= startLimit)
            break; // starts in the overlap with the next window, which will see it whole
        Program p{chunks[k].addr, {}, name};
        p.code.insert(p.code.end(), img.begin() + chunks[k].codeOff, img.begin() + chunks[k].codeOff + chunks[k].bytes);
        size_t last = k;
        while (k + 1 < chunks.size() && chunks[k + 1].addr == p.addr + p.code.size() && chunks[k + 1].off - (chunks[last].codeOff + chunks[last].bytes) <= 64)
        {
            ++k;
            p.code.insert(p.code.end(), img.begin() + chunks[k].codeOff, img.begin() + chunks[k].codeOff + chunks[k].bytes);
            last = k;
        }
        bool hasEnd = false;
        for (size_t o = 0; o + 8 <= p.code.size(); o += 8)
            if (rd32(p.code, o + 4) & 0x40000000u) hasEnd = true;
        if (hasEnd && p.code.size() >= 32)
            progs.push_back(std::move(p));
    }
    return progs;
}

static bool isUpperNop(uint32_t u)
{
    if ((u & 0x3Cu) != 0x3Cu) return false;
    const uint8_t sp = static_cast<uint8_t>((u & 3u) | ((u >> 4) & 0x7Cu));
    return sp == 0x2Fu || sp == 0x30u;
}

// Generation-time timing knowledge within one block. A register written at pair j with latency L
// can stall a read at pair k only if k - j < L, since every pair takes at least one cycle. A register
// not written yet in the block was written before it, at least one cycle before the block began, so
// it cannot stall a read at pair k >= its kind's largest latency. Reads that cannot stall need no
// runtime check (stallPair still recomputes the exact stall whenever one is checked for).
struct BlockTiming
{
    int vf[32][4];
    int vi[16];
    int acc[4];
};

struct MaxLatency
{
    int vf = 0, vi = 0, acc = 0;
};

static MaxLatency findMaxLatency(VU1Fast &dec)
{
    MaxLatency m;
    m.acc = static_cast<int>(VU1Fast::kAccForwardLatency);
    uint8_t code[8];
    auto scan = [&](uint32_t upper, uint32_t lower) {
        std::memcpy(code, &lower, 4);
        std::memcpy(code + 4, &upper, 4);
        const auto d = dec.decodeInstructionPair(code, 0u);
        if (d.upperUsage.reserved || d.lowerUsage.reserved)
            return;
        for (uint32_t k = 0; k < d.writeCount; ++k)
            m.vf = std::max<int>(m.vf, d.writeLatency[k]);
        if (d.viWriteMask)
            m.vi = std::max<int>(m.vi, d.viWriteLatency);
    };
    const uint32_t nopUpper = 0x000002FFu, nopLower = 0x8000033Cu;
    for (uint32_t low = 0; low < 2048u; ++low)
        scan((0x1Fu << 21) | (1u << 16) | (2u << 11) | low, nopLower); // all upper ops, all lanes
    for (uint32_t op = 0; op < 128u; ++op)
        for (uint32_t low = 0; low < 2048u; ++low)
            scan(nopUpper, (op << 25) | (0x1Fu << 21) | (3u << 16) | (4u << 11) | low);
    return m;
}

static MaxLatency g_maxLatency;

static void beginBlockTiming(BlockTiming &t)
{
    for (auto &r : t.vf)
        for (int &l : r)
            l = g_maxLatency.vf;
    for (int &v : t.vi)
        v = g_maxLatency.vi;
    for (int &a : t.acc)
        a = g_maxLatency.acc;
}

static bool pairIsControl(const VU1Fast::DecodedInstructionPair &d)
{
    return d.eBit || d.dBit || d.tBit || (!d.iBit && d.lowerUsage.pipeline == VU1Fast::PipelineBranch);
}

// s3recomp: whether the pair before A may be a control pair whose delay slot A is. Before the
// program's first pair lies memory outside the program, so that is treated as possible (it used to
// be decoded from beyond the program's buffer).
static bool prevPairMayBeControl(VU1Fast &dec, const Program &p, uint32_t A)
{
    if (A <= p.addr)
        return true;
    return pairIsControl(dec.decodeInstructionPair(p.code.data() - p.addr, A - 8u));
}

// Programs that never read or set the status flag apply FMAC status results at issue
// (VU1Fast::m_eagerStatus).
static bool programAllowsEagerStatus(VU1Fast &dec, const Program &p)
{
    const uint32_t end = p.addr + uint32_t(p.code.size());
    for (uint32_t A = p.addr; A < end; A += 8)
    {
        const auto d = dec.decodeInstructionPair(p.code.data() - p.addr, A);
        if (d.iBit)
            continue;
        const uint32_t op = d.lower >> 25;
        if (op == 0x14u || op == 0x15u || op == 0x16u || op == 0x17u)
            return false;
    }
    return true;
}

// Upper instructions that write the MAC flag (when their dest is not empty).
static bool upperWritesMac(uint32_t u)
{
    if (((u >> 21) & 0xFu) == 0u)
        return false;
    if ((u & 0x3Cu) == 0x3Cu)
    {
        const uint32_t c = (u & 3u) | ((u >> 4) & 0x7Cu);
        return c <= 0x0Fu || (c >= 0x18u && c <= 0x1Cu) || c == 0x1Eu || (c >= 0x20u && c <= 0x2Au) || c == 0x2Cu || c == 0x2Du || c == 0x2Eu;
    }
    const uint32_t c = u & 0x3Fu;
    return c <= 0x0Fu || (c >= 0x18u && c <= 0x1Cu) || c == 0x1Eu || (c >= 0x20u && c <= 0x2Au) || c == 0x2Cu || c == 0x2Du || c == 0x2Eu;
}

static bool lowerReadsMac(const VU1Fast::DecodedInstructionPair &d)
{
    if (d.iBit)
        return false;
    const uint32_t op = d.lower >> 25;
    return op == 0x18u || op == 0x1Au || op == 0x1Bu;
}

// For each pair of a straight-line block, whether its MAC result is dead: a later FMAC at pair i2 in
// the block replaces it, and no MAC read is at pairs i+1..i2+3 (a read there may still see it; a read
// at least four pairs after i2 sees i2's result or a later one, whatever the stalls).
static std::vector<bool> deadMacResults(VU1Fast &dec, const Program &p, uint32_t start, uint32_t stop)
{
    const uint32_t n = (stop - start) / 8u;
    std::vector<bool> writes(n), reads(n), dead(n, false);
    for (uint32_t k = 0; k < n; ++k)
    {
        const auto d = dec.decodeInstructionPair(p.code.data() - p.addr, start + k * 8u);
        writes[k] = !d.upperUsage.reserved && upperWritesMac(d.upper);
        reads[k] = lowerReadsMac(d);
    }
    for (uint32_t i = 0; i < n; ++i)
    {
        if (!writes[i])
            continue;
        uint32_t i2 = i + 1;
        while (i2 < n && !writes[i2])
            ++i2;
        if (i2 >= n || i2 + 3u >= n)
            continue;
        bool readSeen = false;
        for (uint32_t j = i + 1; j <= i2 + 3u && !readSeen; ++j)
            readSeen = reads[j];
        dead[i] = !readSeen;
    }
    return dead;
}

// s3recomp: program-wide version of deadMacResults. A MAC result written at pair i is dead when, on
// every path from i, another FMAC writes MAC before any MAC read and no read follows within three
// pairs of that write (reads there might still see i's result), and the program does not end or
// leave (JR/JALR, falling off the end) before that. Paths follow the VU1's delay slots: the pair
// after a branch or E/D/T pair executes, then control goes to the target(s) or ends.
static std::vector<bool> deadMacProgram(VU1Fast &dec, const Program &p, const std::set<uint32_t> &leaders)
{
    const uint32_t n = uint32_t(p.code.size() / 8u);
    const uint32_t base = p.addr;
    std::vector<VU1Fast::DecodedInstructionPair> d(n);
    std::vector<bool> writes(n), reads(n);
    for (uint32_t k = 0; k < n; ++k)
    {
        d[k] = dec.decodeInstructionPair(p.code.data() - p.addr, base + k * 8u);
        writes[k] = !d[k].upperUsage.reserved && !d[k].iBit && upperWritesMac(d[k].upper);
        reads[k] = d[k].upperUsage.reserved || d[k].lowerUsage.reserved || lowerReadsMac(d[k]);
    }
    constexpr int kEnd = -1, kUnknown = -2;
    // successors of pair k (as pair indices, or kEnd / kUnknown)
    auto controlSuccs = [&](uint32_t c, std::vector<int> &out) {
        // c is a control pair; this gives where control goes after its delay slot c+1
        const auto &dc = d[c];
        if (dc.eBit) { out.push_back(kEnd); return; }
        bool fall = dc.dBit || dc.tBit; // a halt that may not trigger
        if (dc.dBit || dc.tBit) out.push_back(kEnd);
        if (!dc.iBit && dc.lowerUsage.pipeline == VU1Fast::PipelineBranch)
        {
            const uint32_t op = dc.lower >> 25;
            if (op == 0x24u || op == 0x25u) { out.push_back(kUnknown); return; } // JR / JALR
            const int32_t imm = static_cast<int32_t>(dc.lower << 21) >> 21;
            const uint32_t target = (base + c * 8u + 8u + uint32_t(imm * 8)) & 0x3FFFu;
            if (target >= base && target < base + n * 8u) out.push_back(int((target - base) / 8u));
            else out.push_back(kUnknown);
            const bool unconditional = op == 0x20u || op == 0x21u; // B / BAL
            if (!unconditional) fall = true;
        }
        if (fall) out.push_back(c + 2u < n ? int(c + 2u) : kUnknown);
    };
    std::vector<std::vector<int>> succ(n);
    for (uint32_t k = 0; k < n; ++k)
    {
        const bool prevControl = k > 0 && pairIsControl(d[k - 1]);
        if (prevControl)
        {
            controlSuccs(k - 1, succ[k]);
            if (leaders.count(base + k * 8u)) // reachable without its control pair too
                succ[k].push_back(k + 1u < n ? int(k + 1u) : kUnknown);
        }
        else
            succ[k].push_back(k + 1u < n ? int(k + 1u) : kUnknown);
    }
    // g(k, w): no MAC read in the next w pairs from k on any path (an end is fine: the pipeline then
    // flushes in order, so the later write wins).
    std::map<std::pair<int, int>, bool> gMemo;
    std::function<bool(int, int)> g = [&](int k, int w) -> bool {
        if (w == 0 || k == kEnd) return true;
        if (k == kUnknown) return false;
        auto key = std::make_pair(k, w);
        auto it = gMemo.find(key);
        if (it != gMemo.end()) return it->second;
        bool ok = !reads[k];
        for (size_t s = 0; ok && s < succ[k].size(); ++s) ok = g(succ[k][s], w - 1);
        gMemo[key] = ok;
        return ok;
    };
    // f(k): from k, on every path a MAC write comes before any read (and g holds after it), without
    // ending or leaving first. Computed as a greatest fixed point without recursion: start with every
    // pair that is not a read as true and clear pairs whose condition fails until nothing changes
    // (a cycle with no reads keeps the old value unread, so it stays true).
    std::vector<int8_t> fState(n);
    for (uint32_t k = 0; k < n; ++k)
        fState[k] = reads[k] ? 3 : 2;
    for (bool changed = true; changed;)
    {
        changed = false;
        for (uint32_t k = 0; k < n; ++k)
        {
            if (fState[k] != 2)
                continue;
            bool ok = true;
            if (writes[k])
                for (int s : succ[k]) ok = ok && g(s, 3);
            else
                for (int s : succ[k]) ok = ok && s >= 0 && fState[s] == 2;
            if (!ok) { fState[k] = 3; changed = true; }
        }
    }
    std::vector<bool> dead(n, false);
    for (uint32_t i = 0; i < n; ++i)
    {
        if (!writes[i]) continue;
        bool ok = true;
        for (int s : succ[i]) ok = ok && s >= 0 && fState[s] == 2;
        dead[i] = ok;
    }
    return dead;
}

static bool g_programWritesZeroRegs = true; // s3recomp: per program, set before its blocks are emitted

static void emitPair(std::string &o, VU1Fast &dec, const Program &p, uint32_t A, BlockTiming &timing, int pairIndex, bool macDead)
{
    const auto d = dec.decodeInstructionPair(p.code.data() - p.addr, A);
    const bool plainEnd = pairIndex > 0 && !pairIsControl(d) &&
                          !prevPairMayBeControl(dec, p, A);
    const uint32_t next = (A + 8u) & 0x3FFFu;
    emit(o, "  { // 0x%04X\n    if (v.m_cycle >= budgetEnd || v.m_stopRequested) return kGenBudget;\n", A);
    if (d.upperUsage.reserved || d.lowerUsage.reserved)
    {
        emit(o, "    return kGenFallback;\n  }\n");
        return;
    }
    // s3recomp: the previous pair in the block ended with stepCycle(), which already committed what
    // was due at this cycle, so only a block's first pair needs the check.
    if (pairIndex == 0)
        emit(o, "    v.commitIfDue();\n");
    std::string ready;
    for (uint32_t k = 0; k < d.readCount; ++k)
        for (int l = 0; l < 4; ++l)
            if ((d.readLanes[k] & (8u >> l)) && pairIndex < timing.vf[d.readReg[k] & 31u][l])
                ready += " r = std::max(r, v.m_vfReady[" + std::to_string(d.readReg[k]) + "][" + std::to_string(l) + "]);";
    for (uint32_t bits = d.viReadMask; bits; bits &= bits - 1)
        if (pairIndex < timing.vi[__builtin_ctz(bits) & 15])
            ready += " r = std::max(r, v.m_viReady[" + std::to_string(__builtin_ctz(bits)) + "]);";
    // ACC results forward in one cycle, so they can never stall a later pair: no ACC readiness
    // checks or updates (the generic core's m_accReady then only holds older, already-ready cycles).
    static_assert(VU1Fast::kAccForwardLatency <= 1u, "ACC readiness is not tracked in generated code");
    if (d.lowerUsage.pipeline == VU1Fast::PipelineFdiv || d.lowerUsage.waitQ)
        ready += " if (v.m_fdiv.valid) r = std::max(r, v.m_fdiv.readyCycle);";
    if (d.lowerUsage.pipeline == VU1Fast::PipelineEfu)
        ready += " r = std::max(r, v.m_efuResourceReady);";
    if (d.lowerUsage.waitP)
        ready += " for (const auto &e : v.m_efu) if (e.valid) r = std::max(r, e.readyCycle);";
    if (d.lowerUsage.pipeline == VU1Fast::PipelineXgkick)
        ready += " if (v.m_xgkick.active) r = std::max(r, v.m_cycle + 1u);";
    if (!ready.empty())
        emit(o, "    { uint64_t r = v.m_cycle;%s if (r > v.m_cycle && !v.stallPair(0x%Xu, budgetEnd)) return kGenBudget; }\n", ready.c_str(), A);
    const uint8_t writtenVi = d.viWriteMask ? static_cast<uint8_t>(__builtin_ctz(d.viWriteMask)) : 0u;
    if (writtenVi && d.lowerUsage.delaysNextBranchRead)
        emit(o, "    const int32_t oldVi = v.m_state.vi[%u];\n", writtenVi);
    const bool upperNop = isUpperNop(d.upper);
    const bool lowerNop = d.lower == 0u || d.lower == 0x8000033Cu;
    if (d.iBit)
    {
        if (!upperNop) emit(o, "    v.execUpperInlDaz(0x%08Xu);\n", d.upper);
        emit(o, "    { uint32_t b = 0x%08Xu; float f; std::memcpy(&f, &b, 4); v.m_state.i = v.normalizeOperand(f); }\n", d.lower);
    }
    else if (d.upperVfShadowReg)
    {
        emit(o, "    { float oldVf[4], upperVf[4]; std::memcpy(oldVf, v.m_state.vf[%u], 16); v.execUpperInlDaz(0x%08Xu); std::memcpy(upperVf, v.m_state.vf[%u], 16); std::memcpy(v.m_state.vf[%u], oldVf, 16);\n",
                     d.upperVfShadowReg, d.upper, d.upperVfShadowReg, d.upperVfShadowReg);
        emit(o, "      v.execLowerInl(0x%08Xu, v.m_activeVuData, v.m_activeVuDataSize, *v.m_activeGs, v.m_activeMemory, 0x%08Xu); std::memcpy(v.m_state.vf[%u], upperVf, 16); }\n",
                     d.lower, d.upper, d.upperVfShadowReg);
    }
    else
    {
        if (!upperNop && macDead) emit(o, "    v.m_skipMac = true; v.execUpperInlDaz(0x%08Xu); v.m_skipMac = false;\n", d.upper);
        else if (!upperNop) emit(o, "    v.execUpperInlDaz(0x%08Xu);\n", d.upper);
        if (!lowerNop) emit(o, "    v.execLowerInl(0x%08Xu, v.m_activeVuData, v.m_activeVuDataSize, *v.m_activeGs, v.m_activeMemory, 0x%08Xu);\n", d.lower, d.upper);
    }
    emit(o, "    v.m_viBranchBackupValid = false;\n");
    if (writtenVi)
        emit(o, "    v.m_state.vi[%u] = static_cast<int16_t>(v.m_state.vi[%u]);\n", writtenVi, writtenVi);
    for (uint32_t k = 0; k < d.writeCount; ++k)
    {
        if (d.writeLanes[k] == 0xFu)
            emit(o, "    { const uint64_t t = v.m_cycle + %uu; auto &w = v.m_vfReady[%u]; w[0] = w[1] = w[2] = w[3] = t; }\n", d.writeLatency[k], d.writeReg[k]);
        else
            for (int l = 0; l < 4; ++l)
                if (d.writeLanes[k] & (8u >> l))
                    emit(o, "    v.m_vfReady[%u][%d] = v.m_cycle + %uu;\n", d.writeReg[k], l, d.writeLatency[k]);
    }
    for (uint32_t bits = d.viWriteMask; bits; bits &= bits - 1)
        emit(o, "    v.m_viReady[%d] = v.m_cycle + %uu;\n", __builtin_ctz(bits), d.viWriteLatency);
    for (uint32_t k = 0; k < d.writeCount; ++k)
        for (int l = 0; l < 4; ++l)
            if (d.writeLanes[k] & (8u >> l))
                timing.vf[d.writeReg[k] & 31u][l] = pairIndex + d.writeLatency[k];
    for (uint32_t bits = d.viWriteMask; bits; bits &= bits - 1)
        timing.vi[__builtin_ctz(bits) & 15] = pairIndex + d.viWriteLatency;
    for (int l = 0; l < 4; ++l)
        if (d.accWriteLanes & (8u >> l))
            timing.acc[l] = pairIndex + static_cast<int>(VU1Fast::kAccForwardLatency);
    if (writtenVi && d.lowerUsage.delaysNextBranchRead)
        emit(o, "    v.recordViWriteForBranch(%u, oldVi);\n", writtenVi);
    const bool isBranch = d.lowerUsage.pipeline == VU1Fast::PipelineBranch;
    if (plainEnd)
    {
        // s3recomp: a program that never writes VF0/VI0 cannot dirty them, so the per-pair reset is skipped.
        emit(o, g_programWritesZeroRegs ? "    v.pairEndPlain(0x%Xu);\n  }\n" : "    v.pairEndPlainNoZero(0x%Xu);\n  }\n", next);
        return;
    }
    emit(o, "    if (v.pairEnd<%s, %s, %s, %s>(0x%Xu) == kGenEnded) return kGenEnded;\n", d.eBit ? "true" : "false", d.dBit ? "true" : "false",
                 d.tBit ? "true" : "false", isBranch ? "true" : "false", next);
    emit(o, "    if (v.m_state.pc != 0x%Xu) return kGenNext;\n  }\n", next);
}


// s3recomp: lean blocks. When a block is entered with nothing pending from earlier code that could
// make any of its pairs stall (checked once at entry), every stall inside it is known when the code is
// generated: they come only from the block's own writes. Such a block then needs no per-pair budget,
// stop, readiness or pc/cycle bookkeeping. Each pair still advances m_cycle and commits what is due,
// so flags, Q and stores happen at exactly the same cycles as in the exact block. The entry check
// falls back to the exact block (emitted as <name>_x) whenever its assumptions do not hold.
static bool g_leanBlocks = true;
static int g_leanCount = 0, g_blockCount = 0;

static void emitPairExec(std::string &o, const VU1Fast::DecodedInstructionPair &d, bool macDead)
{
    const bool upperNop = isUpperNop(d.upper);
    const bool lowerNop = d.lower == 0u || d.lower == 0x8000033Cu;
    if (d.iBit)
    {
        if (!upperNop) emit(o, macDead ? "    v.m_skipMac = true; v.execUpperInlDaz(0x%08Xu); v.m_skipMac = false;\n" : "    v.execUpperInlDaz(0x%08Xu);\n", d.upper);
        emit(o, "    { uint32_t b = 0x%08Xu; float f; std::memcpy(&f, &b, 4); v.m_state.i = v.normalizeOperand(f); }\n", d.lower);
    }
    else if (d.upperVfShadowReg)
    {
        emit(o, "    { float oldVf[4], upperVf[4]; std::memcpy(oldVf, v.m_state.vf[%u], 16); v.execUpperInlDaz(0x%08Xu); std::memcpy(upperVf, v.m_state.vf[%u], 16); std::memcpy(v.m_state.vf[%u], oldVf, 16);\n",
             d.upperVfShadowReg, d.upper, d.upperVfShadowReg, d.upperVfShadowReg);
        emit(o, "      v.execLowerInl(0x%08Xu, v.m_activeVuData, 0x4000u, *v.m_activeGs, v.m_activeMemory, 0x%08Xu); std::memcpy(v.m_state.vf[%u], upperVf, 16); }\n",
             d.lower, d.upper, d.upperVfShadowReg);
    }
    else
    {
        if (!upperNop && macDead) emit(o, "    v.m_skipMac = true; v.execUpperInlDaz(0x%08Xu); v.m_skipMac = false;\n", d.upper);
        else if (!upperNop) emit(o, "    v.execUpperInlDaz(0x%08Xu);\n", d.upper);
        if (!lowerNop) emit(o, "    v.execLowerInl(0x%08Xu, v.m_activeVuData, 0x4000u, *v.m_activeGs, v.m_activeMemory, 0x%08Xu);\n", d.lower, d.upper);
    }
}

// s3recomp: turbo segments (see ps2_vu1_turbo_inl.h). Runs of plain pairs inside a lean block whose
// flag results are scheduled here, at generation time, instead of through the flag pipeline.
static bool g_turbo = true;
static bool g_programEager = false; // per program: no status flag reads or FSSET
static int g_turboSegments = 0, g_turboPairs = 0;

static bool lowerIsFdivIssue(const VU1Fast::DecodedInstructionPair &x)
{
    if (x.iBit || (x.lower >> 25) != 0x40u || (x.lower & 0x3Cu) != 0x3Cu)
        return false;
    const uint32_t sub = (x.lower & 3u) | ((x.lower >> 4) & 0x7Cu);
    return sub == 0x38u || sub == 0x39u || sub == 0x3Au; // DIV, SQRT, RSQRT
}
static bool lowerIsMfp(const VU1Fast::DecodedInstructionPair &x)
{
    if (x.iBit || (x.lower >> 25) != 0x40u || (x.lower & 0x3Cu) != 0x3Cu)
        return false;
    return ((x.lower & 3u) | ((x.lower >> 4) & 0x7Cu)) == 0x64u;
}
static bool upperIsTurboFmac(uint32_t u) { return !isUpperNop(u) && vu1turbo::isTurboFmac(u) && ((u >> 21) & 0xFu) != 0u; }
static bool upperIsClip(uint32_t u) { return !isUpperNop(u) && vu1turbo::isClip(u); }

static bool pairTurboEligible(const VU1Fast::DecodedInstructionPair &x)
{
    if (x.upperUsage.reserved || x.lowerUsage.reserved)
        return false;
    const uint32_t u = x.upper;
    if (!isUpperNop(u))
    {
        const auto f = vu1turbo::fmacInfo(u);
        const bool fmacNoDest = f.kind != vu1turbo::kNone && ((u >> 21) & 0xFu) == 0u;
        if (!(fmacNoDest || vu1turbo::isTurboFmac(u) || vu1turbo::isClip(u) || vu1turbo::isMiscNoFlags(u)))
            return false;
    }
    if (x.iBit)
        return true;
    const uint32_t op = x.lower >> 25;
    if (op == 0x11u || op == 0x14u || op == 0x15u || op == 0x16u || op == 0x17u) // FCSET, status flag ops
        return false;
    if ((op == 0x18u || op == 0x1Au || op == 0x1Bu || op == 0x1Cu) && ((x.lower >> 16) & 0xFu) == 0u)
        return false; // a flag read into VI0 does not commit the flags first (kept on the exact path)
    if (x.lowerUsage.pipeline == VU1Fast::PipelineBranch || x.lowerUsage.pipeline == VU1Fast::PipelineEfu ||
        x.lowerUsage.pipeline == VU1Fast::PipelineXgkick || x.lowerUsage.waitP)
        return false;
    return true;
}

// Emits pairs s..e-1 of a lean block as one turbo segment. issue[] are the static issue cycles
// relative to the block's entry cycle b.
static void emitTurboSegment(std::string &o, const std::vector<VU1Fast::DecodedInstructionPair> &d, uint32_t start,
                             uint32_t s, uint32_t e, const std::vector<int> &issue, const std::vector<bool> &deadAll,
                             uint32_t progAddr, const std::vector<std::vector<std::pair<int, int>>> &storeVf,
                             const std::vector<std::vector<int>> &storeVi, bool clearBackup)
{
    ++g_turboSegments;
    g_turboPairs += int(e - s);
    const int E = issue[e - 1] + 1; // the cycle after the segment's last pair, relative to b
    std::vector<int> macWriters, clipWriters;
    int lastFmac = -1;
    bool anyFmac = false;
    for (uint32_t k = s; k < e; ++k)
    {
        if (upperIsTurboFmac(d[k].upper)) { macWriters.push_back(int(k)); lastFmac = int(k); anyFmac = true; }
        if (upperIsClip(d[k].upper)) clipWriters.push_back(int(k));
    }
    auto deadAt = [&](int k) { return bool(deadAll[(start + uint32_t(k) * 8u - progAddr) / 8u]); };
    // which results are needed: read inside the segment, or published at its end
    std::set<int> needMac, needClip;
    auto visibleWriter = [&](const std::vector<int> &w, int k) {
        int best = -1;
        for (int j : w)
            if (j < k && issue[j] + 4 <= issue[k]) best = j;
        return best;
    };
    for (uint32_t k = s; k < e; ++k)
    {
        const auto &x = d[k];
        if (x.iBit) continue;
        const uint32_t op = x.lower >> 25;
        if (op == 0x18u || op == 0x1Au || op == 0x1Bu)
        { const int j = visibleWriter(macWriters, int(k)); if (j >= 0) needMac.insert(j); }
        if (op == 0x10u || op == 0x12u || op == 0x13u || op == 0x1Cu)
        { const int j = visibleWriter(clipWriters, int(k)); if (j >= 0) needClip.insert(j); }
    }
    int directMac = -1, directClip = -1;
    std::vector<int> late; // pairs whose results are still in flight at E, in issue order
    for (int j : macWriters)
        if (issue[j] + 4 <= E) directMac = j;
    for (int j : clipWriters)
        if (issue[j] + 4 <= E) directClip = j;
    if (directMac >= 0 && deadAt(directMac)) directMac = -1;
    if (directMac >= 0) needMac.insert(directMac);
    if (directClip >= 0) needClip.insert(directClip);
    for (uint32_t k = s; k < e; ++k)
    {
        const bool m = upperIsTurboFmac(d[k].upper) && issue[k] + 4 > E && !deadAt(int(k));
        const bool c = upperIsClip(d[k].upper) && issue[k] + 4 > E;
        if (m) needMac.insert(int(k));
        if (c) needClip.insert(int(k));
        if (m || c) late.push_back(int(k));
    }

    emit(o, "  { // turbo 0x%04X-0x%04X\n", start + s * 8u, start + (e - 1u) * 8u);
    if (anyFmac)
        emit(o, "    vu1turbo::Sticky tst{_mm_setzero_ps(), _mm_setzero_ps(), 0u};\n    uint32_t tcur = v.m_state.status & 0xFu, tdummy = 0u;\n");
    if (!clipWriters.empty())
        emit(o, "    uint32_t twc = v.m_workingClip;\n");
    if (clearBackup)
        emit(o, "    v.m_viBranchBackupValid = false;\n");
    for (uint32_t k = s; k < e; ++k)
    {
        const auto &x = d[k];
        const uint32_t A = start + k * 8u;
        const int c = issue[k];
        emit(o, "    // 0x%04X\n", A);
        const auto f = vu1turbo::fmacInfo(x.upper);
        const bool upperReadsQ = !isUpperNop(x.upper) && upperIsTurboFmac(x.upper) && f.src == vu1turbo::kQ;
        if (upperReadsQ || lowerIsFdivIssue(x) || lowerIsMfp(x))
            emit(o, "    v.m_cycle = b + %du; v.commitIfDue();\n", c);
        const uint8_t writtenVi = x.viWriteMask ? static_cast<uint8_t>(__builtin_ctz(x.viWriteMask)) : 0u;
        const bool records = writtenVi && x.lowerUsage.delaysNextBranchRead && k + 1u == e;
        if (records)
            emit(o, "    const int32_t oldVi = v.m_state.vi[%u];\n", writtenVi);
        // upper
        std::string up;
        if (upperIsTurboFmac(x.upper))
        {
            char mv[32] = "tdummy";
            if (needMac.count(int(k)))
            {
                emit(o, "    uint32_t tmac%u = 0u;\n", k);
                std::snprintf(mv, sizeof mv, "tmac%u", k);
            }
            char buf[200];
            std::snprintf(buf, sizeof buf, "vu1TurboFmac<0x%08Xu, %s, %s>(v, tst, %s, %s);", x.upper, needMac.count(int(k)) ? "true" : "false",
                          int(k) == lastFmac ? "true" : "false", mv, int(k) == lastFmac ? "tcur" : "tdummy");
            up = buf;
        }
        else if (upperIsClip(x.upper))
        {
            char buf[160];
            std::snprintf(buf, sizeof buf, "twc = ((twc << 6) | vu1TurboClipBits<0x%08Xu>(v)) & 0xFFFFFFu;", x.upper);
            up = buf;
            if (needClip.count(int(k)))
                up += " const uint32_t tclip" + std::to_string(k) + " = twc;";
        }
        else if (!isUpperNop(x.upper) && vu1turbo::isMiscNoFlags(x.upper) && ((x.upper >> 21) & 0xFu) != 0u)
        {
            char buf[64];
            std::snprintf(buf, sizeof buf, "v.execUpperInlDaz(0x%08Xu);", x.upper);
            up = buf;
        }
        // lower
        std::string lo;
        const bool lowerNop = x.iBit || x.lower == 0u || x.lower == 0x8000033Cu;
        if (!lowerNop)
        {
            const uint32_t op = x.lower >> 25;
            const unsigned it = (x.lower >> 16) & 0xFu, is = (x.lower >> 11) & 0xFu;
            auto macSrc = [&]() {
                const int j = visibleWriter(macWriters, int(k));
                return j >= 0 ? "tmac" + std::to_string(j) : "(v.commitFlags(b + " + std::to_string(c) + "u), v.m_state.mac)";
            };
            auto clipSrc = [&]() {
                const int j = visibleWriter(clipWriters, int(k));
                return j >= 0 ? "tclip" + std::to_string(j) : "(v.commitFlags(b + " + std::to_string(c) + "u), v.m_state.clip)";
            };
            char buf[256];
            const uint32_t imm24 = x.lower & 0xFFFFFFu;
            if (op == 0x18u)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[%u] = ((%s & 0xFFFFu) == (uint32_t)(uint16_t)v.m_state.vi[%u]) ? 1 : 0;", it, macSrc().c_str(), is);
            else if (op == 0x1Au)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[%u] = (int32_t)(%s & (uint32_t)(uint16_t)v.m_state.vi[%u]);", it, macSrc().c_str(), is);
            else if (op == 0x1Bu)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[%u] = (int32_t)(%s | (uint32_t)(uint16_t)v.m_state.vi[%u]);", it, macSrc().c_str(), is);
            else if (op == 0x10u)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[1] = ((%s & 0xFFFFFFu) == 0x%Xu) ? 1 : 0;", clipSrc().c_str(), imm24);
            else if (op == 0x12u)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[1] = ((%s & 0x%Xu) != 0u) ? 1 : 0;", clipSrc().c_str(), imm24);
            else if (op == 0x13u)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[1] = ((%s | 0x%Xu) == 0xFFFFFFu) ? 1 : 0;", clipSrc().c_str(), imm24);
            else if (op == 0x1Cu)
                std::snprintf(buf, sizeof buf, "v.m_state.vi[%u] = static_cast<int32_t>(%s & 0x0FFFu);", it, clipSrc().c_str());
            else
            {
                if (lowerIsFdivIssue(x) || lowerIsMfp(x))
                    std::snprintf(buf, sizeof buf, "v.execLowerInl(0x%08Xu, v.m_activeVuData, 0x4000u, *v.m_activeGs, v.m_activeMemory, 0x%08Xu);", x.lower, x.upper);
                else
                    std::snprintf(buf, sizeof buf, "v.execLowerInl(0x%08Xu, v.m_activeVuData, 0x4000u, *v.m_activeGs, v.m_activeMemory, 0x%08Xu);", x.lower, x.upper);
            }
            lo = buf;
        }
        if (x.iBit)
        {
            if (!up.empty()) emit(o, "    %s\n", up.c_str());
            emit(o, "    { uint32_t ib = 0x%08Xu; float f; std::memcpy(&f, &ib, 4); v.m_state.i = v.normalizeOperand(f); }\n", x.lower);
        }
        else if (x.upperVfShadowReg && !up.empty())
        {
            const unsigned R = x.upperVfShadowReg;
            emit(o, "    { float oldVf[4], upperVf[4]; std::memcpy(oldVf, v.m_state.vf[%u], 16);\n      %s\n      std::memcpy(upperVf, v.m_state.vf[%u], 16); std::memcpy(v.m_state.vf[%u], oldVf, 16);\n", R, up.c_str(), R, R);
            if (!lo.empty()) emit(o, "      %s\n", lo.c_str());
            emit(o, "      std::memcpy(v.m_state.vf[%u], upperVf, 16); }\n", R);
        }
        else
        {
            if (!up.empty()) emit(o, "    %s\n", up.c_str());
            if (!lo.empty()) emit(o, "    %s\n", lo.c_str());
        }
        if (writtenVi)
            emit(o, "    v.m_state.vi[%u] = static_cast<int16_t>(v.m_state.vi[%u]);\n", writtenVi, writtenVi);
        for (auto &[r, l] : storeVf[k])
        {
            int lat = 0;
            for (uint32_t w = 0; w < x.writeCount; ++w)
                if ((x.writeReg[w] & 31) == r && (x.writeLanes[w] & (8u >> l))) lat = x.writeLatency[w];
            emit(o, "    v.m_vfReady[%d][%d] = b + %du;\n", r, l, c + lat);
        }
        for (int r : storeVi[k])
            emit(o, "    v.m_viReady[%d] = b + %du;\n", r, c + x.viWriteLatency);
        if (records)
            emit(o, "    v.recordViWriteForBranch(%u, oldVi);\n", writtenVi);
        if (g_programWritesZeroRegs)
            emit(o, "    v.resetZeroRegs();\n");
    }
    // End of the segment: the core's cycle, then the flag results as the pipeline holds them now.
    emit(o, "    v.m_cycle = b + %du;\n", E);
    if (directMac >= 0 || directClip >= 0)
        emit(o, "    v.commitFlags(b + %du);\n", E);
    if (directMac >= 0)
        emit(o, "    v.m_state.mac = tmac%d;\n", directMac);
    if (directClip >= 0)
        emit(o, "    v.m_state.clip = tclip%d;\n", directClip);
    for (int j : late)
    {
        const bool m = upperIsTurboFmac(d[j].upper);
        emit(o, "    v.turboPushFlag(b + %du, %s, %s, %s, %s);\n", issue[j], m ? "true" : "false",
             m ? ("tmac" + std::to_string(j)).c_str() : "0u", m ? "false" : "true", m ? "0u" : ("tclip" + std::to_string(j)).c_str());
    }
    if (anyFmac)
        emit(o, "    v.m_state.status = (v.m_state.status & 0xFF0u) | tcur | (vu1TurboStickyBits(tst) << 6);\n");
    if (!clipWriters.empty())
        emit(o, "    v.m_workingClip = twc;\n");
    emit(o, "    v.commitIfDue();\n  }\n");
}

// Returns false (emitting nothing) when the block cannot be lean.
static bool emitLeanBlock(std::string &o, VU1Fast &dec, const Program &p, uint32_t start, uint32_t stop,
                          const std::vector<bool> &deadAll, const std::string &name, const std::string &exactName)
{
    const uint32_t n = (stop - start) / 8u;
    std::vector<VU1Fast::DecodedInstructionPair> d(n);
    for (uint32_t k = 0; k < n; ++k)
    {
        d[k] = dec.decodeInstructionPair(p.code.data() - p.addr, start + k * 8u);
        const auto &x = d[k];
        if (x.upperUsage.reserved || x.lowerUsage.reserved)
            return false;
        if (!x.iBit && (x.lowerUsage.pipeline == VU1Fast::PipelineEfu || x.lowerUsage.waitP ||
                        x.lowerUsage.pipeline == VU1Fast::PipelineXgkick))
            return false;
        if (x.upperUsage.pipeline == VU1Fast::PipelineEfu || x.upperUsage.pipeline == VU1Fast::PipelineXgkick)
            return false;
    }
    // Static issue cycles, relative to the block's entry cycle, mirroring calculatePairReadyCycle.
    std::vector<int> issue(n);
    int lastVf[32][4], latVf[32][4], lastVi[16], latVi[16];
    for (auto &r : lastVf) for (int &x : r) x = -1;
    for (int &x : lastVi) x = -1;
    int lastFdiv = -1, latFdiv = 0;
    std::map<std::pair<int, int>, int> vfGuard; // (reg, lane) -> earliest issue that reads it before any block write
    std::map<int, int> viGuard;
    int fdivGuard = -1;
    for (uint32_t k = 0; k < n; ++k)
    {
        const auto &x = d[k];
        int ready = k == 0 ? 0 : issue[k - 1] + 1;
        std::vector<std::pair<int, int>> preVf;
        std::vector<int> preVi;
        for (uint32_t r = 0; r < x.readCount; ++r)
            for (int l = 0; l < 4; ++l)
                if (x.readLanes[r] & (8u >> l))
                {
                    const int reg = x.readReg[r] & 31;
                    if (lastVf[reg][l] >= 0) ready = std::max(ready, issue[lastVf[reg][l]] + latVf[reg][l]);
                    else preVf.push_back({reg, l});
                }
        for (uint32_t bits = x.viReadMask; bits; bits &= bits - 1)
        {
            const int r = __builtin_ctz(bits) & 15;
            if (lastVi[r] >= 0) ready = std::max(ready, issue[lastVi[r]] + latVi[r]);
            else preVi.push_back(r);
        }
        const bool usesFdiv = !x.iBit && (x.lowerUsage.pipeline == VU1Fast::PipelineFdiv || x.lowerUsage.waitQ);
        bool preFdiv = false;
        if (usesFdiv)
        {
            if (lastFdiv >= 0) ready = std::max(ready, issue[lastFdiv] + latFdiv);
            else preFdiv = true;
        }
        issue[k] = ready;
        // Readiness of earlier code: a write issued before the block (at entry - 1 at the latest) is
        // ready by entry - 1 + its latency, so only early reads need checking.
        for (auto &pr : preVf)
            if (issue[k] < g_maxLatency.vf && !vfGuard.count(pr)) vfGuard[pr] = issue[k];
        for (int r : preVi)
            if (issue[k] < g_maxLatency.vi && !viGuard.count(r)) viGuard[r] = issue[k];
        if (preFdiv && fdivGuard < 0) fdivGuard = issue[k];
        // Writes (after this pair's reads).
        for (uint32_t w = 0; w < x.writeCount; ++w)
            for (int l = 0; l < 4; ++l)
                if (x.writeLanes[w] & (8u >> l))
                {
                    lastVf[x.writeReg[w] & 31][l] = int(k);
                    latVf[x.writeReg[w] & 31][l] = x.writeLatency[w];
                }
        for (uint32_t bits = x.viWriteMask; bits; bits &= bits - 1)
        {
            lastVi[__builtin_ctz(bits) & 15] = int(k);
            latVi[__builtin_ctz(bits) & 15] = x.viWriteLatency;
        }
        if (!x.iBit && x.lowerUsage.pipeline == VU1Fast::PipelineFdiv && !x.lowerUsage.waitQ)
        {
            lastFdiv = int(k);
            latFdiv = x.lowerUsage.latency;
        }
    }
    const int total = issue[n - 1] + 1;
    // Exact block first (same body as before, under another name), then the lean entry.
    std::string cond;
    for (auto &[rl, at] : vfGuard)
        cond += " && v.m_vfReady[" + std::to_string(rl.first) + "][" + std::to_string(rl.second) + "] <= b + " + std::to_string(at) + "u";
    for (auto &[r, at] : viGuard)
        cond += " && v.m_viReady[" + std::to_string(r) + "] <= b + " + std::to_string(at) + "u";
    if (fdivGuard >= 0)
        cond += " && (!v.m_fdiv.valid || v.m_fdiv.readyCycle <= b + " + std::to_string(fdivGuard) + "u)";
    // s3recomp: turbo segments: maximal runs of at least two eligible plain pairs
    std::vector<uint32_t> segEnd(n, 0u); // segEnd[k] > k when a segment starts at k
    bool anySegment = false;
    if (g_turbo && g_programEager)
    {
        std::vector<bool> ok(n, false);
        for (uint32_t k = 0; k < n; ++k)
        {
            const uint32_t A = start + k * 8u;
            const bool prevCtrl = k > 0 ? pairIsControl(d[k - 1])
                                        : prevPairMayBeControl(dec, p, A);
            ok[k] = !pairIsControl(d[k]) && !prevCtrl && pairTurboEligible(d[k]);
        }
        for (uint32_t k = 0; k < n;)
        {
            if (!ok[k]) { ++k; continue; }
            uint32_t e = k;
            while (e < n && ok[e]) ++e;
            if (e - k >= 2u) { segEnd[k] = e; anySegment = true; }
            k = e;
        }
    }
    if (anySegment)
        cond += " && v.m_eagerStatus";
    // s3recomp: a stall of the block's first pair (on results of earlier code) is taken here, as
    // stallPair would take it, and the block then runs from that cycle. Later pairs' conditions are
    // still checked after it.
    std::string first;
    for (auto &[rl, at] : vfGuard)
        if (at == 0) first += " r = std::max(r, v.m_vfReady[" + std::to_string(rl.first) + "][" + std::to_string(rl.second) + "]);";
    for (auto &[r, at] : viGuard)
        if (at == 0) first += " r = std::max(r, v.m_viReady[" + std::to_string(r) + "]);";
    if (fdivGuard == 0)
        first += " if (v.m_fdiv.valid) r = std::max(r, v.m_fdiv.readyCycle);";
    emit(o, "int %s(VU1Fast &v, uint64_t budgetEnd)\n{\n", name.c_str());
    if (!first.empty())
    {
        emit(o, "  { uint64_t r = v.m_cycle;%s\n", first.c_str());
        emit(o, "    if (r > v.m_cycle) { if (r + %du > budgetEnd || v.m_stopRequested || v.m_xgkick.active) return %s(v, budgetEnd); v.advanceTo(r); } }\n", total, exactName.c_str());
    }
    emit(o, "  const uint64_t b = v.m_cycle;\n");
    emit(o, "  if (!(b + %du <= budgetEnd && !v.m_stopRequested && !v.m_xgkick.active%s))\n    return %s(v, budgetEnd);\n", total, cond.c_str(), exactName.c_str());
    emit(o, "  v.commitIfDue();\n");
    // last writer of each (reg, lane) / vi in the block: its readiness is stored for later blocks
    std::vector<std::vector<std::pair<int, int>>> storeVf(n);
    std::vector<std::vector<int>> storeVi(n);
    for (int r = 0; r < 32; ++r)
        for (int l = 0; l < 4; ++l)
            if (lastVf[r][l] >= 0) storeVf[lastVf[r][l]].push_back({r, l});
    for (int r = 0; r < 16; ++r)
        if (lastVi[r] >= 0) storeVi[lastVi[r]].push_back(r);
    bool prevRecorded = false;
    bool lastPlain = false;
    for (uint32_t k = 0; k < n; ++k)
    {
        if (segEnd[k] > k)
        {
            const uint32_t e = segEnd[k];
            emitTurboSegment(o, d, start, k, e, issue, deadAll, p.addr, storeVf, storeVi, k == 0 || prevRecorded);
            const auto &lastPair = d[e - 1];
            prevRecorded = lastPair.viWriteMask && lastPair.lowerUsage.delaysNextBranchRead;
            lastPlain = true;
            k = e - 1;
            continue;
        }
        const auto &x = d[k];
        const uint32_t A = start + k * 8u;
        const uint32_t next = (A + 8u) & 0x3FFFu;
        const bool ctrl = pairIsControl(x);
        // like emitPair: the pair before a block's first pair may be its branch (blocks are capped in
        // length, so a delay slot can start a block)
        const bool prevCtrl = k > 0 ? pairIsControl(d[k - 1])
                                    : prevPairMayBeControl(dec, p, A);
        emit(o, "  { // 0x%04X%s\n", A, ctrl ? " control" : "");
        if (k > 0 && issue[k] > issue[k - 1] + 1)
            emit(o, "    v.advanceTo(b + %du);\n", issue[k]);
        if (ctrl && !prevCtrl)
            emit(o, "    v.m_state.pc = 0x%Xu;\n", A);
        const uint8_t writtenVi = x.viWriteMask ? static_cast<uint8_t>(__builtin_ctz(x.viWriteMask)) : 0u;
        const bool records = writtenVi && x.lowerUsage.delaysNextBranchRead;
        if (records)
            emit(o, "    const int32_t oldVi = v.m_state.vi[%u];\n", writtenVi);
        emitPairExec(o, x, deadAll[(A - p.addr) / 8u]);
        if (k == 0 || prevRecorded)
            emit(o, "    v.m_viBranchBackupValid = false;\n");
        if (writtenVi)
            emit(o, "    v.m_state.vi[%u] = static_cast<int16_t>(v.m_state.vi[%u]);\n", writtenVi, writtenVi);
        for (auto &[r, l] : storeVf[k])
        {
            int lat = 0;
            for (uint32_t w = 0; w < x.writeCount; ++w)
                if ((x.writeReg[w] & 31) == r && (x.writeLanes[w] & (8u >> l))) lat = x.writeLatency[w];
            emit(o, "    v.m_vfReady[%d][%d] = b + %du;\n", r, l, issue[k] + lat);
        }
        for (int r : storeVi[k])
            emit(o, "    v.m_viReady[%d] = b + %du;\n", r, issue[k] + x.viWriteLatency);
        if (records)
            emit(o, "    v.recordViWriteForBranch(%u, oldVi);\n", writtenVi);
        prevRecorded = records;
        if (!ctrl && !prevCtrl)
        {
            emit(o, g_programWritesZeroRegs ? "    v.resetZeroRegs();\n    ++v.m_cycle;\n    v.commitIfDue();\n  }\n" : "    ++v.m_cycle;\n    v.commitIfDue();\n  }\n");
            lastPlain = true;
        }
        else
        {
            const bool isBranch = !x.iBit && x.lowerUsage.pipeline == VU1Fast::PipelineBranch;
            emit(o, "    if (v.pairEnd<%s, %s, %s, %s>(0x%Xu) == kGenEnded) return kGenEnded;\n", x.eBit ? "true" : "false", x.dBit ? "true" : "false",
                 x.tBit ? "true" : "false", isBranch ? "true" : "false", next);
            emit(o, "    if (v.m_state.pc != 0x%Xu) return kGenNext;\n  }\n", next);
            lastPlain = false;
        }
    }
    if (lastPlain)
        emit(o, "  v.m_state.pc = 0x%Xu;\n  v.m_state.cycles = v.m_cycle;\n", stop & 0x3FFFu);
    emit(o, "  return kGenNext;\n}\n\n");
    return true;
}

// Addresses where generated blocks start: program start, branch targets, and the pairs after a
// branch or E-bit delay slot.
static std::set<uint32_t> findLeaders(VU1Fast &dec, const Program &p)
{
    std::set<uint32_t> leaders{p.addr};
    const uint32_t end = p.addr + uint32_t(p.code.size());
    for (uint32_t A = p.addr; A < end; A += 8)
    {
        const auto d = dec.decodeInstructionPair(p.code.data() - p.addr, A);
        if (d.eBit || d.dBit || d.tBit)
            leaders.insert(A + 16u);
        if (d.iBit || d.lowerUsage.pipeline != VU1Fast::PipelineBranch)
            continue;
        leaders.insert(A + 16u);
        const uint32_t op = d.lower >> 25;
        if (op == 0x20 || op == 0x21 || (op >= 0x28 && op <= 0x2F))
        {
            const int32_t imm = static_cast<int32_t>(d.lower << 21) >> 21;
            leaders.insert((A + 8u + uint32_t(imm * 8)) & 0x3FFFu);
        }
    }
    std::set<uint32_t> in;
    for (uint32_t l : leaders)
        if (l >= p.addr && l < end)
            in.insert(l);
    // Cap straight-line blocks so no generated function gets too large for the compiler.
    constexpr uint32_t kMaxBlockPairs = 48u;
    std::vector<uint32_t> extra;
    for (auto it = in.begin(); it != in.end(); ++it)
    {
        const auto nx = std::next(it);
        const uint32_t stop = nx == in.end() ? end : *nx;
        for (uint32_t a = *it + kMaxBlockPairs * 8u; a < stop; a += kMaxBlockPairs * 8u)
            extra.push_back(a);
    }
    in.insert(extra.begin(), extra.end());
    return in;
}

int main(int argc, char **argv)
{
    std::string outDir, tag = "untagged";
    std::vector<std::string> inputs;
    for (int a = 1; a < argc; ++a)
    {
        const std::string arg = argv[a];
        if (arg == "--out" && a + 1 < argc)
            outDir = argv[++a];
        else if (arg == "--no-lean")
            g_leanBlocks = false;
        else if (arg == "--no-turbo")
            g_turbo = false;
        else if (arg == "--tag" && a + 1 < argc)
            tag = argv[++a];
        else
            inputs.push_back(arg);
    }
    if (outDir.empty() || inputs.empty())
    {
        std::fprintf(stderr, "usage: ps2_vu1gen --out <dir> [--tag <text>] <file>...\n"
                             "Finds VU1 microprograms (VIF MPG uploads) in the files and writes them as C++.\n");
        return 2;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outDir, ec);
    VU1State st{};
    VU1Fast dec(st, VU1Fast::Unit::VU1);
    g_maxLatency = findMaxLatency(dec);
    std::vector<Program> all;
    // Files are scanned in windows so large disc archives need little memory.
    constexpr size_t kWindow = 32u << 20, kOverlap = 256u << 10;
    for (const std::string &input : inputs)
    {
        std::ifstream f(input, std::ios::binary);
        if (!f)
        {
            std::fprintf(stderr, "ps2_vu1gen: cannot read %s\n", input.c_str());
            continue;
        }
        const std::string name = fs::path(input).filename().string();
        std::vector<uint8_t> img;
        for (;;)
        {
            const size_t keep = img.size();
            img.resize(keep + kWindow);
            f.read(reinterpret_cast<char *>(img.data() + keep), static_cast<std::streamsize>(kWindow));
            img.resize(keep + static_cast<size_t>(f.gcount()));
            const bool last = !f;
            const size_t limit = last ? img.size() : img.size() - kOverlap;
            for (auto &p : findPrograms(img, limit, name, dec))
                all.push_back(std::move(p));
            if (last)
                break;
            img.erase(img.begin(), img.end() - static_cast<std::ptrdiff_t>(kOverlap));
        }
    }
    // drop duplicates
    std::map<uint64_t, Program> uniq;
    for (auto &p : all) uniq.emplace(vu1RegionHash(p.code.data(), uint32_t(p.code.size())) ^ (uint64_t(p.addr) << 48), p);
    // One source file per program (they are large) plus a registration file.
    const fs::path dir(outDir);
    std::vector<std::string> written;
    const std::string header = "// Generated by ps2_vu1gen (" + tag + ") - do not edit.\n";
    std::vector<std::string> table;
    int idx = 0;
    int eagerPrograms = 0;
    size_t deadMacPairs = 0;
    for (auto &[key, p] : uniq)
    {
        const uint64_t h = vu1RegionHash(p.code.data(), uint32_t(p.code.size()));
        const std::set<uint32_t> leaders = findLeaders(dec, p);
        const std::vector<bool> deadAll = deadMacProgram(dec, p, leaders);
        g_programEager = programAllowsEagerStatus(dec, p);
        g_programWritesZeroRegs = false;
        for (uint32_t A = p.addr; A < p.addr + uint32_t(p.code.size()); A += 8)
        {
            const auto d = dec.decodeInstructionPair(p.code.data() - p.addr, A);
            if (d.upperUsage.reserved || d.lowerUsage.reserved || (d.viWriteMask & 1u))
                g_programWritesZeroRegs = true;
            for (uint32_t k = 0; k < d.writeCount; ++k)
                if ((d.writeReg[k] & 31u) == 0u)
                    g_programWritesZeroRegs = true;
        }
        const uint32_t end = p.addr + uint32_t(p.code.size());
        const std::string prefix = "vu1p" + std::to_string(idx) + "_b";
        // Blocks are spread over several source files of at most kPairsPerFile pairs each, so
        // no translation unit needs much compiler memory.
        constexpr uint32_t kPairsPerFile = 160u;
        int part = 0;
        uint32_t pairsInPart = 0;
        std::string o;
        std::string partName;
        auto flushPart = [&]() {
            if (partName.empty())
                return;
            writeIfChanged(dir / partName, o);
            written.push_back(partName);
            o.clear();
            partName.clear();
        };
        for (auto it = leaders.begin(); it != leaders.end(); ++it)
        {
            const uint32_t start = *it;
            auto nx = std::next(it);
            const uint32_t stop = nx == leaders.end() ? end : *nx;
            if (partName.empty() || pairsInPart >= kPairsPerFile)
            {
                flushPart();
                partName = "vu1_" + std::to_string(idx) + "_" + std::to_string(part++) + ".cpp";
                o = header;
                emit(o, "// VU1 program %d\n#include \"ps2_vu1_fast_inl.h\"\n#include \"ps2_vu1_turbo_inl.h\"\n#include <algorithm>\n#include <cstring>\n\n", idx);
                pairsInPart = 0;
            }
            char nameBuf[64];
            std::snprintf(nameBuf, sizeof(nameBuf), "%s%04X", prefix.c_str(), start);
            const std::string blockName = nameBuf;
            std::string lean;
            const bool isLean = g_leanBlocks && emitLeanBlock(lean, dec, p, start, stop, deadAll, blockName, blockName + "_x");
            ++g_blockCount;
            g_leanCount += isLean ? 1 : 0;
            emit(o, "%sint %s%s(VU1Fast &v, uint64_t budgetEnd)\n{\n", isLean ? "static " : "", blockName.c_str(), isLean ? "_x" : "");
            BlockTiming timing;
            beginBlockTiming(timing);
            for (uint32_t A = start; A < stop; A += 8)
            {
                const uint32_t k = (A - start) / 8u;
                const bool dead = deadAll[(A - p.addr) / 8u];
                emitPair(o, dec, p, A, timing, static_cast<int>(k), dead);
                deadMacPairs += dead ? 1 : 0;
            }
            emit(o, "    return kGenNext;\n}\n\n");
            o += lean;
            pairsInPart += (stop - start) / 8u;
        }
        flushPart();
        partName = "vu1_" + std::to_string(idx) + ".cpp";
        o = header;
        emit(o, "// VU1 program %d: %zu pairs at 0x%X from %s.\n#include \"ps2_vu1_fast.h\"\n\n", idx, p.code.size() / 8, p.addr, p.origin.c_str());
        for (uint32_t l : leaders)
            emit(o, "int %s%04X(VU1Fast &v, uint64_t budgetEnd);\n", prefix.c_str(), l);
        emit(o, "\nnamespace\n{\nusing BlockFn = int (*)(VU1Fast &, uint64_t);\nconst BlockFn kBlocks[%zu] = {\n", p.code.size() / 8);
        for (uint32_t A = p.addr; A < end; A += 8)
        {
            if (leaders.count(A))
                emit(o, "    %s%04X,\n", prefix.c_str(), A);
            else
                emit(o, "    nullptr,\n");
        }
        const bool eager = programAllowsEagerStatus(dec, p);
        eagerPrograms += eager ? 1 : 0;
        emit(o, "};\n\nint run(VU1Fast &v, uint64_t budgetEnd)\n{\n");
        emit(o, "    for (;;)\n    {\n        const uint32_t pc = v.m_state.pc;\n        if (pc < 0x%Xu || pc >= 0x%Xu)\n            return kGenLeft;\n", p.addr, end);
        // s3recomp: eager status can start later in the program (VU1Fast::tryEagerStatus).
        if (eager)
            emit(o, "        if (!v.m_eagerStatus)\n            v.m_eagerStatus = v.tryEagerStatus();\n");
        emit(o, "        const BlockFn f = kBlocks[(pc - 0x%Xu) >> 3];\n        if (!f)\n            return kGenFallback;\n", p.addr);
        emit(o, "        const int r = f(v, budgetEnd);\n        if (r != kGenNext)\n            return r;\n    }\n}\n} // namespace\n\n");
        emit(o, "int vu1prog_%d(VU1Fast &v, uint64_t budgetEnd)\n{\n", idx);
        if (eager)
            emit(o, "    v.m_eagerStatus = v.tryEagerStatus(); // nothing in this program reads the status flag\n    const int r = run(v, budgetEnd);\n    v.m_eagerStatus = false;\n    return r;\n}\n");
        else
            emit(o, "    return run(v, budgetEnd);\n}\n");
        flushPart();
        char line[200];
        std::snprintf(line, sizeof(line), "    {0x%016llXull, 0x%Xu, 0x%zXu, vu1prog_%d}, // %s", (unsigned long long)h, p.addr, p.code.size(), idx, p.origin.c_str());
        table.push_back(line);
        ++idx;
    }
    std::string o = header;
    emit(o, "#include \"ps2_vu1_fast.h\"\n\n");
    for (int k = 0; k < idx; ++k)
        emit(o, "int vu1prog_%d(VU1Fast &v, uint64_t budgetEnd);\n", k);
    if (idx > 0)
    {
        emit(o, "\nnamespace\n{\nconst VU1CompiledRegion kRegions[] = {\n");
        for (auto &l : table)
            emit(o, "%s\n", l.c_str());
        emit(o, "};\nstruct Registrar\n{\n    Registrar() { vu1RegisterCompiledRegions(kRegions, sizeof(kRegions) / sizeof(kRegions[0])); }\n} g_registrar;\n}\n");
    }
    writeIfChanged(dir / "vu1_registry.cpp", o);
    written.push_back("vu1_registry.cpp");
    std::string list;
    for (const auto &w : written)
        list += w + "\n";
    writeIfChanged(dir / "files.txt", list);
    size_t pairs = 0;
    for (auto &[key, p] : uniq)
        pairs += p.code.size() / 8;
    std::printf("ps2_vu1gen: %d VU1 programs (%zu instruction pairs; %d without status reads, %zu dead MAC results; %d of %d blocks lean, %d turbo segments of %d pairs) in %zu source files\n", idx, pairs, eagerPrograms, deadMacPairs, g_leanCount, g_blockCount, g_turboSegments, g_turboPairs, written.size());
    return 0;
}
