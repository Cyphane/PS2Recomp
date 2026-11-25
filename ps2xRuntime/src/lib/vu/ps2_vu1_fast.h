#pragma once
// s3recomp: fast VU1 core. Same instruction semantics as VU1Interpreter (it started as a copy), but
// without the cycle-by-cycle pipeline machinery: register results are written immediately (readers
// stall until they are ready, so programs cannot see the difference) and only the state programs can
// observe on a timeline is tracked - Q, P, MAC/status/clip flags, the branch VI delay and XGKICK.
// It shares VU1State with the interpreter, so either can run a given call.
#include "runtime/ps2_vu1.h"

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define VU1_HOT_INLINE __forceinline
#else
#define VU1_HOT_INLINE inline __attribute__((always_inline))
#endif

class VU1Fast;

// A VU1 program compiled to C++ (see the VU1 program generator). It runs from v.m_state.pc while
// the program's bytes are resident in VU1 code memory, and returns one of the kGen* codes.
using VU1CompiledFn = int (*)(VU1Fast &v, uint64_t budgetEnd);
enum : int
{
    kGenEnded = 0,     // program finished (E-bit / D / T), pipelines not yet flushed
    kGenBudget = 1,    // cycle budget used up or stop requested; m_state.pc is where to resume
    kGenLeft = 2,      // control left the program; m_state.pc is the new address
    kGenFallback = 3,  // this pair needs the generic core (m_state.pc)
    kGenNext = 4       // block finished normally; continue at m_state.pc
};
struct VU1CompiledRegion
{
    uint64_t hash;   // vu1RegionHash of the program bytes
    uint32_t addr;   // byte address in VU1 code memory
    uint32_t size;   // bytes
    VU1CompiledFn fn;
};
void vu1RegisterCompiledRegions(const VU1CompiledRegion *regions, size_t count);
size_t vu1CompiledRegionCount();
inline uint64_t vu1RegionHash(const uint8_t *bytes, uint32_t size)
{
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i + 8u <= size; i += 8u)
    {
        uint64_t w;
        __builtin_memcpy(&w, bytes + i, 8);
        h = (h ^ w) * 1099511628211ull;
        h ^= h >> 29;
    }
    return h ^ size;
}

class VU1Fast
{
public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Fast(VU1State &state, Unit unit = Unit::VU1);

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

public: // generated VU1 code accesses the core directly
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        InstructionUsage lowerUsage{};
        InstructionUsage upperUsage{};
        bool iBit = false;
        bool eBit = false;
        bool mBit = false;
        bool dBit = false;
        bool tBit = false;
        uint8_t upperVfShadowReg = 0;
        uint8_t suppressedLowerVf = 0;
        // Precomputed for the timing model (filled by decodeInstructionPair).
        uint8_t readCount = 0;
        uint8_t readReg[4]{};
        uint8_t readLanes[4]{};
        uint16_t viReadMask = 0;
        uint8_t accReadLanes = 0;
        uint8_t writeCount = 0;
        uint8_t writeReg[2]{};
        uint8_t writeLanes[2]{};
        uint8_t writeLatency[2]{};
        uint16_t viWriteMask = 0;
        uint8_t viWriteLatency = 0;
        uint8_t accWriteLanes = 0;
    };

    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool valid = false;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;
    };

    struct PendingStore
    {
        uint64_t readyCycle = 0;
        uint32_t address = 0;
        std::array<uint32_t, 4> words{};
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct PendingVfWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t reg = 0;
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct PendingViWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        int32_t value = 0;
        uint8_t reg = 0;
        bool valid = false;
    };

    struct PendingAccWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t laneMask = 0;
        bool valid = false;
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;
    };

    static constexpr uint32_t kFmacLatency = 4u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    static constexpr uint32_t kMaxFlagEntries = 8u;
    static constexpr uint32_t kMaxPendingStores = 8u;
    static constexpr uint32_t kMaxPendingVfWrites = 16u;
    static constexpr uint32_t kMaxPendingViWrites = 8u;
    static constexpr uint32_t kMaxPendingAccWrites = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State &m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;

    std::array<FlagPipelineEntry, kMaxFlagEntries> m_flagPipeline{};
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    std::array<PendingStore, kMaxPendingStores> m_storePipeline{};
    std::array<PendingVfWrite, kMaxPendingVfWrites> m_vfWritePipeline{};
    std::array<PendingViWrite, kMaxPendingViWrites> m_viWritePipeline{};
    std::array<PendingAccWrite, kMaxPendingAccWrites> m_accWritePipeline{};
    XgkickPipeline m_xgkick{};
    std::array<std::array<uint64_t, 4>, 32> m_vfReady{};
    std::array<uint64_t, 16> m_viReady{};
    std::array<uint64_t, 4> m_accReady{};
    std::array<std::array<uint64_t, 4>, 32> m_vfLatestWrite{};
    std::array<uint64_t, 16> m_viLatestWrite{};
    std::array<uint64_t, 4> m_accLatestWrite{};

    uint64_t m_cycle = 0;
    uint64_t m_nextCommit = ~0ull;
    // Generated-code support
    VU1_HOT_INLINE void commitIfDue()
    {
        if (m_cycle >= m_nextCommit)
            commitReadyPipelines();
    }
    VU1_HOT_INLINE void stepCycle()
    {
        ++m_cycle;
        m_state.cycles = m_cycle;
        commitIfDue();
        if (m_xgkick.active)
            progressXgkick();
    }
    VU1_HOT_INLINE void resetZeroRegs()
    {
        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;
    }
    // Out-of-line stall for generated code (rare path): waits until the pair at pc may issue.
    bool stallPair(uint32_t pc, uint64_t budgetEnd);
    template <bool EBit, bool DBit, bool TBit, bool IsBranch>
    VU1_HOT_INLINE int pairEnd(uint32_t next)
    {
        resetZeroRegs();
        m_state.pc = next;
        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & 0x3FFFu;
                m_state.branchPending = false;
            }
            else
                --m_state.branchDelay;
        }
        bool ended = false;
        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            ended = true;
        }
        else if (m_state.ebit)
            ended = true;
        else
        {
            const bool dHalt = DBit && m_state.dBitEnabled;
            const bool tHalt = TBit && m_state.tBitEnabled;
            if ((dHalt || tHalt) && !IsBranch)
            {
                m_state.stoppedByD = dHalt;
                m_state.stoppedByT = tHalt;
                ended = true;
            }
            else if (EBit)
                m_state.ebit = true;
            else if (dHalt || tHalt)
            {
                m_state.haltAfterDelaySlot = true;
                m_pendingHaltD = dHalt;
                m_pendingHaltT = tHalt;
            }
        }
        stepCycle();
        return ended ? kGenEnded : kGenNext;
    }
    // Pair end for generated code when no branch, E-bit or halt can be pending: the pair before it in
    // the same block was not a branch and had no E/D/T bit, and this pair has none of those either.
    VU1_HOT_INLINE void pairEndPlain(uint32_t next)
    {
        resetZeroRegs();
        m_state.pc = next;
        stepCycle();
    }
    // s3recomp: the same for programs that never write VF0 or VI0 (they stay 0 / (0,0,0,1)).
    VU1_HOT_INLINE void pairEndPlainNoZero(uint32_t next)
    {
        m_state.pc = next;
        stepCycle();
    }
    const uint8_t *m_activeVuCode = nullptr;
    uint32_t m_activeVuCodeSize = 0;
    int stepPair(uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint64_t budgetEnd);
    VU1CompiledFn compiledFor(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc);
    std::array<VU1CompiledFn, 2048> m_residentFn{};
    uint64_t m_residentGeneration = ~0ull;
    const uint8_t *m_residentCode = nullptr;
    bool m_compiledEnabled = true;
    uint32_t m_flagHead = 0;  // m_flagPipeline is a ring in issue order (all entries have the same latency)
    uint32_t m_flagCount = 0;
    // Flag results (MAC, status, clip) are committed lazily: by the instructions that read them,
    // before a Q result updates status, when the ring is full, and when the pipelines are flushed or
    // stepped cycle by cycle. Commits happen in issue order at the same cycles as eager commits, so
    // every reader sees exactly the same values.
    // Generated code calls these at nearly every pair, so only the checks are inline; the loop that
    // commits entries is out of line (keeps the generated blocks small enough for the i-cache).
    VU1_HOT_INLINE void commitFlags(uint64_t now)
    {
        if (m_flagCount != 0u && m_flagPipeline[m_flagHead].readyCycle <= now)
            commitFlagsSlow(now);
    }
    void commitFlagsSlow(uint64_t now);
    VU1_HOT_INLINE FlagPipelineEntry *pushFlagEntry()
    {
        if (m_flagCount >= kMaxFlagEntries)
        {
            commitFlagsSlow(m_cycle);
            if (m_flagCount >= kMaxFlagEntries)
                return nullptr;
        }
        FlagPipelineEntry *e = &m_flagPipeline[(m_flagHead + m_flagCount) % kMaxFlagEntries];
        ++m_flagCount;
        e->readyCycle = m_cycle + kFmacLatency;
        e->issueCycle = m_cycle;
        e->mac = 0u;
        e->status = 0u;
        e->extraSticky = 0u;
        e->clip = 0u;
        e->valid = true;
        e->writesMac = false;
        e->writesStatus = false;
        e->writesSticky = false;
        e->writesClip = false;
        return e;
    } // earliest readyCycle of any pending pipeline entry
    uint64_t m_nextWriteSequence = 0;
    uint64_t m_efuResourceReady = 0;
    uint32_t m_workingClip = 0;
    uint32_t m_currentUpperInstruction = 0;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    // Generated code support for flags nothing can observe in flight. In programs that never read
    // the status flag (no FSAND/FSEQ/FSOR) and never set it (no FSSET), FMAC status results are
    // applied at issue: applying them in issue order gives the same value once the pipeline would
    // have drained, and no other status writer touches the same bits. m_skipMac marks an FMAC whose
    // MAC result is replaced by a later FMAC in the same block before any instruction can read it.
    // Both are off outside generated code.
    bool m_eagerStatus = false;
    bool m_skipMac = false;
    bool statusFlagsPending() const
    {
        for (uint32_t k = 0; k < m_flagCount; ++k)
        {
            const FlagPipelineEntry &e = m_flagPipeline[(m_flagHead + k) % kMaxFlagEntries];
            if (e.writesStatus || e.writesSticky)
                return true;
        }
        return false;
    }
    // Returns true when the FMAC flag results were handled here.
    VU1_HOT_INLINE bool flagsAtIssue(uint32_t mac, uint32_t status, uint32_t extraSticky)
    {
        if (!m_eagerStatus)
            return false;
        const uint32_t current = status & 0xFu;
        m_state.status = (m_state.status & 0xFF0u) | current | ((current | extraSticky) << 6);
        if (m_skipMac)
            return true;
        FlagPipelineEntry *e = pushFlagEntry();
        if (!e)
        {
            reportReservedInstruction(true, 0xFFFFFFFFu);
            return true;
        }
        e->mac = mac;
        e->writesMac = true;
        return true;
    }
    void resetXgkick();
    void drainXgkick();
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    InstructionUsage decodeUpperUsage(uint32_t upper) const;
    InstructionUsage decodeLowerUsage(uint32_t lower) const;
    static void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    const DecodedInstructionPair &getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc);
    DecodedInstructionPair m_uncachedPair{};
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execUpperInl(uint32_t instr); // defined in ps2_vu1_fast_upper_inl.h
    void execUpperInlDaz(uint32_t instr); // generated code (runs with MXCSR DAZ set)
    void execUpperScalarInl(uint32_t instr); // the same, without the four-lane FMAC fast path
    void execUpperScalar(uint32_t instr);    // out-of-line execUpperScalarInl
    void execLowerInl(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr); // ps2_vu1_fast_lower_inl.h
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest)
    {
        if (dest & 0x8u) dst[0] = result[0];
        if (dest & 0x4u) dst[1] = result[1];
        if (dest & 0x2u) dst[2] = result[2];
        if (dest & 0x1u) dst[3] = result[3];
    }
    void applyDestAcc(const float *result, uint8_t dest);
    void applyFmacDest(float *dst, float *result, uint8_t dest);
    void applyFmacDestAcc(float *result, uint8_t dest);
    void normalizeFmacResult(float *result, uint8_t dest, uint8_t laneFlags[4], const uint8_t productFlags[4]);
    bool calculateFmacExactResult(uint32_t component, double &result) const;
    uint8_t normalizeFmacExactResult(float &value, double exactResult) const;
    uint32_t calculateFmacProductSticky(uint8_t dest, uint8_t productFlags[4]) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    // Inlinable variants (ps2_vu1_fast_upper_inl.h) taking the upper instruction explicitly.
    uint32_t fmacProductStickyInl(uint32_t upper, uint8_t dest, uint8_t productFlags[4]) const;
    void normalizeFmacLaneSlow(uint32_t component, float *result, uint8_t laneFlags[4]);
    void applyFmacDestInl(uint32_t upper, float *dst, float *result, uint8_t dest);
    bool fmacExactInl(uint32_t upper, uint32_t component, double &result) const;
    bool fmacSimdInl(uint32_t instr); // four-lane fast path (ps2_vu1_fast_upper_inl.h)
    template <bool Daz>
    bool fmacSimdInlT(uint32_t instr);
    bool miscSimdInl(uint32_t instr); // four-lane MAX/MINI/ITOF/FTOI/ABS/CLIP
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    void queueVfWrite(uint8_t reg, uint8_t laneMask, const float value[4], uint32_t latency);
    void queueViWrite(uint8_t reg, int32_t value, uint32_t latency);
    void queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);
    bool pipelinesPending() const;

#if defined(_MSC_VER)
    __forceinline
#else
    __attribute__((always_inline))
#endif
    float normalizeOperand(float value) const
    {
        uint32_t bits;
        __builtin_memcpy(&bits, &value, 4);
        const uint32_t exponent = bits & 0x7F800000u;
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0x7F800000u)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        __builtin_memcpy(&value, &bits, 4);
        return value;
    }
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const { return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu; }
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc) { return normalizeOperand(vf[bc & 3u]); }
};

