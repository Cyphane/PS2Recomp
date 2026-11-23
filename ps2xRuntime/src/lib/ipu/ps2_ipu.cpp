// SPDX-License-Identifier: GPL-3.0-or-later
// Synchronous driver for PCSX2's IPU core (IPU/): replaces PCSX2's event scheduler and DMAC
// code (IPUdma.cpp) with direct handling of the runtime's channel 3/4 registers. The DMA
// state machine below follows PCSX2's IPUdma.cpp.
#include "ps2_ipu.h"

#include "IPU/IPU.h"
#include "IPU/IPU_MultiISA.h"
#include "IPU/IPUdma.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace ps2ipu
{
    DMACh ipu0ch{};
    DMACh ipu1ch{};
    CpuRegsCompat cpuRegs{};
    IPUDMAStatus IPU1Status{};

    void ReadFIFO_IPUout(mem128_t *out);
    void WriteFIFO_IPUin(const mem128_t *value);

    namespace
    {
        constexpr uint32_t kFromBase = 0x1000B000u; // D3
        constexpr uint32_t kToBase = 0x1000B400u;   // D4
        constexpr uint32_t kTagCnt = 1u;

        HostInterface g_host{};
        std::recursive_mutex g_mutex;
        bool g_attached = false;
        bool g_process = false, g_to = false, g_from = false, g_toDma = false;
        uint32_t g_tagId = 0u; // ID of the toIPU chain tag being transferred

        // PS2X_IPU_LOG=1: log the first IPU events (commands, DMA starts/ends) to stderr.
        bool logOn()
        {
            static const bool on = std::getenv("PS2X_IPU_LOG") != nullptr;
            return on;
        }
        void ipuLog(const char *fmt, ...)
        {
            static int n = 0;
            if (!logOn() || n > 3000)
                return;
            ++n;
            va_list ap;
            va_start(ap, fmt);
            std::fprintf(stderr, "[ipu] ");
            std::vfprintf(stderr, fmt, ap);
            std::fprintf(stderr, "\n");
            va_end(ap);
        }

        void loadChannels()
        {
            if (!g_host.readReg)
                return;
            ipu0ch.chcr._u32 = g_host.readReg(g_host.user, kFromBase);
            ipu0ch.madr = g_host.readReg(g_host.user, kFromBase + 0x10u);
            ipu0ch.qwc = g_host.readReg(g_host.user, kFromBase + 0x20u) & 0xFFFFu;
            ipu1ch.chcr._u32 = g_host.readReg(g_host.user, kToBase);
            ipu1ch.madr = g_host.readReg(g_host.user, kToBase + 0x10u);
            ipu1ch.qwc = g_host.readReg(g_host.user, kToBase + 0x20u) & 0xFFFFu;
            ipu1ch.tadr = g_host.readReg(g_host.user, kToBase + 0x30u);
        }

        void storeChannels()
        {
            if (!g_host.writeReg)
                return;
            g_host.writeReg(g_host.user, kFromBase, ipu0ch.chcr._u32);
            g_host.writeReg(g_host.user, kFromBase + 0x10u, ipu0ch.madr);
            g_host.writeReg(g_host.user, kFromBase + 0x20u, ipu0ch.qwc);
            g_host.writeReg(g_host.user, kToBase, ipu1ch.chcr._u32);
            g_host.writeReg(g_host.user, kToBase + 0x10u, ipu1ch.madr);
            g_host.writeReg(g_host.user, kToBase + 0x20u, ipu1ch.qwc);
            g_host.writeReg(g_host.user, kToBase + 0x30u, ipu1ch.tadr);
        }

        void channelDone(uint32_t base, uint32_t cause)
        {
            ipuLog("dma%u done madr=%08x qwc=%x", cause, base == kToBase ? ipu1ch.madr : ipu0ch.madr,
                   base == kToBase ? ipu1ch.qwc : ipu0ch.qwc);
            storeChannels();
            if (g_host.dmacDone)
                g_host.dmacDone(g_host.user, base, cause);
        }

        uint8_t *guest(uint32_t address, uint32_t bytes)
        {
            return g_host.guestPtr ? g_host.guestPtr(g_host.user, address, bytes) : nullptr;
        }

        // Reads the source-chain tag at TADR and sets MADR/QWC/TADR; true when the chain ends
        // after this tag (PCSX2's hwDmacSrcChain).
        bool readSourceTag()
        {
            const uint8_t *t = guest(ipu1ch.tadr, 16u);
            if (!t)
            {
                ipu1ch.qwc = 0u;
                return true;
            }
            uint32_t tag0 = 0, tag1 = 0;
            std::memcpy(&tag0, t, 4);
            std::memcpy(&tag1, t + 4, 4);
            ipu1ch.qwc = tag0 & 0xFFFFu;
            ipu1ch.chcr.TAG = tag0 >> 16;
            const uint32_t id = (tag0 >> 28) & 7u;
            const bool irq = (tag0 >> 31) != 0u;
            g_tagId = id;
            bool end = false;
            switch (id)
            {
            case 0: // REFE
                ipu1ch.madr = tag1;
                ipu1ch.tadr += 16u;
                end = true;
                break;
            case 1: // CNT: data follows the tag; TADR moves past it as it is sent
                ipu1ch.madr = ipu1ch.tadr + 16u;
                break;
            case 2: // NEXT
                ipu1ch.madr = ipu1ch.tadr + 16u;
                ipu1ch.tadr = tag1;
                break;
            case 3: // REF
            case 4: // REFS
                ipu1ch.madr = tag1;
                ipu1ch.tadr += 16u;
                break;
            case 7: // END
                ipu1ch.madr = ipu1ch.tadr + 16u;
                end = true;
                break;
            default: // CALL/RET are not used for IPU input
                end = true;
                break;
            }
            if (ipu1ch.chcr.TIE && irq)
                end = true;
            return end;
        }

        int IPU1chain()
        {
            const uint32_t qwc = ipu1ch.qwc;
            const u32 *mem = reinterpret_cast<const u32 *>(guest(ipu1ch.madr, qwc * 16u));
            if (!mem)
            {
                ipu1ch.qwc = 0u;
                IPU1Status.InProgress = false;
                return 0;
            }
            const int n = ipu_fifo.in.write(mem, static_cast<int>(qwc));
            ipu1ch.madr += static_cast<uint32_t>(n) << 4;
            ipu1ch.qwc -= static_cast<uint32_t>(n);
            if (ipu1ch.chcr.MOD == 1 && g_tagId == kTagCnt)
                ipu1ch.tadr = ipu1ch.madr;
            if (!ipu1ch.qwc)
                IPU1Status.InProgress = false;
            return n;
        }

        void IPU1dma()
        {
            if (!ipu1ch.chcr.STR || ipu1ch.chcr.MOD == 2)
                return;

            if (!IPUCoreStatus.DataRequested)
            {
                cpuRegs.eCycle[4] = 0x9999;
                if (IPUCoreStatus.WaitingOnIPUTo)
                {
                    IPUCoreStatus.WaitingOnIPUTo = false;
                    g_process = true;
                }
                return;
            }

            int totalqwc = 0;
            if (!IPU1Status.InProgress)
            {
                IPU1Status.DMAFinished = readSourceTag();
                if (ipu1ch.qwc)
                    IPU1Status.InProgress = true;
            }

            if (IPU1Status.InProgress)
                totalqwc += IPU1chain();

            if (totalqwc == 0 || (IPU1Status.DMAFinished && !IPU1Status.InProgress))
            {
                cpuRegs.eCycle[4] = 0;
                g_to = true;
            }
            else
            {
                cpuRegs.eCycle[4] = 0x9999;
            }

            if (IPUCoreStatus.WaitingOnIPUTo && g_BP.IFC >= 1)
            {
                IPUCoreStatus.WaitingOnIPUTo = false;
                g_process = true;
            }
        }

        void ipu1Interrupt()
        {
            if (!IPU1Status.DMAFinished || IPU1Status.InProgress)
            {
                IPU1dma();
                return;
            }
            ipu1ch.chcr.STR = 0;
            channelDone(kToBase, DMAC_TO_IPU);
        }

        void IPU0dma()
        {
            if (!ipuRegs.ctrl.OFC)
            {
                if (IPUCoreStatus.WaitingOnIPUFrom)
                {
                    IPUCoreStatus.WaitingOnIPUFrom = false;
                    g_process = true;
                }
                return;
            }
            if (!ipu0ch.chcr.STR || ipu0ch.qwc == 0)
            {
                if (IPUCoreStatus.WaitingOnIPUFrom)
                {
                    IPUCoreStatus.WaitingOnIPUFrom = false;
                    g_process = true;
                }
                return;
            }

            const uint32_t readsize = std::min<uint32_t>(ipu0ch.qwc, ipuRegs.ctrl.OFC);
            uint8_t *dst = guest(ipu0ch.madr, readsize * 16u);
            if (dst)
                ipu_fifo.out.read(dst, readsize);
            else
            {
                alignas(16) uint8_t sink[8 * 16];
                ipu_fifo.out.read(sink, readsize);
            }
            ipu0ch.madr += readsize << 4;
            ipu0ch.qwc -= readsize;

            if (!ipu0ch.qwc)
                g_from = true;

            if (ipuRegs.ctrl.BUSY && IPUCoreStatus.WaitingOnIPUFrom)
            {
                IPUCoreStatus.WaitingOnIPUFrom = false;
                g_process = true;
            }
        }

        void ipu0Interrupt()
        {
            if (ipu0ch.qwc > 0)
            {
                IPU0dma();
                return;
            }
            if (!ipu0ch.chcr.STR)
                return;
            ipu0ch.chcr.STR = 0;
            channelDone(kFromBase, DMAC_FROM_IPU);
        }

        void startToIpu()
        {
            if (ipu1ch.chcr.MOD == 1) // chain
            {
                if (ipu1ch.qwc == 0)
                {
                    IPU1Status.InProgress = false;
                    IPU1Status.DMAFinished = false;
                }
                else // continuing a chain: the tag that was in flight is in CHCR.TAG
                {
                    IPU1Status.InProgress = true;
                    const uint32_t tag = ipu1ch.chcr.TAG;
                    const uint32_t id = (tag >> 12) & 7u;
                    const bool irq = (tag >> 15) != 0u;
                    g_tagId = id;
                    IPU1Status.DMAFinished = id == 0u || id == 7u || (irq && ipu1ch.chcr.TIE);
                }
            }
            else
            {
                g_tagId = 0xFFu;
                IPU1Status.InProgress = true;
                IPU1Status.DMAFinished = true;
            }
            IPU1dma();
        }

        void startFromIpu()
        {
            IPU0dma();
            if (ipu0ch.qwc == 0x10000u)
            {
                ipu0ch.qwc = 0;
                ipu0ch.chcr.STR = 0;
                channelDone(kFromBase, DMAC_FROM_IPU);
            }
        }

        // Runs whatever the IPU and its DMA channels can do now.
        void pump()
        {
            for (int guard = 0; guard < 1000000; ++guard)
            {
                if (g_toDma)
                {
                    g_toDma = false;
                    IPU1dma();
                    continue;
                }
                if (g_from)
                {
                    g_from = false;
                    ipu0Interrupt();
                    continue;
                }
                if (g_to)
                {
                    g_to = false;
                    ipu1Interrupt();
                    continue;
                }
                if (g_process)
                {
                    g_process = false;
                    IPUProcessInterrupt();
                    continue;
                }
                break;
            }
        }

        struct Scope
        {
            std::lock_guard<std::recursive_mutex> lock{g_mutex};
            Scope() { loadChannels(); }
            ~Scope()
            {
                pump();
                storeChannels();
            }
        };
    }

    // hooks used by the PCSX2 core
    void bridgeScheduleProcess() { g_process = true; }
    void bridgeScheduleTo() { g_to = true; }
    void bridgeScheduleFrom() { g_from = true; }
    void bridgeScheduleToDma() { g_toDma = true; }
    void hwIntcIrq(int cause)
    {
        if (g_host.intcRaise)
            g_host.intcRaise(g_host.user, static_cast<uint32_t>(cause));
    }
    void ipuDmaReset()
    {
        IPU1Status.InProgress = false;
        IPU1Status.DMAFinished = true;
    }

    void attach(const HostInterface &host)
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_host = host;
        if (!g_attached)
        {
            g_attached = true;
            ipuReset();
        }
    }

    void reset()
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);
        g_process = g_to = g_from = g_toDma = false;
        std::memset(&cpuRegs, 0, sizeof(cpuRegs));
        ipuReset();
    }

    bool handlesRegister(uint32_t address) { return address >= 0x10002000u && address < 0x10002040u; }

    uint32_t read32(uint32_t address)
    {
        Scope s;
        pump();
        return ipuRead32(address);
    }

    uint64_t read64(uint32_t address)
    {
        Scope s;
        pump();
        return ipuRead64(address);
    }

    void write32(uint32_t address, uint32_t value)
    {
        Scope s;
        ipuLog("w %08x = %08x (ctrl=%08x ifc=%u bp=%u ofc=%u)", address, value, ipuRegs.ctrl._u32, g_BP.IFC, g_BP.BP,
               ipuRegs.ctrl.OFC);
        (void)ipuWrite32(address, value); // only CMD and CTRL are writable
    }

    void write64(uint32_t address, uint64_t value)
    {
        Scope s;
        ipuLog("w64 %08x = %llx", address, static_cast<unsigned long long>(value));
        ipuWrite64(address, value);
    }

    void fifoRead(uint8_t out[16])
    {
        Scope s;
        alignas(16) mem128_t v{};
        if (ipuRegs.ctrl.OFC > 0)
            ReadFIFO_IPUout(&v);
        std::memcpy(out, &v, 16);
        if (ipuRegs.ctrl.BUSY && IPUCoreStatus.WaitingOnIPUFrom)
        {
            IPUCoreStatus.WaitingOnIPUFrom = false;
            g_process = true;
        }
    }

    void fifoWrite(const uint8_t in[16])
    {
        Scope s;
        alignas(16) mem128_t v{};
        std::memcpy(&v, in, 16);
        WriteFIFO_IPUin(&v);
    }

    void dmaStart(uint32_t channelBase)
    {
        Scope s;
        ipuLog("dma%c start chcr=%08x madr=%08x qwc=%x tadr=%08x", channelBase == kToBase ? '4' : '3',
               channelBase == kToBase ? ipu1ch.chcr._u32 : ipu0ch.chcr._u32,
               channelBase == kToBase ? ipu1ch.madr : ipu0ch.madr, channelBase == kToBase ? ipu1ch.qwc : ipu0ch.qwc,
               ipu1ch.tadr);
        if (channelBase == kToBase)
            startToIpu();
        else if (channelBase == kFromBase)
            startFromIpu();
    }
}
