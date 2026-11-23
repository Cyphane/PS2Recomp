// SPDX-License-Identifier: GPL-3.0-or-later
// IPU (MPEG decoder) emulation for the runtime. The command/decoder core is PCSX2's
// (GPL-3.0+, see IPU/); this layer drives it synchronously from the EE's register, FIFO and
// DMA (channel 3 = fromIPU, channel 4 = toIPU) accesses.
#pragma once

#include <cstdint>

namespace ps2ipu
{
    struct HostInterface
    {
        void *user = nullptr;
        // raw DMA channel register access (no side effects)
        uint32_t (*readReg)(void *user, uint32_t address) = nullptr;
        void (*writeReg)(void *user, uint32_t address, uint32_t value) = nullptr;
        // host pointer to guest memory for a DMA transfer (RDRAM or scratchpad), or null
        uint8_t *(*guestPtr)(void *user, uint32_t address, uint32_t bytes) = nullptr;
        // channel finished: clears STR, sets D_STAT and queues the DMAC interrupt
        void (*dmacDone)(void *user, uint32_t channelBase, uint32_t cause) = nullptr;
        // INTC interrupt (cause 8 = IPU)
        void (*intcRaise)(void *user, uint32_t cause) = nullptr;
    };

    void attach(const HostInterface &host);
    void reset();

    bool handlesRegister(uint32_t address); // 0x10002000-0x1000203F
    uint32_t read32(uint32_t address);
    uint64_t read64(uint32_t address);
    void write32(uint32_t address, uint32_t value);
    void write64(uint32_t address, uint64_t value);

    void fifoRead(uint8_t out[16]);       // IPUout FIFO, 0x10007000
    void fifoWrite(const uint8_t in[16]); // IPUin FIFO, 0x10007010

    // A D3 (0x1000B000) or D4 (0x1000B400) CHCR write with STR set.
    void dmaStart(uint32_t channelBase);
}
