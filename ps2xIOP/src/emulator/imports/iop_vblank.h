#pragma once

#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopKernel;
    class IopGuestExecutor;

    class IopVblank
    {
    public:
        explicit IopVblank(IopKernel &kernel) noexcept;

        void reset() noexcept;
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle);

        // s3recomp: deliver handlers registered with RegisterVblankHandler at the start and
        // end of every field, as the IOP vblank module does from its interrupt.
        void serviceDue(uint64_t currentCycle, IopGuestExecutor &executor);
        [[nodiscard]] uint64_t nextEventCycle(uint64_t fallback) const noexcept;

    private:
        struct Handler
        {
            uint32_t edge = 0u;
            uint32_t priority = 0u;
            uint32_t function = 0u;
            uint32_t argument = 0u;
            uint32_t gp = 0u;
        };

        void runEdge(uint32_t edge, IopGuestExecutor &executor);

        IopKernel &m_kernel;
        std::vector<Handler> m_handlers;
        uint64_t m_nextStart = 0u;
        uint64_t m_nextEnd = 0u;
        bool m_servicing = false;
    };
}
