#include "iop_vblank.h"

#include "../core/iop_cpu.h"
#include "../iop_emulator_const.h"
#include "../core/iop_kernel.h"
#include "../services/iop_rpc.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    IopVblank::IopVblank(IopKernel &kernel) noexcept
        : m_kernel(kernel)
    {
    }

    bool IopVblank::dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle)
    {
        switch (ordinal)
        {
        case 4: // WaitVblankStart
        case 5: // WaitVblankEnd
        case 6: // WaitVblank
        case 7: // WaitNonVblank
        {
            const bool waitForEnd = ordinal == 5u || ordinal == 7u;
            const uint64_t phase = waitForEnd ? kVblankEndPhaseCycles : 0u;
            const uint64_t fieldStart = currentCycle - (currentCycle % kVblankPeriodCycles);
            uint64_t wakeCycle = fieldStart + phase;
            if (wakeCycle <= currentCycle)
                wakeCycle += kVblankPeriodCycles;
            m_kernel.delayCurrentUntil(wakeCycle, cpu);
            cpu.gpr[2] = 0u;
            return true;
        }
        case 8: // RegisterVblankHandler(edge, priority, handler, arg)
        {
            const uint32_t edge = cpu.gpr[4] ? 1u : 0u;
            const uint32_t function = cpu.gpr[6];
            for (const Handler &existing : m_handlers)
            {
                if (existing.edge == edge && existing.function == function)
                {
                    cpu.gpr[2] = static_cast<uint32_t>(-104); // KE_FOUND_HANDLER
                    return true;
                }
            }
            if (function != 0u)
            {
                m_handlers.push_back({edge, cpu.gpr[5], function, cpu.gpr[7], cpu.gpr[28]});
                std::stable_sort(m_handlers.begin(), m_handlers.end(), [](const Handler &l, const Handler &r)
                                 { return l.priority < r.priority; });
                if (m_nextStart == 0u)
                {
                    const uint64_t fieldStart = currentCycle - (currentCycle % kVblankPeriodCycles);
                    m_nextStart = fieldStart + kVblankPeriodCycles;
                    m_nextEnd = fieldStart + kVblankEndPhaseCycles;
                    if (m_nextEnd <= currentCycle)
                        m_nextEnd += kVblankPeriodCycles;
                }
            }
            cpu.gpr[2] = 0u;
            return true;
        }
        case 9: // ReleaseVblankHandler(edge, handler)
        {
            const uint32_t edge = cpu.gpr[4] ? 1u : 0u;
            const uint32_t function = cpu.gpr[5];
            const auto before = m_handlers.size();
            m_handlers.erase(std::remove_if(m_handlers.begin(), m_handlers.end(), [&](const Handler &h)
                                            { return h.edge == edge && h.function == function; }),
                             m_handlers.end());
            cpu.gpr[2] = m_handlers.size() != before ? 0u : static_cast<uint32_t>(-105); // KE_NOTFOUND_HANDLER
            return true;
        }
        default:
            return false;
        }
    }

    void IopVblank::reset() noexcept
    {
        m_handlers.clear();
        m_nextStart = 0u;
        m_nextEnd = 0u;
        m_servicing = false;
    }

    void IopVblank::runEdge(uint32_t edge, IopGuestExecutor &executor)
    {
        const std::vector<Handler> snapshot = m_handlers;
        for (const Handler &handler : snapshot)
        {
            if (handler.edge != edge)
                continue;
            const uint32_t keep = executor.executeGuestFunctionWithBudget(handler.function, handler.argument,
                                                                          0u, 0u, 0u, handler.gp, 200000u);
            if (keep == 0u)
            {
                m_handlers.erase(std::remove_if(m_handlers.begin(), m_handlers.end(), [&](const Handler &h)
                                                { return h.edge == handler.edge && h.function == handler.function; }),
                                 m_handlers.end());
            }
        }
    }

    void IopVblank::serviceDue(uint64_t currentCycle, IopGuestExecutor &executor)
    {
        if (m_servicing || m_handlers.empty() || m_nextStart == 0u)
            return;
        m_servicing = true;
        struct Guard
        {
            bool &flag;
            ~Guard() { flag = false; }
        } guard{m_servicing};

        if (currentCycle >= m_nextEnd)
        {
            m_nextEnd += kVblankPeriodCycles;
            if (m_nextEnd <= currentCycle)
                m_nextEnd = currentCycle - (currentCycle % kVblankPeriodCycles) + kVblankEndPhaseCycles + kVblankPeriodCycles;
            runEdge(1u, executor);
        }
        if (currentCycle >= m_nextStart)
        {
            m_nextStart += kVblankPeriodCycles;
            if (m_nextStart <= currentCycle)
                m_nextStart = currentCycle - (currentCycle % kVblankPeriodCycles) + kVblankPeriodCycles;
            runEdge(0u, executor);
        }
    }

    uint64_t IopVblank::nextEventCycle(uint64_t fallback) const noexcept
    {
        if (m_handlers.empty() || m_nextStart == 0u)
            return fallback;
        return std::min(fallback, std::min(m_nextStart, m_nextEnd));
    }
}
