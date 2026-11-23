#include "iop_ioman.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "../services/iop_rpc.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <vector>

namespace ps2x::iop::detail
{
    IopIoman::IopIoman(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    void IopIoman::reset()
    {
        m_devices.clear();
        for (auto &[fd, file] : m_files)
            m_host.closeHostFile(file.handle);
        m_files.clear();
        m_nextFd = 3;
    }

    bool IopIoman::dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor &executor)
    {
        constexpr size_t kMaxDevices = 16u;
        const uint32_t a0 = cpu.gpr[4];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };

        constexpr uint32_t kError = 0xFFFFFFFFu; // -1 (EIO-style failure)
        switch (ordinal)
        {
        case 4: // open(name, flags)
        {
            const std::string guestPath = m_memory.readString(a0, 256u);
            const uint32_t flags = cpu.gpr[5];
            constexpr uint32_t kAccessMask = 0x3u; // O_RDONLY=1, O_WRONLY=2, O_RDWR=3
            if (guestPath.empty() || (flags & kAccessMask) != 0x1u)
            {
                m_host.log(LogLevel::Warning, "[IOP] ioman open: only read-only access is supported: " + guestPath);
                setV0(kError);
                return true;
            }
            const std::string hostPath = m_host.translateGuestPath(guestPath);
            const uint64_t handle = hostPath.empty() ? 0u : m_host.openHostFile(hostPath);
            if (handle == 0u)
            {
                m_host.log(LogLevel::Warning, "[IOP] ioman open failed: " + guestPath);
                setV0(kError);
                return true;
            }
            OpenFile file;
            file.handle = handle;
            (void)m_host.hostFileSize(handle, file.size);
            const int32_t fd = m_nextFd++;
            m_files[fd] = file;
            setV0(static_cast<uint32_t>(fd));
            return true;
        }
        case 5: // close(fd)
        {
            const auto it = m_files.find(static_cast<int32_t>(a0));
            if (it == m_files.end())
            {
                setV0(kError);
                return true;
            }
            m_host.closeHostFile(it->second.handle);
            m_files.erase(it);
            setV0(0u);
            return true;
        }
        case 6: // read(fd, buffer, size)
        {
            const auto it = m_files.find(static_cast<int32_t>(a0));
            const uint32_t buffer = cpu.gpr[5];
            const uint32_t size = cpu.gpr[6];
            if (it == m_files.end())
            {
                setV0(kError);
                return true;
            }
            OpenFile &file = it->second;
            const uint64_t remaining = file.position < file.size ? file.size - file.position : 0u;
            const size_t want = static_cast<size_t>(std::min<uint64_t>(size, remaining));
            std::vector<uint8_t> data(want);
            size_t got = 0u;
            if (want != 0u && !m_host.readHostFile(file.handle, file.position, data.data(), want, got))
            {
                setV0(kError);
                return true;
            }
            if (got != 0u && !m_memory.writeRam(buffer, data.data(), got))
            {
                setV0(kError);
                return true;
            }
            file.position += got;
            setV0(static_cast<uint32_t>(got));
            return true;
        }
        case 8: // lseek(fd, offset, whence)
        {
            const auto it = m_files.find(static_cast<int32_t>(a0));
            if (it == m_files.end())
            {
                setV0(kError);
                return true;
            }
            OpenFile &file = it->second;
            const int64_t offset = static_cast<int32_t>(cpu.gpr[5]);
            const uint32_t whence = cpu.gpr[6]; // 0 SET, 1 CUR, 2 END
            int64_t base = 0;
            if (whence == 1u)
                base = static_cast<int64_t>(file.position);
            else if (whence == 2u)
                base = static_cast<int64_t>(file.size);
            else if (whence != 0u)
            {
                setV0(kError);
                return true;
            }
            const int64_t target = base + offset;
            if (target < 0)
            {
                setV0(kError);
                return true;
            }
            file.position = static_cast<uint64_t>(target);
            setV0(static_cast<uint32_t>(target));
            return true;
        }
        case 20: // AddDrv
        {
            if (a0 == 0u || m_devices.size() >= kMaxDevices)
            {
                setV0(0xFFFFFFFFu);
                return true;
            }

            const uint32_t nameAddress = m_memory.read32(a0);
            const uint32_t operations = m_memory.read32(a0 + 16u);
            const std::string name = m_memory.readString(nameAddress, 64u);
            if (nameAddress == 0u || operations == 0u || name.empty())
            {
                setV0(0xFFFFFFFFu);
                return true;
            }

            m_devices.push_back({a0, cpu.gpr[28], name});
            const uint32_t init = m_memory.read32(operations);
            if (init != 0u)
            {
                const int32_t result = static_cast<int32_t>(
                    executor.executeGuestFunction(init, a0, 0u, 0u, 0u, cpu.gpr[28]));
                if (result < 0)
                {
                    m_devices.pop_back();
                    setV0(0xFFFFFFFFu);
                    return true;
                }
            }

            setV0(0u);
            return true;
        }
        case 21: // DelDrv
        {
            const std::string name = m_memory.readString(a0, 64u);
            const auto device = std::find_if(
                m_devices.begin(), m_devices.end(),
                [&](const Device &candidate)
                { return candidate.name == name; });
            if (device == m_devices.end())
            {
                setV0(0xFFFFFFFFu);
                return true;
            }

            const uint32_t operations = m_memory.read32(device->address + 16u);
            const uint32_t deinit = operations != 0u ? m_memory.read32(operations + 4u) : 0u;
            if (deinit != 0u)
                (void)executor.executeGuestFunction(deinit, device->address, 0u, 0u, 0u, device->gp);
            m_devices.erase(device);
            setV0(0u);
            return true;
        }
        default:
            return false;
        }
    }
}
