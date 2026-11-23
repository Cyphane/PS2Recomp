#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopGuestExecutor;
    class IopMemory;

    class IopIoman
    {
    public:
        IopIoman(IopHost &host, IopMemory &memory) noexcept;

        void reset();
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor &executor);

    private:
        struct Device
        {
            uint32_t address = 0u;
            uint32_t gp = 0u;
            std::string name;
        };

        // s3recomp: read-only file descriptors opened through ioman open() (e.g. a sound
        // driver streaming its banks from cdrom0:), backed by host files.
        struct OpenFile
        {
            uint64_t handle = 0u;
            uint64_t size = 0u;
            uint64_t position = 0u;
        };

        IopHost &m_host;
        IopMemory &m_memory;
        std::vector<Device> m_devices;
        std::map<int32_t, OpenFile> m_files;
        int32_t m_nextFd = 3;
    };
}
