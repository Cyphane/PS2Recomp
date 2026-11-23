// s3recomp: Vulkan GS raster backend. See gs_vulkan_backend.h for the design.
#include "gs_vulkan_backend.h"

#include "volk.h"

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace
{
#include "gs_vulkan_shaders.inc"
#include "gs_vulkan_tables.inc"

constexpr uint32_t kGsPages = 512u;
constexpr uint32_t kPageBytes = 8192u;
using PageSet = std::bitset<kGsPages>;

constexpr VkDeviceSize kLocalMemoryBytes = 4ull << 20;
constexpr VkDeviceSize kClutBytes = 4ull << 20;        // 4096 CLUT snapshots of 1 KiB
constexpr VkDeviceSize kStagingBytes = 16ull << 20;
constexpr VkDeviceSize kVertexBytes = 48ull << 20;
constexpr VkDeviceSize kDrawStateBytes = 8ull << 20;
constexpr uint32_t kRenderArea = 2048u;

struct Rect
{
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    bool empty() const { return x1 < x0 || y1 < y0; }
    Rect clip(const Rect &r) const
    {
        Rect o{std::max(x0, r.x0), std::max(y0, r.y0), std::min(x1, r.x1), std::min(y1, r.y1)};
        return o.empty() ? Rect{} : o;
    }
};

void pageDims(uint8_t psm, uint32_t &pw, uint32_t &ph)
{
    switch (psm)
    {
    case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S:
        pw = 64u; ph = 64u; break;
    case GS_PSM_T8:
        pw = 128u; ph = 64u; break;
    case GS_PSM_T4:
        pw = 128u; ph = 128u; break;
    default:
        pw = 64u; ph = 32u; break;
    }
}

// Conservative set of 8 KiB pages touched by a rectangle of a buffer (block base, width in 64-pixel units).
PageSet pagesFor(uint32_t bpBlocks, uint32_t bw, uint8_t psm, const Rect &r)
{
    PageSet s;
    if (r.empty() || r.x1 < 0 || r.y1 < 0)
        return s;
    uint32_t pw, ph;
    pageDims(psm, pw, ph);
    const uint32_t ppr = std::max<uint32_t>(1u, (std::max<uint32_t>(bw, 1u) * 64u) / pw);
    const uint32_t base = bpBlocks >> 5;
    const bool spill = (bpBlocks & 31u) != 0u;
    const uint32_t px0 = static_cast<uint32_t>(std::max(r.x0, 0)) / pw, px1 = static_cast<uint32_t>(r.x1) / pw;
    const uint32_t py0 = static_cast<uint32_t>(std::max(r.y0, 0)) / ph, py1 = static_cast<uint32_t>(r.y1) / ph;
    for (uint32_t py = py0; py <= py1 && py < 4096u; ++py)
        for (uint32_t px = px0; px <= px1 && px < 64u; ++px)
        {
            const uint32_t p = base + py * ppr + px;
            s.set(p % kGsPages);
            if (spill)
                s.set((p + 1u) % kGsPages);
        }
    return s;
}

bool isRenderablePsm(uint8_t psm)
{
    switch (psm)
    {
    case GS_PSM_CT32: case GS_PSM_CT24: case GS_PSM_CT16: case GS_PSM_CT16S:
    case GS_PSM_Z32: case GS_PSM_Z24: case GS_PSM_Z16: case GS_PSM_Z16S:
        return true;
    default:
        return false;
    }
}

bool isFourBit(uint8_t psm) { return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH; }
bool isEightBit(uint8_t psm) { return psm == GS_PSM_T8 || psm == GS_PSM_T8H; }

uint32_t swizzleClutIndexCSM1(uint32_t index)
{
    return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
}

using ReadFn = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
ReadFn readFn(uint8_t psm)
{
    using namespace GSMem;
    switch (psm)
    {
    case GS_PSM_CT32: return ReadCT32;
    case GS_PSM_CT24: return ReadCT24;
    case GS_PSM_CT16: return ReadCT16;
    case GS_PSM_CT16S: return ReadCT16S;
    case GS_PSM_Z32: return ReadZ32;
    case GS_PSM_Z24: return ReadZ24;
    case GS_PSM_Z16: return ReadZ16;
    case GS_PSM_Z16S: return ReadZ16S;
    default: return ReadNull;
    }
}

struct DrawState
{
    uint32_t fbBp, fbBw, fbPsm, fbMask;
    uint32_t zBp, zPsm, flags, test;
    uint32_t alpha, scissor0, scissor1, fogColor;
    uint32_t primFlags, texBp, texBw, texPsm;
    uint32_t texWH, texMode, clampU, clampV;
    uint32_t clut, clutBase, texa, pad;
};
static_assert(sizeof(DrawState) == 96, "DrawState must match the shader layout");

constexpr uint32_t F_FGE = 1u, F_ABE = 2u, F_PABE = 4u, F_FBA = 8u, F_ZMASK = 16u, F_HASZ = 32u, F_HASFB = 64u;
constexpr uint32_t P_TME = 1u, P_FST = 2u, P_LINEAR = 4u, P_ZROUND = 8u;

struct GpuVertex
{
    float pos[4];
    float color[4];
    float tex[4];
    uint32_t draw;
};
static_assert(sizeof(GpuVertex) == 52, "unexpected vertex size");

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    VkDeviceSize size = 0;
    bool coherent = true;
};

// Upload marks a run of page uploads (indices into uploads) recorded between draws: the command
// buffer replays uploads and draws in the order they were recorded.
enum class Topo : uint8_t { Triangles, Lines, Points, Upload };

struct Run
{
    Topo topo;
    uint32_t first;
    uint32_t count;
};

template <typename F>
void forEachRun(const PageSet &pages, F &&f)
{
    uint32_t p = 0;
    while (p < kGsPages)
    {
        if (!pages.test(p))
        {
            ++p;
            continue;
        }
        uint32_t end = p;
        while (end + 1u < kGsPages && pages.test(end + 1u))
            ++end;
        f(p, end - p + 1u);
        p = end + 1u;
    }
}
} // namespace

struct GSVulkanBackend::Impl
{
    std::mutex mutex;
    GSCpuBackend cpu;
    uint8_t *vram = nullptr;
    uint32_t vramSize = 0;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipelines[3] = {};
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    std::string deviceName;

    GpuBuffer localMem, layout, cluts, staging, readback, vertices, drawStates;

    // Recording state for the current submission
    VkDeviceSize stagingUsed = 0;
    uint32_t vertexCount = 0;
    uint32_t drawCount = 0;
    uint32_t clutCount = 0;
    std::vector<VkBufferCopy> uploads;   // staging -> local memory
    std::vector<Run> runs;
    DrawState lastState{};
    bool haveLastState = false;

    // Local memory coherence, per 8 KiB page:
    //   stale - the CPU copy changed and the GPU copy has not been refreshed yet
    //   dirty - the GPU copy holds pixels the CPU copy does not have yet
    //   used  - draws recorded in the current submission read or write the page
    PageSet stale, dirty, used;

    std::array<uint16_t, 512> clut{};
    std::array<uint32_t, 2> clutCbp{};
    uint64_t clutVersion = 1;
    uint64_t clutVersionInRing = 0;
    uint32_t clutRingWord = 0;

    PageSet uploadPages;
    bool uploadActive = false;

    uint64_t statPrims = 0, statFlushes = 0, statPageUploads = 0, statPageDownloads = 0;
    // Why GPU submissions happen (PS2X_VK_STATS): 0 staging full, 1 CPU read (draw/CLUT/transfer),
    // 2 CPU write over GPU-drawn pages, 3 CPU write over pages queued draws use, 4 CLUT ring full,
    // 5 vertex/draw buffers full, 6 presentation read-back, 7 other.
    uint64_t statReason[8] = {};
    int flushReason = 7;

    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags want)
    {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want)
                return i;
        return UINT32_MAX;
    }

    bool createBuffer(GpuBuffer &b, VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible, bool hostCached = false)
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bi, nullptr, &b.buffer) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        uint32_t type = UINT32_MAX;
        if (hostVisible)
        {
            if (hostCached)
                type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (type == UINT32_MAX && hostCached)
                type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (type == UINT32_MAX)
                type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        else
        {
            type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type == UINT32_MAX)
                type = findMemoryType(req.memoryTypeBits, 0);
        }
        if (type == UINT32_MAX)
            return false;
        b.coherent = (memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &ai, nullptr, &b.memory) != VK_SUCCESS)
            return false;
        vkBindBufferMemory(device, b.buffer, b.memory, 0);
        if (hostVisible && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped) != VK_SUCCESS)
            return false;
        b.size = size;
        return true;
    }

    void destroyBuffer(GpuBuffer &b)
    {
        if (b.buffer)
            vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory)
            vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }

    bool init(std::string &why)
    {
        if (volkInitialize() != VK_SUCCESS)
        {
            why = "the Vulkan loader (vulkan-1) is not installed";
            return false;
        }
        if (volkGetInstanceVersion() < VK_API_VERSION_1_3)
        {
            why = "the Vulkan loader is older than 1.3";
            return false;
        }
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Suikoden III Recompiled";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
        {
            why = "vkCreateInstance failed";
            return false;
        }
        volkLoadInstanceOnly(instance);

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        int bestScore = -1;
        std::string rejected;
        const char *wanted = std::getenv("PS2X_VK_DEVICE");
        for (VkPhysicalDevice pd : devices)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.apiVersion < VK_API_VERSION_1_3)
            {
                rejected += std::string(props.deviceName) + " (Vulkan < 1.3); ";
                continue;
            }
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
            bool interlockExt = false;
            for (auto &e : exts)
                if (std::strcmp(e.extensionName, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) == 0)
                    interlockExt = true;
            VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT il{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
            VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            f13.pNext = interlockExt ? &il : nullptr;
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &f13;
            vkGetPhysicalDeviceFeatures2(pd, &f2);
            if (!interlockExt || !il.fragmentShaderPixelInterlock || !f13.dynamicRendering || !f2.features.fragmentStoresAndAtomics)
            {
                rejected += std::string(props.deviceName) + " (no fragment shader pixel interlock); ";
                continue;
            }
            uint32_t qCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
            std::vector<VkQueueFamilyProperties> qs(qCount);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qs.data());
            int family = -1;
            for (uint32_t i = 0; i < qCount; ++i)
                if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    family = static_cast<int>(i);
                    break;
                }
            if (family < 0)
                continue;
            int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
            if (wanted && std::strstr(props.deviceName, wanted))
                score = 100;
            if (score > bestScore)
            {
                bestScore = score;
                physical = pd;
                queueFamily = static_cast<uint32_t>(family);
                deviceName = props.deviceName;
            }
        }
        if (!physical)
        {
            why = rejected.empty() ? "no Vulkan device found" : "no suitable Vulkan device: " + rejected;
            return false;
        }
        vkGetPhysicalDeviceMemoryProperties(physical, &memProps);

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queueFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT il{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
        il.fragmentShaderPixelInterlock = VK_TRUE;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_FALSE;
        f13.pNext = &il;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.features.fragmentStoresAndAtomics = VK_TRUE;
        f2.pNext = &f13;
        const char *exts[] = {VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts;
        if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS)
        {
            why = "vkCreateDevice failed on " + deviceName;
            return false;
        }
        volkLoadDevice(device);
        vkGetDeviceQueue(device, queueFamily, 0, &queue);

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = queueFamily;
        vkCreateCommandPool(device, &cpi, nullptr, &cmdPool);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cmdPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cai, &cmd);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(device, &fci, nullptr, &fence);

        const VkBufferUsageFlags xfer = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!createBuffer(localMem, kLocalMemoryBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | xfer, false) ||
            !createBuffer(layout, sizeof(k_gsLayoutTables), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true) ||
            !createBuffer(cluts, kClutBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true) ||
            !createBuffer(staging, kStagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true) ||
            !createBuffer(readback, kLocalMemoryBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, true) ||
            !createBuffer(vertices, kVertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true) ||
            !createBuffer(drawStates, kDrawStateBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true))
        {
            why = "could not allocate GPU buffers on " + deviceName;
            return false;
        }
        std::memcpy(layout.mapped, k_gsLayoutTables, sizeof(k_gsLayoutTables));

        VkDescriptorSetLayoutBinding binds[4]{};
        for (uint32_t i = 0; i < 4; ++i)
        {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl.bindingCount = 4;
        dsl.pBindings = binds;
        vkCreateDescriptorSetLayout(device, &dsl, nullptr, &setLayout);
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        vkCreateDescriptorPool(device, &dpi, nullptr, &descPool);
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = descPool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &setLayout;
        vkAllocateDescriptorSets(device, &dai, &descSet);
        VkDescriptorBufferInfo bufInfo[4] = {{localMem.buffer, 0, VK_WHOLE_SIZE},
                                             {layout.buffer, 0, VK_WHOLE_SIZE},
                                             {drawStates.buffer, 0, VK_WHOLE_SIZE},
                                             {cluts.buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufInfo[i];
        }
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout;
        vkCreatePipelineLayout(device, &pli, nullptr, &pipeLayout);

        auto makeModule = [&](const uint32_t *code, size_t bytes) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = bytes;
            smi.pCode = code;
            VkShaderModule m = VK_NULL_HANDLE;
            vkCreateShaderModule(device, &smi, nullptr, &m);
            return m;
        };
        vs = makeModule(k_gs_draw_vert, sizeof(k_gs_draw_vert));
        fs = makeModule(k_gs_draw_frag, sizeof(k_gs_draw_frag));
        if (!vs || !fs)
        {
            why = "shader module creation failed";
            return false;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";
        VkVertexInputBindingDescription vb{0, sizeof(GpuVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription va[4] = {
            {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(GpuVertex, pos)},
            {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(GpuVertex, color)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(GpuVertex, tex)},
            {3, 0, VK_FORMAT_R32_UINT, offsetof(GpuVertex, draw)}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vb;
        vi.vertexAttributeDescriptionCount = 4;
        vi.pVertexAttributeDescriptions = va;
        VkViewport viewport{0, 0, float(kRenderArea), float(kRenderArea), 0, 1};
        VkRect2D scissor{{0, 0}, {kRenderArea, kRenderArea}};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.pViewports = &viewport;
        vps.scissorCount = 1;
        vps.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        VkPipelineRenderingCreateInfo pri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        const VkPrimitiveTopology topos[3] = {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
        for (int i = 0; i < 3; ++i)
        {
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = topos[i];
            VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gpi.pNext = &pri;
            gpi.stageCount = 2;
            gpi.pStages = stages;
            gpi.pVertexInputState = &vi;
            gpi.pInputAssemblyState = &ia;
            gpi.pViewportState = &vps;
            gpi.pRasterizationState = &rs;
            gpi.pMultisampleState = &ms;
            gpi.pColorBlendState = &cb;
            gpi.layout = pipeLayout;
            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipelines[i]) != VK_SUCCESS)
            {
                why = "pipeline creation failed";
                return false;
            }
        }
        return true;
    }


    void shutdown()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            for (VkPipeline p : pipelines)
                if (p)
                    vkDestroyPipeline(device, p, nullptr);
            if (vs) vkDestroyShaderModule(device, vs, nullptr);
            if (fs) vkDestroyShaderModule(device, fs, nullptr);
            if (pipeLayout) vkDestroyPipelineLayout(device, pipeLayout, nullptr);
            if (descPool) vkDestroyDescriptorPool(device, descPool, nullptr);
            if (setLayout) vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            for (GpuBuffer *b : {&localMem, &layout, &cluts, &staging, &readback, &vertices, &drawStates})
                destroyBuffer(*b);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance)
        {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }

    // ---------------------------------------------------------------- GPU submission
    // Runs the recorded page uploads and draws, then copies the requested pages of local memory back.
    void execute(const PageSet &download)
    {
        if (runs.empty() && uploads.empty() && download.none())
            return;
        ++statFlushes;
        ++statReason[flushReason & 7];
        flushReason = 7;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        auto barrier = [&](VkPipelineStageFlags src, VkAccessFlags srcA, VkPipelineStageFlags dst, VkAccessFlags dstA) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = srcA;
            mb.dstAccessMask = dstA;
            vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
        };
        barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        if (!runs.empty())
        {
            // Uploads and draws in recorded order: a CPU write to pages that queued draws read no
            // longer needs the GPU to finish first, since its upload runs after those draws.
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {kRenderArea, kRenderArea}};
            ri.layerCount = 1;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &descSet, 0, nullptr);
            VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, &zero);
            int bound = -1;
            bool rendering = false;
            bool drawnSinceUpload = false;
            for (const Run &r : runs)
            {
                if (r.topo == Topo::Upload)
                {
                    if (rendering)
                    {
                        vkCmdEndRendering(cmd);
                        rendering = false;
                    }
                    if (drawnSinceUpload)
                        barrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                    vkCmdCopyBuffer(cmd, staging.buffer, localMem.buffer, r.count, uploads.data() + r.first);
                    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
                    drawnSinceUpload = false;
                    continue;
                }
                if (!rendering)
                {
                    vkCmdBeginRendering(cmd, &ri);
                    rendering = true;
                }
                const int p = static_cast<int>(r.topo);
                if (p != bound)
                {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[p]);
                    bound = p;
                }
                vkCmdDraw(cmd, r.count, 1, r.first, 0);
                drawnSinceUpload = true;
            }
            if (rendering)
                vkCmdEndRendering(cmd);
        }
        std::vector<VkBufferCopy> regions;
        forEachRun(download, [&](uint32_t first, uint32_t count) {
            regions.push_back({VkDeviceSize(first) * kPageBytes, VkDeviceSize(first) * kPageBytes, VkDeviceSize(count) * kPageBytes});
        });
        if (!regions.empty())
        {
            barrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            vkCmdCopyBuffer(cmd, localMem.buffer, readback.buffer, static_cast<uint32_t>(regions.size()), regions.data());
            barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        }
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &fence);
        if (!regions.empty())
        {
            if (!readback.coherent)
            {
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = readback.memory;
                range.size = VK_WHOLE_SIZE;
                vkInvalidateMappedMemoryRanges(device, 1, &range);
            }
            for (const VkBufferCopy &r : regions)
                std::memcpy(vram + r.dstOffset, static_cast<uint8_t *>(readback.mapped) + r.dstOffset, static_cast<size_t>(r.size));
            statPageDownloads += download.count();
        }
        uploads.clear();
        runs.clear();
        vertexCount = 0;
        drawCount = 0;
        clutCount = 0;
        clutVersionInRing = 0;
        stagingUsed = 0;
        haveLastState = false;
        used.reset();
    }

    void flushGpu() { execute(PageSet{}); }

    // ---------------------------------------------------------------- local memory coherence
    // Brings these pages of the GPU copy up to date with CPU-side changes.
    void refreshPages(const PageSet &pages)
    {
        const PageSet need = pages & stale;
        if (need.none())
            return;
        if (stagingUsed + VkDeviceSize(need.count()) * kPageBytes > staging.size)
        {
            flushReason = 0;
            flushGpu();
        }
        const uint32_t firstUpload = static_cast<uint32_t>(uploads.size());
        forEachRun(need, [&](uint32_t first, uint32_t count) {
            const VkDeviceSize bytes = VkDeviceSize(count) * kPageBytes;
            std::memcpy(static_cast<uint8_t *>(staging.mapped) + stagingUsed, vram + VkDeviceSize(first) * kPageBytes, static_cast<size_t>(bytes));
            uploads.push_back({stagingUsed, VkDeviceSize(first) * kPageBytes, bytes});
            stagingUsed += bytes;
        });
        // Its own run (never merged), so a page appears at most once per copy command.
        runs.push_back({Topo::Upload, firstUpload, static_cast<uint32_t>(uploads.size()) - firstUpload});
        statPageUploads += need.count();
        stale &= ~need;
    }

    // The CPU is about to read these pages of local memory.
    void beforeCpuRead(const PageSet &pages)
    {
        const PageSet need = pages & dirty;
        if (need.none())
            return;
        if (flushReason == 7)
            flushReason = 1;
        execute(need);
        dirty &= ~need;
    }

    // The CPU is about to write these pages: pull GPU-drawn pixels back first (the write may be
    // partial) and finish recorded draws that still read the old contents.
    void beforeCpuWrite(const PageSet &pages)
    {
        const PageSet need = pages & dirty;
        if (need.any())
        {
            flushReason = 2;
            execute(need);
            dirty &= ~need;
        }
        // Pages queued draws only read need no flush: the new contents are uploaded in order,
        // after those draws (see execute).
    }

    void afterCpuWrite(const PageSet &pages) { stale |= pages; }

    // ---------------------------------------------------------------- CLUT
    void loadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
    {
        if (!isFourBit(tex0.psm) && !isEightBit(tex0.psm))
            return;
        switch (tex0.cld)
        {
        case 1u: break;
        case 2u: clutCbp[0] = tex0.cbp; break;
        case 3u: clutCbp[1] = tex0.cbp; break;
        case 4u:
            if (clutCbp[0] == tex0.cbp)
                return;
            clutCbp[0] = tex0.cbp;
            break;
        case 5u:
            if (clutCbp[1] == tex0.cbp)
                return;
            clutCbp[1] = tex0.cbp;
            break;
        default:
            return;
        }
        const bool fourBit = isFourBit(tex0.psm);
        const bool sixteen = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
        const bool thirtyTwo = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
        if (!sixteen && !thirtyTwo)
            return;
        const uint32_t cbw = tex0.csm == 0u ? 1u : std::max<uint32_t>(texclut.cbw, 1u);
        const Rect src = tex0.csm == 0u ? Rect{0, 0, 15, 15}
                                        : Rect{int(texclut.cou) * 16, int(texclut.cov), int(texclut.cou) * 16 + 255, int(texclut.cov)};
        beforeCpuRead(pagesFor(tex0.cbp, cbw, tex0.cpsm, src));

        const uint32_t entryCount = fourBit ? 16u : 256u;
        const uint32_t csaMask = sixteen ? 0x1Fu : 0x0Fu;
        const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;
        const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwo && !fourBit;
        const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;
        ReadFn read = readFn(tex0.cpsm);
        for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
        {
            uint32_t sx, sy, sw = 1u;
            if (tex0.csm == 0u)
            {
                const uint32_t si = swizzleClutIndexCSM1(entry);
                sx = si & 0x0Fu;
                sy = si >> 4u;
            }
            else
            {
                sw = cbw;
                sx = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
                sy = texclut.cov;
            }
            const uint32_t raw = read(vram, tex0.cbp, sw, sx, sy);
            const uint32_t dst = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteen ? 0x1FFu : 0x0FFu);
            if (sixteen)
                clut[dst] = static_cast<uint16_t>(raw);
            else
            {
                clut[dst] = static_cast<uint16_t>(raw & 0xFFFFu);
                clut[dst + 256u] = static_cast<uint16_t>(raw >> 16u);
            }
        }
        ++clutVersion;
    }

    uint32_t currentClutWord()
    {
        if (clutVersionInRing != clutVersion)
        {
            if (VkDeviceSize(clutCount + 1u) * sizeof(clut) > cluts.size)
            {
                flushReason = 4;
                flushGpu();
            }
            clutRingWord = clutCount * static_cast<uint32_t>(sizeof(clut) / 4u);
            std::memcpy(static_cast<uint8_t *>(cluts.mapped) + VkDeviceSize(clutCount) * sizeof(clut), clut.data(), sizeof(clut));
            ++clutCount;
            clutVersionInRing = clutVersion;
        }
        return clutRingWord;
    }

    // ---------------------------------------------------------------- primitives
    void submit(const GSPrimitiveBatch &b)
    {
        const GSDrawState &st = b.state;
        const GSContext &ctx = st.context;
        uint32_t needed = 1;
        Topo topo = Topo::Points;
        switch (st.prim.type)
        {
        case GS_PRIM_TRIANGLE: case GS_PRIM_TRISTRIP: case GS_PRIM_TRIFAN:
            needed = 3; topo = Topo::Triangles; break;
        case GS_PRIM_SPRITE:
            needed = 2; topo = Topo::Triangles; break;
        case GS_PRIM_LINE: case GS_PRIM_LINESTRIP:
            needed = 2; topo = Topo::Lines; break;
        default:
            needed = 1; topo = Topo::Points; break;
        }
        if (b.vertexCount < needed || !isRenderablePsm(ctx.frame.psm))
            return;
        ++statPrims;

        const uint32_t emit = st.prim.type == GS_PRIM_SPRITE ? 6u : needed;
        if (VkDeviceSize(vertexCount + emit) * sizeof(GpuVertex) > vertices.size ||
            VkDeviceSize(drawCount + 1u) * sizeof(DrawState) > drawStates.size)
        {
            flushReason = 5;
            flushGpu();
        }

        // Pixel bounds this primitive can write (GS samples at integer coordinates; right and bottom
        // edges are exclusive for triangles and sprites).
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
        for (uint32_t i = 0; i < needed; ++i)
        {
            const float x = b.vertices[i].x - ofx, y = b.vertices[i].y - ofy;
            minX = std::min(minX, x); maxX = std::max(maxX, x);
            minY = std::min(minY, y); maxY = std::max(maxY, y);
        }
        Rect bbox;
        if (topo == Topo::Triangles)
            bbox = Rect{int(std::ceil(minX)), int(std::ceil(minY)), int(std::ceil(maxX)) - 1, int(std::ceil(maxY)) - 1};
        else
            bbox = Rect{int(std::floor(minX)) - 1, int(std::floor(minY)) - 1, int(std::ceil(maxX)) + 1, int(std::ceil(maxY)) + 1};
        bbox = bbox.clip(Rect{ctx.scissor.x0, ctx.scissor.y0, ctx.scissor.x1, ctx.scissor.y1})
                   .clip(Rect{0, 0, int(kRenderArea) - 1, int(kRenderArea) - 1});
        if (bbox.empty())
            return;

        const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
        const uint32_t ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
        const bool hasZ = (ztst >= 2u || !ctx.zbuf.zmask) && isRenderablePsm(ctx.zbuf.psm);
        const bool tme = st.prim.tme;

        // Pages this primitive touches, brought up to date on the GPU before it runs.
        const PageSet fbPages = pagesFor(ctx.frame.fbp << 5, fbw, ctx.frame.psm, bbox);
        const PageSet zPages = hasZ ? pagesFor(ctx.zbuf.zbp << 5, fbw, ctx.zbuf.psm, bbox) : PageSet{};
        PageSet texPages;
        uint32_t texW = 1, texH = 1;
        if (tme)
        {
            texW = std::clamp<uint32_t>(st.textureWidth, 1u, 1024u);
            texH = std::clamp<uint32_t>(st.textureHeight, 1u, 1024u);
            texPages = pagesFor(ctx.tex0.tbp0, std::max<uint32_t>(ctx.tex0.tbw, 1u), ctx.tex0.psm, Rect{0, 0, int(texW) - 1, int(texH) - 1});
        }
        const PageSet touched = fbPages | zPages | texPages;
        refreshPages(touched);
        const uint32_t clutWord = (tme && (isFourBit(ctx.tex0.psm) || isEightBit(ctx.tex0.psm))) ? currentClutWord() : 0u;

        DrawState ds{};
        ds.fbBp = ctx.frame.fbp << 5;
        ds.fbBw = fbw;
        ds.fbPsm = ctx.frame.psm;
        ds.fbMask = ctx.frame.fbmsk;
        ds.flags = F_HASFB;
        if (hasZ)
        {
            ds.zBp = ctx.zbuf.zbp << 5;
            ds.zPsm = ctx.zbuf.psm;
            ds.flags |= F_HASZ;
        }
        if (st.prim.fge) ds.flags |= F_FGE;
        if (st.prim.abe) ds.flags |= F_ABE;
        if (st.pabe) ds.flags |= F_PABE;
        if (ctx.fba & 1u) ds.flags |= F_FBA;
        if (ctx.zbuf.zmask) ds.flags |= F_ZMASK;
        ds.test = static_cast<uint32_t>(ctx.test);
        ds.alpha = static_cast<uint32_t>(ctx.alpha & 0xFFu) | (static_cast<uint32_t>((ctx.alpha >> 32) & 0xFFu) << 8);
        ds.scissor0 = uint32_t(ctx.scissor.x0) | (uint32_t(ctx.scissor.y0) << 16);
        ds.scissor1 = uint32_t(ctx.scissor.x1) | (uint32_t(ctx.scissor.y1) << 16);
        ds.fogColor = uint32_t(st.fogR) | (uint32_t(st.fogG) << 8) | (uint32_t(st.fogB) << 16);
        if (tme)
        {
            ds.primFlags |= P_TME;
            if (st.prim.fst) ds.primFlags |= P_FST;
            if (st.linearFilter) ds.primFlags |= P_LINEAR;
            ds.texBp = ctx.tex0.tbp0;
            ds.texBw = std::max<uint32_t>(ctx.tex0.tbw, 1u);
            ds.texPsm = ctx.tex0.psm;
            ds.texWH = texW | (texH << 16);
            const uint64_t clamp = ctx.clamp;
            ds.texMode = (ctx.tex0.tfx & 3u) | ((ctx.tex0.tcc & 1u) << 2) | (uint32_t(clamp & 3u) << 4) | (uint32_t((clamp >> 2) & 3u) << 6);
            ds.clampU = uint32_t((clamp >> 4) & 0x3FFu) | (uint32_t((clamp >> 14) & 0x3FFu) << 16);
            ds.clampV = uint32_t((clamp >> 24) & 0x3FFu) | (uint32_t((clamp >> 34) & 0x3FFu) << 16);
            ds.clut = uint32_t(ctx.tex0.cpsm) | (uint32_t(ctx.tex0.csm & 1u) << 8) | (uint32_t(ctx.tex0.csa & 0x1Fu) << 16);
            ds.clutBase = clutWord;
            ds.texa = uint32_t(st.texa.ta0) | (uint32_t(st.texa.ta1) << 8) | (uint32_t(st.texa.aem ? 1u : 0u) << 16);
        }
        if (topo == Topo::Triangles && st.prim.type != GS_PRIM_SPRITE)
            ds.primFlags |= P_ZROUND;

        if (!haveLastState || std::memcmp(&ds, &lastState, sizeof(ds)) != 0)
        {
            std::memcpy(static_cast<uint8_t *>(drawStates.mapped) + size_t(drawCount) * sizeof(DrawState), &ds, sizeof(ds));
            lastState = ds;
            haveLastState = true;
            ++drawCount;
        }
        const uint32_t drawIndex = drawCount - 1u;

        GpuVertex *out = static_cast<GpuVertex *>(vertices.mapped) + vertexCount;
        auto fill = [&](GpuVertex &o, const GSVertex &v, const GSVertex &colourSrc, float x, float y, float s, float t, float q) {
            o.pos[0] = x;
            o.pos[1] = y;
            o.pos[2] = static_cast<float>(v.z);
            o.pos[3] = static_cast<float>(v.fog);
            o.color[0] = colourSrc.r;
            o.color[1] = colourSrc.g;
            o.color[2] = colourSrc.b;
            o.color[3] = colourSrc.a;
            o.tex[0] = s;
            o.tex[1] = t;
            o.tex[2] = q;
            o.tex[3] = 0.0f;
            o.draw = drawIndex;
        };
        auto texOf = [&](const GSVertex &v, float &s, float &t, float &q) {
            if (st.prim.fst)
            {
                s = static_cast<float>(v.u);
                t = static_cast<float>(v.v);
                q = 1.0f;
            }
            else
            {
                s = v.s;
                t = v.t;
                q = v.q;
            }
        };
        if (st.prim.type == GS_PRIM_SPRITE)
        {
            const GSVertex &a = b.vertices[0], &c = b.vertices[1];
            const float x0 = a.x - ofx, y0 = a.y - ofy, x1 = c.x - ofx, y1 = c.y - ofy;
            float s0, t0, q0, s1, t1, q1;
            texOf(a, s0, t0, q0);
            texOf(c, s1, t1, q1);
            if (!st.prim.fst)
            {
                // Sprites interpolate texture coordinates linearly between the two corners.
                const float iq0 = std::fabs(q0) > 1e-8f ? 1.0f / q0 : 1.0f, iq1 = std::fabs(q1) > 1e-8f ? 1.0f / q1 : 1.0f;
                s0 *= iq0; t0 *= iq0; s1 *= iq1; t1 *= iq1;
            }
            fill(out[0], c, c, x0, y0, s0, t0, 1.0f);
            fill(out[1], c, c, x1, y0, s1, t0, 1.0f);
            fill(out[2], c, c, x0, y1, s0, t1, 1.0f);
            fill(out[3], c, c, x1, y0, s1, t0, 1.0f);
            fill(out[4], c, c, x1, y1, s1, t1, 1.0f);
            fill(out[5], c, c, x0, y1, s0, t1, 1.0f);
        }
        else
        {
            const GSVertex &provoking = b.vertices[needed - 1u];
            for (uint32_t i = 0; i < needed; ++i)
            {
                const GSVertex &v = b.vertices[i];
                float s, t, q;
                texOf(v, s, t, q);
                fill(out[i], v, st.prim.iip ? v : provoking, v.x - ofx, v.y - ofy, s, t, q);
            }
        }
        if (!runs.empty() && runs.back().topo == topo && runs.back().first + runs.back().count == vertexCount)
            runs.back().count += emit;
        else
            runs.push_back({topo, vertexCount, emit});
        vertexCount += emit;

        used |= touched;
        dirty |= fbPages;
        if (hasZ && !ctx.zbuf.zmask)
            dirty |= zPages;
    }

    PageSet transferDestPages(const GSTransferCommand &c) const
    {
        const Rect r{c.trxpos.dsax, c.trxpos.dsay, c.trxpos.dsax + int(c.trxreg.rrw) - 1, c.trxpos.dsay + int(c.trxreg.rrh) - 1};
        return pagesFor(c.bitbltbuf.dbp, std::max<uint32_t>(c.bitbltbuf.dbw, 1u), c.bitbltbuf.dpsm, r);
    }

    PageSet transferSourcePages(const GSTransferCommand &c) const
    {
        const Rect r{c.trxpos.ssax, c.trxpos.ssay, c.trxpos.ssax + int(c.trxreg.rrw) - 1, c.trxpos.ssay + int(c.trxreg.rrh) - 1};
        return pagesFor(c.bitbltbuf.sbp, std::max<uint32_t>(c.bitbltbuf.sbw, 1u), c.bitbltbuf.spsm, r);
    }

    void report()
    {
        static const bool verbose = std::getenv("PS2X_VK_STATS") != nullptr;
        if (!verbose)
            return;
        static uint64_t lastPrims = 0;
        if (statPrims - lastPrims < 200000u)
            return;
        lastPrims = statPrims;
        std::fprintf(stderr, "[gs-vk] prims=%llu submits=%llu pages up=%llu down=%llu reasons=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", (unsigned long long)statPrims,
                     (unsigned long long)statFlushes, (unsigned long long)statPageUploads, (unsigned long long)statPageDownloads,
                     (unsigned long long)statReason[0], (unsigned long long)statReason[1], (unsigned long long)statReason[2],
                     (unsigned long long)statReason[3], (unsigned long long)statReason[4], (unsigned long long)statReason[5],
                     (unsigned long long)statReason[6], (unsigned long long)statReason[7]);
    }

    void resetState()
    {
        stale.set();
        dirty.reset();
        used.reset();
        clut.fill(0);
        clutCbp.fill(0);
        ++clutVersion;
        uploadActive = false;
    }
};

// -------------------------------------------------------------------- public interface

GSVulkanBackend::GSVulkanBackend() : m(std::make_unique<Impl>()) {}

GSVulkanBackend::~GSVulkanBackend()
{
    if (m)
        m->shutdown();
}

std::unique_ptr<GSRasterBackend> GSVulkanBackend::TryCreate(std::string &whyNot)
{
    std::unique_ptr<GSVulkanBackend> backend(new GSVulkanBackend());
    if (!backend->m->init(whyNot))
        return nullptr;
    std::fprintf(stderr, "[gs-vk] Vulkan renderer on %s\n", backend->m->deviceName.c_str());
    return backend;
}

void GSVulkanBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    m->cpu.Initialize(vram, vramSize);
    std::lock_guard<std::mutex> lock(m->mutex);
    m->vram = vram;
    m->vramSize = vramSize;
    m->resetState();
}

void GSVulkanBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->vram)
    {
        m->beforeCpuRead(m->dirty);
        m->flushGpu();
    }
    m->resetState();
    m->cpu.Reset();
}

void GSVulkanBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (!m->vram)
        return;
    m->submit(batch);
    m->report();
}

void GSVulkanBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->vram)
        m->loadClut(tex0, texclut);
}

void GSVulkanBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    m->uploadActive = false;
    if (m->vram)
    {
        switch (command.direction)
        {
        case 0u:
            m->uploadPages = m->transferDestPages(command);
            m->beforeCpuWrite(m->uploadPages);
            m->uploadActive = true;
            break;
        case 1u:
            m->beforeCpuRead(m->transferSourcePages(command));
            break;
        case 2u:
        {
            const PageSet dst = m->transferDestPages(command);
            m->beforeCpuRead(m->transferSourcePages(command));
            m->beforeCpuWrite(dst);
            m->cpu.BeginTransfer(command);
            m->afterCpuWrite(dst);
            return;
        }
        default:
            break;
        }
    }
    m->cpu.BeginTransfer(command);
}

void GSVulkanBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->uploadActive)
    {
        // Draws between chunks may have used these pages since the transfer started.
        m->beforeCpuWrite(m->uploadPages);
        m->cpu.UploadImage(data, sizeBytes);
        m->afterCpuWrite(m->uploadPages);
        return;
    }
    m->cpu.UploadImage(data, sizeBytes);
}

void GSVulkanBackend::Flush()
{
}

void GSVulkanBackend::TextureFlush()
{
}

void GSVulkanBackend::Sync(GSSyncReason reason)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->vram && (reason == GSSyncReason::Reset || reason == GSSyncReason::DebugReadback))
        m->beforeCpuRead(m->dirty);
}

PresentationFrame GSVulkanBackend::Present(const GSPresentationRequest &request)
{
    // s3recomp: only the GPU write-back and a copy of local memory happen under the backend lock;
    // the display conversion runs on that copy afterwards. Converting under the lock kept the game
    // thread waiting (its draws and uploads take the same lock) for most of every present.
    thread_local std::vector<uint8_t> snapshot;
    {
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->vram)
    {
        // Only the buffers the display circuits (or the preferred source) read need writing back;
        // the frame the game is still drawing stays on the GPU.
        PageSet pages;
        uint32_t displayRows = 0u;
        auto addFrame = [&](uint32_t fbp, uint32_t fbw, uint8_t psm, uint32_t originY, uint32_t rows) {
            if (fbw != 0u)
                pages |= pagesFor(fbp << 5, fbw, psm, Rect{0, int(originY), int(fbw * 64u) - 1, int(originY + rows) - 1});
        };
        auto addCircuit = [&](uint64_t dispfb, uint64_t display) {
            uint32_t rows = static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u;
            if (rows < 64u)
                rows = 448u;
            displayRows = std::max(displayRows, rows);
            addFrame(uint32_t(dispfb & 0x1FFu), uint32_t((dispfb >> 9) & 0x3Fu), uint8_t((dispfb >> 15) & 0x1Fu),
                     uint32_t((dispfb >> 43) & 0x7FFu), rows);
        };
        if (request.pmode & 1u)
            addCircuit(request.dispfb1, request.display1);
        if (request.pmode & 2u)
            addCircuit(request.dispfb2, request.display2);
        if (request.hasPreferredSource)
            addFrame(request.preferredSource.fbp, request.preferredSource.fbw, request.preferredSource.psm, 0u,
                     displayRows ? displayRows : 448u);
        m->flushReason = 6;
        m->beforeCpuRead(pages.none() ? m->dirty : pages);
        m->flushReason = 7;
    }
    m->cpu.SnapshotVram(snapshot);
    }
    if (snapshot.empty())
        return {};
    thread_local GSCpuBackend presenter;
    presenter.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return presenter.Present(request);
}

bool GSVulkanBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (!m->vram)
        return false;
    if (context.frame.fbw != 0u && isRenderablePsm(context.frame.psm))
    {
        // s3recomp: clear on the GPU, as a flat sprite over the scissor rectangle that leaves Z alone.
        // Clearing in local memory on the CPU meant reading the GPU-drawn frame back first and
        // uploading it again afterwards, every frame.
        GSPrimitiveBatch batch{};
        batch.vertexCount = 2;
        batch.state.context = context;
        batch.state.context.test = (1ull << 16) | (1ull << 17); // ZTE on, ZTST always; no alpha/dest-alpha test
        batch.state.context.zbuf.zmask = true;                  // no Z write
        batch.state.prim.type = GS_PRIM_SPRITE;
        const float ofx = static_cast<float>(context.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(context.xyoffset.ofy >> 4);
        GSVertex corner{};
        corner.r = static_cast<uint8_t>(rgba);
        corner.g = static_cast<uint8_t>(rgba >> 8u);
        corner.b = static_cast<uint8_t>(rgba >> 16u);
        corner.a = static_cast<uint8_t>(rgba >> 24u);
        batch.vertices[0] = corner;
        batch.vertices[0].x = static_cast<float>(context.scissor.x0) + ofx;
        batch.vertices[0].y = static_cast<float>(context.scissor.y0) + ofy;
        batch.vertices[1] = corner;
        batch.vertices[1].x = static_cast<float>(std::max(context.scissor.x0, context.scissor.x1) + 1) + ofx;
        batch.vertices[1].y = static_cast<float>(std::max(context.scissor.y0, context.scissor.y1) + 1) + ofy;
        m->submit(batch);
        return true;
    }
    const Rect r{context.scissor.x0, context.scissor.y0, std::max(context.scissor.x0, context.scissor.x1), std::max(context.scissor.y0, context.scissor.y1)};
    const PageSet pages = pagesFor(context.frame.fbp << 5, std::max<uint32_t>(context.frame.fbw, 1u), context.frame.psm, r);
    m->beforeCpuWrite(pages);
    const bool ok = m->cpu.ClearFramebuffer(context, rgba);
    m->afterCpuWrite(pages);
    return ok;
}

uint32_t GSVulkanBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    return m->cpu.ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSVulkanBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->vram)
        m->beforeCpuRead(pagesFor(base, std::max<uint32_t>(bw, 1u), uint8_t(psm), Rect{int(x), int(y), int(x), int(y)}));
    return m->cpu.ReadVram(psm, base, bw, x, y);
}

void GSVulkanBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (!m->vram)
        return;
    const PageSet pages = pagesFor(base, std::max<uint32_t>(bw, 1u), uint8_t(psm), Rect{int(x), int(y), int(x), int(y)});
    m->beforeCpuWrite(pages);
    m->cpu.WriteVram(psm, base, bw, x, y, value);
    m->afterCpuWrite(pages);
}

void GSVulkanBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->vram)
        m->beforeCpuRead(m->dirty);
    m->cpu.SnapshotVram(out);
}

GSTransferSnapshot GSVulkanBackend::GetTransferSnapshot() const
{
    return m->cpu.GetTransferSnapshot();
}
