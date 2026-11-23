#pragma once

// s3recomp: Vulkan GS raster backend.
//
// The GPU keeps a copy of GS local memory (4 MiB) in its native swizzled layout. Primitives are
// rasterised by the GPU; a fragment shader with pixel interlock applies the GS per-pixel pipeline
// (Z test, alpha test / AFAIL, destination alpha test, blending, FBA, FBMSK) and reads textures and
// writes frame/depth pixels with the GS addressing rules, so render-to-texture needs no copies and
// textures need no CPU decoding.
//
// The frontend's 4 MiB copy stays the authority for everything the GPU does not draw. The two copies
// are kept coherent per 8 KiB page: CPU writes (image transfers, clears) mark pages for upload before
// the next draw that touches them; pages the GPU drew are copied back when the CPU side needs them
// (presentation, local->host transfers, CLUT loads, debug snapshots). Image transfers, CLUT handling
// and presentation reuse GSCpuBackend.

#include "runtime/gs/gs_backend.h"

#include <memory>
#include <string>

class GSVulkanBackend final : public GSRasterBackend
{
public:
    // Returns nullptr (and the reason) when no suitable Vulkan device is available.
    static std::unique_ptr<GSRasterBackend> TryCreate(std::string &whyNot);
    ~GSVulkanBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    struct Impl;

private:
    GSVulkanBackend();
    std::unique_ptr<Impl> m;
};
