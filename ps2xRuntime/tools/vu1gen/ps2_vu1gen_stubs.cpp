// ps2_vu1gen links the VU1 fast core only to decode instructions. These stand in for the runtime
// pieces the core refers to but the generator never calls.
#include "runtime/ps2_memory.h"
#include "runtime/gs/gs_frontend.h"
#include <atomic>
namespace s3perf
{
    std::atomic<uint64_t> vu1Ns{0}, vu1Cycles{0}, vu1NativeCycles{0};
    thread_local bool inVu1 = false;
}
void PS2Memory::submitGifPacket(GifPathId, const uint8_t *, uint32_t, bool, bool) {}
void GS::processGIFPacket(const uint8_t *, uint32_t) {}
