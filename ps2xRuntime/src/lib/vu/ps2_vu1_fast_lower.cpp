#include "ps2_vu1_fast_lower_inl.h"

// ============================================================================
// Lower instructions
// ============================================================================
void VU1Fast::execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr)
{
    execLowerInl(instr, vuData, dataSize, gs, memory, upperInstr);
}
