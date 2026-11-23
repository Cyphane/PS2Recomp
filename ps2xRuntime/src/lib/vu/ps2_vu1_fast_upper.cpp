#include "ps2_vu1_fast_upper_inl.h"

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
void VU1Fast::execUpper(uint32_t instr)
{
    if (fmacSimdInl(instr) || miscSimdInl(instr))
        return;
    execUpperScalarInl(instr);
}

void VU1Fast::execUpperScalar(uint32_t instr)
{
    execUpperScalarInl(instr);
}
