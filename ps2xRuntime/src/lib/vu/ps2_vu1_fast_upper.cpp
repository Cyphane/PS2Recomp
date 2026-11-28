#include "ps2_vu1_fast_upper_inl.h"
#include <xmmintrin.h>

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

// s3recomp: the exact path for an FMAC in a turbo segment. It runs without denormals-are-zero, like
// execUpperInlDaz's exact path, and hands its flag results back through m_cap* (see flagsAtIssue).
void VU1Fast::turboUpperSlow(uint32_t instr)
{
    m_captureFlags = true;
    m_capValid = false;
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr & ~0x0040u);
    execUpperScalarInl(instr);
    _mm_setcsr(csr);
    m_captureFlags = false;
}
