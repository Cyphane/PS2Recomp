#version 450
#extension GL_ARB_fragment_shader_interlock : require
// s3recomp Vulkan GS fragment stage.
//
// The GPU holds a copy of GS local memory (4 MiB) in its native swizzled layout. Textures are read
// from it and frame/depth pixels are written into it with the GS addressing rules, so drawing into
// a buffer and sampling it as a texture needs no copies. Pixel interlock keeps each pixel's
// read-modify-write in primitive order; 16-bit pixels share 32-bit words with a neighbour, so those
// stores use atomics on their own bits.
layout(pixel_interlock_ordered) in;

layout(location = 0) noperspective in vec4 vColor;
layout(location = 1) noperspective in vec3 vTex;
layout(location = 2) noperspective in vec2 vZFog;
layout(location = 3) flat in uint vDraw;

struct DrawState
{
    uint fbBp, fbBw, fbPsm, fbMask;
    uint zBp, zPsm, flags, test;
    uint alpha, scissor0, scissor1, fogColor;
    uint primFlags, texBp, texBw, texPsm;
    uint texWH, texMode, clampU, clampV;
    uint clut, clutBase, texa, pad;
};

layout(std430, set = 0, binding = 0) coherent buffer LocalMemory { uint lm[]; };
layout(std430, set = 0, binding = 1) readonly buffer Layout { uint tbl[]; };
layout(std430, set = 0, binding = 2) readonly buffer Draws { DrawState d[]; };
layout(std430, set = 0, binding = 3) readonly buffer Cluts { uint cl[]; };

const uint F_FGE = 1u, F_ABE = 2u, F_PABE = 4u, F_FBA = 8u, F_ZMASK = 16u, F_HASZ = 32u, F_HASFB = 64u;
const uint P_TME = 1u, P_FST = 2u, P_LINEAR = 4u, P_ZROUND = 8u;

// Layout families: 0 CT32 (also CT24, T8H, T4HL, T4HH), 1 Z32/Z24, 2 CT16, 3 CT16S, 4 Z16, 5 Z16S, 6 T8, 7 T4
uint family(uint psm)
{
    switch (psm)
    {
    case 48u: case 49u: return 1u;
    case 2u: return 2u;
    case 10u: return 3u;
    case 50u: return 4u;
    case 58u: return 5u;
    case 19u: return 6u;
    case 20u: return 7u;
    default: return 0u;
    }
}

// Element index (word / halfword / byte / nibble, depending on the family) of pixel (x, y).
uint gsAddr(uint fam, uint bp, uint bw, uint x, uint y)
{
    uint pw, ph, bwd, bh, sh;
    if (fam <= 1u) { pw = 64u; ph = 32u; bwd = 8u; bh = 8u; sh = 6u; }
    else if (fam <= 5u) { pw = 64u; ph = 64u; bwd = 16u; bh = 8u; sh = 7u; }
    else if (fam == 6u) { pw = 128u; ph = 64u; bwd = 16u; bh = 16u; sh = 8u; }
    else { pw = 128u; ph = 128u; bwd = 32u; bh = 16u; sh = 9u; }
    uint ppr = max(1u, (max(bw, 1u) * 64u) / pw);
    uint page = (bp >> 5) + (y / ph) * ppr + (x / pw);
    uint lx = x % pw, ly = y % ph;
    uint blockId = (bp & 31u) + tbl[tbl[fam * 2u] + (ly / bh) * (pw / bwd) + (lx / bwd)];
    uint within = tbl[tbl[fam * 2u + 1u] + (ly % bh) * bwd + (lx % bwd)];
    return ((page * 32u + blockId) << sh) + within;
}

uint rd32(uint e) { return lm[e & 0xFFFFFu]; }
uint rd16(uint e) { e &= 0x1FFFFFu; return (lm[e >> 1] >> ((e & 1u) * 16u)) & 0xFFFFu; }
uint rd8(uint e) { e &= 0x3FFFFFu; return (lm[e >> 2] >> ((e & 3u) * 8u)) & 0xFFu; }
uint rd4(uint e) { e &= 0x7FFFFFu; return (lm[e >> 3] >> ((e & 7u) * 4u)) & 0xFu; }

// Same value GSMem::Read* returns for this format.
uint readRaw(uint psm, uint e)
{
    switch (psm)
    {
    case 0u: case 48u: return rd32(e);
    case 1u: case 49u: return rd32(e) & 0xFFFFFFu;
    case 2u: case 10u: case 50u: case 58u: return rd16(e);
    case 19u: return rd8(e);
    case 20u: return rd4(e);
    case 27u: return rd32(e) >> 24;
    case 36u: return (rd32(e) >> 24) & 0xFu;
    case 44u: return rd32(e) >> 28;
    default: return 0u;
    }
}

void writeRaw(uint psm, uint e, uint v)
{
    uint fam = family(psm);
    if (fam <= 1u)
    {
        uint i = e & 0xFFFFFu;
        if (psm == 1u || psm == 49u)
            lm[i] = (lm[i] & 0xFF000000u) | (v & 0xFFFFFFu);
        else
            lm[i] = v;
    }
    else
    {
        e &= 0x1FFFFFu;
        uint i = e >> 1, s = (e & 1u) * 16u;
        atomicAnd(lm[i], ~(0xFFFFu << s));
        atomicOr(lm[i], (v & 0xFFFFu) << s);
    }
}

uint expand5551(uint c)
{
    return ((c & 0x1Fu) << 3) | (((c >> 5) & 0x1Fu) << 11) | (((c >> 10) & 0x1Fu) << 19) | (((c >> 15) & 1u) << 31);
}
uint pack5551(uint c)
{
    return ((c & 0xFFu) >> 3) | ((((c >> 8) & 0xFFu) >> 3) << 5) | ((((c >> 16) & 0xFFu) >> 3) << 10) | ((c >> 31) << 15);
}
bool is16(uint psm) { return psm == 2u || psm == 10u || psm == 50u || psm == 58u; }

uint applyTexa(uint texa, uint psm, uint c)
{
    if (psm != 1u && psm != 2u && psm != 10u)
        return c;
    bool rgbZero = (c & 0xFFFFFFu) == 0u;
    uint ta0 = texa & 0xFFu, ta1 = (texa >> 8) & 0xFFu;
    bool aem = ((texa >> 16) & 1u) != 0u;
    uint a;
    if (psm == 1u)
        a = (aem && rgbZero) ? 0u : ta0;
    else if ((c >> 31) != 0u)
        a = ta1;
    else
        a = (aem && rgbZero) ? 0u : ta0;
    return (c & 0xFFFFFFu) | (a << 24);
}

uint clutEntry(uint base, uint i) { return (cl[base + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu; }

uint lookupClut(DrawState s, uint index)
{
    uint cpsm = s.clut & 0xFFu, csm = (s.clut >> 8) & 1u, csa = (s.clut >> 16) & 0x1Fu;
    bool sixteen = cpsm == 2u || cpsm == 10u;
    uint clutBase = (csa & (sixteen ? 0x1Fu : 0x0Fu)) << 4;
    bool four = s.texPsm == 20u || s.texPsm == 36u || s.texPsm == 44u;
    bool eight = s.texPsm == 19u || s.texPsm == 27u;
    uint src = four ? (index & 0xFu) : index;
    uint ci = (clutBase + src) & (sixteen ? 0x1FFu : 0xFFu);
    if (!sixteen && csm == 0u && eight)
        ci = min((src & 0xF0u) + clutBase, 240u) + (src & 0xFu);
    if (cpsm == 0u)
        return clutEntry(s.clutBase, ci) | (clutEntry(s.clutBase, ci + 256u) << 16);
    if (cpsm == 1u)
        return applyTexa(s.texa, 1u, (clutEntry(s.clutBase, ci) | (clutEntry(s.clutBase, ci + 256u) << 16)) & 0xFFFFFFu);
    if (sixteen)
        return applyTexa(s.texa, cpsm, expand5551(clutEntry(s.clutBase, ci)));
    return 0xFFFF00FFu;
}

int wrapCoord(int c, int size, uint mode, int rmin, int rmax)
{
    if (mode == 0u) return int(uint(c) & uint(size - 1));
    if (mode == 1u) return clamp(c, 0, size - 1);
    if (mode == 2u) return min(max(c, rmin), rmax);
    return int((uint(c) & uint(rmin)) | uint(rmax));
}

uvec4 fetchTexel(DrawState s, int u, int v)
{
    int w = int(s.texWH & 0xFFFFu), h = int(s.texWH >> 16);
    u = wrapCoord(u, w, (s.texMode >> 4) & 3u, int(s.clampU & 0xFFFFu), int(s.clampU >> 16));
    v = wrapCoord(v, h, (s.texMode >> 6) & 3u, int(s.clampV & 0xFFFFu), int(s.clampV >> 16));
    uint raw = readRaw(s.texPsm, gsAddr(family(s.texPsm), s.texBp, s.texBw, uint(max(u, 0)), uint(max(v, 0))));
    uint c;
    switch (s.texPsm)
    {
    case 0u: case 48u: case 1u: case 49u: c = applyTexa(s.texa, s.texPsm, raw); break;
    case 2u: case 10u: case 50u: case 58u: c = applyTexa(s.texa, s.texPsm, expand5551(raw)); break;
    case 19u: case 20u: case 27u: case 36u: case 44u: c = lookupClut(s, raw); break;
    default: c = 0xFFFF00FFu; break;
    }
    return uvec4(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu, c >> 24);
}

void main()
{
    DrawState s = d[vDraw];
    ivec2 p = ivec2(floor(gl_FragCoord.xy));

    bool alive = p.x >= int(s.scissor0 & 0xFFFFu) && p.x <= int(s.scissor1 & 0xFFFFu) &&
                 p.y >= int(s.scissor0 >> 16) && p.y <= int(s.scissor1 >> 16);

    ivec4 c = clamp(ivec4(vColor), ivec4(0), ivec4(255));

    if ((s.primFlags & P_TME) != 0u)
    {
        int w = int(s.texWH & 0xFFFFu), h = int(s.texWH >> 16);
        float tu, tv;
        if ((s.primFlags & P_FST) != 0u)
        {
            tu = floor(vTex.x) / 16.0;
            tv = floor(vTex.y) / 16.0;
        }
        else
        {
            float q = abs(vTex.z) > 1.0e-8 ? vTex.z : 1.0;
            tu = round(vTex.x / q * float(w) * 16.0) / 16.0;
            tv = round(vTex.y / q * float(h) * 16.0) / 16.0;
        }
        ivec4 t;
        if ((s.primFlags & P_LINEAR) == 0u)
        {
            t = ivec4(fetchTexel(s, int(floor(tu)), int(floor(tv))));
        }
        else
        {
            float su = tu - 0.5, sv = tv - 0.5;
            int u0 = int(floor(su)), v0 = int(floor(sv));
            float fx = su - float(u0), fy = sv - float(v0);
            vec4 c00 = vec4(fetchTexel(s, u0, v0)), c10 = vec4(fetchTexel(s, u0 + 1, v0));
            vec4 c01 = vec4(fetchTexel(s, u0, v0 + 1)), c11 = vec4(fetchTexel(s, u0 + 1, v0 + 1));
            vec4 top = mix(c00, c10, fx), bot = mix(c01, c11, fx);
            t = clamp(ivec4(round(mix(top, bot, fy))), ivec4(0), ivec4(255));
        }
        uint tfx = s.texMode & 3u;
        bool tcc = ((s.texMode >> 2) & 1u) != 0u;
        ivec4 o;
        if (tfx == 0u)
        {
            o.rgb = min((t.rgb * c.rgb) >> 7, ivec3(255));
            o.a = tcc ? min((t.a * c.a) >> 7, 255) : c.a;
        }
        else if (tfx == 1u)
        {
            o.rgb = t.rgb;
            o.a = tcc ? t.a : c.a;
        }
        else if (tfx == 2u)
        {
            o.rgb = min(((t.rgb * c.rgb) >> 7) + c.a, ivec3(255));
            o.a = tcc ? min(t.a + c.a, 255) : c.a;
        }
        else
        {
            o.rgb = min(((t.rgb * c.rgb) >> 7) + c.a, ivec3(255));
            o.a = tcc ? t.a : c.a;
        }
        c = o;
    }

    if ((s.flags & F_FGE) != 0u)
    {
        int fog = clamp(int(vZFog.y), 0, 255);
        ivec3 fc = ivec3(s.fogColor & 0xFFu, (s.fogColor >> 8) & 0xFFu, (s.fogColor >> 16) & 0xFFu);
        c.rgb = ((fog * c.rgb) >> 8) + (((255 - fog) * fc) >> 8);
    }

    bool wRgb = true, wA = true, wZ = true;
    if ((s.test & 1u) != 0u)
    {
        uint atst = (s.test >> 1) & 7u;
        int aref = int((s.test >> 4) & 0xFFu);
        bool pass = atst == 1u || (atst == 2u && c.a < aref) || (atst == 3u && c.a <= aref) ||
                    (atst == 4u && c.a == aref) || (atst == 5u && c.a >= aref) ||
                    (atst == 6u && c.a > aref) || (atst == 7u && c.a != aref);
        if (!pass)
        {
            uint afail = (s.test >> 12) & 3u;
            if (afail == 0u) { wRgb = false; wA = false; wZ = false; }
            else if (afail == 1u) { wZ = false; }
            else if (afail == 2u) { wRgb = false; wA = false; }
            else { wZ = false; if (s.fbPsm == 0u) wA = false; }
        }
    }
    if ((s.flags & F_HASFB) == 0u) { wRgb = false; wA = false; }
    if ((s.flags & F_HASZ) == 0u || (s.flags & F_ZMASK) != 0u) wZ = false;
    alive = alive && (wRgb || wA || wZ);

    uint z = (s.primFlags & P_ZROUND) != 0u ? uint(max(vZFog.x + 0.5, 0.0)) : uint(max(vZFog.x, 0.0));
    uint ztst = (s.test >> 17) & 3u;
    uint fbE = gsAddr(family(s.fbPsm), s.fbBp, s.fbBw, uint(p.x), uint(p.y));
    uint zE = gsAddr(family(s.zPsm), s.zBp, s.fbBw, uint(p.x), uint(p.y));
    bool hasFb = (s.flags & F_HASFB) != 0u;
    bool hasZ = (s.flags & F_HASZ) != 0u;

    beginInvocationInterlockARB();
    if (alive)
    {
        uint raw = hasFb ? readRaw(s.fbPsm, fbE) : 0u;
        uint fb = raw;
        if (is16(s.fbPsm)) fb = expand5551(raw);
        else if (s.fbPsm == 1u) fb = raw | 0x80000000u;

        bool ok = true;
        if (((s.test >> 14) & 1u) != 0u)
        {
            bool datm = ((s.test >> 15) & 1u) != 0u;
            if (s.fbPsm == 0u) ok = ((raw >> 31) != 0u) == datm;
            else if (is16(s.fbPsm)) ok = (((raw >> 15) & 1u) != 0u) == datm;
        }
        if (ok)
        {
            if (ztst == 0u) ok = false;
            else if (ztst >= 2u)
            {
                uint stored = hasZ ? readRaw(s.zPsm, zE) : 0u;
                ok = ztst == 2u ? (z >= stored) : (z > stored);
            }
        }

        if (ok && (wRgb || wA) && hasFb)
        {
            ivec4 r = c;
            if ((s.flags & F_ABE) != 0u && !((s.flags & F_PABE) != 0u && (c.a & 0x80) == 0))
            {
                ivec4 dc = ivec4(fb & 0xFFu, (fb >> 8) & 0xFFu, (fb >> 16) & 0xFFu, fb >> 24);
                uint asel = s.alpha & 3u, bsel = (s.alpha >> 2) & 3u, csel = (s.alpha >> 4) & 3u, dsel = (s.alpha >> 6) & 3u;
                int fix = int((s.alpha >> 8) & 0xFFu);
                ivec3 A = asel == 0u ? c.rgb : (asel == 1u ? dc.rgb : ivec3(0));
                ivec3 B = bsel == 0u ? c.rgb : (bsel == 1u ? dc.rgb : ivec3(0));
                ivec3 D = dsel == 0u ? c.rgb : (dsel == 1u ? dc.rgb : ivec3(0));
                int C = csel == 0u ? c.a : (csel == 1u ? dc.a : fix);
                r.rgb = clamp((((A - B) * C) >> 7) + D, ivec3(0), ivec3(255));
            }
            if (wA && (s.flags & F_FBA) != 0u && s.fbPsm != 1u)
                r.a |= 0x80;
            uint pix = uint(r.r) | (uint(r.g) << 8) | (uint(r.b) << 16) | (uint(r.a) << 24);
            if (s.fbMask != 0u)
                pix = (pix & ~s.fbMask) | (fb & s.fbMask);
            if (wRgb && !wA && s.fbPsm == 0u)
                pix = (pix & 0x00FFFFFFu) | (fb & 0xFF000000u);
            if (is16(s.fbPsm)) pix = pack5551(pix);
            writeRaw(s.fbPsm, fbE, pix);
        }

        if (ok && wZ && hasZ)
        {
            uint zm = s.zPsm == 48u ? 0xFFFFFFFFu : (s.zPsm == 49u ? 0xFFFFFFu : 0xFFFFu);
            writeRaw(s.zPsm, zE, z & zm);
        }
    }
    endInvocationInterlockARB();
}
