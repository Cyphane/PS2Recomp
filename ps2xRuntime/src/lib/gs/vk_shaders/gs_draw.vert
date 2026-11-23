#version 450
// s3recomp Vulkan GS: positions arrive in GS window pixels (offset already removed). The GS samples
// coverage and attributes at integer pixel coordinates; Vulkan samples at pixel centres, so shift by
// half a pixel. The render area is a fixed 2048x2048 window (the GS coordinate space).
layout(location = 0) in vec4 inPos;    // x, y, z, fog
layout(location = 1) in vec4 inColor;  // r, g, b, a (0..255)
layout(location = 2) in vec4 inTex;    // s, t, q (fst=0) or u, v (texels*16), 1 (fst=1)
layout(location = 3) in uint inDraw;

layout(location = 0) noperspective out vec4 vColor;
layout(location = 1) noperspective out vec3 vTex;
layout(location = 2) noperspective out vec2 vZFog;
layout(location = 3) flat out uint vDraw;

void main()
{
    vec2 p = (inPos.xy + 0.5) * (2.0 / 2048.0) - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
    gl_PointSize = 1.0;
    vColor = inColor;
    vTex = inTex.xyz;
    vZFog = inPos.zw;
    vDraw = inDraw;
}
