import argparse, ctypes, json, math, os, sys, time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import moderngl
import moderngl_window as mglw
from PIL import Image, ImageDraw, ImageOps, ImageFont
from scipy import ndimage

try:
    import pyspacemouse
except Exception:
    pyspacemouse = None


# ============================================================
# Best-effort: prefer NVIDIA GPU on Windows
# ============================================================
try:
    ctypes.windll.ntdll.NtSetInformationProcess(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x27, ctypes.byref(ctypes.c_ulong(1)), ctypes.sizeof(ctypes.c_ulong)
    )
except Exception:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"


# ============================================================
# Slice shaders (full-screen view) + HEAP BRUSH
# ============================================================

# V11 full combined file: restored render dispatch, sampled-volume FX, and actual primitive object previews.

SLICE_VERT = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos * 0.5) + 0.5;   // 0..1, bottom-left origin
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Key idea:
# - We compute the oblique plane sample position p in volume coords [0,1]^3
# - Then apply a "heap brush" that offsets p along the plane normal n
#   based on mouse distance: center digs deepest, outside digs none.
# - u_heap_depth is in normalized volume units (like 0.25 means 25% of box)
SLICE_FRAG = r"""
#version 330

uniform sampler2DArray tex_array;
uniform int   u_num_layers;     // Z layers in texture array
uniform vec2  u_slice_px;       // viewport (w,h) in pixels

uniform vec3  u_center;         // plane center in [0,1]^3
uniform vec3  u_axis_u;         // plane axis U (unit)
uniform vec3  u_axis_v;         // plane axis V (unit)
uniform vec3  u_axis_n;         // plane normal N (unit)
uniform float u_scale;          // legacy half-width in normalized volume units
uniform float u_scale_u;        // half-width along axis U
uniform float u_scale_v;        // half-width along axis V
uniform int   u_aspect_correct; // 1 keeps old square-pixel correction, 0 fills panel

// heap brush
uniform int   u_heap_enable;    // 0/1
uniform vec2  u_mouse;          // [0,1], bottom-left origin
uniform float u_radius;         // brush radius in UV
uniform float u_softness;       // feather size (UV)
uniform float u_layer_stretch;  // shaping in radius space
uniform float u_heap_depth;     // max offset along +/-N (normalized volume units)
uniform float u_heap_dir;       // +1 or -1 (direction along N)

// color controls
uniform int   u_flip_y;         // 1 if texture rows are top-left origin (most PNG stacks), else 0
uniform int   u_bgr_input;      // 1 if stored as BGR in RGB channels, else 0
uniform int   u_filter_mode;    // 0 none, 1 isolate, 2 hide, 3 highlight
uniform int   u_filter_target;  // 1 red, 2 green, 3 blue, 4 white/bone, 5 flesh, 6 dark, 7 bright
uniform float u_filter_strength;
uniform int   u_post_mode;      // 0 none, 1 gray, 2 invert, 3 gray+invert

// compositing controls
uniform int   u_black_transparent; // 1 makes black/near-black samples transparent
uniform float u_black_threshold;   // normalized RGB max threshold for transparent black
uniform float u_output_alpha;      // alpha for non-black samples when blending

// Curved-plane editor uniforms.
// The flat slicing plane is bent along its normal before volume sampling.
uniform int   u_curved_enable;     // 0 flat plane, 1 curved plane
uniform int   u_curved_kind;       // 0 paraboloid, 1 saddle, 2 cylinder-U, 3 cylinder-V, 4 ripple
uniform float u_curved_amp;        // signed offset along plane normal, in normalized volume units
uniform float u_curved_radius;     // radius/width normalization for the parabola

in vec2 v_uv;
out vec4 fragColor;

vec4 sample_volume(vec3 p) {
    // p in [0,1]^3
    float zf = clamp(p.z, 0.0, 1.0) * float(u_num_layers - 1);
    int   z0 = int(floor(zf));
    int   z1 = min(z0 + 1, u_num_layers - 1);
    float t  = fract(zf);

    vec2 uv = p.xy;
    if (u_flip_y != 0) uv.y = 1.0 - uv.y;

    vec4 a = texture(tex_array, vec3(uv, float(z0)));
    vec4 b = texture(tex_array, vec3(uv, float(z1)));
    return mix(a, b, t);
}

float color_match(vec3 c, int target) {
    float mx = max(max(c.r, c.g), c.b);
    float mn = min(min(c.r, c.g), c.b);
    float sat = (mx > 1e-6) ? ((mx - mn) / mx) : 0.0;

    if (target == 1) return clamp((c.r - max(c.g, c.b)) * 2.5 + c.r * 0.35, 0.0, 1.0);
    if (target == 2) return clamp((c.g - max(c.r, c.b)) * 2.5 + c.g * 0.35, 0.0, 1.0);
    if (target == 3) return clamp((c.b - max(c.r, c.g)) * 2.5 + c.b * 0.35, 0.0, 1.0);
    if (target == 4) return clamp((mx - sat * 0.65) * 1.2, 0.0, 1.0);
    if (target == 5) return clamp((c.r - 0.5 * c.b) * 1.4 + sat * 0.35, 0.0, 1.0);
    if (target == 6) return clamp((0.35 - mx) * 3.0, 0.0, 1.0);
    if (target == 7) return clamp((mx - 0.55) * 2.2, 0.0, 1.0);
    return 0.0;
}

float heap_edge_fade(float d) {
    float r = max(u_radius, 1e-6);
    float t = clamp(d / r, 0.0, 1.0);

    // feather band near rim
    float rim0 = 1.0 - (u_softness / r);
    rim0 = clamp(rim0, 0.0, 1.0);

    float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

    // radius->depth shaping
    float shaped_t = pow(t, max(u_layer_stretch, 1e-6));

    // Return BOTH: fade controls blend; shaped_t controls depth
    // We'll recompute shaped_t in main to avoid packing.
    return edge_fade;
}

vec4 finalize_color(vec4 c) {
    if (u_bgr_input != 0) c.rgb = c.bgr;

    if (u_post_mode == 1 || u_post_mode == 3) {
        float g = dot(c.rgb, vec3(0.299, 0.587, 0.114));
        c.rgb = vec3(g);
    }
    if (u_post_mode == 2 || u_post_mode == 3) {
        c.rgb = vec3(1.0) - c.rgb;
    }

    if (u_filter_mode != 0 && u_filter_target != 0) {
        float m = color_match(c.rgb, u_filter_target);
        float s = clamp(u_filter_strength, 0.0, 1.0);
        if (u_filter_mode == 1) {
            c.rgb *= mix(1.0, m, s);
            c.a *= max(0.04, mix(1.0, m, s));
        } else if (u_filter_mode == 2) {
            c.rgb *= (1.0 - m * s);
        } else if (u_filter_mode == 3) {
            float keep = mix(1.0, 0.18, s) + m * s * 0.82;
            c.rgb *= keep;
            c.rgb += vec3(1.0, 0.95, 0.20) * (m * s * 0.45);
        }
    }

    if (u_black_transparent != 0) {
        float m = max(max(c.r, c.g), c.b);
        if (m <= u_black_threshold) {
            return vec4(c.rgb, 0.0);
        }
    }

    return vec4(c.rgb, clamp(u_output_alpha, 0.0, 1.0));
}

void main() {
    // map to [-1,1]
    vec2 s = (v_uv * 2.0 - 1.0);

    // Old single-view mode used aspect correction to keep pixels square.
    // Split-view MPR usually wants to fill each panel, so this is now optional.
    if (u_aspect_correct != 0) {
        float aspect = u_slice_px.x / max(u_slice_px.y, 1.0);
        s.x *= aspect;
    }

    float su = (u_scale_u > 0.0) ? u_scale_u : u_scale;
    float sv = (u_scale_v > 0.0) ? u_scale_v : u_scale;

    // base plane position:
    vec3 p0 = u_center + (u_axis_u * (s.x * su)) + (u_axis_v * (s.y * sv));

    // Optional curved-plane replacement.  This turns the screen-domain plane into
    // a parabolic/curved sheet by pushing samples along the plane normal.
    if (u_curved_enable != 0) {
        float r = max(abs(u_curved_radius), 1e-4);
        vec2 q = s / r;
        float h = 0.0;
        if (u_curved_kind == 0) {
            // Bowl/paraboloid: center stays on the red plane, edges bend along N.
            h = dot(q, q);
        } else if (u_curved_kind == 1) {
            // Saddle: one screen axis bends positive, the other negative.
            h = q.x * q.x - q.y * q.y;
        } else if (u_curved_kind == 2) {
            // Cylindrical bend along U.
            h = q.x * q.x;
        } else if (u_curved_kind == 3) {
            // Cylindrical bend along V.
            h = q.y * q.y;
        } else {
            // Parabolic sheet with a small ripple so the curvature is visible.
            h = dot(q, q) + 0.22 * sin(6.2831853 * q.x) * cos(6.2831853 * q.y);
        }
        p0 += u_axis_n * (u_curved_amp * h);
    }

    // quick reject if outside before heap:
    if (any(lessThan(p0, vec3(0.0))) || any(greaterThan(p0, vec3(1.0)))) {
        fragColor = (u_black_transparent != 0)
            ? vec4(0.0, 0.0, 0.0, 0.0)
            : vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 p = p0;

    if (u_heap_enable != 0) {
        float d = distance(v_uv, u_mouse);

        float r = max(u_radius, 1e-6);
        float t = clamp(d / r, 0.0, 1.0);
        float shaped_t = pow(t, max(u_layer_stretch, 1e-6));

        // center -> deepest, rim -> none
        float depth = (1.0 - shaped_t) * u_heap_depth;

        // offset along normal (dir = +/-1)
        p = p0 + (u_axis_n * (depth * u_heap_dir));

        // if dug outside volume, clamp by discarding (hard edge)
        if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) {
            // still show the undug plane (feels better than black):
            p = p0;
        }

        // blend based on rim fade (soft brush)
        vec4 outside = sample_volume(p0);
        vec4 inside  = sample_volume(p);

        float rim0 = 1.0 - (u_softness / r);
        rim0 = clamp(rim0, 0.0, 1.0);
        float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

        vec4 c = mix(outside, inside, edge_fade);

        fragColor = finalize_color(c);
        return;
    }

    // no heap
    vec4 c = sample_volume(p);
    fragColor = finalize_color(c);
}
"""


# ============================================================
# Gizmo shaders (3D box + 3D plane slab + normal arrow)
# ============================================================

GIZMO_VERT = r"""
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

GIZMO_FRAG = r"""
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""


# ============================================================
# HUD textured quad shaders (for help + gizmo UI overlay)
# ============================================================

HUD_TEX_VERT = r"""
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HUD_TEX_FRAG = r"""
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = texture(u_tex, v_uv); }
"""


# ============================================================
# GPU post-process shader (GL 3.3 fragment-pass replacement for compute)
# ============================================================

POST_FX_VERT = HUD_TEX_VERT

POST_FX_FRAG = r"""
#version 330
uniform sampler2D u_scene;
uniform vec2  u_resolution;
uniform float u_time;
uniform float u_strength;
uniform int   u_mode;          // 0 pass,1 swap,2 repeat,3 fragment,4 cuts,5 flipcontour,6 stretch,7 flesh swell,8 mold,9 vessels,10 vein branch,11 grassfire,12 amat,13 springmass,14 steiner,15 poisson,16 meatexpansion,17 inflation,18 myoglobin,19 fibertrack,20 watermobility,21 marbling
uniform int   u_cut_pattern;   // 0 parallel,1 grid,2 radial,3 irregular
uniform float u_cut_parallel;
uniform float u_cut_perp;
uniform float u_cut_angle;      // radians; controlled by cut-angle slider / left-right keys
uniform int   u_cut_motion;     // 0 fixed, 1 sinusoidal, 2 noise-warped
uniform float u_mask_threshold;
uniform float u_fx_param1;
uniform float u_fx_param2;
in vec2 v_uv;
out vec4 fragColor;

float hash12(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p){
    float n = hash12(p);
    return vec2(n, hash12(p + n + 1.37));
}
float noise2(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p){
    float v = 0.0;
    float a = 0.5;
    for(int i=0;i<4;++i){
        v += a * noise2(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        a *= 0.5;
    }
    return v;
}
vec4 scene(vec2 uv){ return texture(u_scene, clamp(uv, 0.0, 1.0)); }
float lum(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
float mask_at(vec2 uv){ return step(u_mask_threshold, lum(scene(uv).rgb)); }
float redish_at(vec2 uv){ vec3 c = scene(uv).rgb; return clamp((c.r - max(c.g, c.b))*2.2 + c.r*0.45, 0.0, 1.0); }
float edge_at(vec2 uv){
    vec2 px = 1.0 / max(u_resolution, vec2(1.0));
    float c = mask_at(uv);
    float d = 0.0;
    d += abs(c - mask_at(uv + vec2(px.x, 0.0)));
    d += abs(c - mask_at(uv - vec2(px.x, 0.0)));
    d += abs(c - mask_at(uv + vec2(0.0, px.y)));
    d += abs(c - mask_at(uv - vec2(0.0, px.y)));
    return clamp(d, 0.0, 1.0);
}
float mask_distance(vec2 uv){
    // approximate distance from edge inside the current visible object
    vec2 px = 1.0 / max(u_resolution, vec2(1.0));
    vec2 dirs[8] = vec2[8](vec2(1,0),vec2(-1,0),vec2(0,1),vec2(0,-1),vec2(0.707,0.707),vec2(-0.707,0.707),vec2(0.707,-0.707),vec2(-0.707,-0.707));
    float best = 0.35;
    for(int k=0;k<8;++k){
        for(int i=1;i<=14;++i){
            float t = float(i) / 14.0;
            vec2 q = uv + dirs[k] * t * 0.35;
            if(mask_at(q) < 0.5){
                best = min(best, t * 0.35);
                break;
            }
        }
    }
    return best;
}

void main(){
    vec2 uv = v_uv;
    vec4 src = scene(uv);
    float m = mask_at(uv);
    if (u_mode == 0){ fragColor = src; return; }

    vec2 uv2 = uv;
    vec2 px = 1.0 / max(u_resolution, vec2(1.0));

    if (u_mode == 1) {
        if (m > 0.5) {
            vec2 rnd = hash22(floor(uv * u_resolution * 0.25) + floor(u_time * 5.0));
            vec2 ofs = (rnd * 2.0 - 1.0) * (0.015 + 0.085 * u_strength);
            vec2 cand = clamp(uv + ofs, 0.0, 1.0);
            if (mask_at(cand) > 0.5) uv2 = cand;
        }
        fragColor = scene(uv2);
        return;
    }
    if (u_mode == 2) {
        float tiles = mix(1.5, 5.0, clamp(u_strength,0.0,1.0));
        vec2 c = uv - 0.5;
        uv2 = fract(c * tiles + 0.5);
        fragColor = scene(uv2);
        return;
    }
    if (u_mode == 3) {
        vec2 p = uv * 8.0;
        vec2 cell = floor(p);
        float best = 1e9; vec2 bestSeed = vec2(0.0);
        for(int j=-1;j<=1;++j) for(int i=-1;i<=1;++i){
            vec2 cc = cell + vec2(i,j);
            vec2 seed = hash22(cc);
            vec2 diff = vec2(i,j) + seed - fract(p);
            float d = dot(diff,diff);
            if (d < best){ best = d; bestSeed = diff; }
        }
        vec2 push = normalize(bestSeed + 1e-6) * (0.02 + 0.08*u_strength);
        uv2 = clamp(uv + push, 0.0, 1.0);
        fragColor = mix(src, scene(uv2), 0.9);
        return;
    }
    if (u_mode == 4) {
        // Cut / diced-meat displacement. The random swap stays constrained to
        // the foreground mask; this cut pass also only displaces visible object pixels.
        vec2 c = uv - 0.5;
        float ang = u_cut_angle;
        if (u_cut_motion == 1) {
            // Sinusoidal knife: the cut direction gently swims over the image.
            ang += 0.55 * sin(u_time * 0.85 + dot(c, vec2(10.0, -6.0)));
        } else if (u_cut_motion == 2) {
            // Noise knife: Perlin-like FBM bends the cut angle locally.
            ang += (fbm(uv * mix(3.0, 15.0, u_strength) + vec2(u_time * 0.06, -u_time * 0.04)) - 0.5) * 1.45;
        }
        vec2 dir = vec2(cos(ang), sin(ang));
        vec2 perp = vec2(-dir.y, dir.x);
        float coord = dot(c, dir);
        float side = dot(c, perp);
        float piece = 0.0;
        if (u_cut_pattern == 0) {
            float bands = mix(4.0, 18.0, u_strength);
            piece = floor((coord + 1.0) * 0.5 * bands);
        } else if (u_cut_pattern == 1) {
            float bands = mix(3.0, 9.0, u_strength);
            float a = floor((coord + 1.0) * 0.5 * bands);
            float b = floor((side + 1.0) * 0.5 * bands);
            piece = a + b * bands;
        } else if (u_cut_pattern == 2) {
            float r = length(c);
            float th = atan(c.y, c.x);
            float ra = floor(r * mix(6.0, 16.0, u_strength));
            float rb = floor((th + 3.14159265) / 6.2831853 * mix(8.0, 22.0, u_strength));
            piece = ra + rb * 32.0;
        } else {
            piece = floor((coord + 1.0) * mix(5.0, 16.0, u_strength) + hash12(floor((uv + 0.5)*12.0))*4.0);
        }
        float localAng = ang;
        if (u_cut_pattern == 3) {
            // Irregular cut clumps get their own slight direction, like uneven knife cuts.
            localAng += (hash12(vec2(piece, piece*2.31)) - 0.5) * 1.8;
            dir = vec2(cos(localAng), sin(localAng));
            perp = vec2(-dir.y, dir.x);
        }
        float rnd = hash12(vec2(piece, piece*1.37));
        vec2 move = dir * ((rnd - 0.5) * 2.0 * u_cut_parallel) + perp * ((hash12(vec2(piece*0.7, piece*2.1)) - 0.5) * 2.0 * u_cut_perp);
        uv2 = clamp(uv - move * m, 0.0, 1.0);
        fragColor = scene(uv2);
        return;
    }
    if (u_mode == 5) {
        // Contour reflection/export: no color inversion. Pixels near the silhouette
        // are sampled from the interior and pushed outward, like the meat surface is
        // folding outside its own outline.
        vec2 c0 = uv - 0.5;
        float r = length(c0) + 1e-6;
        vec2 n = c0 / r;
        float inside = mask_at(uv);
        float amount = 0.035 + 0.18 * u_strength;
        vec2 inward = clamp(uv - n * amount, 0.0, 1.0);
        vec2 deeper = clamp(uv - n * amount * 0.35, 0.0, 1.0);
        vec4 outCol = src;
        if (inside < 0.5 && mask_at(inward) > 0.5) {
            // Outside pixel receives a reflected sample from just inside the silhouette.
            outCol = scene(inward);
        } else if (inside > 0.5) {
            // Interior gets slightly pulled inward, leaving an edge-peel feeling.
            outCol = scene(deeper);
        }
        float e = edge_at(uv);
        outCol.rgb += vec3(0.10) * e;
        fragColor = outCol;
        return;
    }
    if (u_mode == 6) {
        vec3 col = src.rgb;
        float redish = clamp((col.r - max(col.g, col.b)) * 2.0 + col.r * 0.6, 0.0, 1.0);
        float whitish = clamp((max(max(col.r,col.g),col.b) - abs(col.r-col.g)*0.3) * 1.1, 0.0, 1.0);
        vec2 c0 = uv - 0.5;
        vec2 dir = normalize(c0 + vec2(1e-5));
        float amt = (redish * 0.12 + whitish * 0.03) * u_strength;
        uv2 = clamp(uv - dir * amt * m, 0.0, 1.0);
        vec4 shifted = scene(uv2);
        float fillOn = step(0.5, u_fx_param2);
        vec3 interpCol = (
            scene(mix(uv, uv2, 0.20)).rgb +
            scene(mix(uv, uv2, 0.45)).rgb +
            scene(mix(uv, uv2, 0.70)).rgb +
            shifted.rgb
        ) * 0.25;
        fragColor = vec4(mix(shifted.rgb, interpCol, fillOn), shifted.a);
        return;
    }
    if (u_mode == 7) {
        float r0 = redish_at(uv);
        vec2 grad = vec2(redish_at(uv + vec2(px.x,0.0)) - redish_at(uv - vec2(px.x,0.0)), redish_at(uv + vec2(0.0,px.y)) - redish_at(uv - vec2(0.0,px.y)));
        float swellAmt = mix(0.2, 2.0, clamp(u_fx_param1, 0.0, 1.0));
        vec2 flow = normalize(grad + vec2(1e-5));
        vec2 bulge = grad * (0.03 + 0.10 * u_strength) * swellAmt + (uv - 0.5) * (-0.02 * r0 * u_strength) * swellAmt;
        uv2 = clamp(uv - bulge * m - flow * r0 * (0.008 + 0.028 * u_strength) * swellAmt, 0.0, 1.0);
        vec4 s = scene(uv2);
        vec3 s2 = scene(clamp(uv2 - flow * (0.010 + 0.035 * u_strength), 0.0, 1.0)).rgb;
        vec3 swollen = mix(s.rgb, s2, 0.35) * (1.0 + vec3(0.18, 0.05, 0.05) * r0 * (0.4 + u_strength));
        fragColor = vec4(swollen, s.a);
        return;
    }
    if (u_mode == 8) {
        float moldScale = mix(2.0, 18.0, clamp(u_fx_param2, 0.0, 1.0));
        float moldGrowth = mix(0.25, 1.35, clamp(u_fx_param1, 0.0, 1.0));
        float n = fbm(uv * moldScale + vec2(0.0, u_time * 0.08));
        float n2 = fbm(uv.yx * (moldScale * 1.35) + vec2(13.0, -u_time * 0.05));
        float grow = smoothstep(0.55 - 0.25*u_strength, 0.80, (n * 0.7 + n2 * 0.3) * moldGrowth) * m;
        vec3 mold = mix(vec3(0.18, 0.20, 0.10), vec3(0.70, 0.82, 0.62), smoothstep(0.45, 0.95, n2));
        vec3 col = mix(src.rgb, mold, grow * 0.78);
        col *= 1.0 - grow * 0.08;
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 9) {
        float vesselDensity = mix(4.0, 22.0, clamp(u_fx_param2, 0.0, 1.0));
        float vesselThickness = mix(0.03, 0.20, clamp(u_fx_param1, 0.0, 1.0));
        vec2 p = uv * vesselDensity;
        p += vec2(fbm(uv * 4.0 + u_time*0.05), fbm(uv.yx * 4.3 - u_time*0.04)) * 1.8;
        float vein = abs(fract(p.x + fbm(p * 0.65) * 2.2) - 0.5);
        vein = 1.0 - smoothstep(0.0, vesselThickness + 0.07*(1.0-u_strength), vein);
        vein *= m;
        vec2 dir = normalize(vec2(0.7, 0.25) + vec2(fbm(uv*5.0), fbm(uv*5.0+4.0)));
        uv2 = clamp(uv - dir * vein * (0.006 + 0.02*u_strength), 0.0, 1.0);
        vec3 base = scene(uv2).rgb;
        vec3 veinCol = mix(vec3(0.35, 0.0, 0.02), vec3(0.85, 0.08, 0.10), redish_at(uv));
        vec3 col = mix(base, veinCol, vein * 0.80);
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 10) {
        vec2 c = uv - 0.5;
        float ang = atan(c.y, c.x);
        float rad = length(c);
        float branchDensity = mix(2.0, 16.0, clamp(u_fx_param1, 0.0, 1.0));
        float branchSpread = mix(2.0, 12.0, clamp(u_fx_param2, 0.0, 1.0));
        float branches = sin(ang * (branchDensity + 8.0*u_strength) + fbm(c * (8.0 + branchSpread) + 3.0) * 6.0 + u_time * 0.15);
        float rib = 1.0 - smoothstep(0.18, 0.36, abs(branches) + rad * 0.35);
        rib *= smoothstep(0.85, 0.05, rad) * m;
        vec2 growDir = normalize(c + vec2(0.001));
        uv2 = clamp(uv - growDir * rib * (0.008 + 0.03*u_strength), 0.0, 1.0);
        vec3 base = scene(uv2).rgb;
        vec3 extend = mix(vec3(0.95, 0.20, 0.22), vec3(1.0, 0.88, 0.75), smoothstep(0.0,1.0,fbm(uv*20.0)));
        vec3 col = mix(base, extend, rib * 0.78);
        col += rib * 0.08;
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 11) {
        float d = mask_distance(uv);
        float fireRadius = mix(0.06, 0.45, clamp(u_fx_param2, 0.0, 1.0));
        float pulseSpeed = mix(0.4, 6.5, clamp(u_fx_param1, 0.0, 1.0));
        float q = 1.0 - smoothstep(0.0, fireRadius + 0.18*u_strength, d);
        float pulse = 0.5 + 0.5*sin(28.0*d - u_time*pulseSpeed);
        vec3 ember = mix(vec3(0.12,0.04,0.02), vec3(1.0,0.62,0.18), pulse);
        vec3 ash = mix(vec3(0.05,0.05,0.06), vec3(0.95,0.90,0.82), q);
        vec3 col = mix(ash, ember, q * (0.45 + 0.55*pulse));
        col = mix(src.rgb * 0.25, col, m);
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 12) {
        float circleCount = mix(4.0, 28.0, clamp(u_fx_param1, 0.0, 1.0));
        float jitterAmt = mix(0.0, 0.9, clamp(u_fx_param2, 0.0, 1.0));
        vec2 p = uv * circleCount;
        vec2 cell = floor(p);
        vec2 jitter = (hash22(cell + floor(u_time*0.2)) - 0.5) * jitterAmt;
        vec2 center = (cell + 0.5 + jitter * (0.35 + 0.35*u_strength)) / circleCount;
        float rad = (0.012 + 0.030*hash12(cell*1.73)) * (0.55 + 1.2*u_strength) * mix(0.6, 1.8, clamp(u_fx_param1, 0.0, 1.0));
        float circ = 1.0 - smoothstep(rad*0.75, rad, length(uv - center));
        vec3 ccol = scene(center).rgb;
        vec3 col = mix(vec3(0.02), ccol, circ * m);
        col += circ * 0.08;
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 13) {
        float d = mask_distance(uv);
        vec2 c = uv - 0.5;
        float springForce = mix(0.0, 2.5, clamp(u_fx_param2, 0.0, 1.0));
        float springStiff = mix(0.1, 2.2, clamp(u_fx_param1, 0.0, 1.0));
        vec2 force = vec2(sin(uv.y*12.0 + u_time*1.7), cos(uv.x*10.0 - u_time*1.3)) * 0.008 * springForce;
        vec2 anchor = -normalize(c + vec2(0.002)) * d * (0.8 + 1.2*u_strength) * springStiff;
        uv2 = clamp(uv + force + anchor * 0.06, 0.0, 1.0);
        vec3 base = scene(uv2).rgb;
        float ghost = smoothstep(0.0, 0.18, d) * m;
        vec3 col = mix(base * 0.45, base + vec3(0.08,0.11,0.16)*ghost, 0.85);
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 14) {
        vec2 c = uv - 0.5;
        float ang = atan(c.y, c.x);
        float rad = length(c);
        float density = mix(1.0, 12.0, clamp(u_fx_param1, 0.0, 1.0));
        float fungal = mix(6.0, 26.0, clamp(u_fx_param2, 0.0, 1.0));
        float path = abs(sin(ang * (4.0 + density*u_strength) + fbm(c*fungal) * 7.0 + u_time*0.2));
        float slime = 1.0 - smoothstep(0.25, 0.45, path + rad * (0.9 - 0.4*u_strength));
        vec3 growth = mix(vec3(0.12,0.20,0.08), vec3(0.72,0.92,0.58), fbm(uv*15.0));
        vec3 col = mix(src.rgb * 0.15, growth, slime * m);
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 15) {
        float d = mask_distance(uv);
        float glowRadius = mix(0.03, 0.45, clamp(u_fx_param2, 0.0, 1.0));
        float glowAmt = mix(0.1, 2.0, clamp(u_fx_param1, 0.0, 1.0));
        float glow = smoothstep(glowRadius + 0.12*u_strength, 0.0, d) * m * glowAmt;
        float ridge = (0.5 + 0.5*sin(24.0*d - u_time*2.4)) * glow;
        vec3 neon = mix(vec3(0.10,0.62,1.0), vec3(0.55,1.0,0.90), ridge);
        vec3 col = src.rgb * 0.18 + neon * (0.55 + 0.75*glow);
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 16) {
        // Meat expansion / reverse distance transform style regrowth.
        float d = mask_distance(uv);
        vec2 c = uv - 0.5;
        vec2 inward = normalize(-c + vec2(1e-4));
        vec2 tangent = vec2(-inward.y, inward.x);
        float feed = mix(0.55, 1.85, clamp(u_fx_param1, 0.0, 1.0));
        float expansion = (feed - 1.0) * (0.18 + 0.95 * smoothstep(0.0, 0.22, d));
        float flesh0 = redish_at(uv);
        uv2 = clamp(uv + inward * expansion * d * 1.9 + tangent * (flesh0 - 0.5) * (0.012 + 0.040 * u_strength), 0.0, 1.0);
        vec3 regrown = scene(uv2).rgb;
        vec3 side = scene(clamp(uv2 + tangent * (0.010 + 0.028 * u_strength), 0.0, 1.0)).rgb;
        float flesh = redish_at(uv2);
        vec3 col = mix(regrown, side, 0.30) * (0.88 + 0.22 * feed) + vec3(0.10, 0.04, 0.04) * flesh * max(feed - 1.0, 0.0);
        fragColor = vec4(mix(src.rgb * 0.45, col, m), src.a);
        return;
    }
    if (u_mode == 17) {
        // Inflation / implicit blobby tubes around an internal skeletal field.
        vec2 p = uv - 0.5;
        vec2 bones[4] = vec2[4](vec2(0.0, -0.20), vec2(-0.16, 0.05), vec2(0.17, 0.02), vec2(0.0, 0.24));
        float field = 0.0;
        vec2 grad = vec2(0.0);
        for (int i = 0; i < 4; ++i) {
            vec2 b = bones[i] + vec2(0.06 * sin(u_time * 0.1 + float(i)), 0.04 * cos(u_time * 0.08 + float(i) * 1.7));
            vec2 diff = p - b;
            float r2 = dot(diff, diff) + 0.012;
            float w = 0.018 / r2;
            field += w;
            grad += (-2.0 * 0.018) * diff / (r2 * r2);
        }
        float inflateAmt = mix(0.35, 1.85, clamp(u_fx_param1, 0.0, 1.0));
        float tubeRadius = mix(0.45, 2.8, clamp(u_fx_param2, 0.0, 1.0));
        float tube = smoothstep(0.45, 1.65 + tubeRadius * inflateAmt, field) * m;
        vec2 n = normalize(grad + vec2(1e-5));
        vec2 tdir = vec2(-n.y, n.x);
        uv2 = clamp(uv - n * tube * (0.010 + 0.055 * inflateAmt) + tdir * tube * (0.006 + 0.018 * u_strength), 0.0, 1.0);
        vec3 base = scene(uv2).rgb;
        vec3 side = scene(clamp(uv2 + tdir * (0.012 + 0.020 * inflateAmt), 0.0, 1.0)).rgb;
        vec3 inflated = mix(base, side, 0.28);
        inflated = mix(inflated, inflated * 1.12 + vec3(0.10, 0.06, 0.04), tube * 0.72);
        inflated += tube * 0.06;
        fragColor = vec4(inflated, src.a);
        return;
    }
    if (u_mode == 18) {
        float smoothAmt = mix(0.0, 1.0, clamp(u_fx_param1, 0.0, 1.0));
        float modelAmt = mix(0.2, 2.0, clamp(u_fx_param2, 0.0, 1.0));
        vec3 b0 = scene(uv).rgb;
        vec3 b1 = scene(uv + vec2(px.x,0.0)).rgb;
        vec3 b2 = scene(uv - vec2(px.x,0.0)).rgb;
        vec3 b3 = scene(uv + vec2(0.0,px.y)).rgb;
        vec3 b4 = scene(uv - vec2(0.0,px.y)).rgb;
        vec3 sm = mix(b0, (b0 + b1 + b2 + b3 + b4) / 5.0, smoothAmt);
        float oxy = clamp((sm.r - sm.g) * 2.2 + sm.r * 0.45 - sm.b * 0.25, 0.0, 1.0);
        float deoxy = clamp((sm.g - sm.r) * 1.2 + sm.r * 0.18, 0.0, 1.0);
        float freshness = clamp(oxy * 0.75 + (1.0 - deoxy) * 0.45, 0.0, 1.0);
        vec3 falseCol = mix(vec3(0.08, 0.20, 0.95), vec3(0.95, 0.16, 0.18), oxy);
        falseCol = mix(falseCol, vec3(0.95, 0.92, 0.30), freshness * 0.35);
        fragColor = vec4(mix(src.rgb, falseCol, 0.22 + 0.58 * modelAmt / 2.0) * m + src.rgb * (1.0 - m), src.a);
        return;
    }
    if (u_mode == 19) {
        float dens = mix(5.0, 24.0, clamp(u_fx_param1, 0.0, 1.0));
        float lenGain = mix(0.25, 1.8, clamp(u_fx_param2, 0.0, 1.0));
        float gx = lum(scene(uv + vec2(px.x,0.0)).rgb) - lum(scene(uv - vec2(px.x,0.0)).rgb);
        float gy = lum(scene(uv + vec2(0.0,px.y)).rgb) - lum(scene(uv - vec2(0.0,px.y)).rgb);
        float ang = atan(gy, gx) + 1.5707963;
        vec2 dir = vec2(cos(ang), sin(ang));
        vec2 q = fract((uv - 0.5) * dens);
        float line = abs(q.y - 0.5);
        float stripe = 1.0 - smoothstep(0.01, 0.09 / max(lenGain, 0.05), line);
        float anis = clamp(length(vec2(gx, gy)) * 4.0, 0.0, 1.0);
        vec3 fibers = mix(src.rgb * 0.2, vec3(0.92, 0.96, 0.98), stripe * anis * m);
        fragColor = vec4(fibers, src.a);
        return;
    }
    if (u_mode == 20) {
        float decayMix = mix(0.0, 1.0, clamp(u_fx_param1, 0.0, 1.0));
        float sepAmt = mix(0.2, 2.0, clamp(u_fx_param2, 0.0, 1.0));
        vec3 c0 = scene(uv).rgb;
        vec3 c1 = scene(uv + vec2(2.0*px.x,0.0)).rgb;
        vec3 c2 = scene(uv - vec2(2.0*px.x,0.0)).rgb;
        vec3 c3 = scene(uv + vec2(0.0,2.0*px.y)).rgb;
        vec3 c4 = scene(uv - vec2(0.0,2.0*px.y)).rgb;
        vec3 blur = (c0 + c1 + c2 + c3 + c4) / 5.0;
        float intra = clamp(lum(blur) * 1.2, 0.0, 1.0);
        float extra = clamp(abs(lum(c0) - lum(blur)) * 3.5 * sepAmt, 0.0, 1.0);
        float whc = clamp(intra - extra * 0.6, 0.0, 1.0);
        vec3 wetCol = mix(vec3(0.05, 0.16, 0.32), vec3(0.20, 0.88, 0.96), whc);
        vec3 col = mix(src.rgb * (0.8 - 0.35 * decayMix), wetCol, (0.22 + 0.65 * decayMix) * m);
        fragColor = vec4(col, src.a);
        return;
    }
    if (u_mode == 21) {
        float thr = mix(0.52, 0.92, clamp(u_fx_param1, 0.0, 1.0));
        float fuzzy = mix(0.02, 0.35, clamp(u_fx_param2, 0.0, 1.0));
        vec3 c = scene(uv).rgb;
        float fatness = clamp((min(c.r, min(c.g, c.b)) + max(c.r, max(c.g, c.b))) * 0.5 + (1.0 - abs(c.r-c.g) - abs(c.g-c.b)) * 0.25, 0.0, 1.0);
        float prob = smoothstep(thr - fuzzy, thr + fuzzy, fatness);
        vec3 fatCol = mix(vec3(0.32, 0.04, 0.06), vec3(0.98, 0.95, 0.88), prob);
        vec3 col = mix(src.rgb * 0.55, fatCol, prob * m);
        fragColor = vec4(col, src.a);
        return;
    }
    fragColor = src;
}
"""


def _build_post_fx_compute_shader() -> str:
    """Build a GLSL 4.30 compute version of POST_FX_FRAG.

    The fragment shader already contains the visual logic for the frame FX.  To
    keep the two paths identical, this translates the fragment entry point into
    a compute entry point and writes the result into an rgba8 image.  This avoids
    a raster full-screen post draw for frame_fx and gives us a clean path for
    future multi-pass compute effects.
    """
    import re as _re
    src = POST_FX_FRAG
    src = src.replace("#version 330", "#version 430")
    src = src.replace("in vec2 v_uv;\nout vec4 fragColor;", "layout(local_size_x = 16, local_size_y = 16) in;\nlayout(rgba8, binding = 0) writeonly uniform image2D u_out;\n\nvoid writeColor(ivec2 gid, vec4 c){\n    imageStore(u_out, gid, clamp(c, 0.0, 1.0));\n}")
    src = src.replace(
        "void main(){\n    vec2 uv = v_uv;",
        "void main(){\n    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);\n    ivec2 dims = ivec2(int(u_resolution.x), int(u_resolution.y));\n    if (gid.x < 0 || gid.y < 0 || gid.x >= dims.x || gid.y >= dims.y) return;\n    vec2 uv = (vec2(gid) + vec2(0.5)) / max(u_resolution, vec2(1.0));"
    )
    # Convert all early-outs from fragment assignment to compute image writes.
    src = _re.sub(r"fragColor\s*=\s*([^;]+);\s*return;", r"writeColor(gid, \1); return;", src)
    # Convert the final fallback write.
    src = _re.sub(r"fragColor\s*=\s*src;\s*\n}\s*$", "writeColor(gid, src);\n}", src)
    return src



# ============================================================
# Math helpers
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n

def orthonormal_basis_from_normal(n: np.ndarray):
    n = normalize(n)
    if abs(float(n[2])) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = normalize(np.cross(a, n))
    v = normalize(np.cross(n, u))
    return u, v

def yaw_pitch_to_normal(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x = sy * cp
    y = cy * cp
    z = sp
    return normalize(np.array([x, y, z], dtype=np.float32))

def look_at(eye, target, up):
    eye = np.array(eye, np.float32)
    target = np.array(target, np.float32)
    up = np.array(up, np.float32)

    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

def perspective(fovy_deg, aspect, znear, zfar):
    fovy = np.deg2rad(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / max(aspect, 1e-8)
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M



# ============================================================
# Waypoint recorder + interpolation / timeline helpers
# ============================================================

@dataclass
class CameraState:
    t: float
    position: List[float]
    euler_deg: List[float]
    plane_normal: List[float]
    yaw: float
    pitch: float
    scale: float
    view_mode: str
    heuristics: Dict[str, Any] = field(default_factory=dict)
    heuristic_images: Dict[str, str] = field(default_factory=dict)


@dataclass
class BrushState:
    t: float
    mouse_uv: List[float]
    strength: float
    radius: float
    softness: float
    stretch: float
    direction: float
    enabled: bool


class WaypointRecorder:
    """Stores camera/plane, brush, and combined waypoints and writes JSON."""
    def __init__(self) -> None:
        self.camera_waypoints: List[CameraState] = []
        self.brush_waypoints: List[BrushState] = []
        self.combined_waypoints: List[Dict[str, Any]] = []

    def on_key_c(self, state: CameraState) -> None:
        self.camera_waypoints.append(state)

    def on_key_b(self, state: BrushState) -> None:
        self.brush_waypoints.append(state)

    def on_key_v(self, cam: CameraState, brush: BrushState) -> None:
        self.combined_waypoints.append({"camera": asdict(cam), "brush": asdict(brush)})

    def clear(self) -> None:
        self.camera_waypoints.clear()
        self.brush_waypoints.clear()
        self.combined_waypoints.clear()

    def to_payload(self, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "format": "mpr_plane_waypoints_v2",
            "camera_waypoints": [asdict(x) for x in self.camera_waypoints],
            "brush_waypoints": [asdict(x) for x in self.brush_waypoints],
            "combined_waypoints": self.combined_waypoints,
            "settings": settings or {},
        }

    def save(self, out_json: Path, settings: Optional[Dict[str, Any]] = None) -> None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(self.to_payload(settings=settings), indent=2), encoding="utf-8")

    def load_payload(self, payload: Dict[str, Any]) -> None:
        self.clear()
        for c in payload.get("camera_waypoints", []):
            self.camera_waypoints.append(camera_state_from_dict(c))
        for b in payload.get("brush_waypoints", []):
            self.brush_waypoints.append(brush_state_from_dict(b))
        for cb in payload.get("combined_waypoints", []):
            cam = camera_state_from_dict(cb.get("camera", {}))
            brush = brush_state_from_dict(cb.get("brush", {}))
            self.combined_waypoints.append({"camera": asdict(cam), "brush": asdict(brush)})


def camera_state_from_dict(d: Dict[str, Any]) -> CameraState:
    pos = list(map(float, d.get("position", d.get("center", [0.5, 0.5, 0.5]))))
    yaw = float(d.get("yaw", math.radians(float(d.get("euler_deg", [0.0, 0.0, 0.0])[1])) if "euler_deg" in d else 0.0))
    pitch = float(d.get("pitch", math.radians(float(d.get("euler_deg", [0.0, 0.0, 0.0])[0])) if "euler_deg" in d else 0.0))
    euler = list(map(float, d.get("euler_deg", [math.degrees(pitch), math.degrees(yaw), 0.0])))
    n = list(map(float, d.get("plane_normal", yaw_pitch_to_normal(yaw, pitch).tolist())))
    return CameraState(
        t=float(d.get("t", 0.0)),
        position=pos,
        euler_deg=euler,
        plane_normal=n,
        yaw=yaw,
        pitch=pitch,
        scale=float(d.get("scale", 0.55)),
        view_mode=str(d.get("view_mode", "single")),
        heuristics=dict(d.get("heuristics", {})),
        heuristic_images=dict(d.get("heuristic_images", {})),
    )


def brush_state_from_dict(d: Dict[str, Any]) -> BrushState:
    return BrushState(
        t=float(d.get("t", 0.0)),
        mouse_uv=list(map(float, d.get("mouse_uv", [0.5, 0.5]))),
        strength=float(d.get("strength", d.get("heap_depth", 0.22))),
        radius=float(d.get("radius", d.get("heap_radius", 0.18))),
        softness=float(d.get("softness", d.get("heap_softness", 0.06))),
        stretch=float(d.get("stretch", d.get("heap_stretch", 1.0))),
        direction=float(d.get("direction", d.get("heap_dir", -1.0))),
        enabled=bool(d.get("enabled", d.get("heap_enable", True))),
    )


def smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _safe_point(points: np.ndarray, i: int) -> np.ndarray:
    return points[int(np.clip(i, 0, len(points) - 1))]


def interpolate_points(points: np.ndarray, segment_index: int, u: float, mode: str) -> np.ndarray:
    """Evaluate an interpolated point/vector for one segment of a keyframed path."""
    if len(points) == 0:
        return np.zeros(1, dtype=np.float32)
    if len(points) == 1:
        return points[0]

    i = int(np.clip(segment_index, 0, len(points) - 2))
    u = float(np.clip(u, 0.0, 1.0))
    mode = (mode or "catmull").lower()

    p0 = _safe_point(points, i - 1)
    p1 = _safe_point(points, i)
    p2 = _safe_point(points, i + 1)
    p3 = _safe_point(points, i + 2)

    if mode in ("linear", "line"):
        return (1.0 - u) * p1 + u * p2

    if mode in ("smooth", "smoothstep", "ease"):
        s = smoothstep01(u)
        return (1.0 - s) * p1 + s * p2

    if mode in ("hermite", "hamilton"):
        # Cubic Hermite with Catmull-style finite-difference tangents.
        m1 = 0.5 * (p2 - p0)
        m2 = 0.5 * (p3 - p1)
        u2, u3 = u * u, u * u * u
        h00 = 2*u3 - 3*u2 + 1
        h10 = u3 - 2*u2 + u
        h01 = -2*u3 + 3*u2
        h11 = u3 - u2
        return h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2

    if mode in ("bezier", "bezier_catmull"):
        # Convert local Catmull tangents into cubic Bezier handles.
        b0 = p1
        b1 = p1 + (p2 - p0) / 6.0
        b2 = p2 - (p3 - p1) / 6.0
        b3 = p2
        omt = 1.0 - u
        return (omt**3) * b0 + 3 * (omt**2) * u * b1 + 3 * omt * (u**2) * b2 + (u**3) * b3

    # Default: Catmull-Rom spline.
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2*p1) + (-p0 + p2) * u + (2*p0 - 5*p1 + 4*p2 - p3) * u2 + (-p0 + 3*p1 - 3*p2 + p3) * u3)


def cubic_catmull_rom(points: np.ndarray, num: int) -> np.ndarray:
    if len(points) < 2:
        return points
    out = []
    for i in range(len(points) - 1):
        for t in np.linspace(0.0, 1.0, num=max(1, int(num)), endpoint=False):
            out.append(interpolate_points(points, i, float(t), "catmull"))
    out.append(points[-1])
    return np.array(out, dtype=np.float32)


def add_noise(path: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return path
    return path + np.random.normal(0.0, sigma, size=path.shape)


def _hash_noise01(i: int, channel: int = 0) -> float:
    # Deterministic tiny hash, no external noise dependency.
    x = (int(i) * 374761393 + int(channel) * 668265263) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) * 1274126177 & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF) / float(0xFFFFFF)


def value_noise_1d(t: float, channel: int = 0) -> float:
    i0 = math.floor(t)
    i1 = i0 + 1
    f = t - i0
    s = smoothstep01(f)
    a = _hash_noise01(i0, channel) * 2.0 - 1.0
    b = _hash_noise01(i1, channel) * 2.0 - 1.0
    return (1.0 - s) * a + s * b


def camera_noise_vec(t: float, noise_type: str, amp: float, freq: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return position noise and angular noise [yaw,pitch] for the camera path."""
    if amp <= 0.0 or noise_type in (None, "", "none"):
        return np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
    noise_type = str(noise_type).lower()
    freq = max(1e-6, float(freq))
    x = float(t) * freq
    if noise_type in ("perlin", "value"):
        p = np.array([value_noise_1d(x, 0), value_noise_1d(x, 1), value_noise_1d(x, 2)], dtype=np.float32) * float(amp)
        a = np.array([value_noise_1d(x, 3), value_noise_1d(x, 4)], dtype=np.float32) * float(amp) * 0.75
        return p, a
    if noise_type in ("wobble", "warble", "sin"):
        p = np.array([
            math.sin(x * math.tau + 0.1),
            math.sin(x * math.tau * 0.73 + 1.7),
            math.sin(x * math.tau * 1.31 + 2.4),
        ], dtype=np.float32) * float(amp)
        a = np.array([
            math.sin(x * math.tau * 0.61 + 0.4),
            math.sin(x * math.tau * 0.89 + 1.1),
        ], dtype=np.float32) * float(amp) * 0.75
        return p, a
    if noise_type in ("brownian", "fbm", "fractal"):
        p = np.zeros(3, dtype=np.float32)
        a = np.zeros(2, dtype=np.float32)
        gain = 1.0
        norm = 0.0
        for octave in range(5):
            xo = x * (2.0 ** octave)
            p += np.array([value_noise_1d(xo, 10 + octave*5 + c) for c in range(3)], dtype=np.float32) * gain
            a += np.array([value_noise_1d(xo, 30 + octave*5 + c) for c in range(2)], dtype=np.float32) * gain
            norm += gain
            gain *= 0.5
        p = p / max(norm, 1e-6) * float(amp)
        a = a / max(norm, 1e-6) * float(amp) * 0.75
        return p, a
    if noise_type in ("random", "jitter"):
        rng = np.random.default_rng(int(max(0, math.floor(x * 60.0))))
        p = rng.normal(0.0, float(amp), size=3).astype(np.float32)
        a = rng.normal(0.0, float(amp) * 0.75, size=2).astype(np.float32)
        return p, a
    return np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)


def build_timeline(input_json: Path, out_json: Path, samples_per_segment: int, noise_sigma: float, interpolation: str = "catmull") -> None:
    """Offline helper: load waypoint JSON and write sampled frame positions."""
    data = json.loads(input_json.read_text(encoding="utf-8"))
    cams = data.get("camera_waypoints", [])
    if not cams and data.get("combined_waypoints"):
        cams = [x.get("camera", {}) for x in data.get("combined_waypoints", [])]
    if len(cams) < 2:
        raise ValueError("Need at least two camera or combined waypoints for timeline interpolation.")

    pos = np.array([camera_state_from_dict(c).position for c in cams], dtype=np.float32)
    yaw_pitch_scale = np.array([[camera_state_from_dict(c).yaw, camera_state_from_dict(c).pitch, camera_state_from_dict(c).scale] for c in cams], dtype=np.float32)

    curve_pos = []
    curve_yps = []
    for i in range(len(pos) - 1):
        for u in np.linspace(0.0, 1.0, num=max(1, samples_per_segment), endpoint=False):
            curve_pos.append(interpolate_points(pos, i, float(u), interpolation))
            curve_yps.append(interpolate_points(yaw_pitch_scale, i, float(u), interpolation))
    curve_pos.append(pos[-1])
    curve_yps.append(yaw_pitch_scale[-1])

    curve_pos = add_noise(np.array(curve_pos, dtype=np.float32), noise_sigma)
    curve_yps = np.array(curve_yps, dtype=np.float32)
    timeline = []
    for i, (p, r) in enumerate(zip(curve_pos, curve_yps)):
        timeline.append({"frame": i, "position": p.tolist(), "yaw": float(r[0]), "pitch": float(r[1]), "scale": float(r[2])})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"timeline": timeline, "interpolation": interpolation}, indent=2), encoding="utf-8")



# ============================================================
# Lightweight screen/slice heuristics for HUD display
# ============================================================

def _resize_rgb_for_analysis(rgb: np.ndarray, max_w: int = 192, max_h: int = 128) -> np.ndarray:
    """Small CPU preview used for cheap per-frame morphology heuristics."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    h, w = arr.shape[:2]
    if w <= max_w and h <= max_h:
        return arr
    scale = min(max_w / max(1, w), max_h / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return np.asarray(Image.fromarray(arr, mode="RGB").resize((nw, nh), Image.BILINEAR), dtype=np.uint8)


def _component_stats(mask: np.ndarray, min_area: int = 10) -> Dict[str, Any]:
    """8-connected component stats with a simple circularity estimate."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros_like(m, dtype=bool)
    components: List[Dict[str, float]] = []
    nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]

    for yy in range(h):
        for xx in range(w):
            if not m[yy, xx] or seen[yy, xx]:
                continue
            q = deque([(xx, yy)])
            seen[yy, xx] = True
            area = 0
            xmin = xmax = xx
            ymin = ymax = yy
            perimeter = 0
            while q:
                x, y = q.popleft()
                area += 1
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
                # 4-neighbor boundary perimeter estimate.
                for dx4, dy4 in ((1,0), (-1,0), (0,1), (0,-1)):
                    nx, ny = x + dx4, y + dy4
                    if nx < 0 or nx >= w or ny < 0 or ny >= h or not m[ny, nx]:
                        perimeter += 1
                for dx, dy in nbrs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if area >= min_area:
                bw = xmax - xmin + 1
                bh = ymax - ymin + 1
                bbox_area = max(1, bw * bh)
                circularity = (4.0 * math.pi * area / max(1.0, float(perimeter * perimeter))) if perimeter > 0 else 0.0
                aspect = bw / max(1.0, float(bh))
                components.append({
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "bbox_fill": float(area / bbox_area),
                    "circularity": float(circularity),
                    "aspect": float(aspect),
                })

    circle_like = 0
    for c in components:
        aspect_ok = 0.55 <= c["aspect"] <= 1.80
        # Discrete masks rarely reach perfect 1.0 circularity; 0.45 catches round-ish blobs.
        if aspect_ok and c["circularity"] >= 0.45 and c["bbox_fill"] >= 0.42:
            circle_like += 1

    largest = max((c["area"] for c in components), default=0.0)
    return {
        "blob_count": int(len(components)),
        "largest_blob_area": float(largest),
        "circle_count": int(circle_like),
        "components": components,
    }


def analyze_slice_heuristics(rgb: np.ndarray) -> Dict[str, Any]:
    """
    Approximate visual metrics for the currently displayed slice.

    These are deliberately heuristic, not medical segmentation. They estimate:
    foreground fill, connected blobs, round/circle-like components, and whether
    the visible content looks more bone-like or flesh-like from simple color and
    brightness cues.
    """
    small = _resize_rgb_for_analysis(rgb)
    rgbf = small.astype(np.float32)
    gray = 0.2126 * rgbf[..., 0] + 0.7152 * rgbf[..., 1] + 0.0722 * rgbf[..., 2]
    mx = rgbf.max(axis=2)
    mn = rgbf.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)

    nonblack = gray > 8.0
    if np.any(nonblack):
        dynamic = float(np.percentile(gray[nonblack], 25))
        threshold = max(10.0, min(70.0, dynamic * 0.70))
    else:
        threshold = 10.0
    mask = gray > threshold
    fill_ratio = float(mask.mean())

    min_area = max(8, int(mask.size * 0.0007))
    comps = _component_stats(mask, min_area=min_area)

    # Bone heuristic: bright, low-saturation/white-ish structures.
    bone_mask = mask & (gray > 150.0) & (sat < 0.32)
    # Flesh heuristic: red/pink/brown dominant, usually medium luminance and saturated.
    r, g, b = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    flesh_mask = mask & (r > g * 1.06) & (r > b * 1.08) & (gray > 25.0) & (gray < 235.0) & (sat > 0.08)

    denom = max(1, int(mask.sum()))
    bone_ratio = float(bone_mask.sum() / denom)
    flesh_ratio = float(flesh_mask.sum() / denom)

    if fill_ratio < 0.015:
        tissue = "mostly empty / black"
    elif bone_ratio > max(0.12, flesh_ratio * 1.25):
        tissue = "mostly bone-like"
    elif flesh_ratio > max(0.10, bone_ratio * 1.10):
        tissue = "mostly flesh-like"
    elif bone_ratio > 0.05 and flesh_ratio > 0.05:
        tissue = "mixed flesh + bone"
    else:
        tissue = "unclear / low contrast"

    return {
        "fill_ratio": fill_ratio,
        "blob_count": comps["blob_count"],
        "largest_blob_area": comps["largest_blob_area"],
        "circle_count": comps["circle_count"],
        "bone_ratio": bone_ratio,
        "flesh_ratio": flesh_ratio,
        "tissue": tissue,
        "analysis_size": [int(small.shape[1]), int(small.shape[0])],
    }


def build_blob_debug_visual(rgb: np.ndarray, max_w: int = 256, max_h: int = 192) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a debug RGB image showing the connected blobs the heuristic sees."""
    small = _resize_rgb_for_analysis(rgb, max_w=max_w, max_h=max_h)
    rgbf = small.astype(np.float32)
    gray = 0.2126 * rgbf[..., 0] + 0.7152 * rgbf[..., 1] + 0.0722 * rgbf[..., 2]
    nonblack = gray > 8.0
    if np.any(nonblack):
        dynamic = float(np.percentile(gray[nonblack], 25))
        threshold = max(10.0, min(70.0, dynamic * 0.70))
    else:
        threshold = 10.0
    mask = gray > threshold

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    palette = [
        (255, 80, 80), (80, 210, 255), (255, 220, 90), (140, 255, 120),
        (220, 120, 255), (255, 150, 80), (120, 255, 210), (255, 120, 180),
    ]
    comps = []
    nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    for yy in range(h):
        for xx in range(w):
            if not mask[yy, xx] or seen[yy, xx]:
                continue
            q = deque([(xx, yy)])
            seen[yy, xx] = True
            pts = []
            xmin = xmax = xx
            ymin = ymax = yy
            while q:
                x, y = q.popleft()
                pts.append((x, y))
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
                for dx, dy in nbrs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if len(pts) < max(6, int(mask.size * 0.0007)):
                continue
            comps.append({"pts": pts, "bbox": (xmin, ymin, xmax + 1, ymax + 1), "area": len(pts)})

    comps.sort(key=lambda c: c["area"], reverse=True)
    for idx, comp in enumerate(comps):
        col = np.array(palette[idx % len(palette)], dtype=np.uint8)
        for x, y in comp["pts"]:
            vis[y, x] = col
        xmin, ymin, xmax, ymax = comp["bbox"]
        vis[ymin:ymin+1, xmin:xmax] = 255
        vis[max(0, ymax-1):ymax, xmin:xmax] = 255
        vis[ymin:ymax, xmin:xmin+1] = 255
        vis[ymin:ymax, max(0, xmax-1):xmax] = 255
        cx = int(round((xmin + xmax - 1) * 0.5))
        cy = int(round((ymin + ymax - 1) * 0.5))
        vis[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2)] = 255

    img = Image.fromarray(vis, mode="RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, int(min(w, h) * 0.07)))
    except Exception:
        font = ImageFont.load_default()
    for idx, comp in enumerate(comps[:12], start=1):
        xmin, ymin, xmax, ymax = comp["bbox"]
        draw.text((xmin + 2, max(0, ymin - 10)), str(idx), fill=(255, 255, 255), font=font)

    meta = {
        "blob_count": int(len(comps)),
        "threshold": float(threshold),
        "size": [int(w), int(h)],
        "areas": [int(c["area"]) for c in comps[:24]],
    }
    return np.asarray(img, dtype=np.uint8), meta


def prepare_volume_rgb_uint8(vol: np.ndarray) -> np.ndarray:
    """Accept (Z,H,W), (Z,H,W,1), or (Z,H,W,3+) uint8 volumes.

    Important memory rule: do NOT expand grayscale volumes to RGB here.
    A 711x2000x1019 gray auxiliary volume is ~1.45 GB; repeating it
    to RGB would turn each skeleton/gradient volume into ~4.35 GB and
    can make tiny UI uploads fail with MemoryError.  Keep memmaps/views
    alive and expand one sampled slice/layer at a time only when needed.
    """
    arr = np.asarray(vol)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[3] == 1:
        return arr[..., 0]
    if arr.ndim == 4 and arr.shape[3] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unexpected volume shape for RGB conversion: {arr.shape}")


def normal_to_yaw_pitch(n: np.ndarray) -> Tuple[float, float]:
    n = normalize(np.asarray(n, dtype=np.float32))
    pitch = float(math.asin(np.clip(float(n[2]), -1.0, 1.0)))
    cp = max(1e-8, float(math.cos(pitch)))
    yaw = float(math.atan2(float(n[0]) / cp, float(n[1]) / cp))
    return yaw, pitch


def sample_plane_from_volume_rgb(volume_rgb: np.ndarray, center: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray, scale_u: float, scale_v: float, *, out_w: int = 192, out_h: int = 128) -> np.ndarray:
    """CPU nearest-neighbor slice sample from a RGB/BGR or grayscale volume.

    Returns RGB-shaped (out_h,out_w,3).  Grayscale source volumes are expanded
    only for this small sampled image, not for the whole 3D volume.
    """
    Z, H, W = volume_rgb.shape[:3]
    xs = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)
    sx, sy = np.meshgrid(xs, ys)
    p = (np.asarray(center, np.float32)[None, None, :]
         + np.asarray(axis_u, np.float32)[None, None, :] * (sx[..., None] * float(scale_u))
         + np.asarray(axis_v, np.float32)[None, None, :] * (sy[..., None] * float(scale_v)))
    valid = np.all((p >= 0.0) & (p <= 1.0), axis=2)
    xi = np.clip(np.rint(p[..., 0] * (W - 1)).astype(np.int32), 0, W - 1)
    yi = np.clip(np.rint(p[..., 1] * (H - 1)).astype(np.int32), 0, H - 1)
    zi = np.clip(np.rint(p[..., 2] * (Z - 1)).astype(np.int32), 0, Z - 1)
    out = np.asarray(volume_rgb[zi, yi, xi], dtype=np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    else:
        out = np.asarray(out[..., :3], dtype=np.uint8).copy()
    out[~valid] = 0
    return out


def sample_curved_plane_from_volume_rgb(
    volume_rgb: np.ndarray,
    center: Sequence[float],
    axis_u: Sequence[float],
    axis_v: Sequence[float],
    axis_n: Sequence[float],
    scale_u: float,
    scale_v: float,
    *,
    curved_kind: int = 0,
    curved_amp: float = 0.075,
    curved_radius: float = 1.0,
    out_w: int = 192,
    out_h: int = 128,
) -> np.ndarray:
    """CPU nearest-neighbor sample for the curved/parabolic slicing plane."""
    Z, H, W = volume_rgb.shape[:3]
    xs = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)
    sx, sy = np.meshgrid(xs, ys)
    u = np.asarray(axis_u, np.float32)
    v = np.asarray(axis_v, np.float32)
    n = np.asarray(axis_n, np.float32)
    c = np.asarray(center, np.float32)
    p = (c[None, None, :]
         + u[None, None, :] * (sx[..., None] * float(scale_u))
         + v[None, None, :] * (sy[..., None] * float(scale_v)))

    r = max(abs(float(curved_radius)), 1e-4)
    qx = sx / r
    qy = sy / r
    k = int(curved_kind)
    if k == 0:
        h = qx * qx + qy * qy
    elif k == 1:
        h = qx * qx - qy * qy
    elif k == 2:
        h = qx * qx
    elif k == 3:
        h = qy * qy
    else:
        h = qx * qx + qy * qy + 0.22 * np.sin(2.0 * np.pi * qx) * np.cos(2.0 * np.pi * qy)
    p = p + n[None, None, :] * (h[..., None] * float(curved_amp))

    valid = np.all((p >= 0.0) & (p <= 1.0), axis=2)
    xi = np.clip(np.rint(p[..., 0] * (W - 1)).astype(np.int32), 0, W - 1)
    yi = np.clip(np.rint(p[..., 1] * (H - 1)).astype(np.int32), 0, H - 1)
    zi = np.clip(np.rint(p[..., 2] * (Z - 1)).astype(np.int32), 0, Z - 1)
    out = np.asarray(volume_rgb[zi, yi, xi], dtype=np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    else:
        out = np.asarray(out[..., :3], dtype=np.uint8).copy()
    out[~valid] = 0
    return out


def compute_interest_metrics(gradient_rgb: np.ndarray, skeleton_rgb: np.ndarray) -> Dict[str, float]:
    g = _resize_rgb_for_analysis(gradient_rgb, 128, 96)
    s = _resize_rgb_for_analysis(skeleton_rgb, 128, 96)
    ggray = g[..., 0].astype(np.float32) / 255.0
    sgray = s[..., 0].astype(np.float32) / 255.0
    s_mask = sgray > 0.18
    g_mask = ggray > 0.45
    overlap = g_mask & s_mask
    edge = np.abs(np.diff(ggray, axis=1)).mean() + np.abs(np.diff(ggray, axis=0)).mean()
    skeleton_fill = float(s_mask.mean())
    gradient_mean = float(ggray.mean())
    gradient_fill = float(g_mask.mean())
    overlap_ratio = float(overlap.mean())
    # White distance field = more flesh-like bulk away from the skeleton, but
    # we still want some skeleton evidence in-plane so the structure is readable.
    readability = min(1.0, skeleton_fill * 12.0)
    score = (0.48 * gradient_mean + 0.22 * gradient_fill + 0.18 * readability + 0.12 * min(1.0, edge * 4.0))
    if skeleton_fill < 0.002:
        score *= 0.55
    return {
        "score": float(score),
        "gradient_mean": gradient_mean,
        "gradient_fill": gradient_fill,
        "skeleton_fill": skeleton_fill,
        "overlap_ratio": overlap_ratio,
        "edge_strength": float(edge),
    }


def largest_blob_center(mask: np.ndarray) -> Tuple[Tuple[float, float], Tuple[int, int, int, int]]:
    comps = _component_stats(mask, min_area=max(6, int(mask.size * 0.0005)))
    if not comps.get("components"):
        h, w = mask.shape
        return (w * 0.5, h * 0.5), (0, 0, w, h)
    # Recompute bbox of largest component directly.
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros_like(m, dtype=bool)
    best = None
    nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    for yy in range(h):
        for xx in range(w):
            if not m[yy, xx] or seen[yy, xx]:
                continue
            q = deque([(xx, yy)])
            seen[yy, xx] = True
            pts = []
            while q:
                x, y = q.popleft(); pts.append((x, y))
                for dx, dy in nbrs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; q.append((nx, ny))
            if best is None or len(pts) > len(best):
                best = pts
    if not best:
        return (w * 0.5, h * 0.5), (0, 0, w, h)
    xs = np.array([p[0] for p in best], dtype=np.float32)
    ys = np.array([p[1] for p in best], dtype=np.float32)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (float(xs.mean()), float(ys.mean())), bbox


def centered_image_from_mask(rgb: np.ndarray) -> np.ndarray:
    gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    mask = gray > max(8.0, float(np.percentile(gray, 55)))
    (cx, cy), _ = largest_blob_center(mask)
    h, w = mask.shape
    tx = int(round(w * 0.5 - cx))
    ty = int(round(h * 0.5 - cy))
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (tx, ty))
    return np.asarray(canvas, dtype=np.uint8)


def frame_descriptor(rgb: np.ndarray) -> np.ndarray:
    h = analyze_slice_heuristics(rgb)
    return np.array([
        float(h.get("fill_ratio", 0.0)),
        float(h.get("blob_count", 0.0)),
        float(h.get("circle_count", 0.0)),
        float(h.get("bone_ratio", 0.0)),
        float(h.get("flesh_ratio", 0.0)),
    ], dtype=np.float32)

# ============================================================
# Auto motion + SpaceMouse helpers
# ============================================================

class AutoMotionController:
    def __init__(self):
        self.enabled = True
        self.modulate_heap = True
        self.center_vel = np.zeros(3, dtype=np.float32)
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0
        self.seed = time.perf_counter()

        # Ornstein-Uhlenbeck style drift parameters
        self.center_sigma = 0.085
        self.center_damping = 1.75
        self.angular_sigma = 0.55
        self.angular_damping = 1.4
        self.center_speed_cap = 0.22
        self.angular_speed_cap = 0.75

        self.heap_depth_base = 0.22
        self.heap_radius_base = 0.18
        self.heap_softness_base = 0.06
        self.heap_depth_amp = 0.08
        self.heap_radius_amp = 0.05
        self.heap_softness_amp = 0.015
        self.heap_freq_1 = 0.21
        self.heap_freq_2 = 0.13
        self.heap_freq_3 = 0.31

    def reset(self, center, yaw, pitch, heap_depth, heap_radius, heap_softness):
        self.center_vel[:] = 0.0
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0
        self.heap_depth_base = float(heap_depth)
        self.heap_radius_base = float(heap_radius)
        self.heap_softness_base = float(heap_softness)

    def step(self, app, dt, now):
        if not self.enabled:
            return

        dt = float(np.clip(dt, 1.0 / 240.0, 0.05))

        self.center_vel += np.random.normal(
            loc=0.0,
            scale=self.center_sigma * math.sqrt(dt),
            size=3,
        ).astype(np.float32)
        self.center_vel -= self.center_vel * (self.center_damping * dt)
        speed = float(np.linalg.norm(self.center_vel))
        if speed > self.center_speed_cap:
            self.center_vel *= self.center_speed_cap / max(speed, 1e-8)

        self.yaw_vel += float(np.random.normal(0.0, self.angular_sigma * math.sqrt(dt)))
        self.pitch_vel += float(np.random.normal(0.0, self.angular_sigma * math.sqrt(dt)))
        self.yaw_vel -= self.yaw_vel * (self.angular_damping * dt)
        self.pitch_vel -= self.pitch_vel * (self.angular_damping * dt)
        self.yaw_vel = float(np.clip(self.yaw_vel, -self.angular_speed_cap, self.angular_speed_cap))
        self.pitch_vel = float(np.clip(self.pitch_vel, -self.angular_speed_cap, self.angular_speed_cap))

        app.center += self.center_vel * dt
        for axis in range(3):
            if app.center[axis] < 0.03:
                app.center[axis] = 0.03
                self.center_vel[axis] = abs(self.center_vel[axis]) * 0.8
            elif app.center[axis] > 0.97:
                app.center[axis] = 0.97
                self.center_vel[axis] = -abs(self.center_vel[axis]) * 0.8

        app.yaw += self.yaw_vel * dt
        app.pitch += self.pitch_vel * dt
        app.pitch = float(np.clip(app.pitch, -1.45, 1.45))
        app._update_plane_axes()

        if self.modulate_heap:
            phase = now - self.seed
            app.heap_depth = float(np.clip(
                self.heap_depth_base
                + self.heap_depth_amp * math.sin(phase * self.heap_freq_1 * math.tau)
                + 0.025 * math.sin(phase * 0.47 * math.tau + 0.7),
                0.02,
                0.95,
            ))
            app.heap_radius = float(np.clip(
                self.heap_radius_base
                + self.heap_radius_amp * math.sin(phase * self.heap_freq_2 * math.tau + 1.2),
                0.03,
                0.90,
            ))
            app.heap_softness = float(np.clip(
                self.heap_softness_base
                + self.heap_softness_amp * math.sin(phase * self.heap_freq_3 * math.tau + 2.1),
                0.0,
                min(app.heap_radius * 0.9, 0.45),
            ))


class SpaceMouseController:
    def __init__(self):
        self.available = pyspacemouse is not None
        self.enabled = self.available
        self._ctx = None
        self.device = None
        self.last_buttons = []

    def connect(self):
        if not self.available or self.device is not None:
            return
        try:
            ctx = pyspacemouse.open()
            if hasattr(ctx, "__enter__"):
                self._ctx = ctx
                self.device = ctx.__enter__()
            else:
                self.device = ctx
            self.enabled = self.device is not None
            if self.enabled:
                print("[3Dconnexion] connected")
        except Exception as exc:
            self.enabled = False
            self.device = None
            print(f"[3Dconnexion] unavailable: {exc}")

    def close(self):
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._ctx = None
        self.device = None

    def apply(self, app, dt):
        if not self.enabled:
            return False
        if self.device is None:
            self.connect()
            if self.device is None:
                return False

        try:
            state = self.device.read()
        except Exception as exc:
            print(f"[3Dconnexion] read failed: {exc}")
            self.close()
            self.enabled = False
            return False

        if state is None:
            return False

        move_speed = 0.34
        rot_speed = 1.65
        scale_speed = 0.60
        heap_speed = 0.45

        tx = float(getattr(state, "x", 0.0))
        ty = float(getattr(state, "y", 0.0))
        tz = float(getattr(state, "z", 0.0))
        rr = float(getattr(state, "roll", 0.0))
        rp = float(getattr(state, "pitch", 0.0))
        ry = float(getattr(state, "yaw", 0.0))
        buttons = list(getattr(state, "buttons", []) or [])

        dead = 0.05
        def dz(v):
            return 0.0 if abs(v) < dead else v

        tx, ty, tz, rr, rp, ry = map(dz, (tx, ty, tz, rr, rp, ry))

        app.center += app.u * (tx * move_speed * dt)
        app.center += app.v * (-ty * move_speed * dt)
        app.center += app.n * (-tz * move_speed * dt)
        app.center[:] = np.clip(app.center, 0.0, 1.0)

        app.yaw += rr * rot_speed * dt
        app.pitch += rp * rot_speed * dt
        app.pitch = float(np.clip(app.pitch, -1.55, 1.55))
        app.scale = float(np.clip(app.scale * (1.0 - ry * scale_speed * dt), 0.05, 2.0))

        # Extra control: press button 0 / 1 to dig or pull heap depth
        if len(buttons) > 0 and buttons[0]:
            app.heap_depth = float(np.clip(app.heap_depth + heap_speed * dt, 0.0, 1.0))
        if len(buttons) > 1 and buttons[1]:
            app.heap_depth = float(np.clip(app.heap_depth - heap_speed * dt, 0.0, 1.0))

        return True

# ============================================================
# Main app
# ============================================================

class MPRPlaneUI(mglw.WindowConfig):
    # Compute shaders need OpenGL 4.3+. The RTX/NVIDIA path supports this;
    # if compute compilation fails, frame FX falls back to the existing fragment pass.
    gl_version = (4, 3)
    title = "MPR Plane UI — waypoint camera timeline + tri-plane viewer"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    # Set from command-line arguments in __main__.
    waypoint_json_path = Path("out") / "waypoints.json"
    load_waypoints_path: Optional[Path] = None
    playback_on_start = False
    startup_view_mode = "single"
    seconds_per_segment = 2.0
    interpolation_mode = "catmull"
    noise_type = "none"
    noise_amp = 0.0
    noise_freq = 1.0
    playback_loop = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---------- load volumes ----------
        # Main color volume plus optional grayscale aux volumes for skeleton and
        # gradient-distance / signed-distance style views.
        base = Path(__file__).resolve().parent
        main_candidates = [
            base / "threshold_images" / "volume_uint8.npy",
            base.parent / "threshold_images" / "volume_uint8.npy",
            Path("threshold_images") / "volume_uint8.npy",
        ]
        main_path = next((q for q in main_candidates if q.exists()), main_candidates[0])
        if not main_path.exists():
            tried = "\n  ".join(str(q) for q in main_candidates)
            raise FileNotFoundError(f"Missing color volume_uint8.npy. Tried:\n  {tried}")

        def _pick(cands):
            return next((q for q in cands if q.exists()), None)

        gradient_candidates = [
            # grouped-output layout
            base / "threshold_images" / "gradient_distance" / "volume_uint8.npy",
            base.parent / "threshold_images" / "gradient_distance" / "volume_uint8.npy",
            Path("threshold_images") / "gradient_distance" / "volume_uint8.npy",
            base / "gradient_distance" / "volume_uint8.npy",
            base.parent / "gradient_distance" / "volume_uint8.npy",
            # flat-output layout from earlier scripts
            base / "threshold_images" / "dog_gradient_gray_uint8.npy",
            base.parent / "threshold_images" / "dog_gradient_gray_uint8.npy",
            Path("threshold_images") / "dog_gradient_gray_uint8.npy",
            base / "threshold_images" / "gradient_gray_uint8.npy",
            base.parent / "threshold_images" / "gradient_gray_uint8.npy",
            Path("threshold_images") / "gradient_gray_uint8.npy",
        ]
        skeleton_candidates = [
            # grouped-output layout
            base / "threshold_images" / "skeletons" / "volume_uint8.npy",
            base.parent / "threshold_images" / "skeletons" / "volume_uint8.npy",
            Path("threshold_images") / "skeletons" / "volume_uint8.npy",
            base / "skeletons" / "volume_uint8.npy",
            base.parent / "skeletons" / "volume_uint8.npy",
            # flat-output layout from earlier scripts
            base / "threshold_images" / "dog_skeleton_uint8.npy",
            base.parent / "threshold_images" / "dog_skeleton_uint8.npy",
            Path("threshold_images") / "dog_skeleton_uint8.npy",
            base / "threshold_images" / "skeleton_uint8.npy",
            base.parent / "threshold_images" / "skeleton_uint8.npy",
            Path("threshold_images") / "skeleton_uint8.npy",
        ]
        # Wildcard fallback: when batch scripts name files by dataset prefix.
        for root in [base / "threshold_images", base.parent / "threshold_images", Path("threshold_images")]:
            if root.exists():
                gradient_candidates.extend(sorted(root.glob("*gradient*uint8.npy")))
                gradient_candidates.extend(sorted(root.glob("*distance*uint8.npy")))
                skeleton_candidates.extend(sorted(root.glob("*skeleton*uint8.npy")))

        self.main_volume_path = main_path
        self.gradient_volume_path = _pick(gradient_candidates)
        self.skeleton_volume_path = _pick(skeleton_candidates)

        V_main = np.load(main_path, mmap_mode="r")
        self.V = prepare_volume_rgb_uint8(V_main)
        self.Z, self.H, self.W = self.V.shape[:3]

        self.V_gradient = None
        self.V_skeleton = None
        if self.gradient_volume_path is not None:
            vg = prepare_volume_rgb_uint8(np.load(self.gradient_volume_path, mmap_mode="r"))
            if vg.shape[:3] != self.V.shape[:3]:
                raise ValueError(f"Gradient volume shape mismatch: {vg.shape} vs {self.V.shape}")
            self.V_gradient = vg
        if self.skeleton_volume_path is not None:
            vs = prepare_volume_rgb_uint8(np.load(self.skeleton_volume_path, mmap_mode="r"))
            if vs.shape[:3] != self.V.shape[:3]:
                raise ValueError(f"Skeleton volume shape mismatch: {vs.shape} vs {self.V.shape}")
            self.V_skeleton = vs

        # Press P to save an orthogonal tri-plane snapshot here.
        # The saved frontal/sagittal/transverse slices are driven by
        # the current moving plane center / red gizmo slab position.
        self.snapshot_dir = Path("out") / "triplane_snapshots"
        self.snapshot_count = 0
        self.pending_screen_save = False

        # Split-screen mode:
        #   "axis"  = frontal / sagittal / transverse fixed to volume XYZ axes
        #   "local" = three mutually perpendicular oblique planes driven by the red slab rotation
        self.view_mode = str(self.__class__.startup_view_mode)
        self.mouse_px = (0.0, 0.0)

        # Waypoint / timeline state. C, B, V write into this recorder.
        self.record_start_time = time.perf_counter()
        self.recorder = WaypointRecorder()
        self.waypoint_json_path = Path(self.__class__.waypoint_json_path)
        self.playback_loaded_path = self.__class__.load_waypoints_path
        self.playback_enabled = False
        self.playhead_seconds = 0.0
        self.seconds_per_segment_live = float(self.__class__.seconds_per_segment)
        self.interpolation_mode_live = str(self.__class__.interpolation_mode)
        self.noise_type_live = str(self.__class__.noise_type)
        self.noise_amp_live = float(self.__class__.noise_amp)
        self.noise_freq_live = float(self.__class__.noise_freq)
        self.playback_loop_live = bool(self.__class__.playback_loop)
        self.path_scrub_mode = False
        self._drag_path_scrub = False
        self.path_dirty = True
        self.path_vertex_count = 0

        # Display / interaction controls.
        self.show_gizmo = True
        self.auto_hide_cursor = True
        self.cursor_hidden = False
        self.cursor_hide_delay = 0.85
        self._last_input_time = time.perf_counter()

        # Local oblique trails disabled for performance: every frame clears normally.
        # Keeping alpha at 1.0 avoids faded history/ghosting and reduces blend work.
        self.local_accumulate_frames = False
        self.force_clear_next_frame = True
        self.local_slice_alpha = 1.0
        self.black_alpha_threshold = 5.0 / 255.0

        # Built-in PIL/texture HUD. This replaces the ImGui timeline panel.
        # It scales from the actual framebuffer size and is clickable.
        self.ui_visible = True
        self.ui_buttons: List[Dict[str, Any]] = []
        self.ui_scrub_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_fx_slider_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_blob_slider_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_cut_angle_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_fx_param1_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_fx_param2_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_curve_amp_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_hemo_oxy_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_hemo_deoxy_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_hemo_fresh_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_hemo_sg_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_fx_dropdown_rect: Optional[Tuple[int, int, int, int]] = None
        self.ui_show_rect: Optional[Tuple[int, int, int, int]] = None
        self.hide_all_overlays = False
        self._drag_ui_scrub = False
        self._drag_ui_fx_slider = False
        self._drag_ui_blob_slider = False
        self._drag_ui_cut_angle = False
        self._drag_ui_fx_param1 = False
        self._drag_ui_fx_param2 = False
        self._drag_ui_curve_amp = False
        self._drag_ui_hemo_oxy = False
        self._drag_ui_hemo_deoxy = False
        self._drag_ui_hemo_fresh = False
        self._drag_ui_hemo_sg = False
        self.ui_last_build_key = None
        self.ui_force_rebuild = True
        self.ui_tex = None
        self.ui_tex_size = (0, 0)
        # Separate full-screen preview texture. Do not reuse ui_tex here: pixel_grid
        # and CPU frame_fx draw a complete image before the HUD. Reusing ui_tex
        # overwrote the cached HUD texture, so the UI appeared to disappear in
        # pixel_grid/frame_fx modes until it was forced to rebuild.
        self.overlay_tex = None
        self.overlay_tex_size = (0, 0)
        self.ui_tab = "move"
        self.panel_visible = {"move": True, "timeline": True, "heuristics": True, "fx": True, "objects": True, "plane": True}

        # Curved slicing plane / plane-editor mode.
        # It bends the regular red U/V slicing plane along its normal before sampling.
        self.curved_plane_enable = True
        self.curved_plane_kind = 0
        self.curved_plane_kind_names = ["paraboloid", "saddle", "cylinder_u", "cylinder_v", "ripple_parabola"]
        self.curved_plane_amp = 0.075
        self.curved_plane_radius = 1.0
        self.show_curve_side_panels = False
        # Side panel modes:
        #   local_curved   = GPU-sampled mini panels using the same curved-plane sampler
        #   curve_profile  = minimal analytic curve diagrams, no volume sampling
        self.curve_side_panel_mode = "local_curved"
        # Fast path: draw live MPR panels directly to the screen instead of routing
        # through an extra per-panel FBO/composite pass. This is the closest path
        # to the small Brownian viewer and avoids CPU readback/upload.
        self.fast_direct_live_render = True

        self.main_display_variant = "normal"
        self.aux_from_main = False
        self.pixel_grid_x_metric = "hue"
        self.pixel_grid_y_metric = "value"
        self.pixel_grid_layout = "metric"   # metric / similar / swirl

        # Frame transformation / screen-FX modes.
        self.frame_transform_mode = "coral"
        self.frame_transform_strength = 0.65
        self.vector_flow_show_guides = True
        self.fx_backend = "cpu_cached"
        # Prefer compute shader for full-frame FX when OpenGL 4.3 is available.
        # The fragment shader fallback remains useful for older contexts.
        self.use_compute_frame_fx_live = bool(getattr(self.__class__, "use_compute_frame_fx", True))
        self.post_compute_prog = None
        self.post_compute_available = False
        self.post_compute_tex = None
        self.cut_pattern = "parallel"
        self.cut_offset_parallel = 0.08
        self.cut_offset_perp = 0.08
        self.cut_overlap = 0.0
        self.cut_angle_rad = 0.55
        self.cut_motion_mode = "fixed"   # fixed / sine / noise
        self.blob_pack_distance = 0.22
        self.fx_mode_dropdown_open = False
        self.fx_dropdown_scroll = 0
        self.fx_param_defaults = {
            "fleshswell": (0.65, 0.50),
            "mold": (0.65, 0.55),
            "vessels": (0.45, 0.60),
            "veinbranch": (0.70, 0.55),
            "grassfire": (0.55, 0.60),
            "amat": (0.50, 0.55),
            "springmass": (0.60, 0.55),
            "steiner": (0.65, 0.65),
            "poisson": (0.70, 0.45),
            "stretch": (0.50, 1.00),
            "meatexpansion": (0.65, 0.50),
            "inflation": (0.70, 0.60),
            "myoglobin": (0.65, 0.45),
            "fibertrack": (0.60, 0.60),
            "watermobility": (0.55, 0.55),
            "marbling": (0.55, 0.55),
        }
        self.fx_param_values = {k: [float(v[0]), float(v[1])] for k, v in self.fx_param_defaults.items()}

        # Lightweight caches to improve frame rate for CPU effects.
        self._view_state_version = 0
        self._pixel_grid_cache_key = None
        self._pixel_grid_cache_img = None
        self._frame_fx_cache_key = None
        self._frame_fx_cache_img = None
        self._scene3d_cache_key = None
        self._scene3d_cache_img = None

        # 3D primitive object editor (CPU prototype).
        self.scene_objects: List[Dict[str, Any]] = []
        self.selected_object_index = -1
        self.role_dropdown_open = False
        self.scene_objects_affect_image = True
        self.scene3d_show_objects = True

        # Seeded random-slice board view (non-camera layout mode).
        self.seed_slice_specs: List[Dict[str, Any]] = []
        self.seed_slice_layout = "fill"   # fill / similar
        self.seed_slice_base_seed = 1234

        # H now explicitly hides/shows the mouse cursor. Automatic hiding is off
        # by default so the H key has predictable behavior.
        self.auto_hide_cursor = False
        self.cursor_force_hidden = False

        # Current-slice visual heuristics shown in the HUD.
        # These are CPU morphology passes, so they are opt-in by default.
        # Leave them off while navigating for higher FPS, then turn them on from
        # the toolbar/heuristics tab only when you want live blob counters.
        self.analysis_enabled = False
        self.interest_recommend_live_enabled = False
        self.heuristics_visible = True
        self.heuristics_interval = 0.75
        self._last_heuristics_time = 0.0
        self.current_heuristics: Dict[str, Any] = {"paused": True, "reason": "analysis off"}
        self._heuristics_cache_key = None
        self.current_fx_analysis: Dict[str, Any] = {"paused": True, "reason": "waiting for changes"}
        self._fx_analysis_cache_key = None
        self.fx_analysis_visible = True
        self.hemo_thresholds = {"oxy": 0.58, "deoxy": 0.42, "fresh": 0.56, "savgol": 0.50}
        self.analysis_panel_cache: Dict[str, Dict[str, Any]] = {}
        self.analysis_panel_keys: Dict[str, Tuple[Any, ...]] = {}
        self.analysis_dirty_flags: Dict[str, bool] = {
            "heuristics": True,
            "fx": True,
            "myoglobin": True,
            "inflation": True,
            "meatexpansion": True,
            "marbling": True,
            "live_recompute": True,
        }
        self.live_recompute_cache_key = None
        self.live_recompute_cache_img = None
        self.blob_debug_visible = False
        self.current_blob_debug_image: Optional[np.ndarray] = None
        self.current_blob_debug_meta: Dict[str, Any] = {"paused": True}

        # Interest recommendation from gradient-distance + skeleton volumes.
        self.current_interest: Dict[str, Any] = {"paused": True, "reason": "live interest off"}
        self._last_interest_time = 0.0
        self.interest_interval = 1.25
        self.last_interest_recommendation: Optional[Dict[str, Any]] = None
        self.last_blob_dense_recommendation: Optional[Dict[str, Any]] = None

        # Live color filtering / highlighting.
        self.color_filter_mode = "none"      # none / isolate / hide / highlight
        self.color_filter_target = "red"     # red / green / blue / white / flesh / dark / bright
        self.color_filter_strength = 0.85
        self.timeline_color_marks: List[Dict[str, Any]] = []

        # 24fps capture / offline post-processing.
        self.capture24_active = False
        self.capture24_fps = 24.0
        self.capture24_accum = 0.0
        self.capture24_session_index = 0
        self.capture24_root: Optional[Path] = None
        self.capture24_raw_dir: Optional[Path] = None
        self.capture24_sorted_dir: Optional[Path] = None
        self.capture24_aligned_dir: Optional[Path] = None
        self.capture24_meta_path: Optional[Path] = None
        self.capture24_saved: List[Path] = []
        self.capture_scope_live = "whole"   # whole / panels / both
        self.capture_json_index = 0

        # Live display backend for the main viewing modes.
        # gpu = fast direct slice rendering, cpu = robust NumPy/PIL fallback.
        # Default is GPU with no screen readback. The old auto blank-frame check
        # read the framebuffer and could tank FPS on large volumes.
        self.live_display_backend = "gpu"
        self.gpu_blank_check_enabled = False
        self._gpu_live_blank_check_frames = 0

        # HUD/PIL rebuild throttling. Rendering the existing UI texture is cheap;
        # rebuilding the full PIL image every frame is not.
        self.ui_update_fps = 2.0
        # When false, the PIL HUD is rebuilt only after a click/key/resize.
        # Rebuilding a 2K RGBA PIL image every frame was a major FPS killer.
        self.ui_live_refresh_enabled = False
        self._last_ui_build_time = 0.0


        print(f"[volume] Z={self.Z} H={self.H} W={self.W} dtype=uint8 RGB/BGR")
        print(f"[volume] main={self.main_volume_path}")
        print(f"[volume] gradient_distance={self.gradient_volume_path}")
        print(f"[volume] skeleton={self.skeleton_volume_path}")

        # ---------- programs ----------
        self.slice_prog = self.ctx.program(vertex_shader=SLICE_VERT, fragment_shader=SLICE_FRAG)
        self.gizmo_prog = self.ctx.program(vertex_shader=GIZMO_VERT, fragment_shader=GIZMO_FRAG)
        self.hud_prog = self.ctx.program(vertex_shader=HUD_TEX_VERT, fragment_shader=HUD_TEX_FRAG)
        self.post_prog = self.ctx.program(vertex_shader=POST_FX_VERT, fragment_shader=POST_FX_FRAG)
        self.hud_prog["u_tex"].value = 1
        self.post_prog["u_scene"].value = 2
        self.post_prog["u_resolution"].value = (float(self.wnd.width), float(self.wnd.height))
        self.post_prog["u_strength"].value = float(self.frame_transform_strength)
        self.post_prog["u_mode"].value = 0
        self.post_prog["u_cut_pattern"].value = 0
        self.post_prog["u_cut_parallel"].value = float(self.cut_offset_parallel)
        self.post_prog["u_cut_perp"].value = float(self.cut_offset_perp)
        self.post_prog["u_cut_angle"].value = float(self.cut_angle_rad)
        self.post_prog["u_cut_motion"].value = 0
        self.post_prog["u_mask_threshold"].value = 0.06
        self.post_prog["u_fx_param1"].value = 0.5
        self.post_prog["u_fx_param2"].value = 0.5

        # Optional compute backend for frame_fx.  This is the high-throughput
        # path for 4K screens; if compilation fails, the older fragment pass is
        # used automatically.
        try:
            self.post_compute_prog = self.ctx.compute_shader(_build_post_fx_compute_shader())
            self.post_compute_available = True
            print("[gpu] compute frame_fx enabled (OpenGL 4.3 imageStore path)")
        except Exception as e:
            self.post_compute_prog = None
            self.post_compute_available = False
            print(f"[gpu] compute frame_fx unavailable; using fragment fallback: {e}")

        # ---------- fullscreen quad ----------
        fsq = np.array([-1,-1,  1,-1,  -1, 1,
                        -1, 1,  1,-1,   1, 1], dtype="f4")
        self.slice_vbo = self.ctx.buffer(fsq.tobytes())
        self.slice_vao = self.ctx.simple_vertex_array(self.slice_prog, self.slice_vbo, "in_pos")

        # ---------- upload texture arrays (RGBA8) ----------
        # NOTE: sampler2DArray reads normalized floats from uint8 when dtype="f1"
        self.tex_main = self._upload_volume_texture_array(self.V, label="main")
        self.tex_gradient = self._upload_volume_texture_array(self.V_gradient, label="gradient") if self.V_gradient is not None else self.tex_main
        self.tex_skeleton = self._upload_volume_texture_array(self.V_skeleton, label="skeleton") if self.V_skeleton is not None else self.tex_main
        self.tex = self.tex_main

        self.tex.use(location=0)
        self.slice_prog["tex_array"].value = 0
        self.slice_prog["u_num_layers"].value = int(self.Z)
        self.slice_prog["u_filter_mode"].value = 0
        self.slice_prog["u_filter_target"].value = 0
        self.slice_prog["u_filter_strength"].value = 0.0
        self.slice_prog["u_post_mode"].value = 0

        # ---------- plane state ----------
        self.yaw = 0.0
        self.pitch = 0.0
        self.center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.scale  = 0.55

        self._update_plane_axes()

        # ---------- auto motion / SpaceMouse ----------
        self.auto_motion = AutoMotionController()
        self.auto_motion.enabled = False  # manual waypoint recording should not drift unless F12 is pressed
        self.spacemouse = SpaceMouseController()
        self.auto_motion.reset(
            center=self.center,
            yaw=self.yaw,
            pitch=self.pitch,
            heap_depth=0.22,
            heap_radius=0.18,
            heap_softness=0.06,
        )

        # ---------- heap brush state ----------
        self.heap_enable = True
        self.mouse_uv = (0.5, 0.5)     # bottom-left UV
        self.heap_radius = 0.18
        self.heap_softness = 0.06
        self.heap_depth = 0.22         # normalized volume units
        self.heap_stretch = 1.0
        self.heap_dir = -1.0           # -1 digs along -N; +1 digs along +N
        self.auto_motion.reset(
            center=self.center,
            yaw=self.yaw,
            pitch=self.pitch,
            heap_depth=self.heap_depth,
            heap_radius=self.heap_radius,
            heap_softness=self.heap_softness,
        )
        self.spacemouse.connect()

        # flip rules
        # If your stack is PNG-origin top-left and you DID NOT flip on upload, set flip_y=1.
        # If you already fixed orientation by upload/format, set flip_y=0.
        self.flip_y = 1
        self.bgr_input = 1

        self._push_slice_uniforms()

        # ---------- input state ----------
        self._drag_plane = False
        self._drag_pan   = False
        self._held_keys = set()

        # ---------- gizmo camera state ----------
        self.gizmo_yaw = 0.8
        self.gizmo_pitch = 0.5
        self.gizmo_radius = 2.4

        # ---------- gizmo geometry ----------
        def to_gizmo(p01):
            return np.array(p01, np.float32) - 0.5

        corners = [
            to_gizmo([0,0,0]), to_gizmo([1,0,0]),
            to_gizmo([0,1,0]), to_gizmo([1,1,0]),
            to_gizmo([0,0,1]), to_gizmo([1,0,1]),
            to_gizmo([0,1,1]), to_gizmo([1,1,1]),
        ]
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7),
        ]
        box_lines = []
        for a,b in edges:
            box_lines.append(corners[a]); box_lines.append(corners[b])
        box_lines = np.array(box_lines, dtype="f4")  # (24,3)

        self.box_vbo = self.ctx.buffer(box_lines.tobytes())
        self.box_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.box_vbo, "in_pos")

        # Plane slab (extruded quad)
        # Plane/curve preview geometry shown in the top-right corner viewer.
        # Reserve enough space for a tessellated curved sheet, not just a flat slab.
        self.plane_vbo = self.ctx.buffer(reserve=256 * 1024)
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        # Extra wire overlay so curved sheets read clearly in the corner preview.
        self.curve_wire_vbo = self.ctx.buffer(reserve=256 * 1024)
        self.curve_wire_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.curve_wire_vbo, "in_pos")
        self.curve_wire_vertex_count = 0

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

        # Timeline/path curve drawn inside the top-right gizmo.
        self.path_vbo = self.ctx.buffer(reserve=4096 * 3 * 4)
        self.path_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.path_vbo, "in_pos")

        # Fullscreen transparent HUD quad. The PIL UI is uploaded to a texture.
        hud_quad = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype="f4")
        self.hud_vbo = self.ctx.buffer(hud_quad.tobytes())
        self.hud_vao = self.ctx.vertex_array(
            self.hud_prog,
            [(self.hud_vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.post_vao = self.ctx.vertex_array(
            self.post_prog,
            [(self.hud_vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.post_color_tex = None
        self.post_compute_tex = None
        self.post_fbo = None
        self.post_fbo_size = (0, 0)
        self.panel_color_tex = None
        self.panel_fbo = None
        self.panel_fbo_size = (0, 0)

        self._update_gizmo_geometry()
        if self.playback_loaded_path is not None:
            self.load_waypoint_json(Path(self.playback_loaded_path))
        self.playback_enabled = bool(self.__class__.playback_on_start and self.has_playback_path())
        if self.playback_enabled:
            self.auto_motion.enabled = False
            print("[playback] started from loaded waypoint JSON")

        print("Ready.")
        print("  LMB drag (main): rotate plane")
        print("  MMB drag: pan plane (U/V)")
        print("  Wheel: zoom plane size")
        print("  WASDQE: move plane; Shift = faster")
        print("  View modes: T cycles single -> axis tri-plane -> local oblique -> multi-volume triptych")
        print("  Waypoints: C camera/plane | B brush | V combined | F save JSON")
        print("  Playback: SPACE play/pause | Z path-scrub mode | mouse-drag scrubs when enabled")
        print("  Timeline: U cycles interpolation | O cycles noise | -/= seconds per segment")
        print("  Heap brush: move mouse (G toggles); I/K depth; J/L radius; [ ] softness; ,/. stretch; N direction")
        print("  H = hide/show mouse cursor | Y = toggle live heuristics/blob counters | F2 = hide/show top-right gizmo | F3 = suggest/apply interesting next view | F4 = start/stop 24fps capture | F5 = blob debug | F6 = blob-dense low-interest seek | F7/F8 = cycle filter mode/target | F9 = cycle display variant | F10 = toggle aux-from-main | F11 = cycle frame transform")
        print("  Built-in HUD: click timeline buttons; F1 toggles HUD; click/drag timeline bar to scrub")
        print("  M = toggle heap modulation | X = toggle 3Dconnexion input | F12 = toggle Brownian auto motion")
        print("  Y = toggle flip_y (if upside-down)")
        print("  P = save current screen exactly as shown")
        print("  Curve side panels: Plane tab -> SideViews / SideMode, or Shift+F12 cycles side-panel mode")
        print("  Frame FX: hold Up/Down for strength; Left/Right for FX param or cut angle; Shift+Left/Right adjusts param2/cut distance")
        print("  Cuts: FX tab has Cut motion fixed/sine/noise, cut angle slider, and RandCut for random angle")
        print("  Performance: GPU live path is direct by default; F1 hides UI, F2 hides gizmo, Y toggles CPU analysis")
        print("  R: reset view   ESC: quit")

    # ------------------------------------------------------------
    # Plane + uniforms
    # ------------------------------------------------------------

    def _update_plane_axes(self):
        n = yaw_pitch_to_normal(self.yaw, self.pitch)
        u, v = orthonormal_basis_from_normal(n)
        self.n, self.u, self.v = n, u, v

    def _filter_mode_code(self) -> int:
        return {"none": 0, "isolate": 1, "hide": 2, "highlight": 3}.get(str(self.color_filter_mode), 0)

    def _filter_target_code(self) -> int:
        return {"none": 0, "red": 1, "green": 2, "blue": 3, "white": 4, "flesh": 5, "dark": 6, "bright": 7}.get(str(self.color_filter_target), 0)

    def cycle_color_filter_mode(self) -> None:
        modes = ["none", "isolate", "hide", "highlight"]
        i = modes.index(self.color_filter_mode) if self.color_filter_mode in modes else 0
        self.color_filter_mode = modes[(i + 1) % len(modes)]
        self.ui_force_rebuild = True

    def cycle_color_filter_target(self) -> None:
        targets = ["red", "green", "blue", "white", "flesh", "dark", "bright"]
        i = targets.index(self.color_filter_target) if self.color_filter_target in targets else 0
        self.color_filter_target = targets[(i + 1) % len(targets)]
        self.ui_force_rebuild = True

    def add_timeline_color_mark(self) -> None:
        mark = {
            "time": float(self.playhead_seconds),
            "mode": str(self.color_filter_mode),
            "target": str(self.color_filter_target),
            "strength": float(self.color_filter_strength),
        }
        self.timeline_color_marks.append(mark)
        self.ui_force_rebuild = True
        print(f"[filter] marked timeline color event: {mark}")

    def _apply_filter_uniforms(self) -> None:
        self.slice_prog["u_filter_mode"].value = int(self._filter_mode_code())
        self.slice_prog["u_filter_target"].value = int(self._filter_target_code() if self.color_filter_mode != "none" else 0)
        self.slice_prog["u_filter_strength"].value = float(self.color_filter_strength if self.color_filter_mode != "none" else 0.0)

    def cycle_main_display_variant(self) -> None:
        modes = ["normal", "gray", "invert", "gray_invert"]
        i = modes.index(self.main_display_variant) if self.main_display_variant in modes else 0
        self.main_display_variant = modes[(i + 1) % len(modes)]
        self.ui_force_rebuild = True

    def toggle_aux_from_main(self) -> None:
        self.aux_from_main = not self.aux_from_main
        self.force_clear_next_frame = True
        self.ui_force_rebuild = True

    def cycle_pixel_metric(self, axis: str) -> None:
        items = ["hue", "value", "intensity", "saturation", "red", "green", "blue"]
        if axis == "x":
            cur = self.pixel_grid_x_metric
            i = items.index(cur) if cur in items else 0
            self.pixel_grid_x_metric = items[(i + 1) % len(items)]
        else:
            cur = self.pixel_grid_y_metric
            i = items.index(cur) if cur in items else 0
            self.pixel_grid_y_metric = items[(i + 1) % len(items)]
        self.ui_force_rebuild = True

    def cycle_pixel_layout(self) -> None:
        items = ["metric", "similar", "swirl", "corner", "object", "blobswirl"]
        cur = self.pixel_grid_layout
        i = items.index(cur) if cur in items else 0
        self.pixel_grid_layout = items[(i + 1) % len(items)]
        self.ui_force_rebuild = True

    def _arm_live_blank_check(self, frames: int = 8) -> None:
        if bool(getattr(self, "gpu_blank_check_enabled", False)):
            self._gpu_live_blank_check_frames = max(int(frames), 0)
        else:
            self._gpu_live_blank_check_frames = 0

    def _cycle_live_display_backend(self) -> None:
        modes = ["gpu", "auto", "cpu"]
        cur = str(getattr(self, "live_display_backend", "gpu"))
        idx = modes.index(cur) if cur in modes else 0
        self.live_display_backend = modes[(idx + 1) % len(modes)]
        # Only auto mode needs the expensive screen-read blank detector.
        self.gpu_blank_check_enabled = (self.live_display_backend == "auto")
        self._arm_live_blank_check(6)
        self.force_clear_next_frame = True
        self.ui_force_rebuild = True

    def cycle_curve_side_panel_mode(self) -> None:
        modes = ["local_curved", "curve_profile"]
        cur = str(getattr(self, "curve_side_panel_mode", "local_curved"))
        idx = modes.index(cur) if cur in modes else 0
        self.curve_side_panel_mode = modes[(idx + 1) % len(modes)]
        self.force_clear_next_frame = True
        self.ui_force_rebuild = True
        print(f"[curve side panels] mode={self.curve_side_panel_mode}")

    def toggle_analysis_enabled(self) -> None:
        self.analysis_enabled = not bool(getattr(self, "analysis_enabled", False))
        self._last_heuristics_time = 0.0
        self._heuristics_cache_key = None
        self._fx_analysis_cache_key = None
        self._mark_analysis_dirty()
        if not self.analysis_enabled:
            self.current_heuristics = {"paused": True, "reason": "analysis off"}
            self.current_fx_analysis = {"paused": True, "reason": "analysis off"}
            self.current_blob_debug_meta = {"paused": True}
            self.current_blob_debug_image = None
        self.ui_force_rebuild = True

    def toggle_live_interest_enabled(self) -> None:
        self.interest_recommend_live_enabled = not bool(getattr(self, "interest_recommend_live_enabled", False))
        self._last_interest_time = 0.0
        if not self.interest_recommend_live_enabled:
            self.current_interest = {"paused": True, "reason": "live interest off"}
        self.ui_force_rebuild = True

    def _mark_analysis_dirty(self, *names: str) -> None:
        if not names:
            names = tuple(self.analysis_dirty_flags.keys())
        for name in names:
            self.analysis_dirty_flags[str(name)] = True
        if "live_recompute" in names or not names:
            self.live_recompute_cache_key = None
            self.live_recompute_cache_img = None

    def _analysis_panel_state_key(self, name: str) -> Tuple[Any, ...]:
        p1, p2 = self._get_fx_param_values(self.frame_transform_mode)
        ht = self.hemo_thresholds
        return (
            str(name),
            int(getattr(self, "_view_state_version", 0)),
            str(getattr(self, "view_mode", "single")),
            str(getattr(self, "frame_transform_mode", "none")),
            round(float(getattr(self, "frame_transform_strength", 0.0)), 3),
            round(float(p1), 3),
            round(float(p2), 3),
            round(float(ht.get("oxy", 0.58)), 3),
            round(float(ht.get("deoxy", 0.42)), 3),
            round(float(ht.get("fresh", 0.56)), 3),
            round(float(ht.get("savgol", 0.50)), 3),
        )

    def _set_hemo_threshold_from_ui_x(self, which: str, x: float) -> None:
        rect = getattr(self, f"ui_hemo_{which}_rect", None)
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        t = 0.0 if x1 <= x0 else (float(x) - float(x0)) / float(x1 - x0)
        self.hemo_thresholds[which] = float(np.clip(t, 0.0, 1.0))
        self._mark_analysis_dirty("fx", "myoglobin")
        self._fx_analysis_cache_key = None
        self.ui_force_rebuild = True

    def _signed_distance_and_skeleton_images(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        rgb = np.asarray(rgb, dtype=np.uint8)
        mask = self._foreground_mask_simple(rgb)
        H, W = mask.shape
        if not np.any(mask):
            zero = np.zeros((H, W, 3), dtype=np.uint8)
            return zero, zero, {"mask_fill": 0.0, "skel_fill": 0.0, "dist_peak": 0.0}
        dist_in = ndimage.distance_transform_edt(mask)
        dist_out = ndimage.distance_transform_edt(~mask)
        signed = dist_in - dist_out
        maxabs = float(max(1e-6, np.percentile(np.abs(signed), 99.0)))
        norm = np.clip(0.5 + 0.5 * (signed / maxabs), 0.0, 1.0)
        signed_gray = (norm * 255.0).astype(np.uint8)
        signed_rgb = np.repeat(signed_gray[..., None], 3, axis=2)

        mx = ndimage.maximum_filter(dist_in, size=7)
        skel = (dist_in > 1.0) & (dist_in >= mx - 1e-5) & mask
        skel = ndimage.binary_dilation(skel, iterations=1)
        skel_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        skel_rgb[..., 0] = (skel * 255).astype(np.uint8)
        skel_rgb[..., 1] = (skel * 255).astype(np.uint8)
        skel_rgb[..., 2] = (skel * 255).astype(np.uint8)
        meta = {
            "mask_fill": float(mask.mean()),
            "skel_fill": float(skel.mean()),
            "dist_peak": float(np.max(dist_in)) if dist_in.size else 0.0,
        }
        return signed_rgb, skel_rgb, meta

    def _compute_live_recompute_view(self, W: int, H: int) -> Image.Image:
        key = (
            int(getattr(self, "_view_state_version", 0)), int(W), int(H),
            str(getattr(self, "view_mode", "single")),
            str(getattr(self, "main_display_variant", "normal")),
            str(getattr(self, "color_filter_mode", "none")),
            str(getattr(self, "color_filter_target", "red")),
            round(float(getattr(self, "color_filter_strength", 0.0)), 3),
        )
        if self.live_recompute_cache_key == key and self.live_recompute_cache_img is not None:
            return self.live_recompute_cache_img.copy()
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        draw = ImageDraw.Draw(canvas)
        spec = dict(self._single_view_spec())
        spec["aspect_correct"] = 0
        panel_viewports = self._panel_viewports()
        left_vp = panel_viewports[0]
        mid_vp = panel_viewports[1]
        right_vp = panel_viewports[2]
        base_rgb = self._sample_rgb_for_spec("main", spec, out_w=max(1, int(left_vp[2])), out_h=max(1, int(left_vp[3])))
        base_rgb = self._apply_cpu_post_and_filter(base_rgb, "main")
        signed_rgb, skel_rgb, meta = self._signed_distance_and_skeleton_images(base_rgb)
        panels = [
            (left_vp, Image.fromarray(base_rgb, "RGB").convert("RGBA"), "Original volume"),
            (mid_vp, Image.fromarray(signed_rgb, "RGB").convert("RGBA"), "Live signed distance"),
            (right_vp, Image.fromarray(skel_rgb, "RGB").convert("RGBA"), "Live skeleton"),
        ]
        for (vx, vy, vw, vh), panel, label in panels:
            x0 = int(vx); y0 = int(H - (int(vy) + int(vh)))
            canvas.alpha_composite(panel.resize((int(vw), int(vh)), Image.NEAREST), (x0, y0))
            draw.rectangle((x0 + 6, y0 + 6, x0 + 8 + max(110, 7 * len(label)), y0 + 25), fill=(0, 0, 0, 150))
            draw.text((x0 + 10, y0 + 10), label, fill=(255, 255, 255, 255))
        self.analysis_panel_cache["live_recompute"] = {
            "title": "Realtime recompute",
            "lines": [
                f"filled area: {meta['mask_fill']*100.0:.1f}%",
                f"skeleton fill: {meta['skel_fill']*100.0:.2f}%",
                f"distance peak: {meta['dist_peak']:.2f}",
            ],
        }
        self.analysis_panel_keys["live_recompute"] = key
        self.analysis_dirty_flags["live_recompute"] = False
        self.live_recompute_cache_key = key
        self.live_recompute_cache_img = canvas.copy()
        return canvas

    def _cached_mode_panel(self, mode_name: str) -> Dict[str, Any]:
        key = self._analysis_panel_state_key(mode_name)
        if (not self.analysis_dirty_flags.get(mode_name, True)) and self.analysis_panel_keys.get(mode_name) == key:
            return self.analysis_panel_cache.get(mode_name, {"title": mode_name, "lines": ["cached"]})
        metrics = self._compute_fx_quality_metrics(mode_name)
        p1, p2 = self._get_fx_param_values(mode_name)
        lines = []
        if mode_name == "myoglobin":
            ht = self.hemo_thresholds
            lines = [
                f"Oxy threshold: {ht['oxy']:.2f}",
                f"Deoxy threshold: {ht['deoxy']:.2f}",
                f"Freshness gate: {ht['fresh']:.2f}",
                f"SG smoothing: {ht['savgol']:.2f}",
                f"FX params: p1={p1:.2f} p2={p2:.2f}",
            ]
            title = "Hemoglobin / oxygenation"
        elif mode_name == "inflation":
            lines = [f"Strength: {self.frame_transform_strength:.2f}", f"Pressure: {p1:.2f}", f"Tube radius: {p2:.2f}"]
            title = "Inflation panel"
        elif mode_name == "meatexpansion":
            lines = [f"Strength: {self.frame_transform_strength:.2f}", f"Feed amount: {p1:.2f}", f"Lateral shift: {p2:.2f}"]
            title = "Expansion panel"
        elif mode_name == "marbling":
            lines = [f"Segmentation bias: {p1:.2f}", f"Connectivity weight: {p2:.2f}"]
            title = "Marbling panel"
        else:
            title = f"{mode_name} panel"
        for name, value in metrics[:5]:
            lines.append(f"{name}: {value}")
        panel = {"title": title, "lines": lines}
        self.analysis_panel_cache[mode_name] = panel
        self.analysis_panel_keys[mode_name] = key
        self.analysis_dirty_flags[mode_name] = False
        return panel

    def _current_post_mode(self, volume_key: str = "main") -> int:
        if volume_key == "main":
            if self.view_mode == "single_gray":
                return 1
            if self.view_mode == "single_invert":
                return 2
            if self.view_mode == "single_gray_invert":
                return 3
            return {"normal": 0, "gray": 1, "invert": 2, "gray_invert": 3}.get(self.main_display_variant, 0)
        if self.aux_from_main:
            if volume_key == "gradient":
                return 1
            if volume_key == "skeleton":
                return 3
        return 0

    def _apply_post_uniforms(self, volume_key: str = "main") -> None:
        self.slice_prog["u_post_mode"].value = int(self._current_post_mode(volume_key))

    def _pixel_metric_values(self, rgb: np.ndarray, name: str) -> np.ndarray:
        arr = rgb.astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        diff = mx - mn
        if name == "red": return r
        if name == "green": return g
        if name == "blue": return b
        if name == "value": return mx
        if name == "intensity": return (r + g + b) / 3.0
        if name == "saturation": return np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
        # hue
        hue = np.zeros_like(mx)
        mask = diff > 1e-6
        idx = (mx == r) & mask
        hue[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
        idx = (mx == g) & mask
        hue[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
        idx = (mx == b) & mask
        hue[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
        hue = hue / 6.0
        return hue

    def _build_pixel_grid_image(self, out_w: int, out_h: int) -> Image.Image:
        key = (self._view_state_version, int(out_w), int(out_h), self.pixel_grid_layout, self.pixel_grid_x_metric, self.pixel_grid_y_metric)
        if self._pixel_grid_cache_key == key and self._pixel_grid_cache_img is not None:
            return self._pixel_grid_cache_img.copy()
        rgb = self._sample_rgb_for_spec("main", self._single_view_spec(), out_w=220, out_h=160)
        layout = str(self.pixel_grid_layout)
        mask = self._foreground_mask_simple(rgb)
        coords = np.argwhere(mask)
        if len(coords) == 0:
            mask[:] = True
            coords = np.argwhere(mask)
        flat_colors = rgb[coords[:,0], coords[:,1]]
        xvals = self._pixel_metric_values(rgb, self.pixel_grid_x_metric)[coords[:,0], coords[:,1]]
        yvals = self._pixel_metric_values(rgb, self.pixel_grid_y_metric)[coords[:,0], coords[:,1]]
        if layout == "similar":
            order = np.lexsort((flat_colors[:,2], flat_colors[:,1], flat_colors[:,0], flat_colors.mean(axis=1)))
        elif layout == "corner":
            order = np.lexsort((coords[:,1], coords[:,0]))
        else:
            order = np.lexsort((xvals, yvals))
        src = flat_colors[order]
        target = np.argwhere(mask).copy()
        if layout == "swirl":
            H, W = mask.shape
            yy, xx = np.indices((H, W))
            rr = (yy - H/2.0)**2 + (xx - W/2.0)**2
            ang = np.arctan2(yy - H/2.0, xx - W/2.0)
            ids = np.argwhere(mask)
            ord2 = np.lexsort((ang[mask], rr[mask]))
            target = ids[ord2]
        elif layout == "blobswirl":
            lab, n = ndimage.label(mask)
            ordered = []
            for i in range(1, n+1):
                ys, xs = np.where(lab == i)
                if len(xs) == 0:
                    continue
                cy, cx = ys.mean(), xs.mean()
                rr = (ys - cy)**2 + (xs - cx)**2
                ang = np.arctan2(ys - cy, xs - cx)
                ord2 = np.lexsort((ang, rr))
                ordered.extend(list(zip(ys[ord2], xs[ord2])))
            if ordered:
                target = np.array(ordered, dtype=np.int32)
        elif layout == "object":
            ids = np.argwhere(mask)
            target = ids
        out = np.zeros_like(rgb)
        count = min(len(src), len(target))
        out[target[:count,0], target[:count,1]] = src[:count]
        # keep outside black so sorting stays within object confines
        im = Image.fromarray(out, mode="RGB").resize((out_w, out_h), Image.NEAREST)
        draw = ImageDraw.Draw(im)
        f = self._scaled_font(14)
        draw.rounded_rectangle((8, 8, min(out_w - 8, 520), 34), radius=8, fill=(8, 10, 14), outline=(180,180,210))
        draw.text((16, 12), f"Pixel grid  mode:{layout}  X:{self.pixel_grid_x_metric}  Y:{self.pixel_grid_y_metric}", fill=(240,240,250), font=f)
        self._pixel_grid_cache_key = key
        self._pixel_grid_cache_img = im.copy()
        return im

    def _upload_overlay_texture(self, img: Image.Image) -> None:
        """Upload a full-screen preview image without clobbering the HUD texture."""
        img = img.convert("RGBA")
        W, H = img.size
        if self.overlay_tex is None or self.overlay_tex_size != (W, H):
            try:
                if self.overlay_tex is not None:
                    self.overlay_tex.release()
            except Exception:
                pass
            self.overlay_tex = self.ctx.texture((W, H), components=4, dtype="f1")
            self.overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.overlay_tex_size = (W, H)
        self.overlay_tex.write(img.tobytes("raw", "RGBA"))

    def _render_overlay_image(self, img: Image.Image) -> None:
        # Pixel-grid and CPU frame-FX images are screen content, not UI. Keep them
        # in a separate texture so _render_builtin_ui() can still draw its own
        # cached HUD after this pass.
        self._upload_overlay_texture(img)
        self.ctx.viewport = (0, 0, int(self.wnd.width), int(self.wnd.height))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.overlay_tex.use(location=1)
        self.hud_vao.render(mode=moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)

    def _gpu_fx_code(self, mode: Optional[str] = None) -> int:
        mode = str(mode if mode is not None else self.frame_transform_mode)
        table = {
            "swap": 1,
            "repeat": 2,
            "fragment": 3,
            "cuts": 4,
            "flipcontour": 5,
            "stretch": 6,
            "fleshswell": 7,
            "mold": 8,
            "vessels": 9,
            "veinbranch": 10,
            "grassfire": 11,
            "amat": 12,
            "springmass": 13,
            "steiner": 14,
            "poisson": 15,
            "meatexpansion": 16,
            "inflation": 17,
            "myoglobin": 18,
            "fibertrack": 19,
            "watermobility": 20,
            "marbling": 21,
        }
        return int(table.get(mode, 0))

    def _gpu_cut_pattern_code(self) -> int:
        table = {"parallel": 0, "grid": 1, "radial": 2, "irregular": 3}
        return int(table.get(str(self.cut_pattern), 0))

    def _gpu_cut_motion_code(self) -> int:
        table = {"fixed": 0, "sine": 1, "noise": 2}
        return int(table.get(str(getattr(self, "cut_motion_mode", "fixed")), 0))

    def _ensure_postprocess_targets(self, W: int, H: int) -> None:
        if self.post_fbo is not None and self.post_fbo_size == (int(W), int(H)):
            return
        try:
            if self.post_fbo is not None:
                self.post_fbo.release()
            if self.post_color_tex is not None:
                self.post_color_tex.release()
            if self.post_compute_tex is not None:
                self.post_compute_tex.release()
        except Exception:
            pass
        self.post_color_tex = self.ctx.texture((int(W), int(H)), 4, dtype="f1")
        self.post_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.post_fbo = self.ctx.framebuffer(color_attachments=[self.post_color_tex])
        self.post_compute_tex = self.ctx.texture((int(W), int(H)), 4, dtype="f1")
        self.post_compute_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.post_fbo_size = (int(W), int(H))

    def _ensure_panel_targets(self, W: int, H: int) -> None:
        if self.panel_fbo is not None and self.panel_fbo_size == (int(W), int(H)):
            return
        try:
            if self.panel_fbo is not None:
                self.panel_fbo.release()
            if self.panel_color_tex is not None:
                self.panel_color_tex.release()
        except Exception:
            pass
        self.panel_color_tex = self.ctx.texture((int(W), int(H)), 4, dtype="f1")
        self.panel_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.panel_fbo = self.ctx.framebuffer(color_attachments=[self.panel_color_tex])
        self.panel_fbo_size = (int(W), int(H))

    def _draw_texture_to_viewport(self, tex, viewport) -> None:
        vx, vy, vw, vh = viewport
        self.ctx.screen.use()
        self.ctx.viewport = (int(vx), int(vy), int(vw), int(vh))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        tex.use(location=1)
        self.hud_vao.render(mode=moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)

    def _render_slice_panel_composited(self, viewport, spec, volume_key: str = "main") -> None:
        """Render one panel.

        v25 defaults to the direct GPU path because it is substantially faster:
        one slice shader pass, no offscreen panel FBO, no CPU framebuffer readback,
        and no PIL volume resampling.  The old two-pass compositor is kept behind
        fast_direct_live_render=False for debugging only.
        """
        if bool(getattr(self, "fast_direct_live_render", True)):
            self.ctx.screen.use()
            self.ctx.disable(moderngl.BLEND)
            self._render_slice_panel(viewport, spec, volume_key=volume_key)
            return

        vx, vy, vw, vh = viewport
        self._ensure_panel_targets(int(vw), int(vh))
        self.panel_fbo.use()
        self.ctx.viewport = (0, 0, int(vw), int(vh))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._render_slice_panel((0, 0, int(vw), int(vh)), spec, volume_key=volume_key)
        self._draw_texture_to_viewport(self.panel_color_tex, viewport)

    def _render_live_panel_gpu(self, viewport, spec, volume_key: str = "main") -> None:
        """Direct live MPR draw. This is the preferred high-FPS path."""
        self.ctx.screen.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self._render_slice_panel(viewport, spec, volume_key=volume_key)

    def _push_post_fx_uniforms(self, prog, W: int, H: int) -> None:
        """Shared uniforms for fragment and compute frame-FX backends."""
        prog["u_scene"].value = 2
        prog["u_resolution"].value = (float(W), float(H))
        prog["u_time"].value = float(time.perf_counter())
        prog["u_strength"].value = float(self.frame_transform_strength)
        prog["u_mode"].value = int(self._gpu_fx_code())
        prog["u_cut_pattern"].value = int(self._gpu_cut_pattern_code())
        prog["u_cut_parallel"].value = float(self.cut_offset_parallel)
        prog["u_cut_perp"].value = float(self.cut_offset_perp)
        prog["u_cut_angle"].value = float(self.cut_angle_rad)
        prog["u_cut_motion"].value = int(self._gpu_cut_motion_code())
        prog["u_mask_threshold"].value = 0.06
        p1, p2 = self._get_fx_param_values(self.frame_transform_mode)
        prog["u_fx_param1"].value = float(p1)
        prog["u_fx_param2"].value = float(p2)

    def _render_frame_fx_compute(self, W: int, H: int) -> bool:
        """Render frame FX with a compute shader into an output texture.

        Return True when compute succeeded.  On any driver/API issue, return
        False so the caller can fall back to the fragment shader without showing
        a blank frame.
        """
        if not (bool(getattr(self, "use_compute_frame_fx_live", True)) and bool(getattr(self, "post_compute_available", False)) and self.post_compute_prog is not None):
            return False
        W = max(1, int(W)); H = max(1, int(H))
        self._ensure_postprocess_targets(W, H)

        # Source pass: same exact GPU volume slice as the fragment backend.
        self.post_fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._render_slice_panel((0, 0, W, H), self._single_view_spec(), volume_key="main")

        try:
            self.post_color_tex.use(location=2)
            # Write compute result to an rgba8 output image.  ModernGL's default
            # format inference works for a dtype="f1" / 4-component texture.
            self.post_compute_tex.bind_to_image(0, read=False, write=True)
            self._push_post_fx_uniforms(self.post_compute_prog, W, H)
            groups_x = (W + 15) // 16
            groups_y = (H + 15) // 16
            self.post_compute_prog.run(groups_x, groups_y, 1)
            if hasattr(self.ctx, "memory_barrier"):
                self.ctx.memory_barrier()

            # Present the compute texture.  The UI/HUD is drawn later on top.
            self._draw_texture_to_viewport(self.post_compute_tex, (0, 0, W, H))
            self.fx_backend = "gpu_compute"
            return True
        except Exception as e:
            self.post_compute_available = False
            print(f"[gpu] compute frame_fx failed once; falling back to fragment pass: {e}")
            return False

    def _render_frame_fx_gpu(self, W: int, H: int) -> None:
        """Render the current main-volume slice to an offscreen texture, then run the FX shader.

        This v11 version explicitly binds the FBO/sampler every frame so FX modes that
        depend on sampled volume content, such as cuts, AMAT, grassfire, meat expansion,
        inflation, and the chemical/texture visualizers, do not accidentally process a
        blank/default texture.
        """
        if self._render_frame_fx_compute(W, H):
            return

        W = max(1, int(W)); H = max(1, int(H))
        self._ensure_postprocess_targets(W, H)

        # 1) Source pass: sampled MPR slice into post_color_tex using the GPU.
        # _render_slice_panel now re-binds the texture sampler and layer count
        # per pass, so the post-process source stays grounded on the real volume.
        self.post_fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._render_slice_panel((0, 0, W, H), self._single_view_spec(), volume_key="main")

        # 2) Post pass: FX shader to the visible screen.
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self._push_post_fx_uniforms(self.post_prog, W, H)
        self.post_color_tex.use(location=2)
        self.post_vao.render(mode=moderngl.TRIANGLES)
        self.fx_backend = "gpu_fragment"

    def cycle_frame_transform_mode(self) -> None:
        modes = self._fx_mode_list()
        i = modes.index(self.frame_transform_mode) if self.frame_transform_mode in modes else 0
        self.frame_transform_mode = modes[(i + 1) % len(modes)]
        self.ui_force_rebuild = True

    def _current_frame_rgb(self, out_w: int = 320, out_h: int = 240) -> np.ndarray:
        spec = self._single_view_spec()
        if self.view_mode == "object_editor":
            return self._sample_rgb_with_objects(spec, out_w=out_w, out_h=out_h)
        return self._sample_rgb_for_spec("main", spec, out_w=out_w, out_h=out_h)

    def _foreground_mask_simple(self, rgb: np.ndarray) -> np.ndarray:
        gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
        thr = max(10.0, float(np.percentile(gray, 55)) * 0.65)
        return gray > thr

    def _skeleton_mask_and_points(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = self._foreground_mask_simple(rgb)
        if not np.any(mask):
            return np.zeros_like(mask), np.zeros((0, 2), dtype=np.int32)
        dist = ndimage.distance_transform_edt(mask)
        mx = ndimage.maximum_filter(dist, size=7)
        skel = (dist > 1.0) & (dist >= mx - 1e-5) & mask
        coords = np.argwhere(skel)
        if len(coords) == 0:
            coords = np.argwhere(mask)
        if len(coords) > 220:
            coords = coords[:: max(1, len(coords)//220)]
        return skel, coords.astype(np.int32)

    def _coral_growth_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        skel, _ = self._skeleton_mask_and_points(rgb)
        if not np.any(mask) or not np.any(skel):
            return rgb
        idx = ndimage.distance_transform_edt(~skel, return_distances=False, return_indices=True)
        sy, sx = idx[0], idx[1]
        yy, xx = np.indices((H, W))
        vecx = xx - sx
        vecy = yy - sy
        dist = np.sqrt(vecx * vecx + vecy * vecy) + 1e-6
        ang = np.arctan2(vecy, vecx)
        noise = np.sin(ang * 5.0 + dist * 0.35) * 0.5 + np.cos((sx + sy) * 0.15) * 0.5
        mag = (0.18 + 0.22 * noise) * self.frame_transform_strength * np.clip(dist / max(H, W), 0, 1) * max(H, W)
        tx = np.clip(np.rint(xx + (vecx / dist) * mag - (vecy / dist) * 0.25 * mag).astype(int), 0, W - 1)
        ty = np.clip(np.rint(yy + (vecy / dist) * mag + (vecx / dist) * 0.25 * mag).astype(int), 0, H - 1)
        out = np.zeros_like(rgb)
        out[ty[mask], tx[mask]] = rgb[yy[mask], xx[mask]]
        fill = out.max(axis=2) == 0
        out[fill] = (rgb[fill] * 0.15).astype(np.uint8)
        out = ndimage.gaussian_filter(out, sigma=(0.6, 0.6, 0))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _vector_field_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        skel, _ = self._skeleton_mask_and_points(rgb)
        if not np.any(mask) or not np.any(skel):
            return rgb
        idx = ndimage.distance_transform_edt(~skel, return_distances=False, return_indices=True)
        sy, sx = idx[0], idx[1]
        yy, xx = np.indices((H, W))
        vecx = sx - xx
        vecy = sy - yy
        dist = np.sqrt(vecx * vecx + vecy * vecy) + 1e-6
        tt = time.perf_counter()
        shift = self.frame_transform_strength * (0.45 + 0.35 * np.sin(dist * 0.18 + tt * 2.4))
        px = np.clip(np.rint(xx + vecx * shift + 10 * np.sin(yy * 0.05 + tt * 1.7)).astype(int), 0, W - 1)
        py = np.clip(np.rint(yy + vecy * shift + 10 * np.cos(xx * 0.05 + tt * 1.3)).astype(int), 0, H - 1)
        out = rgb[py, px]
        if self.vector_flow_show_guides:
            ridge = ndimage.maximum_filter((skel * 255).astype(np.uint8), size=3)
            out[..., 1] = np.maximum(out[..., 1], ridge)
        return out

    def _voronoi_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        skel, coords = self._skeleton_mask_and_points(rgb)
        if len(coords) == 0:
            return rgb
        # select key points: endpoints/junction-ish
        nbr = ndimage.convolve(skel.astype(np.int32), np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.int32), mode='constant')
        key = np.argwhere(skel & ((nbr <= 12) | (nbr >= 14)))
        if len(key) < 8:
            key = coords
        if len(key) > 64:
            key = key[:: max(1, len(key)//64)]
        yy, xx = np.indices((H, W))
        pts = key.astype(np.float32)
        d2 = (yy[..., None] - pts[:,0])**2 + (xx[..., None] - pts[:,1])**2
        lab = np.argmin(d2, axis=2)
        out = np.zeros_like(rgb)
        for i, pt in enumerate(pts):
            py, px = int(pt[0]), int(pt[1])
            color = rgb[py, px]
            out[lab == i] = color
        edges = ndimage.maximum_filter(lab, size=3) != ndimage.minimum_filter(lab, size=3)
        out[edges] = 255
        return out

    def _branch_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        im = Image.new('RGB', (W, H), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        skel, coords = self._skeleton_mask_and_points(rgb)
        if len(coords) == 0:
            return rgb
        step = max(1, len(coords) // 72)
        seeds = coords[::step]
        tt = time.perf_counter()

        def draw_branch(x0: float, y0: float, ang: float, length: float, depth: int, col: tuple[int, int, int]) -> None:
            if depth <= 0 or length < 2.0:
                return
            x1 = x0 + np.cos(ang) * length
            y1 = y0 + np.sin(ang) * length
            width = max(1, int(0.6 + depth * 0.7))
            draw.line((x0, y0, x1, y1), fill=col, width=width)
            # main continuation
            draw_branch(x1, y1, ang + 0.12 * np.sin(tt + x0 * 0.03 + y0 * 0.02), length * 0.78, depth - 1, col)
            # left / right splits
            split = 0.45 + 0.20 * np.sin(tt * 0.9 + x0 * 0.01)
            draw_branch(x1, y1, ang + split, length * 0.62, depth - 1, col)
            draw_branch(x1, y1, ang - split, length * 0.62, depth - 1, col)

        for idx, (y, x) in enumerate(seeds):
            col = tuple(int(v) for v in rgb[y, x])
            base_ang = ((x * 13 + y * 7 + idx * 11) % 360) * np.pi / 180.0 + 0.45 * np.sin(tt * 1.4 + x * 0.02)
            length = (10 + ((x + y) % 16)) * (1.0 + 1.1 * self.frame_transform_strength)
            depth = max(2, int(3 + 3 * self.frame_transform_strength))
            draw_branch(float(x), float(y), base_ang, length, depth, col)
            draw_branch(float(x), float(y), base_ang + np.pi, length * 0.75, max(2, depth - 1), col)
        return np.asarray(im, dtype=np.uint8)

    def _drift_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        skel, _ = self._skeleton_mask_and_points(rgb)
        if not np.any(mask) or not np.any(skel):
            return rgb
        idx = ndimage.distance_transform_edt(~skel, return_distances=False, return_indices=True)
        sy, sx = idx[0], idx[1]
        yy, xx = np.indices((H, W))
        tt = time.perf_counter()
        t = 0.45 + 0.45 * self.frame_transform_strength
        bx = sx * (1.0 - t) + xx * t + 18.0 * np.sin(yy * 0.07 + sx * 0.02 + tt * 1.8)
        by = sy * (1.0 - t) + yy * t + 18.0 * np.cos(xx * 0.06 + sy * 0.02 + tt * 1.5)
        tx = np.clip(np.rint(bx).astype(int), 0, W - 1)
        ty = np.clip(np.rint(by).astype(int), 0, H - 1)
        out = np.zeros_like(rgb)
        out[ty[mask], tx[mask]] = rgb[yy[mask], xx[mask]]
        out = np.maximum(out, (rgb * 0.10).astype(np.uint8))
        return out

    def _random_swap_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        rng = np.random.default_rng(12345)
        mask = self._foreground_mask_simple(rgb).reshape(-1)
        out = rgb.copy().reshape(-1, 3)
        ids = np.where(mask)[0]
        if len(ids) < 2:
            return out.reshape(H, W, 3)
        swap = rng.permutation(ids)
        sel = ids[::2]
        out[sel] = out[swap[:len(sel)]]
        return out.reshape(H, W, 3)

    def _repeat_grid_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        tile = np.asarray(Image.fromarray(rgb, mode='RGB').resize((max(8, W//3), max(8, H//3)), Image.BILINEAR), dtype=np.uint8)
        reps_y = int(np.ceil(H / tile.shape[0])); reps_x = int(np.ceil(W / tile.shape[1]))
        out = np.tile(tile, (reps_y, reps_x, 1))[:H, :W, :]
        return out

    def _blob_grid_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        lab, n = ndimage.label(mask)
        centers = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) < 40:
                continue
            centers.append((float(xs.mean()), float(ys.mean())))
        if not centers:
            return self._repeat_grid_transform(rgb)
        out = np.zeros_like(rgb)
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        for cx, cy in centers[:8]:
            mul = 1.6 + 3.0 * self.blob_pack_distance
            ox = int(np.clip(W/2 + (cx - W/2) * mul, -W, W))
            oy = int(np.clip(H/2 + (cy - H/2) * mul, -H, H))
            xs = np.clip(grid_x - ox, 0, W-1)
            ys = np.clip(grid_y - oy, 0, H-1)
            out = np.maximum(out, rgb[ys, xs])
        return out

    def _meatpack_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        if not np.any(mask):
            return rgb
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        block = max(8, int(min(H, W) * (0.05 + 0.10 * self.frame_transform_strength)))
        patches = []
        for yy in range(y0, y1, block):
            for xx in range(x0, x1, block):
                yb, xb = min(yy + block, H), min(xx + block, W)
                m = mask[yy:yb, xx:xb]
                if m.mean() < 0.18:
                    continue
                patches.append((yy, yb, xx, xb, rgb[yy:yb, xx:xb].copy(), m.copy()))
        if not patches:
            return rgb
        out = np.zeros_like(rgb)
        cx, cy = W / 2.0, H / 2.0
        spacing = max(4.0, self.blob_pack_distance * min(H, W) * 0.55)
        dests = []
        angle = 0.0
        radius = 0.0
        for i in range(len(patches)):
            dx = np.cos(angle) * radius
            dy = np.sin(angle) * radius
            dests.append((int(round(cx + dx)), int(round(cy + dy))))
            angle += 0.9
            radius += spacing * 0.18
        # sort larger clumps first for denser central packing
        patches = sorted(patches, key=lambda t: (t[1]-t[0])*(t[3]-t[2]), reverse=True)
        for (yy, yb, xx, xb, patch, pmask), (tx, ty) in zip(patches, dests):
            ph, pw = patch.shape[:2]
            dx0 = int(np.clip(tx - pw // 2, 0, max(0, W - pw)))
            dy0 = int(np.clip(ty - ph // 2, 0, max(0, H - ph)))
            roi = out[dy0:dy0+ph, dx0:dx0+pw]
            roi[pmask] = patch[pmask]
        out = np.maximum(out, (rgb * 0.05).astype(np.uint8))
        return out

    def _skeleton_wrap_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        skel, _ = self._skeleton_mask_and_points(rgb)
        if not np.any(mask) or not np.any(skel):
            return rgb
        idx = ndimage.distance_transform_edt(~skel, return_distances=False, return_indices=True)
        sy, sx = idx[0], idx[1]
        yy, xx = np.indices((H, W))
        vecx = sx - xx
        vecy = sy - yy
        dist = np.sqrt(vecx * vecx + vecy * vecy) + 1e-6
        t = 0.45 + 0.45 * self.frame_transform_strength
        wrap = 6.0 * self.frame_transform_strength
        px = np.clip(np.rint(xx + vecx * t + wrap * np.sin(vecy * 0.12)).astype(int), 0, W - 1)
        py = np.clip(np.rint(yy + vecy * t + wrap * np.cos(vecx * 0.12)).astype(int), 0, H - 1)
        out = np.zeros_like(rgb)
        out[py[mask], px[mask]] = rgb[yy[mask], xx[mask]]
        out = np.maximum(out, (rgb * 0.12).astype(np.uint8))
        return out

    def _fragment_transform(self, rgb: np.ndarray) -> np.ndarray:
        # Larger clump packing / swapping rather than per-pixel fragmentation.
        return self._meatpack_transform(rgb)

    def _cuts_transform(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        mask = self._foreground_mask_simple(rgb)
        if not np.any(mask):
            return rgb
        rng = np.random.default_rng(314159)
        angle = rng.uniform(-1.2, 1.2)
        dirx, diry = np.cos(angle), np.sin(angle)
        perpx, perpy = -diry, dirx
        yy, xx = np.indices((H, W))
        coord = xx * dirx + yy * diry
        cmin, cmax = float(coord[mask].min()), float(coord[mask].max())
        cuts = int(5 + 8 * self.frame_transform_strength)
        band = max((cmax - cmin) / max(cuts, 1), 1.0)
        out = np.zeros_like(rgb)
        for i in range(cuts):
            lo = cmin + i * band
            hi = lo + band
            seg = mask & (coord >= lo) & (coord < hi)
            if not np.any(seg):
                continue
            sweep = ((i - 0.5 * cuts) / max(cuts, 1)) * (18.0 + 50.0 * self.frame_transform_strength)
            jitter = rng.uniform(-6.0, 6.0)
            ox = int(round(perpx * (sweep + jitter)))
            oy = int(round(perpy * (sweep + jitter)))
            xs = np.clip(xx[seg] + ox, 0, W - 1)
            ys = np.clip(yy[seg] + oy, 0, H - 1)
            out[ys, xs] = rgb[yy[seg], xx[seg]]
        out = np.maximum(out, (rgb * 0.08).astype(np.uint8))
        return out

    def _build_frame_transform_image(self, out_w: int, out_h: int) -> Image.Image:
        p1, p2 = self._get_fx_param_values(self.frame_transform_mode)
        key = (self._view_state_version, int(out_w), int(out_h), self.frame_transform_mode, round(float(self.frame_transform_strength),3), round(float(self.blob_pack_distance),3), round(float(p1),3), round(float(p2),3), bool(self.vector_flow_show_guides))
        if self._frame_fx_cache_key == key and self._frame_fx_cache_img is not None:
            return self._frame_fx_cache_img.copy()
        rgb = self._sample_rgb_for_spec("main", self._single_view_spec(), out_w=max(160, out_w//3), out_h=max(120, out_h//3))
        mode = str(self.frame_transform_mode)
        if mode == "coral": out = self._coral_growth_transform(rgb)
        elif mode == "vectorflow": out = self._vector_field_transform(rgb)
        elif mode == "voronoi": out = self._voronoi_transform(rgb)
        elif mode == "branch": out = self._branch_transform(rgb)
        elif mode == "drift": out = self._drift_transform(rgb)
        elif mode == "swap": out = self._random_swap_transform(rgb)
        elif mode == "repeat": out = self._repeat_grid_transform(rgb)
        elif mode == "blobgrid": out = self._blob_grid_transform(rgb)
        elif mode == "fragment": out = self._fragment_transform(rgb)
        elif mode == "meatpack": out = self._meatpack_transform(rgb)
        elif mode == "wrapskeleton": out = self._skeleton_wrap_transform(rgb)
        elif mode == "cuts": out = self._cuts_transform(rgb)
        elif mode == "flipcontour": out = np.asarray(Image.fromarray(rgb, mode='RGB').transpose(Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
        elif mode == "stretch":
            out = self._drift_transform(rgb)
            if self._get_fx_param_values("stretch")[1] >= 0.5:
                out = ((out.astype(np.float32) + np.roll(out, 1, axis=1).astype(np.float32) + np.roll(out, -1, axis=1).astype(np.float32)) / 3.0).astype(np.uint8)
        elif mode == "fleshswell": out = self._drift_transform(rgb)
        elif mode == "mold": out = rgb
        elif mode == "vessels": out = rgb
        elif mode == "veinbranch": out = self._branch_transform(rgb)
        elif mode == "grassfire": out = np.asarray(ImageOps.posterize(Image.fromarray((255 - rgb).astype(np.uint8), mode='RGB'), 4), dtype=np.uint8)
        elif mode == "amat": out = self._voronoi_transform(rgb)
        elif mode == "springmass": out = self._drift_transform(rgb)
        elif mode == "steiner": out = self._branch_transform(rgb)
        elif mode == "poisson": out = np.clip((rgb.astype(np.float32) * 0.25 + 120.0 * np.maximum.reduce(rgb, axis=2, keepdims=True) / 255.0), 0, 255).astype(np.uint8)
        elif mode == "meatexpansion":
            out = np.roll(self._skeleton_wrap_transform(rgb), int(3 + 6 * self.frame_transform_strength), axis=1)
        elif mode == "inflation":
            tmp = self._drift_transform(rgb)
            out = np.roll(tmp, int(2 + 4 * self.frame_transform_strength), axis=0)
        elif mode == "myoglobin": out = rgb
        elif mode == "fibertrack": out = self._branch_transform(rgb)
        elif mode == "watermobility": out = np.repeat((np.clip(rgb.mean(axis=2, keepdims=True) * 1.1, 0, 255)).astype(np.uint8), 3, axis=2)
        elif mode == "marbling": out = rgb
        else: out = rgb
        im = Image.fromarray(out, mode='RGB').resize((out_w, out_h), Image.NEAREST)
        d = ImageDraw.Draw(im)
        f = self._scaled_font(14)
        d.rounded_rectangle((8, 8, min(out_w - 8, 520), 34), radius=8, fill=(8,10,14), outline=(180,180,210))
        d.text((16, 12), f"Frame transform: {mode}  strength={self.frame_transform_strength:.2f}  backend={self.fx_backend}", fill=(245,245,255), font=f)
        self._frame_fx_cache_key = key
        self._frame_fx_cache_img = im.copy()
        return im

    def _scene3d_project(self, pts: np.ndarray, W: int, H: int) -> np.ndarray:
        # pts in [0,1]^3
        p = pts.astype(np.float32) - 0.5
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
        Rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], dtype=np.float32)
        q = p @ (Ry @ Rx).T
        z = q[:,2] + 2.2
        sx = W*0.5 + (q[:,0] / np.maximum(z,1e-4)) * min(W,H) * 0.55
        sy2 = H*0.52 - (q[:,1] / np.maximum(z,1e-4)) * min(W,H) * 0.55
        return np.stack([sx, sy2], axis=1)

    def _object_wire_segments_local(self, obj: Dict[str, Any], steps: int = 32) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return local-space line segments for the real primitive shape.

        Sizes are treated as half-extents/radii in normalized volume space, matching
        _inside_scene_object(). The returned segments are later rotated and translated.
        """
        t = str(obj.get("type", "box")).lower()
        s = np.asarray(obj.get("size", [0.12, 0.12, 0.12]), dtype=np.float32)
        s = np.maximum(s, 1e-5)
        segs: List[Tuple[np.ndarray, np.ndarray]] = []

        def p(x, y, z):
            return np.array([x, y, z], dtype=np.float32)

        if t in ("box", "cube"):
            sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
            corners = [
                p(-sx,-sy,-sz), p(sx,-sy,-sz), p(sx,sy,-sz), p(-sx,sy,-sz),
                p(-sx,-sy, sz), p(sx,-sy, sz), p(sx,sy, sz), p(-sx,sy, sz),
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            return [(corners[a], corners[b]) for a, b in edges]

        if t == "sphere":
            # Three great-circle rings, scaled by ellipsoid radii.
            for plane in range(3):
                pts = []
                for i in range(steps + 1):
                    a = 2.0 * math.pi * i / steps
                    if plane == 0:   # xy
                        pts.append(p(math.cos(a)*s[0], math.sin(a)*s[1], 0.0))
                    elif plane == 1: # xz
                        pts.append(p(math.cos(a)*s[0], 0.0, math.sin(a)*s[2]))
                    else:            # yz
                        pts.append(p(0.0, math.cos(a)*s[1], math.sin(a)*s[2]))
                segs.extend((pts[i], pts[i+1]) for i in range(len(pts)-1))
            return segs

        if t == "cylinder":
            # Cylinder axis is local Z, consistent with _inside_scene_object().
            top = []
            bot = []
            for i in range(steps + 1):
                a = 2.0 * math.pi * i / steps
                x = math.cos(a) * float(s[0])
                y = math.sin(a) * float(s[1])
                top.append(p(x, y, float(s[2])))
                bot.append(p(x, y, -float(s[2])))
            segs.extend((top[i], top[i+1]) for i in range(steps))
            segs.extend((bot[i], bot[i+1]) for i in range(steps))
            for i in range(0, steps, max(1, steps // 8)):
                segs.append((bot[i], top[i]))
            return segs

        if t == "cone":
            # Cone base at -Z, tip at +Z, matching _inside_scene_object().
            base = []
            tip = p(0.0, 0.0, float(s[2]))
            for i in range(steps + 1):
                a = 2.0 * math.pi * i / steps
                base.append(p(math.cos(a)*s[0], math.sin(a)*s[1], -float(s[2])))
            segs.extend((base[i], base[i+1]) for i in range(steps))
            for i in range(0, steps, max(1, steps // 8)):
                segs.append((base[i], tip))
            return segs

        return self._object_wire_segments_local({"type": "box", "size": s.tolist()}, steps=steps)

    def _object_wire_segments_world(self, obj: Dict[str, Any], steps: int = 32) -> List[Tuple[np.ndarray, np.ndarray]]:
        c = np.asarray(obj.get("center", [0.5, 0.5, 0.5]), dtype=np.float32)
        R = self._object_rotation_matrix(obj.get("rot", [0.0, 0.0, 0.0]))
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for a, b in self._object_wire_segments_local(obj, steps=steps):
            out.append((c + a @ R.T, c + b @ R.T))
        return out

    def _draw_scene_object_wire(self, draw: ImageDraw.ImageDraw, obj: Dict[str, Any], out_w: int, out_h: int, color, width: int = 2) -> None:
        for a, b in self._object_wire_segments_world(obj, steps=32):
            pa = self._scene3d_project(np.asarray(a, dtype=np.float32)[None, :], out_w, out_h)[0]
            pb = self._scene3d_project(np.asarray(b, dtype=np.float32)[None, :], out_w, out_h)[0]
            draw.line((float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1])), fill=color, width=width)

    def _draw_object_editor_wire(self, draw: ImageDraw.ImageDraw, obj: Dict[str, Any], spec: Dict[str, Any], out_w: int, out_h: int, color, width: int = 2) -> None:
        for a, b in self._object_wire_segments_world(obj, steps=28):
            ax, ay = self._project_to_slice_pixels(a, spec, out_w, out_h)
            bx, by = self._project_to_slice_pixels(b, spec, out_w, out_h)
            # Avoid drawing very long off-panel artifacts.
            if not (-out_w <= ax <= 2*out_w and -out_h <= ay <= 2*out_h and -out_w <= bx <= 2*out_w and -out_h <= by <= 2*out_h):
                continue
            draw.line((ax, ay, bx, by), fill=color, width=width)

    def _build_scene3d_image(self, out_w: int, out_h: int) -> Image.Image:
        key = (
            self._view_state_version, int(out_w), int(out_h), len(self.scene_objects),
            self.selected_object_index, round(self.yaw, 3), round(self.pitch, 3),
            tuple((obj.get("type"), tuple(np.round(obj.get("center", [0,0,0]), 3)), tuple(np.round(obj.get("size", [0,0,0]), 3)), tuple(np.round(obj.get("rot", [0,0,0]), 1)), obj.get("role")) for obj in self.scene_objects),
        )
        if self._scene3d_cache_key == key and self._scene3d_cache_img is not None:
            return self._scene3d_cache_img.copy()

        im = Image.new('RGBA', (out_w, out_h), (10, 12, 16, 255))
        draw = ImageDraw.Draw(im, 'RGBA')

        # Volume cube.
        corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
        proj = self._scene3d_project(corners, out_w, out_h)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            draw.line((proj[a,0], proj[a,1], proj[b,0], proj[b,1]), fill=(150, 165, 195, 255), width=2)

        # Current slice plane preview.
        spec = self._single_view_spec()
        c = np.asarray(spec['center'], dtype=np.float32)
        u = np.asarray(spec['u'], dtype=np.float32) * float(spec['scale_u'])
        v = np.asarray(spec['v'], dtype=np.float32) * float(spec['scale_v'])
        quad = np.stack([c-u-v, c+u-v, c+u+v, c-u+v], axis=0)
        qp = self._scene3d_project(np.clip(quad, 0, 1), out_w, out_h)
        draw.polygon([(float(x), float(y)) for x, y in qp], outline=(70, 190, 255, 255), fill=(70, 190, 255, 36))
        for a, b in [(0,1),(1,2),(2,3),(3,0)]:
            draw.line((qp[a,0], qp[a,1], qp[b,0], qp[b,1]), fill=(70, 190, 255, 255), width=2)

        # Actual primitive shapes, no fake debug spheres.
        if self.scene3d_show_objects:
            for i, obj in enumerate(self.scene_objects):
                selected = (i == self.selected_object_index)
                role = str(obj.get('role', 'masker'))
                base_color = {
                    'masker': (255, 220, 80, 255),
                    'blocker': (255, 110, 90, 255),
                    'reflector': (90, 210, 255, 255),
                    'shifter': (170, 255, 130, 255),
                }.get(role, (190, 220, 255, 255))
                color = (255, 245, 130, 255) if selected else base_color
                self._draw_scene_object_wire(draw, obj, out_w, out_h, color=color, width=3 if selected else 2)
                centerp = self._scene3d_project(np.asarray(obj.get('center', [0.5,0.5,0.5]), dtype=np.float32)[None, :], out_w, out_h)[0]
                draw.text((float(centerp[0]) + 6, float(centerp[1]) - 6), f"{i}:{obj.get('type','?')} {role}", fill=color, font=self._scaled_font(13))

                if selected:
                    # Translation handles: 3 global-axis arrows from object center.
                    c0 = np.asarray(obj.get('center', [0.5,0.5,0.5]), dtype=np.float32)
                    axes = [(np.array([0.16,0,0], np.float32), (255,80,80,255), 'X'),
                            (np.array([0,0.16,0], np.float32), (80,255,100,255), 'Y'),
                            (np.array([0,0,0.16], np.float32), (90,170,255,255), 'Z')]
                    p0 = self._scene3d_project(c0[None, :], out_w, out_h)[0]
                    for vec, col, label in axes:
                        p1 = self._scene3d_project(np.clip((c0 + vec)[None, :], 0, 1), out_w, out_h)[0]
                        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=col, width=3)
                        draw.ellipse((p1[0]-4, p1[1]-4, p1[0]+4, p1[1]+4), fill=col)
                        draw.text((p1[0]+4, p1[1]+2), label, fill=col, font=self._scaled_font(12))

        # Header + labels.
        draw.rounded_rectangle((8, 8, min(out_w-8, 620), 58), radius=8, fill=(8,10,14,230), outline=(180,180,210,255))
        draw.text((16,12), "3D scene preview — real primitive shapes", fill=(245,245,255,255), font=self._scaled_font(14))
        draw.text((16,34), f"Shape types: box / cylinder / sphere / cone. draw3d={self.scene3d_show_objects} affect_image={self.scene_objects_affect_image}", fill=(190,205,230,255), font=self._scaled_font(12))
        self._scene3d_cache_key = key
        self._scene3d_cache_img = im.copy()
        return im

    def _random_seed_slice_spec(self) -> Dict[str, Any]:
        rng = np.random.default_rng(int(self.seed_slice_base_seed + len(self.seed_slice_specs) * 97))
        center = rng.uniform(0.15, 0.85, size=3).astype(np.float32)
        n = rng.normal(size=3).astype(np.float32)
        n = normalize(n)
        u, v = orthonormal_basis_from_normal(n)
        su = float(rng.uniform(0.22, 0.48))
        sv = float(rng.uniform(0.18, 0.42))
        return self._make_spec(center, u, v, n, su, sv, aspect_correct=0)

    def add_seed_slice(self) -> None:
        self.seed_slice_specs.append(self._random_seed_slice_spec())
        self.view_mode = "slice_seed_board"
        self.force_clear_next_frame = True

    def clear_seed_slices(self) -> None:
        self.seed_slice_specs = []
        self.force_clear_next_frame = True

    def cycle_seed_slice_layout(self) -> None:
        items = ["fill", "similar"]
        i = items.index(self.seed_slice_layout) if self.seed_slice_layout in items else 0
        self.seed_slice_layout = items[(i + 1) % len(items)]
        self.force_clear_next_frame = True

    def _slice_thumb_feature(self, rgb: np.ndarray) -> np.ndarray:
        left = rgb[:, :max(1, rgb.shape[1]//10)].mean(axis=(0,1))
        right = rgb[:, -max(1, rgb.shape[1]//10):].mean(axis=(0,1))
        top = rgb[:max(1, rgb.shape[0]//10), :].mean(axis=(0,1))
        bot = rgb[-max(1, rgb.shape[0]//10):, :].mean(axis=(0,1))
        return np.concatenate([left, right, top, bot], axis=0)

    def _build_seed_slice_board_image(self, out_w: int, out_h: int) -> Image.Image:
        im = Image.new("RGB", (out_w, out_h), (8, 10, 14))
        draw = ImageDraw.Draw(im)
        if not self.seed_slice_specs:
            draw.rounded_rectangle((16, 16, min(out_w-16, 620), 56), radius=12, fill=(18,22,30), outline=(120,140,165))
            draw.text((28, 30), "Seed slice board — use SeedAdd to add random slices from the volume.", fill=(240,245,255), font=self._scaled_font(16))
            return im
        thumbs = []
        for i, spec in enumerate(self.seed_slice_specs):
            rgb = self._sample_rgb_for_spec("main", spec, out_w=180, out_h=130)
            thumbs.append((i, rgb, self._slice_thumb_feature(rgb)))
        if self.seed_slice_layout == "similar" and len(thumbs) > 1:
            remaining = thumbs[1:]
            order = [thumbs[0]]
            while remaining:
                prev = order[-1][2]
                j = int(np.argmin([float(np.linalg.norm(prev - cand[2])) for cand in remaining]))
                order.append(remaining.pop(j))
            thumbs = order
        n = len(thumbs)
        cols = max(1, int(np.ceil(np.sqrt(n * out_w / max(out_h,1)))))
        rows = int(np.ceil(n / cols))
        pad = 12
        cell_w = max(80, (out_w - pad * (cols + 1)) // cols)
        cell_h = max(70, (out_h - pad * (rows + 1) - 36) // rows)
        for idx, (src_idx, rgb, feat) in enumerate(thumbs):
            r = idx // cols
            c = idx % cols
            x0 = pad + c * (cell_w + pad)
            y0 = pad + 40 + r * (cell_h + pad)
            thumb = Image.fromarray(rgb, mode='RGB').resize((cell_w, cell_h), Image.NEAREST)
            im.paste(thumb, (x0, y0))
            draw.rounded_rectangle((x0-1, y0-1, x0+cell_w+1, y0+cell_h+1), radius=6, outline=(120,180,255), width=1)
            draw.text((x0 + 6, y0 + 6), f"#{src_idx}", fill=(255,255,255), font=self._scaled_font(13))
        draw.rounded_rectangle((8, 8, min(out_w-8, 640), 34), radius=8, fill=(8,10,14), outline=(180,180,210))
        draw.text((16, 12), f"Seed slice board  count={len(self.seed_slice_specs)}  layout={self.seed_slice_layout}  (add slices from button / N key)", fill=(245,245,255), font=self._scaled_font(14))
        return im

    def add_scene_object(self, kind: str) -> None:
        kind = str(kind).lower()
        if kind in ("cube", "box"):
            kind = "box"
            size = [0.14, 0.14, 0.14]
        elif kind == "sphere":
            size = [0.15, 0.15, 0.15]
        elif kind == "cylinder":
            size = [0.12, 0.12, 0.22]
        elif kind == "cone":
            size = [0.14, 0.14, 0.24]
        else:
            kind = "box"
            size = [0.14, 0.14, 0.14]
        obj = {
            "type": kind,
            "center": [0.5, 0.5, 0.5],
            "size": size,
            "rot": [0.0, 0.0, 0.0],
            "role": "masker",
            "shift_amount": 0.08,
        }
        self.scene_objects.append(obj)
        self.selected_object_index = len(self.scene_objects) - 1
        self.view_mode = "object_editor" if self.view_mode not in ("scene_3d", "object_editor") else self.view_mode
        self.force_clear_next_frame = True
        self._scene3d_cache_key = None
        self._scene3d_cache_img = None
        self.ui_force_rebuild = True

    def cycle_selected_object(self, step: int) -> None:
        if not self.scene_objects:
            self.selected_object_index = -1
            return
        self.selected_object_index = (self.selected_object_index + step) % len(self.scene_objects)
        self.ui_force_rebuild = True

    def delete_selected_object(self) -> None:
        if 0 <= self.selected_object_index < len(self.scene_objects):
            self.scene_objects.pop(self.selected_object_index)
            if not self.scene_objects:
                self.selected_object_index = -1
            else:
                self.selected_object_index %= len(self.scene_objects)
            self.ui_force_rebuild = True

    def _selected_object(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.selected_object_index < len(self.scene_objects):
            return self.scene_objects[self.selected_object_index]
        return None

    def cycle_selected_role(self) -> None:
        obj = self._selected_object()
        if obj is None:
            return
        roles = ["masker", "blocker", "reflector", "shifter"]
        i = roles.index(obj.get("role", "masker")) if obj.get("role", "masker") in roles else 0
        obj["role"] = roles[(i + 1) % len(roles)]
        self.ui_force_rebuild = True

    def set_selected_role(self, role: str) -> None:
        obj = self._selected_object()
        if obj is None:
            return
        obj["role"] = str(role)
        self.role_dropdown_open = False
        self.ui_force_rebuild = True

    def nudge_selected_object(self, axis: int, delta: float) -> None:
        obj = self._selected_object()
        if obj is None:
            return
        obj["center"][axis] = float(np.clip(obj["center"][axis] + delta, 0.0, 1.0))
        self._scene3d_cache_key = None
        self._scene3d_cache_img = None
        self.ui_force_rebuild = True

    def scale_selected_object(self, axis: int, delta: float) -> None:
        obj = self._selected_object()
        if obj is None:
            return
        obj["size"][axis] = float(np.clip(obj["size"][axis] + delta, 0.02, 0.45))
        self._scene3d_cache_key = None
        self._scene3d_cache_img = None
        self.ui_force_rebuild = True

    def rotate_selected_object(self, axis: int, delta_deg: float) -> None:
        obj = self._selected_object()
        if obj is None:
            return
        obj["rot"][axis] = float(obj["rot"][axis] + delta_deg)
        self._scene3d_cache_key = None
        self._scene3d_cache_img = None
        self.ui_force_rebuild = True

    def _object_rotation_matrix(self, rot_deg: Sequence[float]) -> np.ndarray:
        rx, ry, rz = [np.deg2rad(float(v)) for v in rot_deg]
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
        return (Rz @ Ry @ Rx).astype(np.float32)

    def _inside_scene_object(self, pts: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        c = np.asarray(obj["center"], dtype=np.float32)
        s = np.asarray(obj["size"], dtype=np.float32)
        R = self._object_rotation_matrix(obj["rot"])
        q = (pts - c[None, :]) @ R
        t = str(obj.get("type", "sphere"))
        if t == "sphere":
            return np.sum((q / np.maximum(s[None, :], 1e-6))**2, axis=1) <= 1.0
        if t == "box":
            return np.all(np.abs(q) <= s[None, :], axis=1)
        if t == "cylinder":
            rr = (q[:,0] / max(s[0], 1e-6))**2 + (q[:,1] / max(s[1], 1e-6))**2
            return (rr <= 1.0) & (np.abs(q[:,2]) <= s[2])
        if t == "cone":
            z = (q[:,2] + s[2]) / max(2*s[2], 1e-6)
            rad = (1.0 - np.clip(z, 0.0, 1.0))
            rr = np.sqrt((q[:,0] / max(s[0], 1e-6))**2 + (q[:,1] / max(s[1], 1e-6))**2)
            return (z >= 0.0) & (z <= 1.0) & (rr <= rad)
        return np.zeros((len(pts),), dtype=bool)

    def _sample_rgb_with_objects(self, spec: Dict[str, Any], out_w: int = 256, out_h: int = 192) -> np.ndarray:
        vol_arr, _, vol_bgr = self._volume_key_to_assets("main")
        zdim, hdim, wdim = vol_arr.shape[:3]
        xs = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)
        sx, sy = np.meshgrid(xs, ys)
        if spec.get("aspect_correct", 0):
            sx = sx * (out_w / max(1.0, float(out_h)))
        p = (spec["center"][None, None, :]
             + spec["u"][None, None, :] * (sx[..., None] * float(spec["scale_u"]))
             + spec["v"][None, None, :] * (sy[..., None] * float(spec["scale_v"])))
        valid = np.all((p >= 0.0) & (p <= 1.0), axis=2)
        xi = np.clip(np.rint(p[..., 0] * (wdim - 1)).astype(np.int32), 0, wdim - 1)
        yi = np.clip(np.rint(p[..., 1] * (hdim - 1)).astype(np.int32), 0, hdim - 1)
        zi = np.clip(np.rint(p[..., 2] * (zdim - 1)).astype(np.int32), 0, zdim - 1)
        src = np.asarray(vol_arr[zi, yi, xi], dtype=np.uint8)
        rgb = src[..., ::-1].copy() if vol_bgr else src.copy()
        rgb[~valid] = 0
        if (not self.scene_objects) or (not self.scene_objects_affect_image):
            return rgb
        flat_p = p.reshape(-1, 3)
        flat_rgb = rgb.reshape(-1, 3)
        for obj in self.scene_objects:
            inside = self._inside_scene_object(flat_p, obj)
            if not np.any(inside):
                continue
            role = str(obj.get("role", "masker"))
            if role in ("masker", "blocker"):
                flat_rgb[inside] = 0 if role == "masker" else (flat_rgb[inside] * 0.08).astype(np.uint8)
            elif role == "reflector":
                c = np.asarray(obj["center"], dtype=np.float32)
                R = self._object_rotation_matrix(obj["rot"])
                q = (flat_p[inside] - c[None, :]) @ R
                q[:,0] *= -1.0
                pref = c[None, :] + q @ R.T
                pref = np.clip(pref, 0.0, 1.0)
                xri = np.clip(np.rint(pref[:, 0] * (wdim - 1)).astype(np.int32), 0, wdim - 1)
                yri = np.clip(np.rint(pref[:, 1] * (hdim - 1)).astype(np.int32), 0, hdim - 1)
                zri = np.clip(np.rint(pref[:, 2] * (zdim - 1)).astype(np.int32), 0, zdim - 1)
                refl = np.asarray(vol_arr[zri, yri, xri], dtype=np.uint8)
                refl = refl[:, ::-1] if vol_bgr else refl
                flat_rgb[inside] = refl
            elif role == "shifter":
                c = np.asarray(obj["center"], dtype=np.float32)
                vec = flat_p[inside] - c[None, :]
                ln = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
                vec = vec / ln
                pref = np.clip(flat_p[inside] + vec * float(obj.get("shift_amount", 0.08)), 0.0, 1.0)
                xri = np.clip(np.rint(pref[:, 0] * (wdim - 1)).astype(np.int32), 0, wdim - 1)
                yri = np.clip(np.rint(pref[:, 1] * (hdim - 1)).astype(np.int32), 0, hdim - 1)
                zri = np.clip(np.rint(pref[:, 2] * (zdim - 1)).astype(np.int32), 0, zdim - 1)
                sh = np.asarray(vol_arr[zri, yri, xri], dtype=np.uint8)
                sh = sh[:, ::-1] if vol_bgr else sh
                flat_rgb[inside] = sh
        return flat_rgb.reshape(out_h, out_w, 3)

    def _project_to_slice_pixels(self, pt: Sequence[float], spec: Dict[str, Any], W: int, H: int) -> Tuple[float, float]:
        p = np.asarray(pt, dtype=np.float32)
        rel = p - np.asarray(spec["center"], dtype=np.float32)
        x = float(np.dot(rel, spec["u"]) / max(float(spec["scale_u"]), 1e-6))
        y = float(np.dot(rel, spec["v"]) / max(float(spec["scale_v"]), 1e-6))
        if spec.get("aspect_correct", 0):
            x = x / (W / max(1.0, float(H)))
        px = (x * 0.5 + 0.5) * W
        py = (1.0 - (y * 0.5 + 0.5)) * H
        return px, py

    def _build_object_editor_image(self, out_w: int, out_h: int) -> Image.Image:
        spec = self._single_view_spec()
        rgb = self._sample_rgb_with_objects(spec, out_w=max(200, out_w//2), out_h=max(160, out_h//2))
        im = Image.fromarray(rgb, mode='RGB').resize((out_w, out_h), Image.NEAREST)
        draw = ImageDraw.Draw(im)
        small = self._scaled_font(13)
        for i, obj in enumerate(self.scene_objects):
            px, py = self._project_to_slice_pixels(obj["center"], spec, out_w, out_h)
            color = (255, 220, 80) if i == self.selected_object_index else (120, 255, 180)
            # Draw the actual primitive wireframe projected into the current slice,
            # not a generic sphere/ellipse marker.
            self._draw_object_editor_wire(draw, obj, spec, out_w, out_h, color=color, width=3 if i == self.selected_object_index else 2)
            draw.text((px + 8, py - 10), f"{i}:{obj['type']} {obj['role']}", fill=color, font=small)
            # gizmo handles for selected object
            if i == self.selected_object_index:
                for axis_name, axis_vec, col in [("X", (1,0,0), (255,80,80)), ("Y", (0,1,0), (80,255,80)), ("Z", (0,0,1), (90,180,255))]:
                    tip = np.asarray(obj["center"], dtype=np.float32) + np.asarray(axis_vec, dtype=np.float32) * 0.12
                    tx, ty = self._project_to_slice_pixels(tip, spec, out_w, out_h)
                    draw.line((px, py, tx, ty), fill=col, width=3)
                    draw.rectangle((tx - 3, ty - 3, tx + 3, ty + 3), fill=col)
                    draw.text((tx + 2, ty + 2), axis_name, fill=col, font=small)
                draw.arc((px - 38, py - 38, px + 38, py + 38), start=0, end=300, fill=(255,200,120), width=2)
        draw.rounded_rectangle((8, 8, min(out_w - 8, 520), 36), radius=8, fill=(8,10,14), outline=(180,180,210))
        sel = self._selected_object()
        label = f"Object editor  count={len(self.scene_objects)}  selected={self.selected_object_index}"
        if sel is not None:
            label += f"  type={sel['type']} role={sel['role']} affect={self.scene_objects_affect_image}"
        draw.text((16, 12), label, fill=(245,245,255), font=small)
        return im

    def _push_slice_uniforms(self):
        self.slice_prog["u_center"].value = tuple(float(x) for x in self.center)
        self.slice_prog["u_axis_u"].value = tuple(float(x) for x in self.u)
        self.slice_prog["u_axis_v"].value = tuple(float(x) for x in self.v)
        self.slice_prog["u_axis_n"].value = tuple(float(x) for x in self.n)
        self.slice_prog["u_scale"].value  = float(self.scale)
        self.slice_prog["u_scale_u"].value = float(self.scale)
        self.slice_prog["u_scale_v"].value = float(self.scale)
        self.slice_prog["u_aspect_correct"].value = 1
        self.slice_prog["u_slice_px"].value = (float(self.wnd.width), float(self.wnd.height))

        # heap uniforms
        self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
        self.slice_prog["u_mouse"].value = (float(self.mouse_uv[0]), float(self.mouse_uv[1]))
        self.slice_prog["u_radius"].value = float(self.heap_radius)
        self.slice_prog["u_softness"].value = float(self.heap_softness)
        self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
        self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
        self.slice_prog["u_heap_dir"].value = float(self.heap_dir)

        # Curved-plane defaults. Individual panels override these in _render_slice_panel.
        self.slice_prog["u_curved_enable"].value = 0
        self.slice_prog["u_curved_kind"].value = int(getattr(self, "curved_plane_kind", 0))
        self.slice_prog["u_curved_amp"].value = float(getattr(self, "curved_plane_amp", 0.0))
        self.slice_prog["u_curved_radius"].value = float(getattr(self, "curved_plane_radius", 1.0))

        # orientation/color for the default/main volume.
        # Per-panel multi-volume rendering selects bgr_input inside _render_slice_panel(..., volume_key=...).
        self.slice_prog["u_flip_y"].value = int(self.flip_y)
        self.slice_prog["u_bgr_input"].value = int(self.bgr_input)
        self._apply_filter_uniforms()
        self._apply_post_uniforms("main")
        self.slice_prog["u_black_transparent"].value = 0
        self.slice_prog["u_black_threshold"].value = float(self.black_alpha_threshold)
        self.slice_prog["u_output_alpha"].value = 1.0

    # ------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------

    def _set_cursor_visible(self, visible: bool) -> None:
        """Best-effort cursor visibility across moderngl_window backends."""
        visible = bool(visible)

        # Some moderngl_window backends expose a pyglet-like window object.
        backend_window = getattr(self.wnd, "_window", None)
        if backend_window is not None:
            if hasattr(backend_window, "set_mouse_visible"):
                try:
                    backend_window.set_mouse_visible(visible)
                    self.cursor_hidden = not visible
                    return
                except Exception:
                    pass

            # GLFW backend. Import lazily so the script still works with other backends.
            try:
                import glfw
                mode = glfw.CURSOR_NORMAL if visible else glfw.CURSOR_HIDDEN
                glfw.set_input_mode(backend_window, glfw.CURSOR, mode)
                self.cursor_hidden = not visible
                return
            except Exception:
                pass

        # Last-resort backend properties. These are not guaranteed, but harmless.
        for attr in ("cursor", "mouse_visible"):
            try:
                setattr(self.wnd, attr, visible)
                self.cursor_hidden = not visible
                return
            except Exception:
                pass

    def _mark_navigation_input(self) -> None:
        self._last_input_time = time.perf_counter()
        self._view_state_version += 1
        self._pixel_grid_cache_key = None
        self._frame_fx_cache_key = None
        self._scene3d_cache_key = None
        self._mark_analysis_dirty("heuristics", "fx", "myoglobin", "inflation", "meatexpansion", "marbling", "live_recompute")
        if bool(getattr(self, "cursor_force_hidden", False)):
            if not self.cursor_hidden:
                self._set_cursor_visible(False)
            return
        if self.auto_hide_cursor and not self.cursor_hidden:
            self._set_cursor_visible(False)

    def _maybe_restore_cursor(self) -> None:
        if not self.auto_hide_cursor or not self.cursor_hidden:
            return
        if self._drag_plane or self._drag_pan or self._held_keys:
            return
        if time.perf_counter() - self._last_input_time >= self.cursor_hide_delay:
            self._set_cursor_visible(True)


    def toggle_cursor_hidden(self) -> None:
        """Manual mouse visibility toggle. Bound to H."""
        self.cursor_force_hidden = not bool(getattr(self, "cursor_force_hidden", False))
        self.auto_hide_cursor = False
        self._set_cursor_visible(not self.cursor_force_hidden)
        print(f"mouse_hidden={self.cursor_force_hidden}")

    def _scaled_font(self, px: int):
        size = max(8, int(px))
        for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _ui_scale(self) -> float:
        W, H = max(1, int(self.wnd.width)), max(1, int(self.wnd.height))
        return float(np.clip(min(W / 1280.0, H / 720.0), 0.62, 1.85))

    def _add_ui_button(self, draw, rect, label: str, action: str, font, fill=(38, 42, 52, 218), outline=(150, 165, 190, 230)) -> None:
        x0, y0, x1, y1 = [int(v) for v in rect]
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(4, int((y1-y0) * 0.25)), fill=fill, outline=outline, width=1)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(label, font=font)
        draw.text((x0 + max(2, (x1-x0-tw)//2), y0 + max(1, (y1-y0-th)//2 - 1)), label, fill=(245, 248, 255, 255), font=font)
        self.ui_buttons.append({"rect": (x0, y0, x1, y1), "action": action, "label": label})

    def _set_playhead_from_ui_x(self, x: float) -> None:
        if self.ui_scrub_rect is None:
            return
        x0, y0, x1, y1 = self.ui_scrub_rect
        total = self._playback_total_seconds()
        if total <= 0.0:
            return
        t = float(np.clip((float(x) - x0) / max(1.0, float(x1 - x0)), 0.0, 1.0))
        self.playhead_seconds = t * total
        st = self.evaluate_playback(self.playhead_seconds)
        if st is not None:
            self.apply_playback_state(st)
            self.force_clear_next_frame = True
        self.ui_force_rebuild = True

    def _set_fx_strength_from_ui_x(self, x: float) -> None:
        if self.ui_fx_slider_rect is None:
            return
        x0, y0, x1, y1 = self.ui_fx_slider_rect
        u = float(np.clip((float(x) - x0) / max(1.0, float(x1 - x0)), 0.0, 1.0))
        self.frame_transform_strength = 0.05 + u * (1.50 - 0.05)
        self.ui_force_rebuild = True
        self._frame_fx_cache_key = None

    def _set_blob_pack_from_ui_x(self, x: float) -> None:
        if self.ui_blob_slider_rect is None:
            return
        x0, y0, x1, y1 = self.ui_blob_slider_rect
        u = float(np.clip((float(x) - x0) / max(1.0, float(x1 - x0)), 0.0, 1.0))
        self.blob_pack_distance = 0.02 + u * (0.60 - 0.02)
        self.ui_force_rebuild = True
        self._frame_fx_cache_key = None

    def _set_cut_angle_from_ui_x(self, x: float) -> None:
        if self.ui_cut_angle_rect is None:
            return
        x0, y0, x1, y1 = self.ui_cut_angle_rect
        u = float(np.clip((float(x) - x0) / max(1.0, float(x1 - x0)), 0.0, 1.0))
        self.cut_angle_rad = -math.pi + u * (2.0 * math.pi)
        self.ui_force_rebuild = True
        self._frame_fx_cache_key = None

    def _set_curve_amp_from_ui_x(self, x: float) -> None:
        if self.ui_curve_amp_rect is None:
            return
        x0, y0, x1, y1 = self.ui_curve_amp_rect
        u = float(np.clip((float(x) - x0) / max(1.0, float(x1 - x0)), 0.0, 1.0))
        self.curved_plane_amp = -0.35 + u * 0.70
        self._update_gizmo_geometry()
        self.ui_force_rebuild = True

    def _fx_mode_list(self) -> List[str]:
        return ["coral", "vectorflow", "voronoi", "branch", "drift", "swap", "repeat", "blobgrid", "fragment", "meatpack", "wrapskeleton", "cuts", "flipcontour", "stretch", "fleshswell", "mold", "vessels", "veinbranch", "grassfire", "amat", "springmass", "steiner", "poisson", "meatexpansion", "inflation", "myoglobin", "fibertrack", "watermobility", "marbling"]

    def _get_fx_param_values(self, mode: Optional[str] = None) -> Tuple[float, float]:
        mode = str(mode if mode is not None else self.frame_transform_mode)
        vals = self.fx_param_values.get(mode)
        if vals is None:
            return (0.5, 0.5)
        return (float(vals[0]), float(vals[1]))

    def _set_fx_param_from_ui_x(self, which: int, x: float) -> None:
        rect = self.ui_fx_param1_rect if which == 1 else self.ui_fx_param2_rect
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        u = float(np.clip((float(x) - x0) / max(1.0, float(x1 - x0)), 0.0, 1.0))
        vals = list(self.fx_param_values.get(self.frame_transform_mode, [0.5, 0.5]))
        vals[0 if which == 1 else 1] = u
        self.fx_param_values[self.frame_transform_mode] = vals
        self.ui_force_rebuild = True
        self._frame_fx_cache_key = None
        self._fx_analysis_cache_key = None

    def _current_fx_param_specs(self) -> List[Tuple[str, str, float]]:
        m = str(self.frame_transform_mode)
        a, b = self._get_fx_param_values(m)
        specs = {
            "fleshswell": [("Swell amount", "Controls how much red/flesh areas bulge", a)],
            "mold": [("Growth", "How much mold spreads", a), ("Scale", "Patch size / noise frequency", b)],
            "vessels": [("Thickness", "Thickness of vessel lines", a), ("Density", "How many vessel structures appear", b)],
            "veinbranch": [("Density", "How many branches appear", a), ("Spread", "Branch spatial spread", b)],
            "grassfire": [("Pulse speed", "Speed of the burn pulse", a), ("Burn radius", "How far the effect reaches inward", b)],
            "amat": [("Circle size", "Relative circle reconstruction size", a), ("Jitter", "How much the circles wander", b)],
            "springmass": [("Stiffness", "How strongly image is anchored", a), ("Force", "How much it gets pulled around", b)],
            "steiner": [("Density", "How many fungal paths appear", a), ("Fungal spread", "Spatial complexity of growth", b)],
            "poisson": [("Glow amount", "Brightness of internal glow", a), ("Glow radius", "Glow distance from inner ridge", b)],
            "stretch": [("Fill blend", "0 = pure shift, 1 = interpolate new pixels along the stretch", b)],
            "meatexpansion": [("Feed amount", "Underfed to overfed reconstruction", a)],
            "inflation": [("Inflate amount", "Overall inflation pressure", a), ("Tube radius", "Radius of blobby tubes", b)],
            "myoglobin": [("SG smooth", "Savitzky-Golay style smoothing amount", a), ("PLSR model", "Regression / spectral decoupling emphasis", b)],
            "fibertrack": [("Line density", "Density of tracked streamline lines", a), ("Track length", "Strength / continuity of the tract lines", b)],
            "watermobility": [("Decay mix", "Mono vs. multi-exponential wetness emphasis", a), ("Separation", "Intra / extra-cellular separation strength", b)],
            "marbling": [("Otsu thresh", "Threshold between fat and muscle", a), ("Fuzzy blend", "Soft assignment of fat probability", b)],
        }
        return specs.get(m, [])

    def _compute_fx_quality_metrics(self, mode: Optional[str] = None) -> List[Tuple[str, str]]:
        mode = str(mode if mode is not None else self.frame_transform_mode)
        rgb = self._current_frame_rgb(out_w=192, out_h=144).astype(np.float32) / 255.0
        if rgb.size == 0:
            return []
        gray = rgb.mean(axis=2)
        metrics: List[Tuple[str, str]] = []
        if mode == "myoglobin":
            oxy = float(np.clip(np.mean((rgb[...,0] - rgb[...,1]) * 1.8 + rgb[...,0] * 0.3), 0.0, 1.0))
            deoxy = float(np.clip(np.mean((rgb[...,1] - rgb[...,0]) * 1.2 + rgb[...,0] * 0.15), 0.0, 1.0))
            fresh = float(np.clip(0.7 * oxy + 0.3 * (1.0 - deoxy), 0.0, 1.0))
            ht = getattr(self, "hemo_thresholds", {"oxy":0.58, "deoxy":0.42, "fresh":0.56, "savgol":0.50})
            oxy_pass = float(oxy >= float(ht.get("oxy", 0.58)))
            deoxy_pass = float(deoxy <= float(ht.get("deoxy", 0.42)))
            fresh_pass = float(fresh >= float(ht.get("fresh", 0.56)))
            quality = (oxy_pass + deoxy_pass + fresh_pass) / 3.0
            metrics = [("Oxygenation", f"{oxy:.3f}"), ("Deoxy ratio", f"{deoxy:.3f}"), ("Freshness", f"{fresh:.3f}"), ("Pass quality", f"{quality:.3f}")]
        elif mode == "fibertrack":
            gy, gx = np.gradient(gray)
            gmag = np.sqrt(gx * gx + gy * gy)
            ang = np.arctan2(gy, gx)
            coherence = float(np.sqrt(np.mean(np.cos(2*ang))**2 + np.mean(np.sin(2*ang))**2))
            tenderness = float(np.clip(1.0 - coherence * 0.75 + np.mean(gmag) * 0.25, 0.0, 1.0))
            metrics = [("Fiber coherence", f"{coherence:.3f}"), ("Texture energy", f"{float(np.mean(gmag)):.3f}"), ("Tenderness", f"{tenderness:.3f}")]
        elif mode == "watermobility":
            blur = (gray + np.roll(gray,1,0) + np.roll(gray,-1,0) + np.roll(gray,1,1) + np.roll(gray,-1,1)) / 5.0
            intra = float(np.clip(np.mean(blur) * 1.15, 0.0, 1.0))
            extra = float(np.clip(np.mean(np.abs(gray - blur)) * 4.0, 0.0, 1.0))
            whc = float(np.clip(intra - 0.55 * extra, 0.0, 1.0))
            metrics = [("Intra water", f"{intra:.3f}"), ("Drip loss", f"{extra:.3f}"), ("WHC", f"{whc:.3f}")]
        elif mode == "marbling":
            sat = np.max(rgb, axis=2) - np.min(rgb, axis=2)
            fat = ((gray > float(np.percentile(gray, 68))) & (sat < 0.16)).astype(np.uint8)
            imf = float(fat.mean())
            lbl, n = ndimage.label(fat)
            conn = float(np.clip(n / 200.0, 0.0, 1.0))
            score = float(np.clip(imf * 0.6 + conn * 0.4, 0.0, 1.0))
            metrics = [("IMF %", f"{imf*100.0:.1f}%"), ("Connectivity", f"{conn:.3f}"), ("Marbling", f"{score:.3f}")]
        return metrics

    def _handle_ui_action(self, action: str, x: float = 0.0) -> bool:
        if action == "scrub":
            self._drag_ui_scrub = True
            self.playback_enabled = False
            self._set_playhead_from_ui_x(x)
            return True
        if action == "fxslider":
            self._drag_ui_fx_slider = True
            self._set_fx_strength_from_ui_x(x)
            return True
        if action == "blobslider":
            self._drag_ui_blob_slider = True
            self._set_blob_pack_from_ui_x(x)
            return True
        if action == "cutangleslider":
            self._drag_ui_cut_angle = True
            self._set_cut_angle_from_ui_x(x)
            return True
        if action == "fxparam1slider":
            self._drag_ui_fx_param1 = True
            self._set_fx_param_from_ui_x(1, x)
            return True
        if action == "fxparam2slider":
            self._drag_ui_fx_param2 = True
            self._set_fx_param_from_ui_x(2, x)
            return True
        if action == "curveampslider":
            self._drag_ui_curve_amp = True
            self._set_curve_amp_from_ui_x(x)
            return True
        if action == "hemo_oxy_slider":
            self._drag_ui_hemo_oxy = True
            self._set_hemo_threshold_from_ui_x("oxy", x)
            return True
        if action == "hemo_deoxy_slider":
            self._drag_ui_hemo_deoxy = True
            self._set_hemo_threshold_from_ui_x("deoxy", x)
            return True
        if action == "hemo_fresh_slider":
            self._drag_ui_hemo_fresh = True
            self._set_hemo_threshold_from_ui_x("fresh", x)
            return True
        if action == "hemo_sg_slider":
            self._drag_ui_hemo_sg = True
            self._set_hemo_threshold_from_ui_x("savgol", x)
            return True
        if action == "play":
            self.toggle_playback(); self.ui_force_rebuild = True; return True
        if action == "rewind":
            self.playhead_seconds = 0.0
            st = self.evaluate_playback(self.playhead_seconds)
            if st is not None: self.apply_playback_state(st)
            self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "record_camera": self.record_camera_waypoint(); self.ui_force_rebuild = True; return True
        if action == "record_brush": self.record_brush_waypoint(); self.ui_force_rebuild = True; return True
        if action == "record_combined": self.record_combined_waypoint(); self.ui_force_rebuild = True; return True
        if action == "save": self.save_waypoints(); self.ui_force_rebuild = True; return True
        if action == "interp": self.cycle_interpolation(); self.ui_force_rebuild = True; return True
        if action == "noise": self.cycle_noise(); self.ui_force_rebuild = True; return True
        if action == "sec_minus": self.seconds_per_segment_live = max(0.10, self.seconds_per_segment_live - 0.25); self.path_dirty = True; self.ui_force_rebuild = True; return True
        if action == "sec_plus": self.seconds_per_segment_live = min(60.0, self.seconds_per_segment_live + 0.25); self.path_dirty = True; self.ui_force_rebuild = True; return True
        if action == "amp_minus": self.noise_amp_live = max(0.0, self.noise_amp_live - 0.005); self.ui_force_rebuild = True; return True
        if action == "amp_plus": self.noise_amp_live = min(0.35, self.noise_amp_live + 0.005); self.ui_force_rebuild = True; return True
        if action == "freq_minus": self.noise_freq_live = max(0.01, self.noise_freq_live - 0.10); self.ui_force_rebuild = True; return True
        if action == "freq_plus": self.noise_freq_live = min(12.0, self.noise_freq_live + 0.10); self.ui_force_rebuild = True; return True
        if action == "view":
            modes = ["single", "single_gray", "single_invert", "single_gray_invert", "axis", "local", "multi_volume", "live_recompute", "pixel_grid", "frame_fx", "object_editor", "scene_3d", "curved_plane_editor", "slice_seed_board"]
            cur = modes.index(self.view_mode) if self.view_mode in modes else 0
            self.view_mode = modes[(cur + 1) % len(modes)]
            self._update_gizmo_geometry()
            self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "view_single": self.view_mode = "single"; self._update_gizmo_geometry(); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "view_axis": self.view_mode = "axis"; self._update_gizmo_geometry(); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "view_local": self.view_mode = "local"; self._update_gizmo_geometry(); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "view_multivol": self.view_mode = "multi_volume"; self._update_gizmo_geometry(); self.force_clear_next_frame = True; self.ui_force_rebuild = True; self._mark_analysis_dirty("live_recompute"); return True
        if action == "view_recompute": self.view_mode = "live_recompute"; self._update_gizmo_geometry(); self.force_clear_next_frame = True; self.ui_force_rebuild = True; self._mark_analysis_dirty("live_recompute"); return True
        if action == "view_framefx": self.view_mode = "frame_fx"; self._arm_live_blank_check(8); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "view_curved": self.view_mode = "curved_plane_editor"; self.ui_tab = "plane"; self._update_gizmo_geometry(); self._arm_live_blank_check(8); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "toggle_display_backend": self._cycle_live_display_backend(); return True
        if action == "tab_move": self.ui_tab = None if self.ui_tab == "move" else "move"; self.ui_force_rebuild = True; return True
        if action == "tab_timeline": self.ui_tab = None if self.ui_tab == "timeline" else "timeline"; self.ui_force_rebuild = True; return True
        if action == "tab_heuristics": self.ui_tab = None if self.ui_tab == "heuristics" else "heuristics"; self.ui_force_rebuild = True; return True
        if action == "tab_fx": self.ui_tab = None if self.ui_tab == "fx" else "fx"; self.ui_force_rebuild = True; return True
        if action == "tab_objects": self.ui_tab = None if self.ui_tab == "objects" else "objects"; self.ui_force_rebuild = True; return True
        if action == "tab_plane": self.ui_tab = None if self.ui_tab == "plane" else "plane"; self.ui_force_rebuild = True; return True
        if action == "toggle_move_panel": self.panel_visible["move"] = not self.panel_visible.get("move", True); self.ui_force_rebuild = True; return True
        if action == "toggle_timeline_panel": self.panel_visible["timeline"] = not self.panel_visible.get("timeline", True); self.ui_force_rebuild = True; return True
        if action == "toggle_heuristics_panel": self.panel_visible["heuristics"] = not self.panel_visible.get("heuristics", True); self.ui_force_rebuild = True; return True
        if action == "toggle_fx_analysis_panel": self.fx_analysis_visible = not bool(getattr(self, "fx_analysis_visible", True)); self.ui_force_rebuild = True; return True
        if action == "analysis_toggle": self.toggle_analysis_enabled(); return True
        if action == "interest_toggle": self.toggle_live_interest_enabled(); return True
        if action == "gpu_blank_check": self.gpu_blank_check_enabled = not bool(getattr(self, "gpu_blank_check_enabled", False)); self._arm_live_blank_check(6); self.ui_force_rebuild = True; return True
        if action == "fxmode": self.cycle_frame_transform_mode(); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "fxmode_dropdown": self.fx_mode_dropdown_open = not self.fx_mode_dropdown_open; self.ui_force_rebuild = True; return True
        if action == "fxscroll_up":
            self.fx_dropdown_scroll = max(0, int(self.fx_dropdown_scroll) - 1); self.ui_force_rebuild = True; return True
        if action == "fxscroll_down":
            max_scroll = max(0, len(self._fx_mode_list()) - 10)
            self.fx_dropdown_scroll = min(max_scroll, int(self.fx_dropdown_scroll) + 1); self.ui_force_rebuild = True; return True
        if action.startswith("setfx_"):
            self.frame_transform_mode = action[len("setfx_"):]
            self.fx_mode_dropdown_open = False
            self.ui_force_rebuild = True
            self._frame_fx_cache_key = None
            return True
        if action == "fxstrength_minus": self.frame_transform_strength = max(0.05, self.frame_transform_strength - 0.05); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "fxstrength_plus": self.frame_transform_strength = min(1.5, self.frame_transform_strength + 0.05); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "fxguides": self.vector_flow_show_guides = not self.vector_flow_show_guides; self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "stretchfill_toggle":
            vals = list(self.fx_param_values.get("stretch", [0.5, 1.0]))
            vals[1] = 0.0 if vals[1] >= 0.5 else 1.0
            self.fx_param_values["stretch"] = vals
            self.ui_force_rebuild = True; self._frame_fx_cache_key = None; self._fx_analysis_cache_key = None; return True
        if action == "cutpattern":
            items = ["parallel", "grid", "radial", "irregular"]
            i = items.index(self.cut_pattern) if self.cut_pattern in items else 0
            self.cut_pattern = items[(i + 1) % len(items)]
            self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutmotion":
            items = ["fixed", "sine", "noise"]
            i = items.index(self.cut_motion_mode) if self.cut_motion_mode in items else 0
            self.cut_motion_mode = items[(i + 1) % len(items)]
            self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutrandomangle":
            import random as _random
            self.cut_angle_rad = float(_random.uniform(-math.pi, math.pi))
            self.cut_pattern = "irregular"
            self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutangle_minus": self.cut_angle_rad = float(max(-math.pi, self.cut_angle_rad - math.radians(5.0))); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutangle_plus": self.cut_angle_rad = float(min(math.pi, self.cut_angle_rad + math.radians(5.0))); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutpar_minus": self.cut_offset_parallel = max(0.0, self.cut_offset_parallel - 0.01); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutpar_plus": self.cut_offset_parallel = min(0.5, self.cut_offset_parallel + 0.01); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutperp_minus": self.cut_offset_perp = max(0.0, self.cut_offset_perp - 0.01); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "cutperp_plus": self.cut_offset_perp = min(0.5, self.cut_offset_perp + 0.01); self.ui_force_rebuild = True; self._frame_fx_cache_key = None; return True
        if action == "seedview": self.view_mode = "slice_seed_board"; self.ui_force_rebuild = True; self.force_clear_next_frame = True; return True
        if action == "seedadd": self.add_seed_slice(); self.ui_force_rebuild = True; return True
        if action == "seedclear": self.clear_seed_slices(); self.ui_force_rebuild = True; return True
        if action == "seedlayout": self.cycle_seed_slice_layout(); self.ui_force_rebuild = True; return True
        if action == "curvetoggle": self.curved_plane_enable = not bool(self.curved_plane_enable); self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curvekind": self.curved_plane_kind = (int(self.curved_plane_kind) + 1) % max(1, len(self.curved_plane_kind_names)); self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curveamp_minus": self.curved_plane_amp = max(-0.35, float(self.curved_plane_amp) - 0.01); self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curveamp_plus": self.curved_plane_amp = min(0.35, float(self.curved_plane_amp) + 0.01); self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curverad_minus": self.curved_plane_radius = max(0.15, float(self.curved_plane_radius) - 0.05); self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curverad_plus": self.curved_plane_radius = min(3.0, float(self.curved_plane_radius) + 0.05); self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curvereset": self.curved_plane_enable = True; self.curved_plane_kind = 0; self.curved_plane_amp = 0.075; self.curved_plane_radius = 1.0; self._update_gizmo_geometry(); self.ui_force_rebuild = True; return True
        if action == "curvesideviews": self.show_curve_side_panels = not bool(self.show_curve_side_panels); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "curvesidemode": self.cycle_curve_side_panel_mode(); self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "objaffect":
            self.scene_objects_affect_image = not bool(self.scene_objects_affect_image)
            self._scene3d_cache_key = None
            self._scene3d_cache_img = None
            self.ui_force_rebuild = True
            self.force_clear_next_frame = True
            return True
        if action == "objshow3d":
            self.scene3d_show_objects = not bool(self.scene3d_show_objects)
            self._scene3d_cache_key = None
            self._scene3d_cache_img = None
            self.ui_force_rebuild = True
            self.force_clear_next_frame = True
            return True
        if action == "save_image": self.pending_screen_save = True; return True
        if action == "scene3d": self.view_mode = "scene_3d"; self.ui_force_rebuild = True; self.force_clear_next_frame = True; return True
        if action == "add_sphere": self.add_scene_object("sphere"); return True
        if action == "add_box": self.add_scene_object("box"); return True
        if action == "add_cylinder": self.add_scene_object("cylinder"); return True
        if action == "add_cone": self.add_scene_object("cone"); return True
        if action == "obj_prev": self.cycle_selected_object(-1); return True
        if action == "obj_next": self.cycle_selected_object(1); return True
        if action == "obj_del": self.delete_selected_object(); return True
        if action == "obj_role_dropdown": self.role_dropdown_open = not self.role_dropdown_open; self.ui_force_rebuild = True; return True
        if action == "setrole_masker": self.set_selected_role("masker"); return True
        if action == "setrole_blocker": self.set_selected_role("blocker"); return True
        if action == "setrole_reflector": self.set_selected_role("reflector"); return True
        if action == "setrole_shifter": self.set_selected_role("shifter"); return True
        if action == "obj_shift_minus":
            obj = self._selected_object();
            if obj is not None: obj["shift_amount"] = max(0.0, float(obj.get("shift_amount", 0.08)) - 0.01); self.ui_force_rebuild = True
            return True
        if action == "obj_shift_plus":
            obj = self._selected_object();
            if obj is not None: obj["shift_amount"] = min(0.4, float(obj.get("shift_amount", 0.08)) + 0.01); self.ui_force_rebuild = True
            return True
        if action == "obj_role": self.cycle_selected_role(); return True
        if action == "obj_move_x_minus": self.nudge_selected_object(0, -0.02); return True
        if action == "obj_move_x_plus": self.nudge_selected_object(0, 0.02); return True
        if action == "obj_move_y_minus": self.nudge_selected_object(1, -0.02); return True
        if action == "obj_move_y_plus": self.nudge_selected_object(1, 0.02); return True
        if action == "obj_move_z_minus": self.nudge_selected_object(2, -0.02); return True
        if action == "obj_move_z_plus": self.nudge_selected_object(2, 0.02); return True
        if action == "obj_scale_minus": self.scale_selected_object(0, -0.01); self.scale_selected_object(1, -0.01); self.scale_selected_object(2, -0.01); return True
        if action == "obj_scale_plus": self.scale_selected_object(0, 0.01); self.scale_selected_object(1, 0.01); self.scale_selected_object(2, 0.01); return True
        if action == "obj_rx": self.rotate_selected_object(0, 10.0); return True
        if action == "obj_ry": self.rotate_selected_object(1, 10.0); return True
        if action == "obj_rz": self.rotate_selected_object(2, 10.0); return True
        if action == "dispmode": self.cycle_main_display_variant(); self._push_slice_uniforms(); return True
        if action == "auxmain": self.toggle_aux_from_main(); return True
        if action == "pixx": self.cycle_pixel_metric("x"); return True
        if action == "pixy": self.cycle_pixel_metric("y"); return True
        if action == "pixlayout": self.cycle_pixel_layout(); return True
        if action == "mouse": self.toggle_cursor_hidden(); self.ui_force_rebuild = True; return True
        if action == "gizmo": self.show_gizmo = not self.show_gizmo; self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "heap": self.heap_enable = not self.heap_enable; self.slice_prog["u_heap_enable"].value = int(self.heap_enable); self.ui_force_rebuild = True; return True
        if action == "suggest": self.apply_interest_recommendation(); self.ui_force_rebuild = True; return True
        if action == "blobseek": self.apply_blob_dense_uninteresting_recommendation(); self.ui_force_rebuild = True; return True
        if action == "filtermode": self.cycle_color_filter_mode(); self._push_slice_uniforms(); return True
        if action == "filtertarget": self.cycle_color_filter_target(); self._push_slice_uniforms(); return True
        if action == "filterminus": self.color_filter_strength = max(0.0, self.color_filter_strength - 0.05); self._push_slice_uniforms(); self.ui_force_rebuild = True; return True
        if action == "filterplus": self.color_filter_strength = min(1.0, self.color_filter_strength + 0.05); self._push_slice_uniforms(); self.ui_force_rebuild = True; return True
        if action == "markcolor": self.add_timeline_color_mark(); return True
        if action == "capture24": self.toggle_capture24(); self.ui_force_rebuild = True; return True
        if action == "capturescope":
            scopes = ["whole", "panels", "both"]
            i = scopes.index(self.capture_scope_live) if self.capture_scope_live in scopes else 0
            self.capture_scope_live = scopes[(i + 1) % len(scopes)]
            self.ui_force_rebuild = True; return True
        if action == "sort_size": self.sort_waypoints("size"); return True
        if action == "sort_blobs": self.sort_waypoints("blobs"); return True
        if action == "sort_fleshbone": self.sort_waypoints("fleshbone"); return True
        if action == "sort_interest": self.sort_waypoints("interest"); return True
        if action == "sort_blobleast": self.sort_waypoints("blobleast"); return True
        if action == "jsoncap": self.export_capture_json_from_waypoints(); self.ui_force_rebuild = True; return True
        if action == "pack_equal": self._postprocess_capture24_manual("equal"); self.ui_force_rebuild = True; return True
        if action == "pack_close": self._postprocess_capture24_manual("close"); self.ui_force_rebuild = True; return True
        if action == "thickstack": self._postprocess_capture24_manual("thick"); self.ui_force_rebuild = True; return True
        if action == "blobdebug":
            self.blob_debug_visible = not self.blob_debug_visible
            if self.blob_debug_visible:
                self.analysis_enabled = True
            self.ui_force_rebuild = True
            return True
        if action == "heuristics": self.heuristics_visible = not self.heuristics_visible; self.ui_force_rebuild = True; return True
        if action == "hide_all_ui": self.hide_all_overlays = True; self.ui_visible = False; self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        if action == "show_ui": self.hide_all_overlays = False; self.ui_visible = True; self.force_clear_next_frame = True; self.ui_force_rebuild = True; return True
        return False

    def _handle_ui_click(self, x: float, y: float) -> bool:
        show_rect = getattr(self, "ui_show_rect", None)
        if show_rect is not None:
            x0, y0, x1, y1 = show_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("show_ui", x=x)
        if bool(getattr(self, "hide_all_overlays", False)) or not self.ui_visible:
            return False
        # Timeline bar / FX slider hit-test first.
        if self.ui_scrub_rect is not None:
            x0, y0, x1, y1 = self.ui_scrub_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("scrub", x=x)
        if self.ui_fx_slider_rect is not None:
            x0, y0, x1, y1 = self.ui_fx_slider_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("fxslider", x=x)
        if self.ui_blob_slider_rect is not None:
            x0, y0, x1, y1 = self.ui_blob_slider_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("blobslider", x=x)
        if self.ui_cut_angle_rect is not None:
            x0, y0, x1, y1 = self.ui_cut_angle_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("cutangleslider", x=x)
        if self.ui_fx_param1_rect is not None:
            x0, y0, x1, y1 = self.ui_fx_param1_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("fxparam1slider", x=x)
        if self.ui_fx_param2_rect is not None:
            x0, y0, x1, y1 = self.ui_fx_param2_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("fxparam2slider", x=x)
        if self.ui_curve_amp_rect is not None:
            x0, y0, x1, y1 = self.ui_curve_amp_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action("curveampslider", x=x)
        for rect_name, action_name in (("ui_hemo_oxy_rect", "hemo_oxy_slider"), ("ui_hemo_deoxy_rect", "hemo_deoxy_slider"), ("ui_hemo_fresh_rect", "hemo_fresh_slider"), ("ui_hemo_sg_rect", "hemo_sg_slider")):
            rect = getattr(self, rect_name, None)
            if rect is not None:
                x0, y0, x1, y1 = rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return self._handle_ui_action(action_name, x=x)
        for b in self.ui_buttons:
            x0, y0, x1, y1 = b["rect"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._handle_ui_action(str(b["action"]), x=x)
        return False

    def _sample_current_analysis_rgb(self) -> np.ndarray:
        """CPU sample matching the currently controlled slice center/orientation."""
        # Axis mode: use the frontal fixed-Z panel because it is easiest to interpret.
        if self.view_mode == "axis":
            z, y, x = self._volume_index_from_center()
            return np.asarray(self._to_pil_rgb(self.V[z, :, :]).resize((192, 128), Image.BILINEAR), dtype=np.uint8)

        if self.view_mode == "multi_volume":
            spec = self._single_view_spec()
            return self._sample_rgb_for_spec("gradient", spec, out_w=192, out_h=128)

        spec = self._single_view_spec()
        out_w, out_h = 192, 128
        xs = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)
        sx, sy = np.meshgrid(xs, ys)
        if spec.get("aspect_correct", 0):
            sx = sx * (out_w / max(1.0, float(out_h)))
        p = (spec["center"][None, None, :]
             + spec["u"][None, None, :] * (sx[..., None] * float(spec["scale_u"]))
             + spec["v"][None, None, :] * (sy[..., None] * float(spec["scale_v"])))
        valid = np.all((p >= 0.0) & (p <= 1.0), axis=2)
        xi = np.clip(np.rint(p[..., 0] * (self.W - 1)).astype(np.int32), 0, self.W - 1)
        yi = np.clip(np.rint(p[..., 1] * (self.H - 1)).astype(np.int32), 0, self.H - 1)
        zi = np.clip(np.rint(p[..., 2] * (self.Z - 1)).astype(np.int32), 0, self.Z - 1)
        bgr = np.asarray(self.V[zi, yi, xi], dtype=np.uint8)
        bgr[~valid] = 0
        return bgr[..., ::-1].copy() if self.bgr_input else bgr.copy()

    def _analysis_state_key(self) -> Tuple[Any, ...]:
        p1, p2 = self._get_fx_param_values(self.frame_transform_mode)
        return (
            int(getattr(self, "_view_state_version", 0)),
            str(getattr(self, "view_mode", "single")),
            str(getattr(self, "main_display_variant", "normal")),
            str(getattr(self, "color_filter_mode", "none")),
            str(getattr(self, "color_filter_target", "red")),
            round(float(getattr(self, "color_filter_strength", 0.0)), 3),
            str(getattr(self, "frame_transform_mode", "none")),
            round(float(getattr(self, "frame_transform_strength", 0.0)), 3),
            round(float(p1), 3),
            round(float(p2), 3),
            round(float(getattr(self, "cut_angle_rad", 0.0)), 3),
            round(float(getattr(self, "cut_offset_parallel", 0.0)), 3),
            round(float(getattr(self, "cut_offset_perp", 0.0)), 3),
            bool(getattr(self, "vector_flow_show_guides", False)),
        )

    def _update_fx_analysis_if_needed(self) -> None:
        if not bool(getattr(self, "analysis_enabled", False)):
            self.current_fx_analysis = {"paused": True, "reason": "analysis off"}
            return
        key = self._analysis_state_key()
        if self._fx_analysis_cache_key == key and self.current_fx_analysis and not self.analysis_dirty_flags.get("fx", True):
            return
        self._fx_analysis_cache_key = key
        try:
            metrics = self._compute_fx_quality_metrics(self.frame_transform_mode)
            data = {"mode": str(self.frame_transform_mode), "metrics": metrics, "thresholds": []}
            p1, p2 = self._get_fx_param_values(self.frame_transform_mode)
            if str(self.frame_transform_mode) == "myoglobin":
                ht = self.hemo_thresholds
                data["thresholds"] = [("Oxy threshold", f"{ht['oxy']:.2f}"), ("Deoxy threshold", f"{ht['deoxy']:.2f}"), ("Fresh gate", f"{ht['fresh']:.2f}"), ("SG smooth", f"{ht['savgol']:.2f}"), ("PLSR weight", f"{p2:.2f}")]
            elif str(self.frame_transform_mode) in ("inflation", "meatexpansion", "fleshswell", "stretch"):
                data["thresholds"] = [("Strength", f"{float(self.frame_transform_strength):.2f}"), ("Param 1", f"{float(p1):.2f}"), ("Param 2", f"{float(p2):.2f}")]
            elif str(self.frame_transform_mode) == "marbling":
                data["thresholds"] = [("Fat thresh", f"{float(p1):.2f}"), ("Conn. weight", f"{float(p2):.2f}")]
            self.current_fx_analysis = data
            self.analysis_dirty_flags["fx"] = False
        except Exception as exc:
            self.current_fx_analysis = {"error": str(exc), "mode": str(self.frame_transform_mode)}

    def _update_heuristics_if_needed(self) -> None:
        if not bool(getattr(self, "analysis_enabled", False)):
            self.current_heuristics = {"paused": True, "reason": "analysis off"}
            self.current_blob_debug_meta = {"paused": True}
            self.current_blob_debug_image = None
            self._heuristics_cache_key = None
            self._fx_analysis_cache_key = None
            return
        key = self._analysis_state_key()
        if self._heuristics_cache_key == key and self.current_heuristics and not self.analysis_dirty_flags.get("heuristics", True):
            return
        self._heuristics_cache_key = key
        self._last_heuristics_time = time.perf_counter()
        try:
            rgb = self._sample_current_analysis_rgb()
            self.current_heuristics = analyze_slice_heuristics(rgb)
            if self.blob_debug_visible:
                dbg, meta = build_blob_debug_visual(rgb)
                self.current_blob_debug_image = dbg
                self.current_blob_debug_meta = meta
        except Exception as exc:
            self.current_heuristics = {"error": str(exc)}
            self.current_blob_debug_meta = {"error": str(exc)}
        self.analysis_dirty_flags["heuristics"] = False
        self._update_fx_analysis_if_needed()

    def _build_ui_image(self) -> Image.Image:
        W, H = max(1, int(self.wnd.width)), max(1, int(self.wnd.height))
        S = self._ui_scale()
        self.ui_buttons = []
        self.ui_scrub_rect = None
        self.ui_fx_slider_rect = None
        self.ui_blob_slider_rect = None
        self.ui_cut_angle_rect = None
        self.ui_fx_param1_rect = None
        self.ui_fx_param2_rect = None
        self.ui_curve_amp_rect = None
        self.ui_fx_dropdown_rect = None
        self.ui_show_rect = None
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        font = self._scaled_font(13 * S)
        small = self._scaled_font(11 * S)
        title_font = self._scaled_font(15 * S)

        pad = int(10 * S)
        btn_h = int(28 * S)
        gap = int(6 * S)

        # Top toolbar / tabs (mac-style dropdown bar)
        tb_h = int(38 * S)
        draw.rounded_rectangle((pad, pad, W - pad, pad + tb_h), radius=int(12*S), fill=(10, 12, 16, 210), outline=(120, 140, 165, 220), width=1)
        bx = pad + int(8 * S)
        top_buttons = [
            ("Move / Brush", "tab_move", 112),
            ("Timeline", "tab_timeline", 86),
            ("Screen FX", "tab_fx", 88),
            ("Objects", "tab_objects", 82),
            ("Plane", "tab_plane", 68),
            ("Heuristics", "tab_heuristics", 92),
            ("Hide UI", "hide_all_ui", 76),
            (f"Move:{'on' if self.panel_visible.get('move', True) else 'off'}", "toggle_move_panel", 88),
            (f"Time:{'on' if self.panel_visible.get('timeline', True) else 'off'}", "toggle_timeline_panel", 86),
            (f"Heur:{'on' if self.panel_visible.get('heuristics', True) else 'off'}", "toggle_heuristics_panel", 88),
            (f"Chem:{'on' if getattr(self, 'fx_analysis_visible', True) else 'off'}", "toggle_fx_analysis_panel", 86),
            (f"Analysis:{'on' if getattr(self, 'analysis_enabled', False) else 'off'}", "analysis_toggle", 102),
            (f"LiveInt:{'on' if getattr(self, 'interest_recommend_live_enabled', False) else 'off'}", "interest_toggle", 96),
            ("Single", "view_single", 62),
            ("Axis", "view_axis", 52),
            ("Local", "view_local", 58),
            ("MultiVol", "view_multivol", 76),
            ("LiveSDT", "view_recompute", 76),
            ("FXView", "view_framefx", 66),
            ("Curved", "view_curved", 68),
            (f"Live:{getattr(self, 'live_display_backend', 'gpu').upper()}", "toggle_display_backend", 94),
            (f"BlankChk:{'on' if getattr(self, 'gpu_blank_check_enabled', False) else 'off'}", "gpu_blank_check", 104),
            (f"Disp:{self.main_display_variant}", "dispmode", 108),
            (f"AuxMain:{'on' if self.aux_from_main else 'off'}", "auxmain", 110),
            (f"PixX:{self.pixel_grid_x_metric}", "pixx", 96),
            (f"PixY:{self.pixel_grid_y_metric}", "pixy", 96),
            (f"PixMode:{self.pixel_grid_layout}", "pixlayout", 118),
        ]
        for label, action, bw in top_buttons:
            w = int(bw * S)
            if bx + w > W - pad - 4:
                break
            self._add_ui_button(draw, (bx, pad + int(5*S), bx + w, pad + int(5*S) + int(26*S)), label, action, small)
            bx += w + gap

        # determine active bottom tab
        active_tab = self.ui_tab
        if active_tab in ("move", "timeline", "fx", "objects", "plane") and not self.panel_visible.get(active_tab if active_tab in self.panel_visible else "move", True):
            active_tab = None

        panel_w = int(min(W - 2 * pad, max(420 * S, W * 0.52)))
        panel_h = int(min(300 * S, H * 0.34))
        x0 = pad
        y0 = max(pad + tb_h + gap, H - panel_h - pad)
        x1 = x0 + panel_w
        y1 = y0 + panel_h

        if active_tab is not None:
            draw.rounded_rectangle((x0, y0, x1, y1), radius=int(12*S), fill=(8, 10, 14, 190), outline=(110, 130, 160, 210), width=1)
            title = f"{active_tab.title()} panel"
            draw.text((x0 + pad, y0 + int(7*S)), title, fill=(245, 248, 255, 255), font=title_font)

            total = self._playback_total_seconds()
            pct = 0.0 if total <= 0.0 else float(np.clip(self.playhead_seconds / total, 0.0, 1.0))
            if active_tab == "timeline":
                bar_x0 = x0 + pad
                bar_y0 = y0 + int(34 * S)
                bar_x1 = x1 - pad
                bar_y1 = bar_y0 + int(14 * S)
                self.ui_scrub_rect = (bar_x0, bar_y0 - int(5*S), bar_x1, bar_y1 + int(5*S))
                draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=int(7*S), fill=(32, 36, 45, 230), outline=(110, 125, 150, 230), width=1)
                draw.rounded_rectangle((bar_x0, bar_y0, int(bar_x0 + pct * (bar_x1 - bar_x0)), bar_y1), radius=int(7*S), fill=(70, 190, 255, 230))
                knob_x = int(bar_x0 + pct * (bar_x1 - bar_x0))
                draw.ellipse((knob_x-int(6*S), bar_y0-int(4*S), knob_x+int(6*S), bar_y1+int(4*S)), fill=(255,255,255,245))
                row_y = bar_y1 + int(12*S)
                bx = x0 + pad
                buttons = [
                    (f"{"Pause" if self.playback_enabled else "Play"}", "play", 70), ("Restart", "rewind", 68), ("Cam C", "record_camera", 68),
                    ("Brush B", "record_brush", 78), ("Both V", "record_combined", 70), ("Save F", "save", 64),
                    ("JsonCap", "jsoncap", 84), ("Capture F4", "capture24", 104), (f"Scope:{self.capture_scope_live}", "capturescope", 108),
                ]
                for label, action, bw in buttons:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row_y, bx + w, row_y + btn_h), label, action, font)
                    bx += w + gap
                row2_y = row_y + btn_h + gap
                bx = x0 + pad
                controls = [
                    (f"Interp:{self.interpolation_mode_live}", "interp", 124), (f"Noise:{self.noise_type_live}", "noise", 116),
                    ("sec-", "sec_minus", 48), ("sec+", "sec_plus", 48), ("amp-", "amp_minus", 52), ("amp+", "amp_plus", 52), ("freq-", "freq_minus", 58), ("freq+", "freq_plus", 58),
                    ("SortArea", "sort_size", 82), ("SortBlob", "sort_blobs", 82), ("SortF/B", "sort_fleshbone", 78), ("SortInt", "sort_interest", 76), ("SortB-L", "sort_blobleast", 78),
                ]
                for label, action, bw in controls:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row2_y, bx + w, row2_y + btn_h), label, action, small)
                    bx += w + gap
                info = f"seg={self.seconds_per_segment_live:.2f}s   amp={self.noise_amp_live:.3f}   freq={self.noise_freq_live:.2f}   cap24={'ON' if self.capture24_active else 'OFF'}   scope={self.capture_scope_live}"
                draw.text((x0 + pad, y1 - int(19*S)), info, fill=(210, 225, 245, 235), font=small)
            elif active_tab == "fx":
                row_y = y0 + int(34*S)
                bx = x0 + pad
                buttons = [
                    (f"FX:{self.frame_transform_mode}", "fxmode_dropdown", 190), ("Next", "fxmode", 56), ("▲", "fxscroll_up", 34), ("▼", "fxscroll_down", 34), ("str-", "fxstrength_minus", 52), ("str+", "fxstrength_plus", 52),
                    (f"Guides:{'on' if self.vector_flow_show_guides else 'off'}", "fxguides", 94), (f"Cuts:{self.cut_pattern}", "cutpattern", 110), (f"Move:{self.cut_motion_mode}", "cutmotion", 96), ("RandCut", "cutrandomangle", 76), ("∠-", "cutangle_minus", 42), ("∠+", "cutangle_plus", 42), ("∥-", "cutpar_minus", 42), ("∥+", "cutpar_plus", 42), ("⊥-", "cutperp_minus", 42), ("⊥+", "cutperp_plus", 42),
                    ("SeedView", "seedview", 86), ("SeedAdd", "seedadd", 82), ("SeedClr", "seedclear", 76), (f"Seed:{self.seed_slice_layout}", "seedlayout", 104),
                    (f"StretchFill:{'on' if bool(self._get_fx_param_values('stretch')[1] >= 0.5) else 'off'}", "stretchfill_toggle", 116), ("SaveImg", "save_image", 82), ("View->FX", "view", 84), ("PackEq", "pack_equal", 74), ("PackClose", "pack_close", 88), ("ThickStack", "thickstack", 94),
                ]
                for label, action, bw in buttons:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row_y, bx + w, row_y + btn_h), label, action, font)
                    bx += w + gap

                slider_base_y = row_y + btn_h + int(10*S)
                if self.fx_mode_dropdown_open:
                    ddx0 = x0 + pad
                    ddy0 = slider_base_y
                    col_w = int(140 * S)
                    item_h = int(22 * S)
                    visible_rows = 10
                    modes = self._fx_mode_list()
                    max_scroll = max(0, len(modes) - visible_rows)
                    self.fx_dropdown_scroll = int(np.clip(self.fx_dropdown_scroll, 0, max_scroll))
                    visible = modes[self.fx_dropdown_scroll:self.fx_dropdown_scroll + visible_rows]
                    dd_h = visible_rows * (item_h + gap) - gap
                    self.ui_fx_dropdown_rect = (ddx0, ddy0, ddx0 + col_w, ddy0 + dd_h)
                    draw.rounded_rectangle((ddx0 - int(4*S), ddy0 - int(4*S), ddx0 + col_w + int(22*S), ddy0 + dd_h + int(4*S)), radius=int(8*S), fill=(14,18,24,225), outline=(120,140,165,230), width=1)
                    for i, mode_name in enumerate(visible):
                        ry0 = ddy0 + i * (item_h + gap)
                        fill = (70, 88, 110, 230) if mode_name == self.frame_transform_mode else (32, 36, 45, 230)
                        self._add_ui_button(draw, (ddx0, ry0, ddx0 + col_w, ry0 + item_h), mode_name, f"setfx_{mode_name}", small, fill=fill)
                    # scrollbar
                    sbx0 = ddx0 + col_w + int(6*S)
                    sbx1 = sbx0 + int(10*S)
                    draw.rounded_rectangle((sbx0, ddy0, sbx1, ddy0 + dd_h), radius=int(4*S), fill=(28,32,40,220), outline=(100,110,130,220), width=1)
                    if max_scroll > 0:
                        # Knob position is based on scroll index; knob height is based on visible fraction.
                        knob_h = max(int(14*S), int(dd_h * min(1.0, visible_rows / max(1, len(modes)))))
                        travel = max(0, dd_h - knob_h)
                        k0 = int(ddy0 + (self.fx_dropdown_scroll / max_scroll) * travel)
                        k1 = int(k0 + knob_h)
                    else:
                        k0, k1 = int(ddy0), int(ddy0 + dd_h)
                    k0 = int(np.clip(k0, ddy0, ddy0 + dd_h))
                    k1 = int(np.clip(max(k1, k0 + 1), ddy0, ddy0 + dd_h))
                    draw.rounded_rectangle((sbx0, k0, sbx1, k1), radius=int(4*S), fill=(150,190,255,230))
                    draw.text((sbx1 + int(8*S), ddy0), f"scroll {self.fx_dropdown_scroll+1}/{max(1, len(modes)-visible_rows+1)}", fill=(190,205,225,225), font=small)
                    slider_base_y = ddy0 + dd_h + int(14*S)

                # strength slider
                s_x0 = x0 + pad
                s_y0 = slider_base_y
                s_x1 = min(x1 - pad, s_x0 + int(300*S))
                s_y1 = s_y0 + int(14*S)
                self.ui_fx_slider_rect = (s_x0, s_y0 - int(4*S), s_x1, s_y1 + int(4*S))
                draw.rounded_rectangle((s_x0, s_y0, s_x1, s_y1), radius=int(7*S), fill=(32,36,45,230), outline=(110,125,150,230), width=1)
                frac = (self.frame_transform_strength - 0.05) / (1.50 - 0.05)
                fx = int(s_x0 + np.clip(frac,0.0,1.0)*(s_x1-s_x0))
                draw.rounded_rectangle((s_x0, s_y0, fx, s_y1), radius=int(7*S), fill=(255,180,90,230))
                draw.ellipse((fx-int(6*S), s_y0-int(4*S), fx+int(6*S), s_y1+int(4*S)), fill=(255,255,255,245))
                draw.text((s_x1 + int(10*S), s_y0 - int(2*S)), f"FX strength {self.frame_transform_strength:.2f}", fill=(210,225,245,235), font=small)

                b_x0 = x0 + pad
                b_y0 = s_y1 + int(16*S)
                b_x1 = min(x1 - pad, b_x0 + int(300*S))
                b_y1 = b_y0 + int(14*S)
                self.ui_blob_slider_rect = (b_x0, b_y0 - int(4*S), b_x1, b_y1 + int(4*S))
                draw.rounded_rectangle((b_x0, b_y0, b_x1, b_y1), radius=int(7*S), fill=(32,36,45,230), outline=(110,125,150,230), width=1)
                fracb = (self.blob_pack_distance - 0.02) / (0.60 - 0.02)
                bxv = int(b_x0 + np.clip(fracb,0.0,1.0)*(b_x1-b_x0))
                draw.rounded_rectangle((b_x0, b_y0, bxv, b_y1), radius=int(7*S), fill=(110,220,255,230))
                draw.ellipse((bxv-int(6*S), b_y0-int(4*S), bxv+int(6*S), b_y1+int(4*S)), fill=(255,255,255,245))
                draw.text((b_x1 + int(10*S), b_y0 - int(2*S)), f"Pack dist {self.blob_pack_distance:.2f}", fill=(210,225,245,235), font=small)

                # Cut angle slider. It is useful outside cut mode too because the
                # value is remembered until the user switches back to cuts.
                a_x0 = x0 + pad
                a_y0 = b_y1 + int(16*S)
                a_x1 = min(x1 - pad, a_x0 + int(300*S))
                a_y1 = a_y0 + int(14*S)
                self.ui_cut_angle_rect = (a_x0, a_y0 - int(4*S), a_x1, a_y1 + int(4*S))
                draw.rounded_rectangle((a_x0, a_y0, a_x1, a_y1), radius=int(7*S), fill=(32,36,45,230), outline=(110,125,150,230), width=1)
                fraca = (float(self.cut_angle_rad) + math.pi) / (2.0 * math.pi)
                axv = int(a_x0 + np.clip(fraca,0.0,1.0)*(a_x1-a_x0))
                draw.rounded_rectangle((a_x0, a_y0, axv, a_y1), radius=int(7*S), fill=(210,210,210,230))
                draw.ellipse((axv-int(6*S), a_y0-int(4*S), axv+int(6*S), a_y1+int(4*S)), fill=(255,255,255,245))
                draw.text((a_x1 + int(10*S), a_y0 - int(2*S)), f"Cut angle {math.degrees(self.cut_angle_rad):+.0f}°  {self.cut_motion_mode}", fill=(210,225,245,235), font=small)

                param_specs = self._current_fx_param_specs()
                current_y = a_y1 + int(16*S)
                if len(param_specs) >= 1:
                    p1_name, p1_desc, p1_val = param_specs[0]
                    p1_x0, p1_y0 = x0 + pad, current_y
                    p1_x1, p1_y1 = min(x1 - pad, p1_x0 + int(300*S)), p1_y0 + int(14*S)
                    self.ui_fx_param1_rect = (p1_x0, p1_y0 - int(4*S), p1_x1, p1_y1 + int(4*S))
                    draw.rounded_rectangle((p1_x0, p1_y0, p1_x1, p1_y1), radius=int(7*S), fill=(32,36,45,230), outline=(110,125,150,230), width=1)
                    p1k = int(p1_x0 + np.clip(p1_val, 0.0, 1.0) * (p1_x1 - p1_x0))
                    draw.rounded_rectangle((p1_x0, p1_y0, p1k, p1_y1), radius=int(7*S), fill=(180,220,120,230))
                    draw.ellipse((p1k-int(6*S), p1_y0-int(4*S), p1k+int(6*S), p1_y1+int(4*S)), fill=(255,255,255,245))
                    draw.text((p1_x1 + int(10*S), p1_y0 - int(2*S)), f"{p1_name} {p1_val:.2f}", fill=(210,225,245,235), font=small)
                    current_y = p1_y1 + int(16*S)
                if len(param_specs) >= 2:
                    p2_name, p2_desc, p2_val = param_specs[1]
                    p2_x0, p2_y0 = x0 + pad, current_y
                    p2_x1, p2_y1 = min(x1 - pad, p2_x0 + int(300*S)), p2_y0 + int(14*S)
                    self.ui_fx_param2_rect = (p2_x0, p2_y0 - int(4*S), p2_x1, p2_y1 + int(4*S))
                    draw.rounded_rectangle((p2_x0, p2_y0, p2_x1, p2_y1), radius=int(7*S), fill=(32,36,45,230), outline=(110,125,150,230), width=1)
                    p2k = int(p2_x0 + np.clip(p2_val, 0.0, 1.0) * (p2_x1 - p2_x0))
                    draw.rounded_rectangle((p2_x0, p2_y0, p2k, p2_y1), radius=int(7*S), fill=(220,160,255,230))
                    draw.ellipse((p2k-int(6*S), p2_y0-int(4*S), p2k+int(6*S), p2_y1+int(4*S)), fill=(255,255,255,245))
                    draw.text((p2_x1 + int(10*S), p2_y0 - int(2*S)), f"{p2_name} {p2_val:.2f}", fill=(210,225,245,235), font=small)
                    current_y = p2_y1 + int(14*S)

                if len(param_specs) >= 1:
                    desc_txt = param_specs[0][1]
                    if len(param_specs) >= 2:
                        desc_txt += f" | {param_specs[1][0]}: {param_specs[1][1]}"
                    draw.text((x0 + pad, current_y), desc_txt, fill=(185,205,225,225), font=small)
                    current_y += int(16*S)

                metrics = self._compute_fx_quality_metrics(self.frame_transform_mode)
                if metrics:
                    draw.text((x0 + pad, current_y), "Estimated values:", fill=(240,245,255,235), font=small)
                    current_y += int(15*S)
                    for name, value in metrics[:4]:
                        draw.text((x0 + pad + int(10*S), current_y), f"{name}: {value}", fill=(205,225,245,235), font=small)
                        current_y += int(14*S)

                draw.text((x0 + pad, y1 - int(19*S)), f"current transform={self.frame_transform_mode}  strength={self.frame_transform_strength:.2f}  pack={self.blob_pack_distance:.2f}  cuts=({self.cut_pattern}/{self.cut_motion_mode}, angle={math.degrees(self.cut_angle_rad):+.0f}°, ∥{self.cut_offset_parallel:.2f}, ⊥{self.cut_offset_perp:.2f})  seed={len(self.seed_slice_specs)}:{self.seed_slice_layout}  backend={self.fx_backend}", fill=(210, 225, 245, 235), font=small)
            elif active_tab == "plane":
                yy = y0 + int(34*S)
                kind_names = getattr(self, "curved_plane_kind_names", ["paraboloid"])
                kind = int(getattr(self, "curved_plane_kind", 0)) % max(1, len(kind_names))
                draw.text((x0 + pad, yy), "Curved slicing plane replaces the flat red U/V plane.", fill=(210,225,245,235), font=small)
                yy += int(24*S)
                row_h = int(24*S)
                self._add_ui_button(draw, (x0+pad, yy, x0+pad+int(92*S), yy+row_h), f"Enable:{'on' if self.curved_plane_enable else 'off'}", "curvetoggle", small)
                self._add_ui_button(draw, (x0+pad+int(100*S), yy, x0+pad+int(260*S), yy+row_h), f"Shape:{kind_names[kind]}", "curvekind", small)
                self._add_ui_button(draw, (x0+pad+int(270*S), yy, x0+pad+int(356*S), yy+row_h), "Show View", "view_curved", small)
                self._add_ui_button(draw, (x0+pad+int(366*S), yy, x0+pad+int(476*S), yy+row_h), f"SideViews:{'on' if self.show_curve_side_panels else 'off'}", "curvesideviews", small)
                self._add_ui_button(draw, (x0+pad+int(484*S), yy, x0+pad+int(606*S), yy+row_h), f"SideMode:{str(getattr(self, 'curve_side_panel_mode', 'local_curved'))[:10]}", "curvesidemode", small)
                yy += int(34*S)

                draw.text((x0 + pad, yy), f"Amplitude along plane normal: {self.curved_plane_amp:+.3f}", fill=(235,240,255,245), font=small)
                self._add_ui_button(draw, (x0+pad+int(300*S), yy-int(4*S), x0+pad+int(342*S), yy+row_h-int(4*S)), "-Amp", "curveamp_minus", small)
                self._add_ui_button(draw, (x0+pad+int(348*S), yy-int(4*S), x0+pad+int(390*S), yy+row_h-int(4*S)), "+Amp", "curveamp_plus", small)
                yy += int(18*S)
                s_x0 = x0 + pad
                s_x1 = x1 - pad - int(10*S)
                s_y0 = yy
                s_y1 = yy + int(12*S)
                self.ui_curve_amp_rect = (s_x0, s_y0 - int(4*S), s_x1, s_y1 + int(4*S))
                draw.rounded_rectangle((s_x0, s_y0, s_x1, s_y1), radius=max(2, int(4*S)), fill=(34,42,54,235), outline=(96,112,132,255), width=max(1, int(1*S)))
                zero_u = (0.0 - (-0.35)) / 0.70
                zx = int(s_x0 + zero_u * (s_x1 - s_x0))
                draw.line((zx, s_y0 - int(2*S), zx, s_y1 + int(2*S)), fill=(180,190,205,220), width=max(1, int(1*S)))
                u_amp = float(np.clip((self.curved_plane_amp + 0.35) / 0.70, 0.0, 1.0))
                kx = int(s_x0 + u_amp * (s_x1 - s_x0))
                draw.rounded_rectangle((s_x0, s_y0, kx, s_y1), radius=max(2, int(4*S)), fill=(220,90,90,220))
                draw.ellipse((kx - int(6*S), s_y0 - int(3*S), kx + int(6*S), s_y1 + int(3*S)), fill=(255,235,120,255), outline=(20,20,20,255), width=max(1, int(1*S)))
                yy += int(30*S)

                draw.text((x0 + pad, yy), f"Curve radius / width: {self.curved_plane_radius:.2f}", fill=(235,240,255,245), font=small)
                self._add_ui_button(draw, (x0+pad+int(300*S), yy-int(4*S), x0+pad+int(342*S), yy+row_h-int(4*S)), "-Rad", "curverad_minus", small)
                self._add_ui_button(draw, (x0+pad+int(348*S), yy-int(4*S), x0+pad+int(390*S), yy+row_h-int(4*S)), "+Rad", "curverad_plus", small)
                yy += int(34*S)

                self._add_ui_button(draw, (x0+pad, yy, x0+pad+int(110*S), yy+row_h), "Reset Curve", "curvereset", small)
                yy += int(34*S)
                draw.text((x0 + pad, yy), "Tip: move/rotate the red plane as usual. Curvature is applied after U/V mapping.", fill=(185,205,225,225), font=small)

            elif active_tab == "objects":
                row_y = y0 + int(34*S)
                bx = x0 + pad
                buttons = [
                    ("+Sphere", "add_sphere", 78), ("+Box", "add_box", 64), ("+Cylinder", "add_cylinder", 88), ("+Cone", "add_cone", 70),
                    ("Prev", "obj_prev", 54), ("Next", "obj_next", 54), ("Role▼", "obj_role_dropdown", 72), ("Delete", "obj_del", 62), ("3D View", "scene3d", 78), (f"Affect:{'on' if self.scene_objects_affect_image else 'off'}", "objaffect", 92), (f"Show3D:{'on' if self.scene3d_show_objects else 'off'}", "objshow3d", 98), ("SaveImg", "save_image", 82),
                ]
                for label, action, bw in buttons:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row_y, bx + w, row_y + btn_h), label, action, font)
                    bx += w + gap
                row2_y = row_y + btn_h + gap
                bx = x0 + pad
                controls = [
                    ("X-", "obj_move_x_minus", 40), ("X+", "obj_move_x_plus", 40), ("Y-", "obj_move_y_minus", 40), ("Y+", "obj_move_y_plus", 40), ("Z-", "obj_move_z_minus", 40), ("Z+", "obj_move_z_plus", 40),
                    ("S-", "obj_scale_minus", 40), ("S+", "obj_scale_plus", 40), ("Sh-", "obj_shift_minus", 42), ("Sh+", "obj_shift_plus", 42), ("Rx", "obj_rx", 40), ("Ry", "obj_ry", 40), ("Rz", "obj_rz", 40),
                ]
                for label, action, bw in controls:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row2_y, bx + w, row2_y + btn_h), label, action, small)
                    bx += w + gap
                if self.role_dropdown_open:
                    ddx0 = x0 + pad + int(250*S)
                    ddy0 = row2_y + btn_h + gap
                    roles = [("masker", "setrole_masker"), ("blocker", "setrole_blocker"), ("reflector", "setrole_reflector"), ("shifter", "setrole_shifter")]
                    for ridx, (rlabel, raction) in enumerate(roles):
                        ry = ddy0 + ridx * (btn_h + gap)
                        self._add_ui_button(draw, (ddx0, ry, ddx0 + int(110*S), ry + btn_h), rlabel, raction, small)
                sel = self._selected_object()
                info = f"objects={len(self.scene_objects)}"
                if sel is not None:
                    info += f"  selected={self.selected_object_index}  type={sel['type']} role={sel['role']} c={tuple(round(v,2) for v in sel['center'])} s={tuple(round(v,2) for v in sel['size'])} shift={sel.get('shift_amount',0.08):.2f}"
                draw.text((x0 + pad, y1 - int(19*S)), info, fill=(210, 225, 245, 235), font=small)
            else:  # move / brush
                row_y = y0 + int(34*S)
                bx = x0 + pad
                buttons = [
                    (f"View:{self.view_mode}", "view", 106), ("Mouse H", "mouse", 82), ("Gizmo F2", "gizmo", 86), ("Heap", "heap", 60),
                    ("Suggest F3", "suggest", 98), ("BlobSeek F6", "blobseek", 108), (f"Analysis:{'on' if getattr(self, 'analysis_enabled', False) else 'off'}", "analysis_toggle", 104), ("BlobDbg", "blobdebug", 74), ("SeedView", "seedview", 86), ("SeedAdd", "seedadd", 82), ("SaveImg", "save_image", 82),
                ]
                for label, action, bw in buttons:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row_y, bx + w, row_y + btn_h), label, action, font)
                    bx += w + gap
                row2_y = row_y + btn_h + gap
                bx = x0 + pad
                controls = [
                    (f"Filter:{self.color_filter_mode}", "filtermode", 128), (f"Target:{self.color_filter_target}", "filtertarget", 126),
                    ("flt-", "filterminus", 48), ("flt+", "filterplus", 48), ("MarkClr", "markcolor", 76),
                    ("PackEq", "pack_equal", 74), ("PackClose", "pack_close", 88), ("ThickStack", "thickstack", 94), (f"Seed:{self.seed_slice_layout}", "seedlayout", 104), ("SeedClr", "seedclear", 76),
                ]
                for label, action, bw in controls:
                    w = int(bw * S)
                    if bx + w > x1 - pad:
                        break
                    self._add_ui_button(draw, (bx, row2_y, bx + w, row2_y + btn_h), label, action, small)
                    bx += w + gap
                info = f"display={self.main_display_variant}   view={self.view_mode}   filter={self.color_filter_mode}:{self.color_filter_target}@{self.color_filter_strength:.2f}   aux_from_main={self.aux_from_main}   seed_slices={len(self.seed_slice_specs)}:{self.seed_slice_layout}"
                draw.text((x0 + pad, y1 - int(19*S)), info, fill=(210, 225, 245, 235), font=small)

        # Panel labels for any three-panel mode.
        if self.view_mode not in ("single", "single_gray", "single_invert", "single_gray_invert", "pixel_grid", "frame_fx", "object_editor", "scene_3d", "curved_plane_editor", "slice_seed_board"):
            if self.view_mode == "multi_volume":
                label_specs = [spec for _, spec in self._multi_volume_specs()]
            elif self.view_mode == "live_recompute":
                label_specs = [{"name": n} for n in ["Original volume", "Live signed distance", "Live skeleton"]]
            else:
                label_specs = self._split_view_specs()
            for viewport, spec in zip(self._panel_viewports(), label_specs):
                vx, vy, vw, vh = viewport
                tx = int(vx + 12)
                ty = int(max(8, H - (vy + vh) + 10))
                label = str(spec.get("name", "panel"))
                try:
                    bbox = draw.textbbox((0, 0), label, font=small)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = draw.textsize(label, font=small)
                draw.rounded_rectangle((tx - 6, ty - 4, tx + tw + 8, ty + th + 4), radius=max(4, int(5 * S)), fill=(10, 10, 14, 150), outline=(180, 180, 210, 180), width=1)
                draw.text((tx, ty - 1), label, fill=(245, 245, 255, 245), font=small)

        top_panel_bottom = pad + tb_h + gap
        if self.panel_visible.get("heuristics", True):
            self._update_heuristics_if_needed()
            self._update_interest_if_needed()
            hw = int(min(W - 2 * pad, max(360 * S, W * 0.31)))
            hx0, hy0 = pad, top_panel_bottom
            hdata = self.current_heuristics or {}
            lines = []
            if "error" in hdata:
                lines = ["Slice heuristics", f"error: {hdata['error'][:42]}"]
            elif hdata.get("paused"):
                lines = [
                    "Slice heuristics",
                    f"paused: {hdata.get('reason', 'analysis off')}",
                    "click Analysis:on to run blob/area counters",
                ]
            else:
                lines = [
                    "Slice heuristics",
                    f"filled area: {100.0 * float(hdata.get('fill_ratio', 0.0)):.1f}%",
                    f"continuous shapes/blobs: {int(hdata.get('blob_count', 0))}",
                    f"circle-like blobs: {int(hdata.get('circle_count', 0))}",
                    f"bone-like: {100.0 * float(hdata.get('bone_ratio', 0.0)):.1f}%   flesh-like: {100.0 * float(hdata.get('flesh_ratio', 0.0)):.1f}%",
                    f"classification: {hdata.get('tissue', 'unknown')}",
                ]
            line_h = int(18 * S)
            hh = int((len(lines) + 1) * line_h + 2 * pad)
            draw.rounded_rectangle((hx0, hy0, hx0 + hw, hy0 + hh), radius=int(12*S), fill=(8, 10, 14, 170), outline=(110, 130, 160, 180), width=1)
            yy = hy0 + pad
            for i, line in enumerate(lines):
                draw.text((hx0 + pad, yy), line, fill=(255, 245, 210, 255) if i == 0 else (235, 240, 245, 235), font=title_font if i == 0 else small)
                yy += line_h

            idata = self.current_interest or {}
            i_lines = ["Interest / next camera"]
            if idata.get("paused"):
                i_lines += [idata.get("reason", "live interest off")]
            elif idata.get("unavailable"):
                i_lines += [idata.get("reason", "unavailable")]
            elif "error" in idata:
                i_lines += [f"error: {idata['error'][:42]}"]
            else:
                i_lines += [
                    f"best plane: {idata.get('name', 'unknown')}",
                    f"score: {float(idata.get('score', 0.0)):.3f}",
                    f"grad mean: {float(idata.get('gradient_mean', 0.0)):.3f}   grad fill: {float(idata.get('gradient_fill', 0.0)):.3f}",
                    f"skeleton fill: {float(idata.get('skeleton_fill', 0.0)):.3f}   overlap: {float(idata.get('overlap_ratio', 0.0)):.3f}",
                ]
            if self.last_blob_dense_recommendation and not self.last_blob_dense_recommendation.get("unavailable"):
                i_lines += [f"blobseek: {self.last_blob_dense_recommendation.get('name','')}  score={float(self.last_blob_dense_recommendation.get('blobseek_score',0.0)):.3f}  blobs={int(self.last_blob_dense_recommendation.get('blob_count',0))}"]
            ix0, iy0 = pad, hy0 + hh + gap
            iw = hw
            ih = int((len(i_lines) + 1) * line_h + 2 * pad)
            draw.rounded_rectangle((ix0, iy0, ix0 + iw, iy0 + ih), radius=int(12*S), fill=(8, 12, 18, 165), outline=(150, 120, 90, 180), width=1)
            yy = iy0 + pad
            for i, line in enumerate(i_lines):
                draw.text((ix0 + pad, yy), line, fill=(255, 230, 190, 255) if i == 0 else (235, 240, 245, 235), font=title_font if i == 0 else small)
                yy += line_h

            if bool(getattr(self, "fx_analysis_visible", True)):
                self._update_fx_analysis_if_needed()
                fdata = self.current_fx_analysis or {}
                f_lines = [f"FX analysis / {str(getattr(self, 'frame_transform_mode', 'none'))}"]
                if "error" in fdata:
                    f_lines += [f"error: {fdata['error'][:42]}"]
                elif fdata.get("paused"):
                    f_lines += [str(fdata.get("reason", "analysis off"))]
                else:
                    for name, value in list(fdata.get("metrics", []))[:4]:
                        f_lines.append(f"{name}: {value}")
                    for name, value in list(fdata.get("thresholds", []))[:4]:
                        f_lines.append(f"{name}: {value}")
                    f_lines.append("updates only after movement or UI changes")
                fx0 = pad
                fy0 = iy0 + ih + gap
                fw = hw
                fh = int((len(f_lines) + 1) * line_h + 2 * pad)
                draw.rounded_rectangle((fx0, fy0, fx0 + fw, fy0 + fh), radius=int(12*S), fill=(10, 12, 18, 165), outline=(120, 145, 170, 180), width=1)
                yy = fy0 + pad
                for i, line in enumerate(f_lines):
                    draw.text((fx0 + pad, yy), line, fill=(220, 240, 255, 255) if i == 0 else (235, 240, 245, 235), font=title_font if i == 0 else small)
                    yy += line_h
                next_y = fy0 + fh + gap

                current_mode = str(getattr(self, 'frame_transform_mode', 'none'))
                panel_modes = []
                if current_mode == 'myoglobin':
                    panel_modes.append('myoglobin')
                if current_mode == 'inflation':
                    panel_modes.append('inflation')
                if current_mode == 'meatexpansion':
                    panel_modes.append('meatexpansion')
                if current_mode == 'marbling':
                    panel_modes.append('marbling')
                if self.view_mode == 'live_recompute':
                    panel_modes.append('live_recompute')
                for pmode in panel_modes:
                    pdata = self.analysis_panel_cache.get(pmode) if (pmode == 'live_recompute' and not self.analysis_dirty_flags.get('live_recompute', True)) else self._cached_mode_panel(pmode) if pmode != 'live_recompute' else self.analysis_panel_cache.get('live_recompute', {"title": "Realtime recompute", "lines": ["updates after movement"]})
                    p_lines = [str(pdata.get('title', pmode))] + list(pdata.get('lines', []))
                    ph = int((len(p_lines) + 1) * line_h + 2 * pad)
                    draw.rounded_rectangle((fx0, next_y, fx0 + fw, next_y + ph), radius=int(12*S), fill=(10, 12, 18, 165), outline=(150, 170, 120, 180), width=1)
                    yy = next_y + pad
                    for i, line in enumerate(p_lines):
                        draw.text((fx0 + pad, yy), line, fill=(225, 245, 220, 255) if i == 0 else (235, 240, 245, 235), font=title_font if i == 0 else small)
                        yy += line_h
                    if pmode == 'myoglobin':
                        slider_specs = [('oxy', 'Oxy', self.hemo_thresholds['oxy']), ('deoxy', 'Deoxy', self.hemo_thresholds['deoxy']), ('fresh', 'Fresh', self.hemo_thresholds['fresh']), ('savgol', 'SG', self.hemo_thresholds['savgol'])]
                        sx0 = fx0 + pad
                        sx1 = fx0 + fw - pad
                        sy = next_y + ph - int(4 * line_h)
                        for idx, (key_name, short_name, sval) in enumerate(slider_specs):
                            yb = sy + idx * int(0.85 * line_h)
                            draw.text((sx0, yb - int(12*S)), f"{short_name} {sval:.2f}", fill=(220, 230, 240, 240), font=small)
                            draw.rounded_rectangle((sx0, yb, sx1, yb + int(10*S)), radius=int(5*S), fill=(34, 40, 50, 255), outline=(90, 100, 120, 255), width=1)
                            kx = int(sx0 + (sx1 - sx0) * float(sval))
                            draw.rounded_rectangle((kx - int(5*S), yb - int(3*S), kx + int(5*S), yb + int(13*S)), radius=int(4*S), fill=(150, 190, 255, 230))
                            setattr(self, f'ui_hemo_{key_name}_rect', (sx0, yb - int(4*S), sx1, yb + int(14*S)))
                    next_y += ph + gap

        if self.blob_debug_visible:
            self._update_heuristics_if_needed()
            dbg = self.current_blob_debug_image
            meta = self.current_blob_debug_meta or {}
            if dbg is not None:
                dbg_im = Image.fromarray(dbg, mode="RGB")
                panel_w = int(min(max(220 * S, dbg_im.width * 1.4), W * 0.30))
                scale = panel_w / max(1, dbg_im.width)
                panel_h = int(dbg_im.height * scale)
                dbg_im = dbg_im.resize((panel_w, panel_h), Image.NEAREST)
                dx1 = W - pad
                dx0 = dx1 - panel_w - 2 * pad
                dy0 = pad + tb_h + gap
                extra_lines = [
                    "Blob debug",
                    f"connected blobs: {int(meta.get('blob_count', 0))}",
                    f"threshold: {float(meta.get('threshold', 0.0)):.1f}",
                    "white boxes/colors = detected continuous shapes",
                ]
                line_h = int(18 * S)
                dy1 = dy0 + panel_h + 2 * pad + line_h * len(extra_lines)
                draw.rounded_rectangle((dx0, dy0, dx1, dy1), radius=int(12*S), fill=(8, 10, 14, 175), outline=(110, 130, 160, 180), width=1)
                draw.text((dx0 + pad, dy0 + pad - 2), extra_lines[0], fill=(255, 230, 190, 255), font=title_font)
                img.paste(dbg_im, (dx0 + pad, dy0 + int(22 * S)), None)
                ty = dy0 + int(22 * S) + panel_h + int(6 * S)
                for line in extra_lines[1:]:
                    draw.text((dx0 + pad, ty), line, fill=(235, 240, 245, 235), font=small)
                    ty += line_h

        return img

    def _upload_ui_texture(self, img: Image.Image) -> None:
        W, H = img.size
        if self.ui_tex is None or self.ui_tex_size != (W, H):
            try:
                if self.ui_tex is not None:
                    self.ui_tex.release()
            except Exception:
                pass
            self.ui_tex = self.ctx.texture((W, H), components=4, dtype="f1")
            self.ui_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.ui_tex_size = (W, H)
        try:
            data = img.tobytes("raw", "RGBA")
        except MemoryError:
            # Keep the renderer alive under extreme memory pressure. The root cause
            # was usually grayscale aux volumes expanded to RGB; v24 fixes that, but
            # this guard prevents a HUD upload from crashing the whole viewer.
            self.ui_visible = False
            self.ui_force_rebuild = True
            print("[ui] skipped HUD upload due to MemoryError; press F1/Show UI after memory pressure drops")
            return
        self.ui_tex.write(data)

    def _render_builtin_ui(self) -> None:
        W, H = max(1, int(self.wnd.width)), max(1, int(self.wnd.height))
        now = time.perf_counter()
        interval = 1.0 / max(1.0, float(getattr(self, "ui_update_fps", 2.0)))
        live_ui_refresh = bool(getattr(self, "ui_live_refresh_enabled", False))

        if bool(getattr(self, "hide_all_overlays", False)) or not self.ui_visible:
            need_rebuild = (
                self.ui_tex is None
                or self.ui_tex_size != (W, H)
                or bool(getattr(self, "ui_force_rebuild", False))
                or (live_ui_refresh and (now - float(getattr(self, "_last_ui_build_time", 0.0))) >= interval)
            )
            if need_rebuild:
                S = self._ui_scale()
                img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img, "RGBA")
                font = self._scaled_font(12 * S)
                pad = int(10 * S)
                bw = int(92 * S)
                bh = int(28 * S)
                x1 = W - pad
                y0 = pad
                self.ui_show_rect = (x1 - bw, y0, x1, y0 + bh)
                draw.rounded_rectangle(self.ui_show_rect, radius=int(8 * S), fill=(10, 12, 16, 190), outline=(140, 160, 180, 220), width=1)
                draw.text((x1 - bw + int(12 * S), y0 + int(6 * S)), "Show UI", fill=(245, 248, 255, 255), font=font)
                self._upload_ui_texture(img)
                self._last_ui_build_time = now
                self.ui_force_rebuild = False
            if self.ui_tex is None:
                return
            self.ctx.viewport = (0, 0, W, H)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ui_tex.use(location=1)
            self.hud_vao.render(mode=moderngl.TRIANGLES)
            self.ctx.disable(moderngl.BLEND)
            return

        need_rebuild = (
            self.ui_tex is None
            or self.ui_tex_size != (W, H)
            or bool(getattr(self, "ui_force_rebuild", False))
            or (live_ui_refresh and (now - float(getattr(self, "_last_ui_build_time", 0.0))) >= interval)
        )
        if need_rebuild:
            img = self._build_ui_image()
            self._upload_ui_texture(img)
            self._last_ui_build_time = now
            self.ui_force_rebuild = False
        if self.ui_tex is None:
            return
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ui_tex.use(location=1)
        self.hud_vao.render(mode=moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)

    # ------------------------------------------------------------
    # Gizmo helpers
    # ------------------------------------------------------------

    def _gizmo_eye(self):
        cy, sy = np.cos(self.gizmo_yaw), np.sin(self.gizmo_yaw)
        cp, sp = np.cos(self.gizmo_pitch), np.sin(self.gizmo_pitch)
        x = self.gizmo_radius * sy * cp
        y = self.gizmo_radius * cy * cp
        z = self.gizmo_radius * sp
        return [x, y, z]

    def _gizmo_viewport(self):
        W, H = self.wnd.width, self.wnd.height
        giz_px = int(min(W, H) * 0.28)
        pad = 12
        gx0 = W - giz_px - pad
        gy0 = H - giz_px - pad
        return gx0, gy0, giz_px

    def _build_curved_preview_mesh(self, c, u, v, n, s, kind, amp, radius, grid=22):
        """Build a curved sheet preview for the top-right corner viewer.

        The preview uses the same curved-plane math as the slice shader so the
        little 3D corner view matches the actual curved sampling surface.
        Returns triangle vertices and line vertices in gizmo-space coordinates.
        """
        r = max(abs(float(radius)), 1e-4)
        kind = int(kind)
        grid = max(4, int(grid))

        def height(sx, sy):
            qx = float(sx) / r
            qy = float(sy) / r
            if kind == 0:
                return qx * qx + qy * qy
            if kind == 1:
                return qx * qx - qy * qy
            if kind == 2:
                return qx * qx
            if kind == 3:
                return qy * qy
            return (qx * qx + qy * qy) + 0.22 * np.sin(2.0 * np.pi * qx) * np.cos(2.0 * np.pi * qy)

        us = np.linspace(-1.0, 1.0, grid + 1, dtype=np.float32)
        vs = np.linspace(-1.0, 1.0, grid + 1, dtype=np.float32)
        pts = np.zeros((grid + 1, grid + 1, 3), dtype=np.float32)

        for iy, sy in enumerate(vs):
            for ix, sx in enumerate(us):
                p = c + u * (sx * s) + v * (sy * s)
                p = p + n * (float(amp) * height(sx, sy))
                pts[iy, ix, :] = p - 0.5

        tris = []
        lines = []
        for iy in range(grid):
            for ix in range(grid):
                p00 = pts[iy, ix]
                p10 = pts[iy, ix + 1]
                p01 = pts[iy + 1, ix]
                p11 = pts[iy + 1, ix + 1]
                tris += [p00, p10, p01, p01, p10, p11]

        for iy in range(grid + 1):
            for ix in range(grid):
                lines += [pts[iy, ix], pts[iy, ix + 1]]
        for ix in range(grid + 1):
            for iy in range(grid):
                lines += [pts[iy, ix], pts[iy + 1, ix]]

        tri_arr = np.array(tris, dtype=np.float32) if tris else np.zeros((0, 3), dtype=np.float32)
        line_arr = np.array(lines, dtype=np.float32) if lines else np.zeros((0, 3), dtype=np.float32)
        return tri_arr, line_arr

    def _update_gizmo_geometry(self):
        # plane corners in gizmo space (volume coords minus 0.5)
        c = self.center
        u = self.u
        v = self.v
        n = self.n
        s = self.scale

        curved_preview = (
            getattr(self, "view_mode", "single") == "curved_plane_editor"
            and bool(getattr(self, "curved_plane_enable", False))
        )

        if curved_preview:
            curve_kind = int(getattr(self, "curved_plane_kind", 0))
            curve_amp = float(getattr(self, "curved_plane_amp", 0.075))
            curve_radius = float(getattr(self, "curved_plane_radius", 1.0))
            plane, wire = self._build_curved_preview_mesh(c, u, v, n, s, curve_kind, curve_amp, curve_radius, grid=22)
            self.plane_vbo.orphan(size=max(int(plane.nbytes), 12))
            if plane.size:
                self.plane_vbo.write(plane.astype(np.float32).tobytes())

            self.curve_wire_vertex_count = int(len(wire))
            self.curve_wire_vbo.orphan(size=max(int(wire.nbytes), 12))
            if wire.size:
                self.curve_wire_vbo.write(wire.astype(np.float32).tobytes())
        else:
            p00 = (c - u*s - v*s) - 0.5
            p10 = (c + u*s - v*s) - 0.5
            p01 = (c - u*s + v*s) - 0.5
            p11 = (c + u*s + v*s) - 0.5

            # extrude to a slab so it reads as a 3D object
            t = 0.02
            off = n * (t * 0.5)
            a00, a10, a01, a11 = p00+off, p10+off, p01+off, p11+off
            b00, b10, b01, b11 = p00-off, p10-off, p01-off, p11-off

            tris = []
            # top
            tris += [a00,a10,a01,  a01,a10,a11]
            # bottom (reverse)
            tris += [b00,b01,b10,  b01,b11,b10]
            # sides
            tris += [a00,b00,a10,  a10,b00,b10]
            tris += [a10,b10,a11,  a11,b10,b11]
            tris += [a11,b11,a01,  a01,b11,b01]
            tris += [a01,b01,a00,  a00,b01,b00]

            plane = np.array(tris, dtype=np.float32)
            self.plane_vbo.orphan(size=max(int(plane.nbytes), 12))
            self.plane_vbo.write(plane.tobytes())
            self.curve_wire_vertex_count = 0
            self.curve_wire_vbo.orphan(size=12)

        # normal arrow
        start = (c - 0.5)
        end   = (c + self.n * 0.35 - 0.5)
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())


    def _upload_volume_texture_array(self, volume_rgb: Optional[np.ndarray], label: str = "vol"):
        if volume_rgb is None:
            return None

        # Upload grayscale auxiliary volumes as one-channel texture arrays and
        # use texture swizzle so the shader still receives vec3(gray). This avoids
        # allocating/uploading huge repeated RGB skeleton/gradient volumes.
        is_gray = (volume_rgb.ndim == 3) or (volume_rgb.ndim == 4 and volume_rgb.shape[-1] == 1)
        components = 1 if is_gray else 3
        tex = self.ctx.texture_array((self.W, self.H, self.Z), components=components, dtype="f1")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        if is_gray:
            try:
                tex.swizzle = "RRR1"
            except Exception:
                pass

        for z in range(self.Z):
            layer = volume_rgb[z]
            if is_gray:
                if layer.ndim == 3:
                    layer = layer[..., 0]
                slab = np.ascontiguousarray(layer.astype(np.uint8, copy=False))
            else:
                slab = np.ascontiguousarray(layer[..., :3].astype(np.uint8, copy=False))
            tex.write(slab.tobytes(), viewport=(0, 0, z, self.W, self.H, 1))
            if label == "main" and (z % 25 == 0 or z == self.Z - 1):
                print(f"  uploaded {label} {z+1}/{self.Z}")
        return tex

    def _volume_key_to_assets(self, key: str):
        key = str(key)
        if self.aux_from_main and key in ("gradient", "skeleton"):
            return self.V, self.tex_main, int(self.bgr_input)
        if key == "gradient":
            return self.V_gradient if self.V_gradient is not None else self.V, self.tex_gradient if self.tex_gradient is not None else self.tex_main, 0
        if key == "skeleton":
            return self.V_skeleton if self.V_skeleton is not None else self.V, self.tex_skeleton if self.tex_skeleton is not None else self.tex_main, 0
        return self.V, self.tex_main, int(self.bgr_input)

    def _make_spec(self, center, u, v, n, scale_u, scale_v, *, aspect_correct=1, black_transparent=0, alpha=1.0, name=""):
        return {
            "name": name,
            "center": np.asarray(center, np.float32).copy(),
            "u": np.asarray(u, np.float32).copy(),
            "v": np.asarray(v, np.float32).copy(),
            "n": np.asarray(n, np.float32).copy(),
            "scale_u": float(scale_u),
            "scale_v": float(scale_v),
            "aspect_correct": int(aspect_correct),
            "black_transparent": int(black_transparent),
            "alpha": float(alpha),
        }

    def _multi_volume_specs(self):
        base = self._single_view_spec()
        if self.aux_from_main:
            return [
                ("main", {**base, "name": "Color volume"}),
                ("gradient", {**base, "name": "Main as gradient"}),
                ("skeleton", {**base, "name": "Main as skeleton/invert"}),
            ]
        return [
            ("main", {**base, "name": "Color volume"}),
            ("gradient", {**base, "name": "Gradient distance"}),
            ("skeleton", {**base, "name": "Skeleton"}),
        ]

    def _sample_rgb_for_spec(self, volume_key: str, spec: Dict[str, Any], out_w: int = 192, out_h: int = 128) -> np.ndarray:
        vol, _, bgr_flag = self._volume_key_to_assets(volume_key)
        if int(spec.get("curved_enable", 0)) != 0:
            rgb = sample_curved_plane_from_volume_rgb(
                vol,
                spec["center"], spec["u"], spec["v"], spec.get("n", self.n),
                spec["scale_u"], spec["scale_v"],
                curved_kind=int(spec.get("curved_kind", 0)),
                curved_amp=float(spec.get("curved_amp", 0.075)),
                curved_radius=float(spec.get("curved_radius", 1.0)),
                out_w=out_w,
                out_h=out_h,
            )
        else:
            rgb = sample_plane_from_volume_rgb(vol, spec["center"], spec["u"], spec["v"], spec["scale_u"], spec["scale_v"], out_w=out_w, out_h=out_h)
        return rgb[..., ::-1].copy() if bgr_flag else rgb.copy()

    def _update_interest_if_needed(self) -> None:
        if not bool(getattr(self, "interest_recommend_live_enabled", False)):
            self.current_interest = {"paused": True, "reason": "live interest off; press Suggest/F3 for one-shot"}
            return
        now = time.perf_counter()
        if (now - self._last_interest_time) < self.interest_interval and self.current_interest:
            return
        self._last_interest_time = now
        try:
            self.current_interest = self.compute_interest_recommendation(apply_recommendation=False)
        except Exception as exc:
            self.current_interest = {"error": str(exc)}

    def compute_interest_recommendation(self, apply_recommendation: bool = False) -> Dict[str, Any]:
        if self.V_gradient is None or self.V_skeleton is None:
            data = {"unavailable": True, "reason": "load gradient_distance and skeleton volumes to enable interest suggestion"}
            self.last_interest_recommendation = data
            return data

        ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        candidates = []

        # Current plane.
        candidates.append(("current", self._single_view_spec(), self.center.copy(), self.n.copy()))

        # Sample horizontal/vertical families around the volume to find a strong intersection.
        samples = np.linspace(0.10, 0.90, 9, dtype=np.float32)
        for z in samples:
            c = np.array([0.5, 0.5, float(z)], dtype=np.float32)
            candidates.append((f"horizontal z={z:.2f}", self._make_spec(c, ex, ey, ez, 0.5, 0.5, aspect_correct=0), c, ez))
        for x in samples:
            c = np.array([float(x), 0.5, 0.5], dtype=np.float32)
            candidates.append((f"vertical yz x={x:.2f}", self._make_spec(c, ey, ez, ex, 0.5, 0.5, aspect_correct=0), c, ex))
        for y in samples:
            c = np.array([0.5, float(y), 0.5], dtype=np.float32)
            candidates.append((f"vertical xz y={y:.2f}", self._make_spec(c, ex, ez, ey, 0.5, 0.5, aspect_correct=0), c, ey))

        scored = []
        for name, spec, center, normal in candidates:
            grad_rgb = self._sample_rgb_for_spec("gradient", spec, out_w=128, out_h=96)
            skel_rgb = self._sample_rgb_for_spec("skeleton", spec, out_w=128, out_h=96)
            m = compute_interest_metrics(grad_rgb, skel_rgb)
            m.update({"name": name, "spec": spec, "center": center, "normal": normal})
            scored.append(m)
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0] if scored else {"unavailable": True}
        if best and "spec" in best:
            yaw, pitch = normal_to_yaw_pitch(best["normal"])
            best["yaw"] = yaw; best["pitch"] = pitch
            best["suggested_view_mode"] = "multi_volume"
            best["reason"] = (
                f"grad_mean={best['gradient_mean']:.3f}, grad_fill={best['gradient_fill']:.3f}, "
                f"skeleton_fill={best['skeleton_fill']:.3f}, overlap={best['overlap_ratio']:.3f}"
            )
            if apply_recommendation:
                self.center[:] = np.asarray(best["center"], dtype=np.float32)
                self.yaw = float(yaw)
                self.pitch = float(pitch)
                self._update_plane_axes()
                self._push_slice_uniforms()
                self._update_gizmo_geometry()
                self.view_mode = "multi_volume"
                self.force_clear_next_frame = True
        self.last_interest_recommendation = best
        return best

    def apply_interest_recommendation(self) -> None:
        best = self.compute_interest_recommendation(apply_recommendation=True)
        if best.get("unavailable"):
            print(f"[interest] unavailable: {best.get('reason','')}")
        else:
            self.record_camera_waypoint()
            print(f"[interest] applied {best.get('name')} score={best.get('score',0.0):.3f} {best.get('reason','')} -> recorded new camera waypoint")
        self.ui_force_rebuild = True

    def compute_blob_dense_uninteresting_recommendation(self, apply_recommendation: bool = False) -> Dict[str, Any]:
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        candidates = [("current", self._single_view_spec(), self.center.copy(), self.n.copy())]
        samples = np.linspace(0.10, 0.90, 9, dtype=np.float32)
        for z in samples:
            c = np.array([0.5, 0.5, float(z)], dtype=np.float32)
            candidates.append((f"horizontal z={z:.2f}", self._make_spec(c, ex, ey, ez, 0.5, 0.5, aspect_correct=0), c, ez))
        for x in samples:
            c = np.array([float(x), 0.5, 0.5], dtype=np.float32)
            candidates.append((f"vertical yz x={x:.2f}", self._make_spec(c, ey, ez, ex, 0.5, 0.5, aspect_correct=0), c, ex))
        for y in samples:
            c = np.array([0.5, float(y), 0.5], dtype=np.float32)
            candidates.append((f"vertical xz y={y:.2f}", self._make_spec(c, ex, ez, ey, 0.5, 0.5, aspect_correct=0), c, ey))

        scored = []
        for name, spec, center, normal in candidates:
            main_rgb = self._sample_rgb_for_spec("main", spec, out_w=128, out_h=96)
            h = analyze_slice_heuristics(main_rgb)
            interest_score = 0.0
            if self.V_gradient is not None and self.V_skeleton is not None:
                grad_rgb = self._sample_rgb_for_spec("gradient", spec, out_w=128, out_h=96)
                skel_rgb = self._sample_rgb_for_spec("skeleton", spec, out_w=128, out_h=96)
                interest_score = float(compute_interest_metrics(grad_rgb, skel_rgb).get("score", 0.0))
            blob_norm = min(1.0, float(h.get("blob_count", 0)) / 12.0)
            fill_bonus = min(1.0, float(h.get("fill_ratio", 0.0)) / 0.25)
            anti_interest = 1.0 - max(0.0, min(1.0, interest_score))
            score = 0.55 * blob_norm + 0.20 * fill_bonus + 0.25 * anti_interest
            scored.append({
                "name": name, "spec": spec, "center": center, "normal": normal,
                "blob_count": int(h.get("blob_count", 0)),
                "fill_ratio": float(h.get("fill_ratio", 0.0)),
                "interest_score": interest_score,
                "blobseek_score": float(score),
            })
        scored.sort(key=lambda x: x["blobseek_score"], reverse=True)
        best = scored[0] if scored else {"unavailable": True}
        if best and "spec" in best:
            yaw, pitch = normal_to_yaw_pitch(best["normal"])
            best["yaw"] = yaw; best["pitch"] = pitch
            best["suggested_view_mode"] = "single"
            best["reason"] = f"blob_count={best['blob_count']} fill={best['fill_ratio']:.3f} interest={best['interest_score']:.3f}"
            if apply_recommendation:
                self.center[:] = np.asarray(best["center"], dtype=np.float32)
                self.yaw = float(yaw)
                self.pitch = float(pitch)
                self._update_plane_axes()
                self._push_slice_uniforms()
                self._update_gizmo_geometry()
                self.view_mode = "single"
                self.force_clear_next_frame = True
        self.last_blob_dense_recommendation = best
        return best

    def apply_blob_dense_uninteresting_recommendation(self) -> None:
        best = self.compute_blob_dense_uninteresting_recommendation(apply_recommendation=True)
        if best.get("unavailable"):
            print(f"[blobseek] unavailable: {best.get('reason','')}")
        else:
            self.record_camera_waypoint()
            print(f"[blobseek] applied {best.get('name')} score={best.get('blobseek_score',0.0):.3f} {best.get('reason','')} -> recorded new camera waypoint")
        self.ui_force_rebuild = True

    def _start_capture24(self) -> None:
        if self.capture24_active:
            return
        self.capture24_session_index += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path("out") / "capture_sessions" / f"session_{stamp}_{self.capture24_session_index:03d}"
        raw = root / "raw"
        sorted_dir = root / "sorted_by_area"
        aligned_dir = root / "aligned_by_shape"
        for p in (raw, sorted_dir, aligned_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.capture24_root = root
        self.capture24_raw_dir = raw
        self.capture24_sorted_dir = sorted_dir
        self.capture24_aligned_dir = aligned_dir
        self.capture24_meta_path = root / "frame_metrics.json"
        self.capture24_saved = []
        self.capture24_accum = 0.0
        self.capture24_active = True
        print(f"[capture24] started -> {root}")

    def _capture_frame_to_session(self, image: Optional[Image.Image] = None) -> Optional[Path]:
        if not self.capture24_active or self.capture24_raw_dir is None:
            return None
        img = image if image is not None else self._capture_scene_image()
        stem = f"frame_{len(self.capture24_saved):06d}"
        outputs = self._capture_outputs_from_image(img, self.capture24_raw_dir, stem)
        out = outputs[0] if outputs else None
        if out is not None:
            self.capture24_saved.append(out)
        return out

    def _stop_capture24(self) -> None:
        if not self.capture24_active:
            return
        self.capture24_active = False
        print(f"[capture24] stopped ({len(self.capture24_saved)} frames). post-processing...")
        try:
            self._postprocess_capture24()
        except Exception as exc:
            print(f"[capture24] post-process failed: {exc}")

    def toggle_capture24(self) -> None:
        if self.capture24_active:
            self._stop_capture24()
        else:
            self._start_capture24()
        self.ui_force_rebuild = True

    def _foreground_mask_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
        return gray > max(8.0, float(np.percentile(gray, 35)) * 0.55)

    def _build_thickest_stack_art(self, ordered_paths: List[Path], out_path: Path) -> Optional[Path]:
        strips = []
        thicknesses = []
        for p in ordered_paths:
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            mask = self._foreground_mask_from_rgb(rgb)
            if not np.any(mask):
                continue
            row_counts = mask.sum(axis=1)
            y = int(np.argmax(row_counts))
            xs = np.where(mask[y])[0]
            if xs.size == 0:
                continue
            strip = rgb[y:y+1, xs[0]:xs[-1]+1, :]
            strips.append(strip)
            thicknesses.append(strip.shape[1])
        if not strips:
            return None
        max_w = max(s.shape[1] for s in strips)
        strip_h = 6
        canvas = np.zeros((len(strips) * strip_h, max_w, 3), dtype=np.uint8)
        for i, s in enumerate(strips):
            line = np.asarray(Image.fromarray(s[0], mode="RGB").resize((max_w, strip_h), Image.NEAREST), dtype=np.uint8)
            canvas[i*strip_h:(i+1)*strip_h, :, :] = line
        Image.fromarray(canvas, mode="RGB").save(out_path)
        return out_path

    def _build_packed_equal_art(self, ordered_paths: List[Path], out_path: Path) -> Optional[Path]:
        crops = []
        max_w = max_h = 0
        for p in ordered_paths:
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            mask = self._foreground_mask_from_rgb(rgb)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            crop = rgb[ys.min():ys.max()+1, xs.min():xs.max()+1, :]
            crops.append(crop)
            max_h = max(max_h, crop.shape[0]); max_w = max(max_w, crop.shape[1])
        if not crops:
            return None
        cols = max(1, int(np.ceil(np.sqrt(len(crops)))))
        rows = int(np.ceil(len(crops) / cols))
        cell_w, cell_h = max_w + 8, max_h + 8
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for idx, crop in enumerate(crops):
            r, c = divmod(idx, cols)
            y0 = r * cell_h + (cell_h - crop.shape[0]) // 2
            x0 = c * cell_w + (cell_w - crop.shape[1]) // 2
            canvas[y0:y0+crop.shape[0], x0:x0+crop.shape[1], :] = crop
        Image.fromarray(canvas, mode="RGB").save(out_path)
        return out_path

    def _build_packed_close_art(self, ordered_paths: List[Path], out_path: Path) -> Optional[Path]:
        items = []
        for p in ordered_paths:
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            mask = self._foreground_mask_from_rgb(rgb)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            crop = rgb[ys.min():ys.max()+1, xs.min():xs.max()+1, :]
            items.append(crop)
        if not items:
            return None
        items.sort(key=lambda a: a.shape[0], reverse=True)
        max_width = int(sum(a.shape[1] for a in items) ** 0.7) + 32
        shelves = []
        placements = []
        for crop in items:
            h, w = crop.shape[:2]
            placed = False
            for si, (sx, sy, sh, used) in enumerate(shelves):
                if used + w + 4 <= max_width and h <= sh:
                    placements.append((crop, used, sy))
                    shelves[si] = (sx, sy, sh, used + w + 4)
                    placed = True
                    break
            if not placed:
                y = 0 if not shelves else shelves[-1][1] + shelves[-1][2] + 4
                shelves.append((0, y, h, w + 4))
                placements.append((crop, 0, y))
        total_h = (shelves[-1][1] + shelves[-1][2]) if shelves else 1
        canvas = np.zeros((total_h, max_width, 3), dtype=np.uint8)
        for crop, x0, y0 in placements:
            h, w = crop.shape[:2]
            canvas[y0:y0+h, x0:x0+w, :] = crop
        # trim right edge
        used_w = max((x0 + crop.shape[1] for crop, x0, y0 in placements), default=1)
        canvas = canvas[:, :used_w, :]
        Image.fromarray(canvas, mode="RGB").save(out_path)
        return out_path

    def _postprocess_capture24_manual(self, mode: str) -> None:
        if not self.capture24_saved or self.capture24_root is None:
            print("[capture24] no captured frames available for manual postprocess")
            return
        ordered = list(self.capture24_saved)
        art_dir = self.capture24_root / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        if mode == "thick":
            p = self._build_thickest_stack_art(ordered, art_dir / "thickest_stack_manual.png")
            print(f"[capture24] wrote {p}")
        elif mode == "equal":
            p = self._build_packed_equal_art(ordered, art_dir / "packed_equal_manual.png")
            print(f"[capture24] wrote {p}")
        elif mode == "close":
            p = self._build_packed_close_art(ordered, art_dir / "packed_close_manual.png")
            print(f"[capture24] wrote {p}")

    def _postprocess_capture24(self) -> None:
        if not self.capture24_saved or self.capture24_root is None:
            return
        metrics = []
        aligned = []
        for p in self.capture24_saved:
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            h = analyze_slice_heuristics(rgb)
            desc = frame_descriptor(rgb)
            centered = centered_image_from_mask(rgb)
            aligned.append((p, centered, desc, h))
            metrics.append({
                "file": p.name,
                "fill_ratio": float(h.get("fill_ratio", 0.0)),
                "blob_count": int(h.get("blob_count", 0)),
                "largest_blob_area": int(h.get("largest_blob_area", 0)),
                "circle_count": int(h.get("circle_count", 0)),
                "bone_ratio": float(h.get("bone_ratio", 0.0)),
                "flesh_ratio": float(h.get("flesh_ratio", 0.0)),
            })

        # Sort most area -> least area.
        metrics_sorted = sorted(metrics, key=lambda x: x["fill_ratio"], reverse=True)
        raw_lookup = {m[0].name: m for m in aligned}
        for rank, item in enumerate(metrics_sorted, 1):
            src = self.capture24_raw_dir / item["file"]
            dst = self.capture24_sorted_dir / f"{rank:04d}_{item['file']}"
            Image.open(src).save(dst)

        # Match by connected-shape similarity using greedy nearest-neighbor on descriptors.
        remaining = aligned[:]
        order = []
        if remaining:
            order.append(remaining.pop(0))
            while remaining:
                last_desc = order[-1][2]
                best_i = min(range(len(remaining)), key=lambda i: float(np.linalg.norm(remaining[i][2] - last_desc)))
                order.append(remaining.pop(best_i))
        matched_names = []
        for rank, (src, centered, desc, h) in enumerate(order, 1):
            name = f"{rank:04d}_{src.name}"
            Image.fromarray(centered, mode="RGB").save(self.capture24_aligned_dir / name)
            matched_names.append(name)

        art_dir = self.capture24_root / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        ordered_paths = [self.capture24_raw_dir / item["file"] for item in metrics_sorted if self.capture24_raw_dir is not None]
        thick_path = self._build_thickest_stack_art(ordered_paths, art_dir / "thickest_stack.png")
        equal_path = self._build_packed_equal_art(ordered_paths, art_dir / "packed_equal.png")
        close_path = self._build_packed_close_art(ordered_paths, art_dir / "packed_close.png")

        payload = {
            "fps": self.capture24_fps,
            "frame_count": len(self.capture24_saved),
            "sorted_by_area": metrics_sorted,
            "matched_aligned_sequence": matched_names,
            "artifacts": {
                "thickest_stack": str(thick_path) if thick_path is not None else None,
                "packed_equal": str(equal_path) if equal_path is not None else None,
                "packed_close": str(close_path) if close_path is not None else None,
            },
        }
        if self.capture24_meta_path is not None:
            self.capture24_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[capture24] wrote sorted frames to {self.capture24_sorted_dir}")
        print(f"[capture24] wrote aligned frames to {self.capture24_aligned_dir}")
        print(f"[capture24] wrote packing/thickness artifacts to {art_dir}")

    # ------------------------------------------------------------
    # Waypoint / playback helpers
    # ------------------------------------------------------------

    def _elapsed_record_time(self) -> float:
        return float(time.perf_counter() - self.record_start_time)

    def _make_camera_state(self) -> CameraState:
        return CameraState(
            t=self._elapsed_record_time(),
            position=[float(x) for x in self.center],
            euler_deg=[float(math.degrees(self.pitch)), float(math.degrees(self.yaw)), 0.0],
            plane_normal=[float(x) for x in self.n],
            yaw=float(self.yaw),
            pitch=float(self.pitch),
            scale=float(self.scale),
            view_mode=str(self.view_mode),
        )

    def _make_brush_state(self) -> BrushState:
        return BrushState(
            t=self._elapsed_record_time(),
            mouse_uv=[float(self.mouse_uv[0]), float(self.mouse_uv[1])],
            strength=float(self.heap_depth),
            radius=float(self.heap_radius),
            softness=float(self.heap_softness),
            stretch=float(self.heap_stretch),
            direction=float(self.heap_dir),
            enabled=bool(self.heap_enable),
        )

    def _timeline_settings(self) -> Dict[str, Any]:
        return {
            "seconds_per_segment": float(self.seconds_per_segment_live),
            "interpolation": str(self.interpolation_mode_live),
            "noise_type": str(self.noise_type_live),
            "noise_amp": float(self.noise_amp_live),
            "noise_freq": float(self.noise_freq_live),
            "loop": bool(self.playback_loop_live),
            "capture_scope": str(self.capture_scope_live),
            "color_filter_mode": str(self.color_filter_mode),
            "color_filter_target": str(self.color_filter_target),
            "color_filter_strength": float(self.color_filter_strength),
            "timeline_color_marks": list(self.timeline_color_marks),
            "live_display_backend": str(self.live_display_backend),
            "scene_objects_affect_image": bool(self.scene_objects_affect_image),
            "scene3d_show_objects": bool(self.scene3d_show_objects),
            "curve_side_panel_mode": str(getattr(self, "curve_side_panel_mode", "local_curved")),
            "fast_direct_live_render": bool(getattr(self, "fast_direct_live_render", True)),
        }

    def _heuristic_storage_dir(self) -> Path:
        return Path("out") / "waypoint_heuristics"

    def _camera_state_to_spec(self, cam: CameraState) -> Dict[str, Any]:
        center = np.asarray(cam.position, dtype=np.float32)
        n = yaw_pitch_to_normal(float(cam.yaw), float(cam.pitch))
        u, v = orthonormal_basis_from_normal(n)
        return {
            "name": str(cam.view_mode),
            "center": center,
            "u": u, "v": v, "n": n,
            "scale_u": float(cam.scale),
            "scale_v": float(cam.scale),
            "aspect_correct": 1,
        }

    def _collect_waypoint_heuristics(self, cam: Optional[CameraState] = None, index_hint: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
        if cam is None:
            cam = self._make_camera_state()
        spec = self._camera_state_to_spec(cam)
        main_rgb = self._sample_rgb_for_spec("main", spec, out_w=192, out_h=128)
        heur = analyze_slice_heuristics(main_rgb)
        dbg_rgb, dbg_meta = build_blob_debug_visual(main_rgb)
        heur["blob_debug"] = dbg_meta
        if self.V_gradient is not None and self.V_skeleton is not None:
            grad_rgb = self._sample_rgb_for_spec("gradient", spec, out_w=192, out_h=128)
            skel_rgb = self._sample_rgb_for_spec("skeleton", spec, out_w=192, out_h=128)
            heur["interest"] = compute_interest_metrics(grad_rgb, skel_rgb)
        else:
            heur["interest"] = {"score": 0.0}
        idx = int(index_hint if index_hint is not None else len(self.get_camera_waypoints()) + 1)
        base_dir = self._heuristic_storage_dir() / f"wp_{idx:04d}"
        base_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        Image.fromarray(main_rgb, mode="RGB").save(base_dir / "main_slice.png")
        paths["main_slice"] = str((base_dir / "main_slice.png").as_posix())
        Image.fromarray(dbg_rgb, mode="RGB").save(base_dir / "blob_debug.png")
        paths["blob_debug"] = str((base_dir / "blob_debug.png").as_posix())
        if self.V_gradient is not None:
            grad_rgb = self._sample_rgb_for_spec("gradient", spec, out_w=192, out_h=128)
            Image.fromarray(grad_rgb, mode="RGB").save(base_dir / "gradient_slice.png")
            paths["gradient_slice"] = str((base_dir / "gradient_slice.png").as_posix())
        if self.V_skeleton is not None:
            skel_rgb = self._sample_rgb_for_spec("skeleton", spec, out_w=192, out_h=128)
            Image.fromarray(skel_rgb, mode="RGB").save(base_dir / "skeleton_slice.png")
            paths["skeleton_slice"] = str((base_dir / "skeleton_slice.png").as_posix())
        return heur, paths

    def _camera_metric(self, cam: CameraState, mode: str) -> float:
        h = cam.heuristics or {}
        if mode == "size":
            return float(h.get("fill_ratio", 0.0))
        if mode == "blobs":
            return float(h.get("blob_count", 0.0))
        if mode == "fleshbone":
            return float(h.get("flesh_ratio", 0.0) - h.get("bone_ratio", 0.0))
        if mode == "interest":
            return float(((h.get("interest", {}) or {}).get("score", 0.0)))
        if mode == "blobleast":
            return float(h.get("blob_count", 0.0)) - 4.0 * float(((h.get("interest", {}) or {}).get("score", 0.0)))
        return 0.0

    def sort_waypoints(self, mode: str) -> None:
        if self.recorder.combined_waypoints:
            self.recorder.combined_waypoints.sort(key=lambda cb: self._camera_metric(camera_state_from_dict(cb.get("camera", {})), mode), reverse=True)
        else:
            self.recorder.camera_waypoints.sort(key=lambda cam: self._camera_metric(cam, mode), reverse=True)
        self.path_dirty = True
        self.ui_force_rebuild = True
        print(f"[waypoint] sorted by {mode}")

    def export_capture_json_from_waypoints(self, source_json: Optional[Path] = None) -> Path:
        if source_json is not None:
            self.load_waypoint_json(source_json)
        cams = self.get_camera_waypoints()
        if not cams:
            raise ValueError("Need at least one waypoint to export capture JSON")
        self.capture_json_index += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path("out") / "waypoint_capture_json" / f"capture_{stamp}_{self.capture_json_index:03d}"
        img_dir = root / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for i, cam in enumerate(cams, start=1):
            spec = self._camera_state_to_spec(cam)
            rgb = self._sample_rgb_for_spec("main", spec, out_w=320, out_h=240)
            heur = analyze_slice_heuristics(rgb)
            dbg, dbg_meta = build_blob_debug_visual(rgb)
            heur["blob_debug"] = dbg_meta
            if self.V_gradient is not None and self.V_skeleton is not None:
                grad = self._sample_rgb_for_spec("gradient", spec, out_w=320, out_h=240)
                skel = self._sample_rgb_for_spec("skeleton", spec, out_w=320, out_h=240)
                heur["interest"] = compute_interest_metrics(grad, skel)
            else:
                heur["interest"] = {"score": 0.0}
            img_path = img_dir / f"wp_{i:04d}.png"
            dbg_path = img_dir / f"wp_{i:04d}_blob.png"
            Image.fromarray(rgb, mode="RGB").save(img_path)
            Image.fromarray(dbg, mode="RGB").save(dbg_path)
            entries.append({
                "index": i - 1,
                "time": float(cam.t),
                "position": list(map(float, cam.position)),
                "yaw": float(cam.yaw),
                "pitch": float(cam.pitch),
                "scale": float(cam.scale),
                "view_mode": str(cam.view_mode),
                "image": str(img_path.as_posix()),
                "blob_image": str(dbg_path.as_posix()),
                "heuristics": heur,
            })
        sorted_orders = {
            "size": [e["index"] for e in sorted(entries, key=lambda e: float(e["heuristics"].get("fill_ratio", 0.0)), reverse=True)],
            "blobs": [e["index"] for e in sorted(entries, key=lambda e: float(e["heuristics"].get("blob_count", 0.0)), reverse=True)],
            "fleshbone": [e["index"] for e in sorted(entries, key=lambda e: float(e["heuristics"].get("flesh_ratio", 0.0) - e["heuristics"].get("bone_ratio", 0.0)), reverse=True)],
            "interest": [e["index"] for e in sorted(entries, key=lambda e: float(((e["heuristics"].get("interest", {}) or {}).get("score", 0.0))), reverse=True)],
            "blobleast": [e["index"] for e in sorted(entries, key=lambda e: float(e["heuristics"].get("blob_count", 0.0)) - 4.0 * float(((e["heuristics"].get("interest", {}) or {}).get("score", 0.0))), reverse=True)],
        }
        payload = {
            "format": "mpr_waypoint_capture_timeline_v1",
            "source_json": str(source_json.as_posix()) if source_json is not None else str(self.waypoint_json_path.as_posix()),
            "entries": entries,
            "sorted_orders": sorted_orders,
        }
        out_json = root / "timeline_capture.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[waypoint] exported capture timeline json -> {out_json}")
        return out_json

    def record_camera_waypoint(self) -> None:
        cam = self._make_camera_state()
        heur, paths = self._collect_waypoint_heuristics(cam=cam, index_hint=len(self.recorder.camera_waypoints) + 1)
        cam.heuristics = heur
        cam.heuristic_images = paths
        self.recorder.on_key_c(cam)
        self.path_dirty = True
        print(f"[waypoint] camera #{len(self.recorder.camera_waypoints)} center={self.center.tolist()} yaw={self.yaw:.3f} pitch={self.pitch:.3f}")

    def record_brush_waypoint(self) -> None:
        self.recorder.on_key_b(self._make_brush_state())
        print(f"[waypoint] brush #{len(self.recorder.brush_waypoints)} uv={self.mouse_uv} depth={self.heap_depth:.3f}")

    def record_combined_waypoint(self) -> None:
        cam = self._make_camera_state()
        heur, paths = self._collect_waypoint_heuristics(cam=cam, index_hint=len(self.recorder.combined_waypoints) + 1)
        cam.heuristics = heur
        cam.heuristic_images = paths
        self.recorder.on_key_v(cam, self._make_brush_state())
        self.path_dirty = True
        print(f"[waypoint] combined #{len(self.recorder.combined_waypoints)}")

    def save_waypoints(self) -> None:
        self.recorder.save(self.waypoint_json_path, settings=self._timeline_settings())
        print(f"[waypoint] saved {self.waypoint_json_path}")

    def load_waypoint_json(self, path: Path) -> None:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        # If this is an offline sampled timeline, convert frames into camera states.
        if "timeline" in payload and "camera_waypoints" not in payload:
            self.recorder.clear()
            for fr in payload.get("timeline", []):
                yaw = float(fr.get("yaw", 0.0))
                pitch = float(fr.get("pitch", 0.0))
                cam = CameraState(
                    t=float(fr.get("frame", 0.0)),
                    position=list(map(float, fr.get("position", [0.5, 0.5, 0.5]))),
                    euler_deg=[math.degrees(pitch), math.degrees(yaw), 0.0],
                    plane_normal=list(map(float, yaw_pitch_to_normal(yaw, pitch))),
                    yaw=yaw,
                    pitch=pitch,
                    scale=float(fr.get("scale", self.scale)),
                    view_mode=str(fr.get("view_mode", self.view_mode)),
                )
                self.recorder.camera_waypoints.append(cam)
        else:
            self.recorder.load_payload(payload)
            settings = payload.get("settings", {})
            self.seconds_per_segment_live = float(settings.get("seconds_per_segment", self.seconds_per_segment_live))
            self.interpolation_mode_live = str(settings.get("interpolation", self.interpolation_mode_live))
            self.noise_type_live = str(settings.get("noise_type", self.noise_type_live))
            self.noise_amp_live = float(settings.get("noise_amp", self.noise_amp_live))
            self.noise_freq_live = float(settings.get("noise_freq", self.noise_freq_live))
            self.playback_loop_live = bool(settings.get("loop", self.playback_loop_live))
            self.capture_scope_live = str(settings.get("capture_scope", self.capture_scope_live))
            self.curved_plane_enable = bool(settings.get("curved_plane_enable", self.curved_plane_enable))
            self.curved_plane_kind = int(settings.get("curved_plane_kind", self.curved_plane_kind)) % max(1, len(self.curved_plane_kind_names))
            self.curved_plane_amp = float(settings.get("curved_plane_amp", self.curved_plane_amp))
            self.curved_plane_radius = float(settings.get("curved_plane_radius", self.curved_plane_radius))
            self.show_curve_side_panels = bool(settings.get("show_curve_side_panels", self.show_curve_side_panels))
            self.color_filter_mode = str(settings.get("color_filter_mode", self.color_filter_mode))
            self.color_filter_target = str(settings.get("color_filter_target", self.color_filter_target))
            self.color_filter_strength = float(settings.get("color_filter_strength", self.color_filter_strength))
            self.timeline_color_marks = list(settings.get("timeline_color_marks", self.timeline_color_marks))
            self.live_display_backend = str(settings.get("live_display_backend", self.live_display_backend))
            self.scene_objects_affect_image = bool(settings.get("scene_objects_affect_image", self.scene_objects_affect_image))
            self.scene3d_show_objects = bool(settings.get("scene3d_show_objects", self.scene3d_show_objects))
            self.curve_side_panel_mode = str(settings.get("curve_side_panel_mode", getattr(self, "curve_side_panel_mode", "local_curved")))
            self.fast_direct_live_render = bool(settings.get("fast_direct_live_render", getattr(self, "fast_direct_live_render", True)))
            self._arm_live_blank_check(8)

        self.path_dirty = True
        print(f"[waypoint] loaded {path}")
        print(f"           cameras={len(self.get_camera_waypoints())} brushes={len(self.get_brush_waypoints())} combined={len(self.recorder.combined_waypoints)}")

    def get_camera_waypoints(self) -> List[CameraState]:
        if self.recorder.combined_waypoints:
            return [camera_state_from_dict(x.get("camera", {})) for x in self.recorder.combined_waypoints]
        return list(self.recorder.camera_waypoints)

    def get_brush_waypoints(self) -> List[BrushState]:
        if self.recorder.combined_waypoints:
            return [brush_state_from_dict(x.get("brush", {})) for x in self.recorder.combined_waypoints]
        return list(self.recorder.brush_waypoints)

    def has_playback_path(self) -> bool:
        return len(self.get_camera_waypoints()) >= 2

    def _playback_total_seconds(self) -> float:
        cams = self.get_camera_waypoints()
        if len(cams) < 2:
            return 0.0
        return max(0.001, (len(cams) - 1) * max(0.05, float(self.seconds_per_segment_live)))

    def _playback_segment(self, t_seconds: float) -> Tuple[int, float]:
        cams = self.get_camera_waypoints()
        if len(cams) < 2:
            return 0, 0.0
        seg_dur = max(0.05, float(self.seconds_per_segment_live))
        total = self._playback_total_seconds()
        if self.playback_loop_live:
            t_seconds = t_seconds % total
        else:
            t_seconds = float(np.clip(t_seconds, 0.0, total))
        seg = min(len(cams) - 2, int(t_seconds // seg_dur))
        u = (t_seconds - seg * seg_dur) / seg_dur
        return seg, float(np.clip(u, 0.0, 1.0))

    def evaluate_playback(self, t_seconds: float) -> Optional[Dict[str, Any]]:
        cams = self.get_camera_waypoints()
        if len(cams) < 2:
            return None
        brushes = self.get_brush_waypoints()
        seg, u = self._playback_segment(t_seconds)

        pos = np.array([c.position for c in cams], dtype=np.float32)
        yps = np.array([[c.yaw, c.pitch, c.scale] for c in cams], dtype=np.float32)
        center = interpolate_points(pos, seg, u, self.interpolation_mode_live).astype(np.float32)
        yps_val = interpolate_points(yps, seg, u, self.interpolation_mode_live).astype(np.float32)

        npos, nang = camera_noise_vec(t_seconds, self.noise_type_live, self.noise_amp_live, self.noise_freq_live)
        center = np.clip(center + npos, 0.0, 1.0)
        yaw = float(yps_val[0] + nang[0])
        pitch = float(np.clip(yps_val[1] + nang[1], -1.55, 1.55))
        scale = float(np.clip(yps_val[2], 0.05, 2.0))

        result: Dict[str, Any] = {"center": center, "yaw": yaw, "pitch": pitch, "scale": scale}

        if len(brushes) >= 2:
            bvals = np.array([[b.mouse_uv[0], b.mouse_uv[1], b.strength, b.radius, b.softness, b.stretch, b.direction] for b in brushes], dtype=np.float32)
            bseg = min(seg, len(brushes) - 2)
            b = interpolate_points(bvals, bseg, u, self.interpolation_mode_live).astype(np.float32)
            result["brush"] = b
        elif len(brushes) == 1:
            b = brushes[0]
            result["brush"] = np.array([b.mouse_uv[0], b.mouse_uv[1], b.strength, b.radius, b.softness, b.stretch, b.direction], dtype=np.float32)

        return result

    def apply_playback_state(self, state: Dict[str, Any]) -> None:
        self.center[:] = np.clip(state["center"], 0.0, 1.0)
        self.yaw = float(state["yaw"])
        self.pitch = float(np.clip(state["pitch"], -1.55, 1.55))
        self.scale = float(np.clip(state["scale"], 0.05, 2.0))
        self._update_plane_axes()

        if "brush" in state:
            b = state["brush"]
            self.mouse_uv = (float(np.clip(b[0], 0.0, 1.0)), float(np.clip(b[1], 0.0, 1.0)))
            self.heap_depth = float(np.clip(b[2], 0.0, 1.0))
            self.heap_radius = float(np.clip(b[3], 0.01, 0.9))
            self.heap_softness = float(np.clip(b[4], 0.0, 0.5))
            self.heap_stretch = float(np.clip(b[5], 0.1, 10.0))
            self.heap_dir = float(-1.0 if b[6] < 0.0 else 1.0)

        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    def toggle_playback(self) -> None:
        if not self.has_playback_path():
            print("[playback] need at least two camera/combined waypoints")
            return
        self.playback_enabled = not self.playback_enabled
        if self.playback_enabled:
            self.auto_motion.enabled = False
        print(f"[playback] enabled={self.playback_enabled} interp={self.interpolation_mode_live} seconds/segment={self.seconds_per_segment_live:.2f}")

    def cycle_interpolation(self) -> None:
        modes = ["linear", "smoothstep", "catmull", "bezier", "hermite", "hamilton"]
        i = modes.index(self.interpolation_mode_live) if self.interpolation_mode_live in modes else 1
        self.interpolation_mode_live = modes[(i + 1) % len(modes)]
        self.path_dirty = True
        print(f"[timeline] interpolation={self.interpolation_mode_live}")

    def cycle_noise(self) -> None:
        modes = ["none", "perlin", "brownian", "wobble", "random"]
        i = modes.index(self.noise_type_live) if self.noise_type_live in modes else 0
        self.noise_type_live = modes[(i + 1) % len(modes)]
        print(f"[timeline] noise={self.noise_type_live} amp={self.noise_amp_live:.4f} freq={self.noise_freq_live:.2f}")

    def _scrub_playhead_from_x(self, x: float) -> None:
        total = self._playback_total_seconds()
        if total <= 0.0:
            return
        self.playhead_seconds = float(np.clip(x / max(1, self.wnd.width), 0.0, 1.0) * total)
        st = self.evaluate_playback(self.playhead_seconds)
        if st is not None:
            self.apply_playback_state(st)
            self.force_clear_next_frame = True

    def _update_path_geometry(self) -> None:
        if not self.path_dirty:
            return
        self.path_dirty = False
        cams = self.get_camera_waypoints()
        if len(cams) < 2:
            self.path_vertex_count = 0
            return

        samples_per_segment = 24
        pts = np.array([c.position for c in cams], dtype=np.float32)
        curve = []
        for i in range(len(pts) - 1):
            for u in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
                curve.append(interpolate_points(pts, i, float(u), self.interpolation_mode_live))
        curve.append(pts[-1])
        curve = np.array(curve, dtype=np.float32) - 0.5
        if curve.size == 0:
            self.path_vertex_count = 0
            return
        self.path_vbo.orphan(size=max(curve.nbytes, 12))
        self.path_vbo.write(curve.astype("f4").tobytes())
        self.path_vertex_count = len(curve)

    # ------------------------------------------------------------
    # Input: mouse
    # ------------------------------------------------------------

    def on_mouse_position_event(self, x, y, dx, dy):
        self._mark_navigation_input()
        # Store window pixel mouse position. Each split viewport converts this
        # into its own local [0,1] brush UV before rendering.
        self.mouse_px = (float(x), float(y))
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))

    def on_mouse_press_event(self, x, y, button):
        self._mark_navigation_input()
        if self.ui_visible and self._handle_ui_click(x, y):
            return
        if self.view_mode == "slice_seed_board":
            return
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if button == LEFT:
            if self.path_scrub_mode and self.has_playback_path():
                self._drag_path_scrub = True
                self._scrub_playhead_from_x(x)
            else:
                self._drag_plane = True
        if button == MIDDLE:
            self._drag_pan = True

    def on_mouse_release_event(self, x, y, button):
        self._mark_navigation_input()
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle
        if button == LEFT:
            self._drag_plane = False
            self._drag_path_scrub = False
            self._drag_ui_scrub = False
            self._drag_ui_fx_slider = False
            self._drag_ui_blob_slider = False
            self._drag_ui_cut_angle = False
            self._drag_ui_fx_param1 = False
            self._drag_ui_fx_param2 = False
            self._drag_ui_curve_amp = False
            self._drag_ui_hemo_oxy = False
            self._drag_ui_hemo_deoxy = False
            self._drag_ui_hemo_fresh = False
            self._drag_ui_hemo_sg = False
        self._drag_ui_curve_amp = False
        if button == MIDDLE:
            self._drag_pan = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        self._mark_navigation_input()
        if self._drag_ui_scrub:
            self._set_playhead_from_ui_x(x)
            return
        if self._drag_ui_fx_slider:
            self._set_fx_strength_from_ui_x(x)
            return
        if self._drag_ui_blob_slider:
            self._set_blob_pack_from_ui_x(x)
            return
        if self._drag_ui_cut_angle:
            self._set_cut_angle_from_ui_x(x)
            return
        if self._drag_ui_fx_param1:
            self._set_fx_param_from_ui_x(1, x)
            return
        if self._drag_ui_fx_param2:
            self._set_fx_param_from_ui_x(2, x)
            return
        if self._drag_ui_curve_amp:
            self._set_curve_amp_from_ui_x(x)
            return
        if self._drag_ui_hemo_oxy:
            self._set_hemo_threshold_from_ui_x("oxy", x)
            return
        if self._drag_ui_hemo_deoxy:
            self._set_hemo_threshold_from_ui_x("deoxy", x)
            return
        if self._drag_ui_hemo_fresh:
            self._set_hemo_threshold_from_ui_x("fresh", x)
            return
        if self._drag_ui_hemo_sg:
            self._set_hemo_threshold_from_ui_x("savgol", x)
            return
        if self._drag_path_scrub:
            self._scrub_playhead_from_x(x)
            return

        if self.view_mode == "slice_seed_board":
            return

        # rotate plane
        if self._drag_plane:
            self.yaw   += dx * 0.005
            self.pitch += -dy * 0.005
            self.pitch = float(np.clip(self.pitch, -1.55, 1.55))
            self._update_plane_axes()
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

        # pan plane center
        if self._drag_pan:
            W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)
            aspect = W / max(H, 1)

            du = (dx / W) * (2.0 * self.scale) * aspect
            dv = (-dy / H) * (2.0 * self.scale)

            self.center += self.u * du + self.v * dv
            self.center[:] = np.clip(self.center, 0.0, 1.0)

            self._push_slice_uniforms()
            self._update_gizmo_geometry()

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self._mark_navigation_input()
        mx, my = getattr(self, 'mouse_px', (None, None))
        if self.fx_mode_dropdown_open and self.ui_fx_dropdown_rect is not None and mx is not None:
            x0, y0, x1, y1 = self.ui_fx_dropdown_rect
            if x0 <= mx <= x1 and y0 <= my <= y1:
                max_scroll = max(0, len(self._fx_mode_list()) - 10)
                self.fx_dropdown_scroll = int(np.clip(self.fx_dropdown_scroll - int(np.sign(y_offset)), 0, max_scroll))
                self.ui_force_rebuild = True
                return
        if self.view_mode == "slice_seed_board":
            return
        self.scale *= float(0.92 ** y_offset)
        self.scale = float(np.clip(self.scale, 0.05, 2.0))
        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    # ------------------------------------------------------------
    # Input: keys (continuous while holding + heap controls)
    # ------------------------------------------------------------

    def _adjust_fx_param_by_delta(self, which: int, delta: float) -> None:
        vals = list(self.fx_param_values.get(self.frame_transform_mode, [0.5, 0.5]))
        idx = 0 if which == 1 else 1
        vals[idx] = float(np.clip(vals[idx] + float(delta), 0.0, 1.0))
        self.fx_param_values[self.frame_transform_mode] = vals
        self.ui_force_rebuild = True
        self._frame_fx_cache_key = None

    def _apply_held_frame_fx_key(self, key, dt: float, is_shift: bool) -> bool:
        k = self.wnd.keys
        step = float(dt) * (1.8 if is_shift else 1.0)
        if key == k.UP:
            self.frame_transform_strength = float(np.clip(self.frame_transform_strength + 0.85 * step, 0.05, 1.50))
        elif key == k.DOWN:
            self.frame_transform_strength = float(np.clip(self.frame_transform_strength - 0.85 * step, 0.05, 1.50))
        elif key == k.LEFT or key == k.RIGHT:
            sign = -1.0 if key == k.LEFT else 1.0
            if self.frame_transform_mode == "cuts":
                if is_shift:
                    self.cut_offset_parallel = float(np.clip(self.cut_offset_parallel + sign * 0.22 * dt, 0.0, 0.5))
                else:
                    self.cut_angle_rad = float(np.clip(self.cut_angle_rad + sign * 1.65 * dt, -math.pi, math.pi))
            else:
                # Param 1 by default, Param 2 while Shift is held. If the FX has
                # no dedicated params, this still stores a value and will apply
                # when the user switches into a parametric mode.
                self._adjust_fx_param_by_delta(2 if is_shift else 1, sign * 0.75 * dt)
        else:
            return False
        self.ui_force_rebuild = True
        self._frame_fx_cache_key = None
        return True

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys
        if action in (k.ACTION_PRESS, k.ACTION_RELEASE):
            self._mark_navigation_input()
        if action == k.ACTION_PRESS and key == k.ESCAPE:
            self.wnd.close()
            return

        # one-shot toggles + heap params
        if action == k.ACTION_PRESS:
            def is_key(name: str) -> bool:
                return hasattr(k, name) and key == getattr(k, name)

            if key == k.P:
                # Save the full framebuffer after this frame renders.
                self.pending_screen_save = True
                return

            if hasattr(k, "N") and key == k.N:
                self.add_seed_slice()
                return

            if is_key("C"):
                self.record_camera_waypoint()
                return

            if key == k.B:
                self.record_brush_waypoint()
                return

            if is_key("V"):
                self.record_combined_waypoint()
                return

            if is_key("F"):
                self.save_waypoints()
                return

            if is_key("SPACE"):
                self.toggle_playback()
                return

            if is_key("Z"):
                self.path_scrub_mode = not self.path_scrub_mode
                self.playback_enabled = False if self.path_scrub_mode else self.playback_enabled
                print(f"[path editor] scrub_mode={self.path_scrub_mode}")
                return

            if key == k.T:
                modes = ["single", "single_gray", "single_invert", "single_gray_invert", "axis", "local", "multi_volume", "pixel_grid", "frame_fx", "object_editor", "scene_3d", "curved_plane_editor", "slice_seed_board"]
                cur = modes.index(self.view_mode) if self.view_mode in modes else 0
                self.view_mode = modes[(cur + 1) % len(modes)]
                self.force_clear_next_frame = True
                print(f"view_mode={self.view_mode}")
                return

            if key == k.H:
                self.toggle_cursor_hidden()
                self.ui_force_rebuild = True
                return

            if hasattr(k, "Y") and key == k.Y:
                self.toggle_analysis_enabled()
                print(f"analysis_enabled={self.analysis_enabled}")
                return

            if is_key("F1"):
                if bool(getattr(self, "hide_all_overlays", False)):
                    self.hide_all_overlays = False
                    self.ui_visible = True
                else:
                    self.ui_visible = not self.ui_visible
                self.force_clear_next_frame = True
                self.ui_force_rebuild = True
                print(f"ui_visible={self.ui_visible} hide_all_overlays={self.hide_all_overlays}")
                return

            if is_key("F2"):
                self.show_gizmo = not self.show_gizmo
                self.force_clear_next_frame = True
                self.ui_force_rebuild = True
                print(f"show_gizmo={self.show_gizmo}")
                return

            if is_key("F3"):
                self.apply_interest_recommendation()
                return

            if is_key("F4"):
                self.toggle_capture24()
                return

            if is_key("F5"):
                self.blob_debug_visible = not self.blob_debug_visible
                if self.blob_debug_visible:
                    self.analysis_enabled = True
                self.ui_force_rebuild = True
                print(f"blob_debug_visible={self.blob_debug_visible} analysis_enabled={self.analysis_enabled}")
                return

            if is_key("F6"):
                self.apply_blob_dense_uninteresting_recommendation()
                return

            if is_key("F7"):
                self.cycle_color_filter_mode()
                self._push_slice_uniforms()
                return

            if is_key("F8"):
                self.cycle_color_filter_target()
                self._push_slice_uniforms()
                return

            if is_key("F9"):
                self.cycle_main_display_variant()
                self._push_slice_uniforms()
                return

            if is_key("F10"):
                self.toggle_aux_from_main()
                return

            if is_key("F11"):
                self.cycle_frame_transform_mode()
                return

            if is_key("F12") and (int(modifiers) & int(getattr(k, "MOD_SHIFT", 0))):
                self.cycle_curve_side_panel_mode()
                return

            if self.view_mode == "object_editor":
                if key == k.LEFT: self.nudge_selected_object(0, -0.02); return
                if key == k.RIGHT: self.nudge_selected_object(0, 0.02); return
                if key == k.UP: self.nudge_selected_object(1, 0.02); return
                if key == k.DOWN: self.nudge_selected_object(1, -0.02); return
                if hasattr(k, "PAGE_UP") and key == k.PAGE_UP: self.nudge_selected_object(2, 0.02); return
                if hasattr(k, "PAGE_DOWN") and key == k.PAGE_DOWN: self.nudge_selected_object(2, -0.02); return
            if key in {k.UP, k.DOWN, k.LEFT, k.RIGHT}:
                self._held_keys.add((key, bool(modifiers.shift)))
                return
            if hasattr(k, "G") and key == k.G:
                self.heap_enable = not self.heap_enable
                self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
                print(f"heap_enable={self.heap_enable}")
                return

            # Backspace/Delete used to clear local-oblique accumulation trails.
            # Trails are disabled now, so these keys are left free for normal editor use.

            if is_key("F12"):
                self.auto_motion.enabled = not self.auto_motion.enabled
                if self.auto_motion.enabled:
                    self.playback_enabled = False
                print(f"brownian_auto_motion={self.auto_motion.enabled}")
                return

            if is_key("U"):
                self.cycle_interpolation()
                return

            if is_key("O"):
                self.cycle_noise()
                return

            if is_key("EQUAL") or is_key("NUM_ADD"):
                self.seconds_per_segment_live = min(60.0, self.seconds_per_segment_live + 0.25)
                self.path_dirty = True
                print(f"[timeline] seconds_per_segment={self.seconds_per_segment_live:.2f}")
                return

            if is_key("MINUS") or is_key("NUM_SUBTRACT"):
                self.seconds_per_segment_live = max(0.10, self.seconds_per_segment_live - 0.25)
                self.path_dirty = True
                print(f"[timeline] seconds_per_segment={self.seconds_per_segment_live:.2f}")
                return

            if key == k.M:
                self.auto_motion.modulate_heap = not self.auto_motion.modulate_heap
                print(f"heap_modulation={self.auto_motion.modulate_heap}")
                return

            if key == k.X:
                self.spacemouse.enabled = not self.spacemouse.enabled
                if self.spacemouse.enabled and self.spacemouse.device is None:
                    self.spacemouse.connect()
                print(f"spacemouse_enabled={self.spacemouse.enabled}")
                return

            if key == k.N:
                self.heap_dir *= -1.0
                self.slice_prog["u_heap_dir"].value = float(self.heap_dir)
                print(f"heap_dir={self.heap_dir:+.0f}")
                return

            if key == k.Y:
                self.flip_y = 0 if self.flip_y else 1
                self.slice_prog["u_flip_y"].value = int(self.flip_y)
                print(f"flip_y={self.flip_y}")
                return

            # heap tweak keys
            if key == k.J:
                self.heap_radius = max(0.01, self.heap_radius - 0.01)
                self.slice_prog["u_radius"].value = float(self.heap_radius)
                return
            if key == k.L:
                self.heap_radius = min(0.9, self.heap_radius + 0.01)
                self.slice_prog["u_radius"].value = float(self.heap_radius)
                return

            if key == k.LEFT_BRACKET:
                self.heap_softness = max(0.0, self.heap_softness - 0.005)
                self.slice_prog["u_softness"].value = float(self.heap_softness)
                return
            if key == k.RIGHT_BRACKET:
                self.heap_softness = min(0.5, self.heap_softness + 0.005)
                self.slice_prog["u_softness"].value = float(self.heap_softness)
                return

            if key == k.I:
                self.heap_depth = min(1.0, self.heap_depth + 0.02)
                self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
                return
            if key == k.K:
                self.heap_depth = max(0.0, self.heap_depth - 0.02)
                self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
                return

            if key == k.COMMA:
                self.heap_stretch = max(0.1, self.heap_stretch - 0.1)
                self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
                return
            if key == k.PERIOD:
                self.heap_stretch = min(10.0, self.heap_stretch + 0.1)
                self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
                return

            if key == k.R:
                self.yaw = 0.0
                self.pitch = 0.0
                self.center[:] = (0.5, 0.5, 0.5)
                self.scale = 0.55 
                self.heap_enable = True
                self.heap_radius = 0.18
                self.heap_softness = 0.06
                self.heap_depth = 0.22
                self.heap_stretch = 1.0
                self.heap_dir = -1.0
                self.flip_y = 1
                self.show_gizmo = True
                self.cursor_force_hidden = False
                self._set_cursor_visible(True)
                self.playback_enabled = False
                self.path_scrub_mode = False
                self.force_clear_next_frame = True
                self.auto_motion.reset(
                    center=self.center,
                    yaw=self.yaw,
                    pitch=self.pitch,
                    heap_depth=self.heap_depth,
                    heap_radius=self.heap_radius,
                    heap_softness=self.heap_softness,
                )
                self._update_plane_axes()
                self._push_slice_uniforms()
                self._update_gizmo_geometry()
                return

        # held movement keys + continuous frame-FX sliders.
        move_keys = {k.W, k.S, k.A, k.D, k.Q, k.E, k.UP, k.DOWN, k.LEFT, k.RIGHT}
        if key in move_keys:
            if action == k.ACTION_PRESS:
                self._held_keys.add((key, bool(modifiers.shift)))
            elif action == k.ACTION_RELEASE:
                self._held_keys = {km for km in self._held_keys if km[0] != key}

    def _apply_held_keys(self, dt: float):
        if not self._held_keys:
            return
        self._mark_navigation_input()
        k = self.wnd.keys
        base = 0.22  # normalized units per second

        moved_plane = False
        for key, is_shift in list(self._held_keys):
            if key in {k.UP, k.DOWN, k.LEFT, k.RIGHT}:
                self._apply_held_frame_fx_key(key, dt, is_shift)
                continue
            step = base * dt * (3.0 if is_shift else 1.0)
            if key == k.W: self.center += self.n * step; moved_plane = True
            if key == k.S: self.center -= self.n * step; moved_plane = True
            if key == k.A: self.center -= self.u * step; moved_plane = True
            if key == k.D: self.center += self.u * step; moved_plane = True
            if key == k.Q: self.center -= self.v * step; moved_plane = True
            if key == k.E: self.center += self.v * step; moved_plane = True

        if moved_plane:
            self.center[:] = np.clip(self.center, 0.0, 1.0)
            self._push_slice_uniforms()
            self._update_gizmo_geometry()


    # ------------------------------------------------------------
    # Snapshot export
    # ------------------------------------------------------------

    def _volume_index_from_center(self):
        """Convert normalized plane center [0,1]^3 to integer volume indices."""
        x = int(np.clip(round(float(self.center[0]) * (self.W - 1)), 0, self.W - 1))
        y = int(np.clip(round(float(self.center[1]) * (self.H - 1)), 0, self.H - 1))
        z = int(np.clip(round(float(self.center[2]) * (self.Z - 1)), 0, self.Z - 1))
        return z, y, x

    def _to_pil_rgb(self, arr: np.ndarray) -> Image.Image:
        """Convert a volume slice to PIL RGB, respecting the stored BGR input."""
        a = np.asarray(arr)
        if a.ndim == 2:
            return Image.fromarray(a.astype(np.uint8), mode="L").convert("RGB")
        if a.ndim == 3 and a.shape[-1] == 3:
            # The uploaded volume is documented as BGR; convert to RGB for PIL output.
            if self.bgr_input:
                a = a[..., ::-1]
            return Image.fromarray(a.astype(np.uint8), mode="RGB")
        raise ValueError(f"Unexpected slice shape for snapshot: {a.shape}")

    def save_triplane_snapshot(self, scale: int = 1) -> Path:
        """
        Save frontal/sagittal/transverse views at the current moving plane center.

        This does not change the interactive renderer. It only samples the CPU-side
        volume using the current red plane/gizmo center, then writes a triptych PNG.
        """
        z, y, x = self._volume_index_from_center()

        # Axis convention matches the earlier offline TriPlaneViewer:
        #   frontal:   fixed z, image axes x/y       -> V[z, :, :]
        #   sagittal:  fixed x, image axes y/z       -> V[:, :, x]
        #   transverse: fixed y, image axes x/z      -> V[:, y, :]
        frontal = self.V[z, :, :]       # (H, W, 3)
        sagittal = self.V[:, :, x]      # (Z, H, 3), width is y, height is z
        transverse = self.V[:, y, :]    # (Z, W, 3), width is x, height is z

        panels = [
            self._to_pil_rgb(frontal),
            self._to_pil_rgb(sagittal),
            self._to_pil_rgb(transverse),
        ]
        labels = [
            f"Frontal  z={z}/{self.Z - 1}",
            f"Sagittal  x={x}/{self.W - 1}",
            f"Transverse  y={y}/{self.H - 1}",
        ]

        scale = max(1, int(scale))
        if scale > 1:
            panels = [im.resize((im.width * scale, im.height * scale), Image.NEAREST) for im in panels]

        canvas_w = sum(im.width for im in panels)
        canvas_h = max(im.height for im in panels)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        xoff = 0
        offsets = []
        for im, label in zip(panels, labels):
            canvas.paste(im, (xoff, 0))
            offsets.append(xoff)
            draw.text((xoff + 8, 8), label, fill=(255, 64, 64))
            xoff += im.width

        # Crosshair projections using the same coordinates that drive the moving plane center.
        sx = int(np.clip(x * scale, 0, panels[0].width - 1))
        sy = int(np.clip(y * scale, 0, panels[0].height - 1))
        sz = int(np.clip(z * scale, 0, max(panels[1].height - 1, 1)))

        # Frontal panel: x horizontal, y vertical.
        draw.line([(offsets[0] + sx, 0), (offsets[0] + sx, panels[0].height)], fill=(0, 255, 0))
        draw.line([(offsets[0], sy), (offsets[0] + panels[0].width, sy)], fill=(0, 255, 0))

        # Sagittal panel: y horizontal, z vertical.
        draw.line([(offsets[1] + sy, 0), (offsets[1] + sy, panels[1].height)], fill=(0, 255, 255))
        draw.line([(offsets[1], sz), (offsets[1] + panels[1].width, sz)], fill=(0, 255, 255))

        # Transverse panel: x horizontal, z vertical.
        draw.line([(offsets[2] + sx, 0), (offsets[2] + sx, panels[2].height)], fill=(255, 255, 0))
        draw.line([(offsets[2], sz), (offsets[2] + panels[2].width, sz)], fill=(255, 255, 0))

        # Add normalized center / plane orientation metadata in the bottom-left if there is room.
        meta = (
            f"center=({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f})  "
            f"normal=({self.n[0]:.3f}, {self.n[1]:.3f}, {self.n[2]:.3f})"
        )
        draw.text((8, max(22, canvas_h - 20)), meta, fill=(255, 255, 255))

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_count += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.snapshot_dir / f"triplane_{stamp}_{self.snapshot_count:04d}_z{z:04d}_y{y:04d}_x{x:04d}.png"
        canvas.save(out_path)
        print(f"[snapshot] saved {out_path}")
        return out_path


    # ------------------------------------------------------------
    # Split-screen slice view helpers
    # ------------------------------------------------------------

    def _single_view_spec(self):
        """Regular one-screen red-plane camera mode."""
        return {
            "name": "Single red plane / U-V",
            "center": self.center.copy(),
            "u": self.u.copy(), "v": self.v.copy(), "n": self.n.copy(),
            "scale_u": float(self.scale), "scale_v": float(self.scale),
            "aspect_correct": 1,
        }

    def _curved_view_spec(self):
        """Curved/parabolic replacement for the regular red U/V slicing plane."""
        spec = self._single_view_spec()
        kind_names = getattr(self, "curved_plane_kind_names", ["paraboloid"])
        kind = int(getattr(self, "curved_plane_kind", 0)) % max(1, len(kind_names))
        spec.update({
            "name": f"Curved plane / {kind_names[kind]}",
            "curved_enable": int(getattr(self, "curved_plane_enable", True)),
            "curved_kind": kind,
            "curved_amp": float(getattr(self, "curved_plane_amp", 0.075)),
            "curved_radius": float(getattr(self, "curved_plane_radius", 1.0)),
        })
        return spec

    def _panel_viewports(self):
        """Return three left-to-right viewport rectangles in OpenGL bottom-left coords."""
        W, H = int(self.wnd.width), int(self.wnd.height)
        gutter = 4
        panel_w = max(1, (W - 2 * gutter) // 3)
        return [
            (0, 0, panel_w, H),
            (panel_w + gutter, 0, panel_w, H),
            (2 * (panel_w + gutter), 0, max(1, W - 2 * (panel_w + gutter)), H),
        ]

    def _split_view_specs(self):
        """
        Build the three slice planes for the current screen mode.

        Axis mode uses the current red slab center to choose fixed X/Y/Z
        positions, then fills each panel with an axis-aligned MPR slice.

        Local mode uses the red slab's rotating basis directly:
          1. U/V plane: the red slice itself, normal N.
          2. U/N plane: perpendicular to the red slice, normal V.
          3. V/N plane: perpendicular to the red slice, normal -U.
        """
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if self.view_mode == "axis":
            cx, cy, cz = (float(self.center[0]), float(self.center[1]), float(self.center[2]))
            return [
                {
                    "name": "Frontal / XY, fixed Z",
                    "center": np.array([0.5, 0.5, cz], dtype=np.float32),
                    "u": ex, "v": ey, "n": ez,
                    "scale_u": 0.5, "scale_v": 0.5,
                    "aspect_correct": 0,
                },
                {
                    "name": "Sagittal / YZ, fixed X",
                    "center": np.array([cx, 0.5, 0.5], dtype=np.float32),
                    "u": ey, "v": ez, "n": ex,
                    "scale_u": 0.5, "scale_v": 0.5,
                    "aspect_correct": 0,
                },
                {
                    "name": "Transverse / XZ, fixed Y",
                    "center": np.array([0.5, cy, 0.5], dtype=np.float32),
                    "u": ex, "v": ez, "n": ey,
                    "scale_u": 0.5, "scale_v": 0.5,
                    "aspect_correct": 0,
                },
            ]

        return [
            {
                "name": "Local red plane / U-V",
                "center": self.center.copy(),
                "u": self.u.copy(), "v": self.v.copy(), "n": self.n.copy(),
                "scale_u": float(self.scale), "scale_v": float(self.scale),
                "aspect_correct": 1,
                "black_transparent": 0,
                "alpha": 1.0,
            },
            {
                "name": "Perpendicular / U-N",
                "center": self.center.copy(),
                "u": self.u.copy(), "v": self.n.copy(), "n": self.v.copy(),
                "scale_u": float(self.scale), "scale_v": float(self.scale),
                "aspect_correct": 1,
                "black_transparent": 0,
                "alpha": 1.0,
            },
            {
                "name": "Perpendicular / V-N",
                "center": self.center.copy(),
                "u": self.v.copy(), "v": self.n.copy(), "n": -self.u.copy(),
                "scale_u": float(self.scale), "scale_v": float(self.scale),
                "aspect_correct": 1,
                "black_transparent": 0,
                "alpha": 1.0,
            },
        ]

    def _mouse_uv_for_viewport(self, viewport):
        """Convert current mouse pixel position into local viewport UV."""
        vx, vy, vw, vh = viewport
        mx, my_top = self.mouse_px
        my = float(self.wnd.height) - my_top  # top-left event coords -> bottom-left GL coords
        inside = (vx <= mx < vx + vw) and (vy <= my < vy + vh)
        if not inside:
            return (0.5, 0.5), False
        u = (mx - vx) / max(1.0, float(vw))
        v = (my - vy) / max(1.0, float(vh))
        return (float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))), True

    def _render_slice_panel(self, viewport, spec, volume_key: str = "main"):
        """Render one MPR/oblique slice into a given viewport."""
        vx, vy, vw, vh = viewport
        self.ctx.viewport = (int(vx), int(vy), int(vw), int(vh))
        self.ctx.disable(moderngl.DEPTH_TEST)

        local_mouse_uv, mouse_inside = self._mouse_uv_for_viewport(viewport)

        self.slice_prog["u_center"].value = tuple(float(x) for x in spec["center"])
        self.slice_prog["u_axis_u"].value = tuple(float(x) for x in spec["u"])
        self.slice_prog["u_axis_v"].value = tuple(float(x) for x in spec["v"])
        self.slice_prog["u_axis_n"].value = tuple(float(x) for x in spec["n"])
        self.slice_prog["u_scale"].value = float(max(spec["scale_u"], spec["scale_v"]))
        self.slice_prog["u_scale_u"].value = float(spec["scale_u"])
        self.slice_prog["u_scale_v"].value = float(spec["scale_v"])
        self.slice_prog["u_aspect_correct"].value = int(spec.get("aspect_correct", 0))
        self.slice_prog["u_slice_px"].value = (float(vw), float(vh))

        # Heap brush is local to the panel currently under the cursor.
        self.slice_prog["u_heap_enable"].value = int(self.heap_enable and mouse_inside)
        self.slice_prog["u_mouse"].value = local_mouse_uv
        self.slice_prog["u_radius"].value = float(self.heap_radius)
        self.slice_prog["u_softness"].value = float(self.heap_softness)
        self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
        self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
        self.slice_prog["u_heap_dir"].value = float(self.heap_dir)
        vol_arr, vol_tex, vol_bgr = self._volume_key_to_assets(volume_key)
        # Re-bind all texture-state uniforms every panel.  This avoids blank panels
        # after HUD/postprocess rendering has used other texture units/programs.
        self.slice_prog["tex_array"].value = 0
        self.slice_prog["u_num_layers"].value = int(vol_arr.shape[0] if vol_arr is not None else self.Z)
        self.slice_prog["u_flip_y"].value = int(self.flip_y)
        self.slice_prog["u_bgr_input"].value = int(vol_bgr)
        self._apply_filter_uniforms()
        self._apply_post_uniforms(volume_key)
        self.slice_prog["u_black_transparent"].value = int(spec.get("black_transparent", 0))
        self.slice_prog["u_black_threshold"].value = float(self.black_alpha_threshold)
        self.slice_prog["u_output_alpha"].value = float(spec.get("alpha", 1.0))

        self.slice_prog["u_curved_enable"].value = int(spec.get("curved_enable", 0))
        self.slice_prog["u_curved_kind"].value = int(spec.get("curved_kind", 0))
        self.slice_prog["u_curved_amp"].value = float(spec.get("curved_amp", 0.0))
        self.slice_prog["u_curved_radius"].value = float(max(1e-4, spec.get("curved_radius", 1.0)))

        (vol_tex if vol_tex is not None else self.tex_main).use(location=0)
        self.slice_vao.render()

    def _cpu_color_match(self, rgb: np.ndarray, target: str) -> np.ndarray:
        arr = rgb.astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        lumv = 0.299 * r + 0.587 * g + 0.114 * b
        target = str(target)
        if target == "red":
            m = np.clip((r - np.maximum(g, b)) * 2.2 + r * 0.35, 0.0, 1.0)
        elif target == "green":
            m = np.clip((g - np.maximum(r, b)) * 2.2 + g * 0.35, 0.0, 1.0)
        elif target == "blue":
            m = np.clip((b - np.maximum(r, g)) * 2.2 + b * 0.35, 0.0, 1.0)
        elif target == "white":
            m = np.clip((mn - 0.55) * 2.6, 0.0, 1.0)
        elif target == "flesh":
            m = np.clip(r * 0.75 + g * 0.20 - b * 0.25, 0.0, 1.0)
        elif target == "dark":
            m = np.clip(1.0 - lumv * 1.6, 0.0, 1.0)
        elif target == "bright":
            m = np.clip((lumv - 0.35) * 1.8, 0.0, 1.0)
        else:
            m = np.zeros_like(lumv)
        return m.astype(np.float32)

    def _apply_cpu_post_and_filter(self, rgb: np.ndarray, volume_key: str = "main") -> np.ndarray:
        """CPU equivalent of the simple slice shader post/filter path.

        This is used by the v13 live-volume fallback so that single/local/axis/
        multi_volume always show real volume samples even if the GLSL slice path
        gets into a bad texture state on a specific driver/window backend.
        """
        out = np.asarray(rgb, dtype=np.uint8).copy()
        post = int(self._current_post_mode(volume_key))
        if post == 1 or post == 3:
            lumv = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]).astype(np.uint8)
            out = np.repeat(lumv[..., None], 3, axis=2)
        if post == 2 or post == 3:
            out = (255 - out).astype(np.uint8)

        mode = str(getattr(self, "color_filter_mode", "none"))
        if mode != "none":
            strength = float(np.clip(getattr(self, "color_filter_strength", 0.0), 0.0, 1.0))
            m = self._cpu_color_match(out, getattr(self, "color_filter_target", "none"))[..., None]
            arr = out.astype(np.float32)
            if mode == "isolate":
                arr *= (1.0 - strength) + strength * m
            elif mode == "hide":
                arr *= (1.0 - strength * m)
            elif mode == "highlight":
                keep = (1.0 - 0.82 * strength) + 0.82 * strength * m
                arr = arr * keep + np.array([255.0, 242.0, 55.0], dtype=np.float32) * (m * strength * 0.35)
            out = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return out

    def _sample_panel_image_cpu(self, volume_key: str, spec: Dict[str, Any], out_w: int, out_h: int) -> Image.Image:
        out_w = max(1, int(out_w)); out_h = max(1, int(out_h))
        rgb = self._sample_rgb_for_spec(volume_key, spec, out_w=out_w, out_h=out_h)
        rgb = self._apply_cpu_post_and_filter(rgb, volume_key)
        im = Image.fromarray(rgb, "RGB").convert("RGBA")
        if int(spec.get("black_transparent", 0)) != 0:
            arr = np.asarray(im, dtype=np.uint8).copy()
            mx = arr[..., :3].max(axis=2)
            a = (mx > int(np.clip(self.black_alpha_threshold, 0.0, 1.0) * 255.0)).astype(np.uint8)
            arr[..., 3] = (a * int(np.clip(float(spec.get("alpha", 1.0)), 0.0, 1.0) * 255.0)).astype(np.uint8)
            im = Image.fromarray(arr, "RGBA")
        return im

    def _build_live_volume_view_image(self, W: int, H: int) -> Image.Image:
        """Build the currently selected volume view from CPU sampling.

        Seed views were already working because they sample the NumPy volume on
        the CPU.  This makes the regular live modes use the same reliable source:
        single/local/axis/multi_volume are composited as real sampled images and
        then uploaded as one screen texture.
        """
        W = max(1, int(W)); H = max(1, int(H))
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        draw = ImageDraw.Draw(canvas)

        if self.view_mode == "curved_plane_editor":
            spec = dict(self._curved_view_spec())
            spec["aspect_correct"] = 0
            im = self._sample_panel_image_cpu("main", spec, W, H)
            canvas.alpha_composite(im, (0, 0))
            draw.text((10, 10), str(spec.get("name", "Curved")), fill=(255, 80, 80, 255))
            return canvas

        if self.view_mode in ("single", "single_gray", "single_invert", "single_gray_invert"):
            spec = dict(self._single_view_spec())
            spec["aspect_correct"] = 0
            im = self._sample_panel_image_cpu("main", spec, W, H)
            canvas.alpha_composite(im, (0, 0))
            draw.text((10, 10), str(spec.get("name", "Single")), fill=(255, 80, 80, 255))
            return canvas

        if self.view_mode == "live_recompute":
            return self._compute_live_recompute_view(W, H)
        if self.view_mode == "multi_volume":
            specs = self._multi_volume_specs()
        else:
            specs = [("main", spec) for spec in self._split_view_specs()]

        for viewport, item in zip(self._panel_viewports(), specs):
            vx, vy, vw, vh = viewport
            vol_key, spec = item
            x0 = int(vx)
            y0 = int(H - (int(vy) + int(vh)))
            panel = self._sample_panel_image_cpu(vol_key, spec, int(vw), int(vh))
            canvas.alpha_composite(panel, (x0, y0))
            label = str(spec.get("name", vol_key))
            draw.rectangle((x0 + 6, y0 + 6, x0 + 8 + max(80, 7 * len(label)), y0 + 25), fill=(0, 0, 0, 150))
            draw.text((x0 + 10, y0 + 10), label, fill=(255, 80, 80, 255))
        return canvas

    def _build_frame_fx_source_cpu_image(self, W: int, H: int) -> Image.Image:
        spec = dict(self._single_view_spec())
        spec["aspect_correct"] = 0
        return self._sample_panel_image_cpu("main", spec, max(1, int(W)), max(1, int(H)))

    def _maybe_live_volume_cpu_fallback(self, W: int, H: int) -> bool:
        """If the GPU live slice path came up blank, overlay the reliable CPU image.

        This is mainly for the single/axis/local/multi-volume viewers. In auto
        mode we permanently switch to CPU after detecting a blank frame.
        """
        backend = str(getattr(self, "live_display_backend", "gpu"))
        live_modes = {"single", "single_gray", "single_invert", "single_gray_invert", "axis", "local", "multi_volume", "curved_plane_editor"}
        if self.view_mode not in live_modes or backend == "cpu":
            return False
        if backend != "auto" or not bool(getattr(self, "gpu_blank_check_enabled", False)):
            return False
        if int(getattr(self, "_gpu_live_blank_check_frames", 0)) <= 0:
            return False
        self._gpu_live_blank_check_frames = max(0, int(self._gpu_live_blank_check_frames) - 1)
        try:
            raw = self.ctx.screen.read(components=3, alignment=1)
            gpu = np.frombuffer(raw, dtype=np.uint8)
            if gpu.size != int(W) * int(H) * 3:
                return False
            gpu = gpu.reshape(int(H), int(W), 3)
            gpu_nonblack = float((gpu.max(axis=2) > 3).mean())
            if gpu_nonblack > 0.0025:
                return False
            cpu_img = self._build_live_volume_view_image(int(W), int(H))
            cpu_rgb = np.asarray(cpu_img, dtype=np.uint8)[..., :3]
            cpu_nonblack = float((cpu_rgb.max(axis=2) > 3).mean())
            if cpu_nonblack <= 0.0025:
                return False
            self._render_overlay_image(cpu_img)
            self.fx_backend = "cpu_live_autofallback"
            if backend == "auto":
                self.live_display_backend = "cpu"
            print(f"[live-display] GPU frame looked blank in mode={self.view_mode}; using CPU fallback (backend={backend} -> {self.live_display_backend})")
            return True
        except Exception as exc:
            print(f"[live-display] blank-frame check failed: {exc}")
            return False

    def _capture_scene_image(self) -> Image.Image:
        W, H = int(self.wnd.width), int(self.wnd.height)
        data = self.ctx.screen.read(components=3, alignment=1)
        return Image.frombytes("RGB", (W, H), data).transpose(Image.FLIP_TOP_BOTTOM)

    def _current_panel_labels(self) -> List[str]:
        if self.view_mode == "multi_volume":
            return [spec.get("name", f"panel{i+1}") for i, (_, spec) in enumerate(self._multi_volume_specs())]
        if self.view_mode == "live_recompute":
            return ["Original volume", "Live signed distance", "Live skeleton"]
        if self.view_mode != "single":
            return [spec.get("name", f"panel{i+1}") for i, spec in enumerate(self._split_view_specs())]
        return ["single"]

    def _save_panel_crops(self, img: Image.Image, base_path: Path) -> List[Path]:
        if self.view_mode == "single":
            return []
        H = img.height
        labels = self._current_panel_labels()
        paths = []
        for i, viewport in enumerate(self._panel_viewports()):
            vx, vy, vw, vh = viewport
            left = int(vx)
            upper = int(H - (vy + vh))
            right = int(vx + vw)
            lower = int(H - vy)
            crop = img.crop((left, upper, right, lower))
            safe = ''.join(ch if ch.isalnum() else '_' for ch in str(labels[i]))[:48].strip('_') or f'panel{i+1}'
            out = base_path.parent / f"{base_path.stem}_{safe}{base_path.suffix}"
            crop.save(out)
            paths.append(out)
        return paths

    def _capture_outputs_from_image(self, img: Image.Image, root_dir: Path, stem: str) -> List[Path]:
        root_dir.mkdir(parents=True, exist_ok=True)
        main_path = root_dir / f"{stem}.png"
        outputs: List[Path] = []
        scope = self.capture_scope_live
        triple = self.view_mode != "single"
        if scope in ("whole", "both") or not triple:
            img.save(main_path)
            outputs.append(main_path)
        if triple and scope in ("panels", "both"):
            outputs.extend(self._save_panel_crops(img, main_path))
        return outputs

    def save_screen_snapshot(self, image: Optional[Image.Image] = None) -> Path:
        """Save the current scene without the HUD UI overlay."""
        img = image if image is not None else self._capture_scene_image()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_count += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"screen_{self.view_mode}_{stamp}_{self.snapshot_count:04d}"
        outputs = self._capture_outputs_from_image(img, self.snapshot_dir, stem)
        if outputs:
            print(f"[snapshot] saved {len(outputs)} image(s) under {self.snapshot_dir}")
            return outputs[0]
        out_path = self.snapshot_dir / f"{stem}.png"
        img.save(out_path)
        print(f"[snapshot] saved {out_path}")
        return out_path

    def close(self):
        self._set_cursor_visible(True)
        try:
            if self.recorder.camera_waypoints or self.recorder.brush_waypoints or self.recorder.combined_waypoints:
                self.save_waypoints()
        except Exception as exc:
            print(f"[waypoint] save on close failed: {exc}")
        try:
            if self.capture24_active:
                self._stop_capture24()
        except Exception as exc:
            print(f"[capture24] close failed: {exc}")
        self.spacemouse.close()

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    def resize(self, width, height):
        self._push_slice_uniforms()
        self._arm_live_blank_check(4)

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------


    def _render_curve_side_panels(self) -> None:
        """Render optional curve inspection panels without CPU volume resampling.

        v24 built these panels by NumPy/PIL-sampling the volume every frame.
        That was useful for debugging, but it could dominate frame time at 2K.
        v25 uses the same GPU slice shader as the main viewport.

        Modes:
          local_curved  - three local curved views sampled from the volume
          curve_profile - analytic U/V/perspective diagrams of the curve itself
        """
        if bool(getattr(self, "hide_all_overlays", False)):
            return
        if not bool(getattr(self, "show_curve_side_panels", False)):
            return
        if getattr(self, "view_mode", "single") != "curved_plane_editor":
            return

        W, H = int(self.wnd.width), int(self.wnd.height)
        if W <= 16 or H <= 16:
            return

        S = self._ui_scale()
        pad = max(8, int(10 * S))
        gap = max(6, int(6 * S))
        panel_w = int(np.clip(W * 0.22, 180, 320))
        panel_h = int(np.clip(H * 0.20, 130, 240))
        x0 = W - panel_w - pad
        top_reserved = pad + int(88 * S) if bool(getattr(self, "ui_visible", True)) else pad
        y_top = max(pad, top_reserved)

        # Convert top-left layout to GL bottom-left viewports.
        viewports = []
        for idx in range(3):
            py_top = y_top + idx * (panel_h + gap)
            gl_y = H - (py_top + panel_h)
            if gl_y < pad:
                break
            viewports.append((x0, gl_y, panel_w, panel_h))
        if len(viewports) < 1:
            return

        mode = str(getattr(self, "curve_side_panel_mode", "local_curved"))
        base_center = self.center.copy()
        base_u = self.u.copy()
        base_v = self.v.copy()
        base_n = self.n.copy()
        s = float(self.scale)
        amp = float(getattr(self, "curved_plane_amp", 0.0))
        rad = float(getattr(self, "curved_plane_radius", 1.0))
        kind = int(getattr(self, "curved_plane_kind", 0))
        side_scale_n = float(max(0.10, min(0.80, abs(amp) * 3.0 + 0.18)))

        # Label overlay only; no CPU volume sampling.
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        font = self._scaled_font(11)
        small = self._scaled_font(10)

        if mode == "curve_profile":
            labels = ["Curve profile U", "Curve profile V", "Curve perspective"]
            for idx, viewport in enumerate(viewports):
                vx, gy, vw, vh = [int(v) for v in viewport]
                py_top = H - gy - vh
                draw.rounded_rectangle((vx, py_top, vx + vw, py_top + vh), radius=int(10*S), fill=(8,10,14,180), outline=(210,220,245,220), width=1)
                draw.text((vx + int(8*S), py_top + int(6*S)), labels[idx], fill=(245,248,255,245), font=font)
                cx = vx + vw * 0.5
                cy = py_top + vh * 0.58
                half = vw * 0.40
                pts = []
                steps = 90
                for i in range(steps):
                    t = -1.0 + 2.0 * i / max(1, steps - 1)
                    if idx == 0:
                        h = (t / max(rad, 1e-4)) ** 2
                    elif idx == 1:
                        h = -(t / max(rad, 1e-4)) ** 2 if kind == 1 else (t / max(rad, 1e-4)) ** 2
                    else:
                        h = (t / max(rad, 1e-4)) ** 2 + 0.20 * math.sin(6.2831853 * t)
                    px = cx + t * half
                    py = cy - amp * h * vh * 2.0
                    pts.append((px, py))
                if len(pts) >= 2:
                    draw.line(pts, fill=(255,225,90,250), width=max(2, int(2*S)))
                draw.line((vx + int(16*S), cy, vx + vw - int(16*S), cy), fill=(120,135,155,180), width=1)
            draw.text((x0, max(pad, y_top - int(24*S))), f"Curve side panels: profile  amp {amp:+.3f}  radius {rad:.2f}", fill=(235,240,255,235), font=small)
            self._render_overlay_image(overlay)
            return

        # GPU local curved volume views.  All three panels use the same shader
        # curved-plane displacement, so the side views are actually sampled from
        # the volume, not precomputed CPU thumbnails.
        specs = [
            ("Curved U/V", self._curved_view_spec()),
            ("Curved U/N", {
                "name": "Curved U/N", "center": base_center,
                "u": base_u, "v": base_n, "n": base_v,
                "scale_u": s, "scale_v": side_scale_n, "aspect_correct": 1,
                "curved_enable": int(bool(getattr(self, "curved_plane_enable", True))),
                "curved_kind": kind, "curved_amp": amp, "curved_radius": rad,
            }),
            ("Curved V/N", {
                "name": "Curved V/N", "center": base_center,
                "u": base_v, "v": base_n, "n": -base_u,
                "scale_u": s, "scale_v": side_scale_n, "aspect_correct": 1,
                "curved_enable": int(bool(getattr(self, "curved_plane_enable", True))),
                "curved_kind": kind, "curved_amp": amp, "curved_radius": rad,
            }),
        ]

        self.ctx.screen.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        for viewport, (label, spec) in zip(viewports, specs):
            self._render_slice_panel(viewport, spec, volume_key="main")
            vx, gy, vw, vh = [int(v) for v in viewport]
            py_top = H - gy - vh
            draw.rounded_rectangle((vx, py_top, vx + vw, py_top + int(24*S)), radius=int(7*S), fill=(5,8,14,170), outline=(210,220,245,170), width=1)
            draw.text((vx + int(8*S), py_top + int(5*S)), label, fill=(245,248,255,245), font=font)
        draw.text((x0, max(pad, y_top - int(24*S))), f"Curve side panels: local curved volume  amp {amp:+.3f}  radius {rad:.2f}", fill=(235,240,255,235), font=small)
        self._render_overlay_image(overlay)

    def on_render(self, time: float, frame_time: float):
        W, H = self.wnd.width, self.wnd.height

        if self.playback_enabled and self.has_playback_path():
            self.playhead_seconds += float(frame_time)
            total_play = self._playback_total_seconds()
            if total_play > 0.0 and self.playback_loop_live and self.playhead_seconds >= total_play:
                self.playhead_seconds = self.playhead_seconds % total_play
                self.force_clear_next_frame = True
            st = self.evaluate_playback(self.playhead_seconds)
            if st is not None:
                self.apply_playback_state(st)
        else:
            # smooth movement while held
            self._apply_held_keys(frame_time)

            changed = False
            self.auto_motion.step(self, frame_time, time)
            changed = True if self.auto_motion.enabled else changed
            if self.spacemouse.apply(self, frame_time):
                self._mark_navigation_input()
                self._update_plane_axes()
                changed = True

            if changed:
                self._push_slice_uniforms()
                self._update_gizmo_geometry()

        self._maybe_restore_cursor()

        # ---- single camera / split-screen tri-plane views
        # v11 robust render dispatch:
        #   single_*      -> one full-screen sampled main-volume slice
        #   axis/local    -> three sampled MPR panels
        #   multi_volume  -> main / gradient / skeleton panels
        #   frame_fx      -> always renders a source slice before GPU/CPU FX
        #   object modes  -> draw their own preview overlays later
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)

        overlay_only_modes = {"pixel_grid", "object_editor", "scene_3d", "slice_seed_board", "live_recompute"}
        single_panel_modes = {"single", "single_gray", "single_invert", "single_gray_invert"}

        local_transparent_mode = (self.view_mode == "local") and bool(getattr(self, "local_accumulate_frames", False))
        should_clear = True if self.view_mode == "local" else (self.force_clear_next_frame or (not local_transparent_mode))
        if should_clear:
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.force_clear_next_frame = False

        if local_transparent_mode:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        else:
            self.ctx.disable(moderngl.BLEND)

        if self.view_mode in single_panel_modes:
            self.fx_backend = "slice_panel_composite"
            self._render_slice_panel_composited((0, 0, W, H), self._single_view_spec(), volume_key="main")

        elif self.view_mode == "axis":
            self.fx_backend = "axis_panel_composite"
            for viewport, spec in zip(self._panel_viewports(), self._split_view_specs()):
                self._render_slice_panel_composited(viewport, spec, volume_key="main")

        elif self.view_mode == "local":
            self.fx_backend = "local_oblique_panel_composite"
            for viewport, spec in zip(self._panel_viewports(), self._split_view_specs()):
                self._render_slice_panel_composited(viewport, spec, volume_key="main")

        elif self.view_mode == "multi_volume":
            self.fx_backend = "multi_volume_panel_composite"
            for viewport, item in zip(self._panel_viewports(), self._multi_volume_specs()):
                vol_key, spec = item
                self._render_slice_panel_composited(viewport, spec, volume_key=vol_key)

        elif self.view_mode == "frame_fx":
            # GPU FX such as cuts / AMAT / grassfire / inflation need a real source slice.
            # _render_frame_fx_gpu renders the slice into an FBO, then applies the post shader.
            if self._gpu_fx_code() != 0:
                self._render_frame_fx_gpu(W, H)
            else:
                # CPU fallback still needs the source slice on screen before the overlay image is composed.
                self.fx_backend = "cpu_source_slice"
                self._render_slice_panel((0, 0, W, H), self._single_view_spec(), volume_key="main")

        elif self.view_mode == "curved_plane_editor":
            self.fx_backend = "curved_plane_panel_composite"
            self._render_slice_panel_composited((0, 0, W, H), self._curved_view_spec(), volume_key="main")

        elif self.view_mode in overlay_only_modes:
            # These modes intentionally build a full-screen PIL preview below.
            pass

        else:
            # Unknown modes no longer go blank: fall back to the regular single sampled volume.
            self.fx_backend = "fallback_single"
            self._render_slice_panel((0, 0, W, H), self._single_view_spec(), volume_key="main")

        # Fast live display path: default to GPU slice rendering for the main
        # viewing modes.  Keep the CPU compositor as an optional fallback that
        # can be toggled from the UI if a backend/driver goes blank.
        if self.view_mode in ("single", "single_gray", "single_invert", "single_gray_invert", "axis", "local", "multi_volume", "curved_plane_editor"):
            backend = str(getattr(self, "live_display_backend", "gpu"))
            if backend == "cpu":
                self._render_overlay_image(self._build_live_volume_view_image(W, H))
                self.fx_backend = "cpu_live_volume"
            else:
                # GPU live path: no CPU framebuffer readback unless backend is explicitly AUTO.
                self.fx_backend = f"gpu_direct_{self.view_mode}"
                if str(getattr(self, "live_display_backend", "gpu")) == "auto":
                    self._maybe_live_volume_cpu_fallback(W, H)

        # Restore the canonical red-plane uniforms after per-panel rendering so
        # movement/key handlers still see the main red slab state.
        self._push_slice_uniforms()

        if self.view_mode == "pixel_grid":
            self._render_overlay_image(self._build_pixel_grid_image(W, H))
        elif self.view_mode == "frame_fx":
            if self._gpu_fx_code() == 0:
                self.fx_backend = "cpu_cached"
                self._render_overlay_image(self._build_frame_transform_image(W, H))
        elif self.view_mode == "object_editor":
            self._render_overlay_image(self._build_object_editor_image(W, H))
        elif self.view_mode == "scene_3d":
            self._render_overlay_image(self._build_scene3d_image(W, H))
        elif self.view_mode == "slice_seed_board":
            self._render_overlay_image(self._build_seed_slice_board_image(W, H))
        elif self.view_mode == "live_recompute":
            self._render_overlay_image(self._compute_live_recompute_view(W, H))

        # ---- gizmo (top-right)
        if self.show_gizmo and not bool(getattr(self, "hide_all_overlays", False)) and self.view_mode not in ("pixel_grid", "scene_3d", "slice_seed_board"):
            gx0, gy0, giz_px = self._gizmo_viewport()
            self.ctx.viewport = (gx0, gy0, giz_px, giz_px)
            self.ctx.enable(moderngl.DEPTH_TEST)

            P = perspective(45.0, 1.0, 0.05, 10.0)
            V = look_at(eye=self._gizmo_eye(), target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
            MVP = (P @ V).astype(np.float32)
            self.gizmo_prog["u_mvp"].write(MVP.tobytes())

            # box wireframe
            self.gizmo_prog["u_color"].value = (0.85, 0.90, 0.98, 1.0)
            self.box_vao.render(mode=moderngl.LINES)

            # plane slab / curved sheet preview
            curved_preview = (self.view_mode == "curved_plane_editor" and bool(getattr(self, "curved_plane_enable", False)))
            if curved_preview:
                self.gizmo_prog["u_color"].value = (1.0, 0.22, 0.22, 0.65)
                self.plane_vao.render(mode=moderngl.TRIANGLES)
                if getattr(self, "curve_wire_vertex_count", 0) >= 2:
                    self.gizmo_prog["u_color"].value = (1.0, 0.82, 0.22, 1.0)
                    self.curve_wire_vao.render(mode=moderngl.LINES, vertices=int(self.curve_wire_vertex_count))
            else:
                self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.80)
                self.plane_vao.render(mode=moderngl.TRIANGLES)

            # normal arrow
            self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
            self.n_vao.render(mode=moderngl.LINES)

            # sampled interpolation path through recorded/loaded camera waypoints
            self._update_path_geometry()
            if self.path_vertex_count >= 2:
                self.gizmo_prog["u_color"].value = (0.20, 1.0, 0.35, 1.0)
                self.path_vao.render(mode=moderngl.LINE_STRIP, vertices=self.path_vertex_count)

        # Optional side previews for the curved-plane editor: U / V / perspective.
        self._render_curve_side_panels()

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Capture the rendered scene before drawing the HUD so saved images do not include the UI.
        scene_img = None
        need_capture = self.capture24_active or self.pending_screen_save
        if need_capture:
            scene_img = self._capture_scene_image()

        if self.capture24_active and scene_img is not None:
            self.capture24_accum += float(frame_time)
            step = 1.0 / max(1e-6, float(self.capture24_fps))
            while self.capture24_accum >= step:
                self.capture24_accum -= step
                self._capture_frame_to_session(scene_img)

        if self.pending_screen_save and scene_img is not None:
            self.pending_screen_save = False
            self.save_screen_snapshot(scene_img)

        # Draw built-in non-ImGui HUD last so it scales with the current screen.
        self._render_builtin_ui()


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)