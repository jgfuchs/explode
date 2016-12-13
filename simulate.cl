
// for reading exact int coords
__constant sampler_t samp_i =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST;

// for reading float coords, with interpolation
__constant sampler_t samp_f =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_LINEAR;

// for reading normalized float coords, with interpolation
__constant sampler_t samp_n =
    CLK_NORMALIZED_COORDS_TRUE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_LINEAR;

// for reading normalized float coords, no interpolation
__constant sampler_t samp_ni =
    CLK_NORMALIZED_COORDS_TRUE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;


// convention: c* = coefficient, t* = temperature, r* = rate
__constant const float
    // general constants
    h           = 0.25,     // cell side length (m)
    hinv        = 1.0f/h,   // cells per unit length
    grav        = 9.8,      // acceleration due to gravity (m/s^2)
    cVort       = 8.0,     // vorticity confinement
    // heat-related
    cBuoy       = 0.04*h,   // buoyancy multiplier
    cSink       = 0.3,      // smoke sinking
    cCooling    = 1200,     // cooling
    tAmb        = 300,      // ambient temperature (K)
    tMax        = 6000,     // "maximum" temperature (K)
    // combustion-related
    tIgnite     = 500,      // (auto)ignition temperature (K)
    rBurn       = 4,        // fuel burn rate (amt/sec)
    rHeat       = 2400,     // heat production rate (K/s/fuel)
    rSmoke      = 1.0,      // smoke/soot production rate
    rDvg        = 18,       // extra divergence = "explosiveness"
    rSmokeDiss  = 0.008;    // smoke dissipation/dissappearance


__constant int3 dx = {1, 0, 0},
                dy = {0, 1, 0},
                dz = {0, 0, 1};

struct Object {
    float3 pos, dim;
};


// lookup value at coordinate (i, j, k) in image
inline float4 ix(image3d_t img, int3 c) {
    return read_imagef(img, samp_i, c);
}


void __kernel init_grid(
    uint walls,
    uint nobjs,
    __global const struct Object *objects,
    __write_only image3d_t U,       // velocity
    __write_only image3d_t T,       // thermo
    __write_only image3d_t B)       // boundaries
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    write_imagef(U, pos, (float4)(0));
    write_imagef(T, pos, (float4)(0));

    int nx = get_image_width(B),
        ny = get_image_height(B),
        nz = get_image_depth(B);

    // set walls as boundaries
    uint b = 0;
    if (walls) {
        if (pos.x == 0 || pos.x == nx-1
         || pos.y == 0 || pos.y == ny-1
         || pos.z == 0 || pos.z == nz-1)
         {
             b = 2;
         }
    }

    // voxelize objects
    for (int i = 0; i < nobjs; i++) {
        float3 p = objects[i].pos;
        float3 dim = objects[i].dim;
        if (pos.x >= p.x && pos.x <= p.x+dim.x
         && pos.y >= p.y && pos.y <= p.y+dim.y
         && pos.z >= p.z && pos.z <= p.z+dim.z)
         {
             b = 1;
         }
    }

    write_imageui(B, pos, b);
}


void __kernel advect(
    const float dt,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __write_only image3d_t U_out,
    __write_only image3d_t T_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float3 fpos = convert_float3(pos) + 0.5f;
    float3 p0 = fpos - dt * hinv * read_imagef(U, pos).xyz;

    write_imagef(U_out, pos, read_imagef(U, samp_f, p0));
    write_imagef(T_out, pos, read_imagef(T, samp_f, p0));
}


void __kernel curl(
    __read_only image3d_t U,
    __write_only image3d_t Curl)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int i = pos.x, j = pos.y, k = pos.z;

    // "prefetch" to avoid unecessary lookups
    float4 x1 = ix(U, pos + dx);
    float4 x2 = ix(U, pos - dx);
    float4 y1 = ix(U, pos + dy);
    float4 y2 = ix(U, pos - dy);
    float4 z1 = ix(U, pos + dz);
    float4 z2 = ix(U, pos - dz);

    float4 curl = {
        y1.z - y2.z - z1.y + z2.y,
        z1.x - z2.x - x1.z + x2.z,
        x1.y - x2.y - y1.x + y2.x,
        0
    };

    curl.xyz *= 0.5f * hinv;
    curl.w = length(curl.xyz);

    write_imagef(Curl, pos, curl);
}


void __kernel add_forces(
    const float dt,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __read_only image3d_t Curl,
    __write_only image3d_t U_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float4 therm = read_imagef(T,  pos);

    // force accumulator
    float3 f = 0;

    // buoyancy - hot air rises
    f.y += cBuoy * (therm.x - tAmb);

    // smoke sinks
    f.y -= cSink * grav * therm.y;

    // vorticity confinement
    float3 eta = {
        ix(Curl, pos + dx).w - ix(Curl, pos - dx).w,
        ix(Curl, pos + dy).w - ix(Curl, pos - dy).w,
        ix(Curl, pos + dz).w - ix(Curl, pos - dz).w,
    };
    // eta = norm(grad(abs(curl(U))))
    eta = normalize(eta * 0.5f * hinv);
    // force = eps * (|eta| x curl U) * dh
    f.xyz += cVort * cross(eta, ix(Curl, pos).xyz) * h;

    float4 v = read_imagef(U, pos);
    v.xyz += dt * f;
    write_imagef(U_out, pos, v);
}


void __kernel reaction(
    const float dt,
    __read_only image3d_t T,
    __write_only image3d_t T_out,
    __write_only image3d_t Dvg)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    // x = temperature, y = smoke, z = fuel
    float4 f = read_imagef(T, pos);

    // cooling
    float r  = (f.x - tAmb) / (tMax - tAmb);
    f.x = max(f.x - cCooling * pown(r, 4), tAmb);   // don't go below ambient

    // combustion
    float dvg = 0;
    if (f.x > tIgnite && f.z > 0.0f) {
        float df = min(f.z, rBurn * dt);
        f.x += rHeat * df;
        f.y += rSmoke * df;
        f.z -= df;
        dvg = rDvg * df;
    }

    f.y *= 1.0f - rSmokeDiss;

    write_imagef(T_out, pos, f);
    write_imagef(Dvg, pos, (float4)(dvg, 0, 0, 0));
}


void __kernel divergence(
    __read_only image3d_t U,
    __read_only image3d_t Dvg,
    __write_only image3d_t Dvg_out,
    __write_only image3d_t P)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float d0 = read_imagef(Dvg, pos).x;
    float d = -0.5f * h *
         ((ix(U, pos + dx).x - ix(U, pos - dx).x)
        + (ix(U, pos + dy).y - ix(U, pos - dy).y)
        + (ix(U, pos + dz).z - ix(U, pos - dz).z));
    write_imagef(Dvg_out, pos, d0 + d);

    // avoid a call to enqueueFillImage by zeroing pressure field here
    write_imagef(P, pos, 0);
}


void __kernel jacobi(
    __read_only image3d_t P,        // pressure
    __read_only image3d_t Dvg,      // divergence
    __write_only image3d_t P_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float f = ((ix(P, pos + dx).x + ix(P, pos - dx).x
              + ix(P, pos + dy).x + ix(P, pos - dy).x
              + ix(P, pos + dz).x + ix(P, pos - dz).x) + ix(Dvg, pos).x) / 6.0;
    write_imagef(P_out, pos, f);
}


void __kernel project(
    __read_only image3d_t U,        // velocity
    __read_only image3d_t P,        // pressure
    __write_only image3d_t U_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float3 gradP = {
        ix(P, pos + dx).x - ix(P, pos - dx).x,
        ix(P, pos + dy).x - ix(P, pos - dy).x,
        ix(P, pos + dz).x - ix(P, pos - dz).x
    };

    float3 vOld = read_imagef(U, pos).xyz;
    float3 vNew = vOld - 0.5f * hinv * gradP;
    write_imagef(U_out, pos, (float4)(vNew, 0));
}


void __kernel set_bounds(
    __read_only image3d_t B,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __write_only image3d_t U_out,
    __write_only image3d_t T_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float4 u = ix(U, pos);
    float4 t = ix(T, pos);

    int b = read_imageui(B, pos).x;
    if (b) {
        u = 0;
        t = 0;
    }

    write_imagef(U_out, pos, u);
    write_imagef(T_out, pos, t);
}


void __kernel add_explosion(
    const float3 loc,
    const float size,
    __read_only image3d_t T,
    __write_only image3d_t T_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int id = (get_image_depth(T) * pos.z + pos.y) * get_image_height(T) + pos.x;
    float4 f = read_imagef(T, pos);

    // explosion positions are normalized coords
    float3 fpos = convert_float3(pos) / get_image_width(T);
    float d = distance(loc, fpos);
    if (d < size) {
        f.xyz = (float3)(3000, 0, 1.25);
    }

    write_imagef(T_out, pos, f);
}


#include "render.cl"
