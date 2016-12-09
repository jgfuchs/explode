
struct Camera {
    float3 pos, center, up;
    uint width, height;
};

struct Light {
    float3 pos;
    float intensity;
};

// for reading exact int coords
__constant sampler_t samp_i =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST;

// for reading interpolated float coords
__constant sampler_t samp_f =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_LINEAR;

// physics constants
__constant float
    h           = 0.5,      // cell side length
    grav        = 9.8,      // acceleration due to gravity
    cBuoy       = 0.03,     // buoyancy multiplier
    cSink       = 0.0,      // smoke sinking
    tAmb        = 300,      // ambient temperature
    tMax        = 6000,     // maximum temperature
    cCooling    = 1200,     // cooling factor
    vortEps     = 8.0;      // vorticity confinement

__constant int3 dx = {1, 0, 0},
                dy = {0, 1, 0},
                dz = {0, 0, 1};

// lookup value at coordinate (i, j, k) in image
inline float4 ix(image3d_t img, int3 c) {
    return read_imagef(img, samp_i, c);
}

void __kernel init_grid(
    __write_only image3d_t U,       // velocity
    __write_only image3d_t T,       // thermo
    __write_only image3d_t B)       // boundaries
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    write_imagef(U, pos, (float4)(0));
    write_imagef(T, pos, (float4)(tAmb, 0, 0, 0));

    int nx = get_image_width(B),
        ny = get_image_height(B),
        nz = get_image_depth(B);

    uint val;
    if (pos.x == 0 || pos.x == nx-1 || pos.z == 0 || pos.z == nz-1 || pos.y == 0 || pos.y == ny-1) {
        val = 1;    // walls (incl. floor & ceiling) are solid
    }/* else if (5 > distance(convert_float3(pos), (float3)(32, 40, 32))) {
        val = 2;
    } */else {
        val = 0;    // fluid everywhere else
    }
    write_imageui(B, pos, val);
}


void __kernel advect(
    const float dt,
    __read_only image3d_t U,
    __read_only image3d_t in,
    __write_only image3d_t out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float3 p = convert_float3(pos) + 0.5f;
    float3 pback = p - dt * (1.0f / h) * read_imagef(U, pos).xyz;
    float4 val = read_imagef(in, samp_f, pback);

    write_imagef(out, pos, val);
}


void __kernel curl(
    __read_only image3d_t U,
    __write_only image3d_t Curl)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int i = pos.x, j = pos.y, k = pos.z;

    // prefetch to avoid unecessary lookups
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

    curl.xyz *= 0.5f / h;
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
    {
        // eta = norm(grad(abs(curl(U))))
        float3 eta = {
            ix(Curl, pos + dx).w - ix(Curl, pos - dx).w,
            ix(Curl, pos + dy).w - ix(Curl, pos - dy).w,
            ix(Curl, pos + dz).w - ix(Curl, pos - dz).w,
        };
        eta = normalize(eta * 0.5f / h);

        // force = eps * (|eta| x curl U) * dh
        f.xyz += vortEps * cross(eta, ix(Curl, pos).xyz) * h;
    }

    float4 v = read_imagef(U, pos);
    v.xyz += dt * f;
    write_imagef(U_out, pos, v);
}


void __kernel reaction(
    const float dt,
    __read_only image3d_t T,
    __write_only image3d_t T_out,
    const float4 ex)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float4 f = read_imagef(T, pos);

    // cooling
    float r  = (f.x - tAmb) / (tMax - tAmb);
    f.x = max(f.x - cCooling * pown(r, 4), tAmb);   // don't go below ambient

    // heat & smoke source
    if (ex.w > 0) {
        float d = distance(convert_float3(pos), ex.xyz);
        float radius = 6;
        if (d < radius) {
            float a = min(radius - d, 1.0f);
            f.x = 3000 * a;
            f.y = 10 * a;
        }
    }

    write_imagef(T_out, pos, f);
}


void __kernel divergence(
    __read_only image3d_t U,
    __write_only image3d_t Dvg,
    __write_only image3d_t P)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float d = -0.5f * h *
         ((ix(U, pos + dx).x - ix(U, pos - dx).x)
        + (ix(U, pos + dy).y - ix(U, pos - dy).y)
        + (ix(U, pos + dz).z - ix(U, pos - dz).z));
    write_imagef(Dvg, pos, d);

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
    float3 vNew = vOld - 0.5f * (1.0f / h) * gradP;
    write_imagef(U_out, pos, (float4)(vNew, 0));
}



void __kernel set_vel_bounds(
    __read_only image3d_t B,
    __read_only image3d_t U,
    __write_only image3d_t U_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float4 v = ix(U, pos);

    int b = read_imageui(B, pos).x;
    if (b == 1) {
        int nx = get_image_width(B),
            ny = get_image_height(B),
            nz = get_image_depth(B);

        if (pos.x == 0)         { v = ix(U, pos + dx); v.x *= -1; }
        else if (pos.x == nx-1) { v = ix(U, pos - dx); v.x *= -1; }
        else if (pos.y == 0)    { v = ix(U, pos + dy); v.y *= -1; }
        else if (pos.y == ny-1) { v = ix(U, pos - dy); v.y *= -1; }
        else if (pos.z == 0)    { v = ix(U, pos + dz); v.z *= -1; }
        else if (pos.z == nz-1) { v = ix(U, pos - dz); v.z *= -1; }
    }

    write_imagef(U_out, pos, v);
}


void __kernel set_bounds(
    __read_only image3d_t B,
    __read_only image3d_t in,
    __write_only image3d_t out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float4 f = ix(in, pos);

    int b = read_imageui(B, pos).x;
    if (b == 1) {
        int nx = get_image_width(B),
            ny = get_image_height(B),
            nz = get_image_depth(B);

        if (pos.x == 0)         f = ix(in, pos + dx);
        else if (pos.x == nx-1) f = ix(in, pos - dx);
        else if (pos.y == 0)    f = ix(in, pos + dy);
        else if (pos.y == ny-1) f = ix(in, pos - dy);
        else if (pos.z == 0)    f = ix(in, pos + dz);
        else if (pos.z == nz-1) f = ix(in, pos - dz);
    }

    write_imagef(out, pos, f);
}


void __kernel render(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float2 fpos = convert_float2(pos) * get_image_width(U) / get_image_width(img);

    // float4 sp = (float4)(fpos, 0.5, 0.0);
    // float acc = 0.0f;
    // while (sp.z < 64.0f) {
    //     float s = read_imagef(T, samp_f, sp).y;
    //     acc += s;
    //     sp.z += 0.5f;
    // }
    // uint4 color = {convert_uint3((float3)(acc)), 255};

    float4 sp = (float4)(fpos, 32, 0);
    float3 t = read_imagef(T, samp_f, sp).xyz;
    uint4 color = {convert_uint3((float3)(t.y*60)), 255};
    // uint4 color = {convert_uint3((float3)((t.x-tAmb)*0.4)), 255};

    // float4 sp = (float4)(fpos, 32, 0);
    // float p = read_imagef(B, samp_f, sp).x;
    // uint4 color = {convert_uint3((float3)(p)*60), 255};

    // float3 vel = read_imagef(U, samp_f, (float4)(fpos, 32, 0)).xyz;
    // uint4 color = {convert_uint3((float3)(fabs(vel*10))), 255};

    write_imageui(img, (int2)(pos.x, get_image_height(img)-1-pos.y), color);
}
