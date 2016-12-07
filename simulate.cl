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
    h           = 0.5,      // cell side length (m)
    grav        = 9.8,     // acceleration due to gravity (m/s^2)
    cBuoy       = 0.15,     // buoyancy multiplier (m/Ks^2)
    airDens     = 1.29,     // density of air  (kg/m^3)
    tAmb        = 300,      // ambient temperature (K)
    vortEps     = 5.0;      // vorticity confinement coefficient

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
    if (pos.x < 1 || pos.x > nx-1 || pos.z < 1 || pos.z > nz-1 || pos.y > ny-1) {
        val = 1;    // walls & ceiling are open
    } else if (pos.y < 1) {
        val = 2;    // floor is solid
    } else {
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
    float temp = therm.x;
    f.y += cBuoy * (temp - tAmb);

    // smoke sinks
    float smoke = therm.y;
    f.y += smoke / (smoke + airDens) * grav;

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
    const float3 xy)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float4 f = read_imagef(T, pos);
    float r = distance(convert_float3(pos), xy);
    f.x += 500 * exp(-r*r / 2);
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

    float d = read_imagef(Dvg, pos).x;

    float p1 = ix(P, pos + dx).x;
    float p2 = ix(P, pos - dx).x;
    float p3 = ix(P, pos + dy).x;
    float p4 = ix(P, pos - dy).x;
    float p5 = ix(P, pos + dz).x;
    float p6 = ix(P, pos - dz).x;

    float f = (d + (p1 + p2 + p3 + p4 + p5 + p6)) / 6.0;
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


void __kernel render(
    __read_only image3d_t U,
    __read_only image3d_t T,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float2 fpos = convert_float2(pos) * get_image_width(U) / get_image_width(img);

    float4 sp = (float4)(fpos, 1.0, 0.0);
    float acc = 0.0f;
    while (sp.z < 64.0f) {
        float temp = read_imagef(T, samp_f, sp).x;
        acc += 0.5f * (temp - tAmb) / (0.2f * sp.z);
        sp.z += 1.0f;
    }

    // printf("%f\n", acc);
    uint4 color = {convert_uint3((float3)(acc)), 255};

    // float3 vel = read_imagef(U, samp_f, sp).xyz;
    // float p = read_imagef(T, samp_f, sp).x;
    // uint4 color = {convert_uint3(vel*10), 255};

    write_imageui(img, (int2)(pos.x, get_image_height(img)-1-pos.y), color);
}
