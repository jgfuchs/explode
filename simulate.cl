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
    g           = -9.8,     // acceleration due to gravity (m/s^2)
    cBuoy       = 0.15,     // buoyancy multiplier (m/Ks^2)
    airDens     = 1.29,     // density of air  (kg/m^3)
    tAmb        = 300;      // ambient temperature (K)

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

void __kernel add_forces(
    const float dt,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __write_only image3d_t U_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float4 v = read_imagef(U, pos);

    // add buoyancy
    float t = read_imagef(T, pos).x;
    v.y += dt * cBuoy * (t - tAmb);

    write_imagef(U_out, pos, v);
}

void __kernel reaction(
    const float dt,
    __read_only image3d_t T,
    __write_only image3d_t T_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float4 f = read_imagef(T, pos);
    float r = distance(convert_float3(pos), (float3)(64, 16, 64));
    f.x += 2000 * exp(-r*r / 16);
    write_imagef(T_out, pos, f);
}

void __kernel divergence(
    __read_only image3d_t U,
    __write_only image3d_t Dvg,
    __write_only image3d_t P)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float vx1 = read_imagef(U, samp_i, pos + (int3)(1, 0, 0)).x;
    float vx2 = read_imagef(U, samp_i, pos - (int3)(1, 0, 0)).x;
    float vy1 = read_imagef(U, samp_i, pos + (int3)(0, 1, 0)).y;
    float vy2 = read_imagef(U, samp_i, pos - (int3)(0, 1, 0)).y;
    float vz1 = read_imagef(U, samp_i, pos + (int3)(0, 0, 1)).z;
    float vz2 = read_imagef(U, samp_i, pos - (int3)(0, 0, 1)).z;

    float d = -0.5f * h * ((vx1 - vx2) + (vy1 - vy2) + (vz1 - vz2));
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

    float p1 = read_imagef(P, samp_i, pos + (int3)(1, 0, 0)).x;
    float p2 = read_imagef(P, samp_i, pos - (int3)(1, 0, 0)).x;
    float p3 = read_imagef(P, samp_i, pos + (int3)(0, 1, 0)).x;
    float p4 = read_imagef(P, samp_i, pos - (int3)(0, 1, 0)).x;
    float p5 = read_imagef(P, samp_i, pos + (int3)(0, 0, 1)).x;
    float p6 = read_imagef(P, samp_i, pos - (int3)(0, 0, 1)).x;

    float f = (d + (p1 + p2 + p3 + p4 + p5 + p6)) / 6.0;
    write_imagef(P_out, pos, f);
}

void __kernel project(
    __read_only image3d_t U,        // velocity
    __read_only image3d_t P,        // pressure
    __write_only image3d_t U_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

    float p1 = read_imagef(P, samp_i, pos + (int3)(1, 0, 0)).x;
    float p2 = read_imagef(P, samp_i, pos - (int3)(1, 0, 0)).x;
    float p3 = read_imagef(P, samp_i, pos + (int3)(0, 1, 0)).x;
    float p4 = read_imagef(P, samp_i, pos - (int3)(0, 1, 0)).x;
    float p5 = read_imagef(P, samp_i, pos + (int3)(0, 0, 1)).x;
    float p6 = read_imagef(P, samp_i, pos - (int3)(0, 0, 1)).x;

    float3 grad = 0.5f * (1.0f / h) * (float3)(p1 - p2, p3 - p4, p5 - p6);
    float3 vNew = read_imagef(U, pos).xyz - grad;
    write_imagef(U_out, pos, (float4)(vNew, 0));
}

void __kernel render(
    __read_only image3d_t U,
    __read_only image3d_t T,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float2 fpos = convert_float2(pos) * get_image_width(U) / get_image_width(img);

    float4 sp = (float4)(fpos, 64, 0);

    float temp = read_imagef(T, samp_f, sp).x;
    temp = (temp - tAmb);
    uint4 color = {convert_uint3((float3)(temp)), 255};

    // float3 vel = read_imagef(U, samp_f, sp).xyz;
    // float p = read_imagef(T, samp_f, sp).x;
    // uint4 color = {convert_uint(length(vel)*0), convert_uint(p*100), 0, 255};

    write_imageui(img, (int2)(pos.x, get_image_height(img)-1-pos.y), color);
}
