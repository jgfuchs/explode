
__constant sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST;

void __kernel init_grid(
    __write_only image3d_t U,
    __write_only image3d_t P,
    __write_only image3d_t T)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    write_imagef(U, pos, 0.0);
    write_imagef(P, pos, 0.0);
    write_imagef(T, pos, (float4)(25, 0, 0, 0));
}

void __kernel advect(
    const float dt,
    const float h,
    __read_only image3d_t F,
    __read_only image3d_t V,
    __write_only image3d_t V_out)
{
    int3 coord = {get_global_id(0), get_global_id(1), get_global_id(2)};
    float3 pos = convert_float3(coord) - dt * (1/h) * read_imagef(F, coord).xyz;
    write_imagef(V_out, coord, read_imagef(V, samp, pos));
}

void __kernel project()
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
}

void __kernel render(
    __read_only image3d_t U,
    __read_only image3d_t P,
    __read_only image3d_t T,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float temp = read_imagef(T, (int4){pos.xy, 64, 0}).x;
    uint4 color = {temp, 0, 0, 255};
    write_imageui(img, pos, color);
}
