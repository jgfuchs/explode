
__constant sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_NEAREST;

void __kernel init_grid(
    __write_only image3d_t U,
    __write_only image3d_t P)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    write_imagef(U, pos, (float4)(0.0));
    write_imagef(P, pos, 0.0);
}

void __kernel advect_field(
    const float dt,
    __read_only image3d_t F,
    __read_only image3d_t V,
    __write_only image3d_t V_out)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)}

}

void __kernel advect_scalar(
    const float dt,
    __read_only image3d_t F,
    __read_only image3d_t V,
    __write_only image3d_t V_out)
{

}

void __kernel project()
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};

}
