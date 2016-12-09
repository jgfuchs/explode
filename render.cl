void __kernel render(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};

    uint4 color = {0, 0, 0, 255};
    write_imageui(img, (int2)(pos.x, get_image_height(img)-1-pos.y), color);
}


void __kernel render_slice(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t U,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float2 fpos = convert_float2(pos) * get_image_width(U) / get_image_width(img);

    float4 sp = (float4)(fpos, 32, 0);
    uint b = read_imageui(B, samp_f, sp).x;
    float4 t = read_imagef(T, samp_f, sp);

    float s = clamp(t.y*60, 0.0f, 255.0f);
    uint4 color = {255 - convert_uint3((float3)(s)), 255};
    if (b > 0) {
        color.xyz = (uint3)(255, 128, 0);
    }

    write_imageui(img, (int2)(pos.x, get_image_height(img)-1-pos.y), color);
}
