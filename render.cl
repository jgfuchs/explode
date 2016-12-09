
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
