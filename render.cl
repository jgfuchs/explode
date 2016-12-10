
struct Camera {
    float3 pos, center, up;
    uint width, height;
};

struct Light {
    float3 pos;
    float intensity;
};

void __kernel render(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __write_only image2d_t img)
{
    int2 imgPos = {get_global_id(0), get_global_id(1)};

    const int   nsamp = 64;
    const int   nlsamp = 32;
    const float maxDist = sqrt(3.0f);
    const float scale = maxDist / nsamp;
    const float lscale = maxDist / nlsamp;
    const float absorption = 10.0;
    const float lightLum = 100.0;
    const float rhoEps = 0.0001;

    const float3 eyePos = {0.5, 0.5, -1.0};
    const float3 lightPos = {1.0, 2.0, 0.2};

    float3 pos = {1.0f*imgPos.x/cam.width, 1.0f*imgPos.y/cam.height, 0};
    float3 dir = normalize(pos - eyePos) * scale;

    float tx = 1.0;
    float Lo = 0.0;

    int i, j;
    for (i = 0; i < nsamp; i++) {
        float rho = read_imagef(T, samp_n, pos).y;
        if (rho > rhoEps) {
            tx *= 1.0f - rho * scale * absorption;
            if (tx < 0.01f) break;

            float3 ldir = normalize(lightPos - pos) * scale;
            float3 lpos = pos + ldir;
            float txl = 1.0 / 10;

            for (j = 0; j < nlsamp; j++) {
                float lrho = read_imagef(T, samp_n, lpos).y;
                txl *= 1.0 - lrho * scale * absorption;
                if (txl < 0.01) break;

                lpos += ldir;
            }

            float Li = lightLum * txl;
            Lo += Li * tx * rho * scale;
        }

        pos += dir;
    }

    uint4 color = {0, 0, 0, 255};
    color.xyz = convert_uint3((float3)(Lo*255));

    write_imageui(img, (int2)(imgPos.x, cam.height-1-imgPos.y), color);
}


void __kernel render_slice(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float2 fpos = convert_float2(pos) * get_image_width(T) / cam.width;

    float4 sp = (float4)(fpos, 32, 0);
    uint b = read_imageui(B, samp_f, sp).x;
    float4 t = read_imagef(T, samp_f, sp);

    float s = clamp(t.y*400, 0.0f, 255.0f);
    uint4 color = {255 - convert_uint3((float3)(s)), 255};
    if (b > 0) {
        color.xyz = (uint3)(255, 128, 0);
    }

    write_imageui(img, (int2)(pos.x, cam.height-1-pos.y), color);
}
