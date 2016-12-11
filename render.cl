
struct Camera {
    float3 pos;
    uint2 size;
};

struct Light {
    float3 pos;
    float intensity;
};

#define RHO_EPS     0.001f
#define TX_EPS      0.01f

void __kernel render(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __write_only image2d_t img)
{
    int2 imgPos = {get_global_id(0), get_global_id(1)};

    const int   nsamp = 128;            // main ray samples
    const int   nlsamp = 64;            // light ray samples
    const float maxDist = sqrt(3.0f);   // cube diagonal
    const float ds = maxDist / nsamp;   // main ray step size
    const float dsl = maxDist / nlsamp; // light ray step size
    const float absorption = 50.0;

    float3 pos = {1.0f*imgPos.x/cam.size.x, 1.0f*imgPos.y/cam.size.y, 0};
    float3 dir = normalize(pos - cam.pos) * ds;

    float tx = 1.0;      // transmittance along ray
    float3 Lo = 0.0;     // total light output from ray

    int i, j;
    for (i = 0; i < nsamp; i++) {
        float4 Tsamp = read_imagef(T, samp_n, pos);
        float temp = Tsamp.x;
        float rho = Tsamp.y;
        if (rho > RHO_EPS) {
            tx *= 1.0f - rho * ds * absorption;
            if (tx < TX_EPS) break;

            float3 ldir = normalize(light.pos - pos) * dsl;
            float3 lpos = pos + ldir;
            float txl = 1.0f;

            for (j = 0; j < nlsamp; j++) {
                float rhol = read_imagef(T, samp_n, lpos).y;
                txl *= 1.0f - rhol * dsl * absorption;
                if (txl < TX_EPS) break;

                lpos += ldir;
            }

            float Li = light.intensity * txl;
            float3 Le = (float3)(1.0, 0.65, 0.0) * 60 * (temp - tAmb) / tMax;
            Lo += (Li + Le) * tx * rho * ds;
        }

        pos += dir;
    }

    float3 bg = {0.5, 0.5, 0.9};
    // if (pos.y < 0.0f) {
    //     bg = (float3)(0.5, 0.25, 0.0);
    //     if (tx > 4.0f*TX_EPS) {
    //         float3 ldir = normalize(light.pos - pos) * dsl;
    //         float3 lpos = pos + ldir;
    //         float txf = 1.0;
    //         for (j = 0; j < nlsamp; j++) {
    //             float rhol = read_imagef(T, samp_n, lpos).y;
    //             txf *= 1.0f - 0.5f * rhol * dsl * absorption;
    //             if (txf < TX_EPS) break;
    //             lpos += ldir;
    //         }
    //
    //         bg *= txf;
    //     }
    // }

    float3 color = Lo + tx * bg;

    uint4 rgba = {convert_uint3(color*255), 255};
    write_imageui(img, (int2)(imgPos.x, cam.size.y-1-imgPos.y), rgba);
}


void __kernel gen_spectra(
    __write_only image1d_t S)
{
    int pos = get_global_id(0);
    float temp = tMax * pos / get_global_size(0);

    write_imagef(S, pos, 0.0f);
}
