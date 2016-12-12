
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
    __global float3 *Spec,
    __write_only image2d_t img)
{
    int2 imgPos = {get_global_id(0), get_global_id(1)};

    const int   nsamp = 128;            // main ray samples
    const int   nlsamp = 64;            // light ray samples
    const float maxDist = sqrt(3.0f);   // cube diagonal
    const float ds = maxDist / nsamp;   // main ray step size
    const float dsl = maxDist / nlsamp; // light ray step size
    const float absorption = 30.0;

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
            float3 Le = Spec[(int)(temp / tMax * 256.0f)]*8.0f;
            Lo += (Li + Le) * tx * rho * ds;
        }

        pos += dir;

        // terminate if out-of-bounds
        if (pos.x < 0.0f || pos.x > 1.0f
         || pos.y < 0.0f || pos.y > 1.0f
         || pos.z < 0.0f || pos.z > 1.0f) {
            break;
        }
    }

    float3 bg = {0.5, 0.5, 0.9};
    float3 color = Lo + tx * bg;
    // color /= max(max(1.0f, color.x), max(color.y, color.z));

    uint4 rgba = {convert_uint3(color*255), 255};
    write_imageui(img, (int2)(imgPos.x, cam.size.y-1-imgPos.y), rgba);
}


void __kernel render_slice(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __global float3 *Spec,
    __write_only image2d_t img)
{
    int2 pos = {get_global_id(0), get_global_id(1)};
    float2 fpos = convert_float2(pos) * get_image_width(T) / cam.size.x;
    float4 sp = (float4)(fpos, 32, 0);
    uint b = read_imageui(B, samp_f, sp).x;
    uint4 color = {0, 0, 0, 255};

    float4 samp = read_imagef(T, samp_f, sp);

    // temperature
    // color.x = (int)(8.0f * samp.x / 4000.0f) * 32;

    // smoke
    // color.xyz = convert_uint3((float3)(samp.y*200));

    // fuel
    // color.y = (int) 255 * samp.z;

    // black body
    color.xyz = convert_uint3(255 * Spec[(int)(samp.x / tMax * 256.0f)]);

    // mark walls
    if (b > 0) {
        color.xyz = (uint3)(255, 128, 0);
    }

    write_imageui(img, (int2)(pos.x, cam.size.y-1-pos.y), color);
}


// Planck's equation for blackbody radiation
//  (wl=wavelength  in nm, t=temperature in K)
float planck(float wl, float t) {
    const float C1 = 3.74183e-16;   // 2*pi*h*c^2
    const float C2 = 1.4388e-2;     // h*c/k

    wl *= 1e-9;
    return C1 * pown(wl, -5) / (exp(C2 / (wl * t)) - 1.0f);
}

void __kernel gen_blackbody(
    __global float3 *CIE,
    __global float3 *S)
{
    int pos = get_global_id(0);
    float temp = tMax * pos / get_global_size(0);
    if (temp < 1000.0f) {
        S[pos] = (float3)(0);
        return;
    }

    const int wl_min = 380;
    const int wl_max = 780;

    // integrate blackbody spectrum against CIE color matching
    float3 xyz = 0;
    float wl = wl_min;
    for (int i = 0; i < 81; i++) {
        xyz += CIE[i] * planck((float) wl, temp);
        wl += 5.0f;
    }
    xyz = xyz / (xyz.x + xyz.y + xyz.z);

    // XYZ to sRGB conversion matrix
    const float3 sRGB[3] = {
        { 3.2404542, -1.5371385, -0.4985314},
        {-0.9692660,  1.8760108,  0.0415560},
        { 0.0556434, -0.2040259,  1.0572252}
    };

    float3 rgb = {dot(sRGB[0], xyz), dot(sRGB[1], xyz), dot(sRGB[2], xyz)};
    rgb += -min(min(0.0f, rgb.x), min(rgb.y, rgb.z));
    float rgbMax = max(max(rgb.x, rgb.y), rgb.z);
    if (rgbMax > 0.0f) rgb /= rgbMax;

    S[pos] = rgb;

    // printf("%.0f %f %f %f\n", temp, xyz.x, xyz.y, xyz.z);
    // printf("%.0f %f %f %f\n", temp, rgb.x, rgb.y, rgb.z);
}
