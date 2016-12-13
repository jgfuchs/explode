
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

__constant const int
    nsamp = 128,        // main ray samples
    nlsamp = 64;        // light ray samples
__constant const float
    maxDist = 1.73205080,       // cube diagonal = sqrt(3)
    ds = maxDist / nsamp,       // main ray step size
    dsl = maxDist / nlsamp,     // light ray step size
    absorption = 30.0;

inline float4 getBlackbody(image2d_t Spec, float temp) {
    return read_imagef(Spec, samp_n, (float2)(temp / tMax, 0));
}

float3 trace_to_light(
    image3d_t T,
    image2d_t Spec,
    const struct Light *light,
    float3 pos0)
{
    float3 dir = normalize(light->pos - pos0) * dsl;
    float3 pos = pos0 + dir;
    float tx = 1.0f;

    for (int i = 0; i < nlsamp; i++) {
        float rho = read_imagef(T, samp_n, pos).y;
        tx *= 1.0f - rho * dsl * absorption;
        if (tx < TX_EPS) break;

        pos += dir;
    }

    return tx * light->intensity;
}

void __kernel render(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __read_only image3d_t BN,
    __read_only image2d_t Spec,
    __write_only image2d_t img)
{
    int2 imgPos = {get_global_id(0), get_global_id(1)};

    float3 pos = {1.0f*imgPos.x/cam.size.x, 1.0f*imgPos.y/cam.size.y, 0};
    float3 dir = normalize(pos - cam.pos) * ds;

    float tx = 1.0;      // transmittance along ray
    float3 Lo = 0.0;     // total light output from ray

    int i, j;
    float3 bg = {0.5, 0.5, 0.9};
    for (i = 0; i < nsamp; i++) {
        if (read_imageui(B, samp_ni, pos).x == 1) {
            float3 Li = trace_to_light(T, Spec, &light, pos);

            float3 L = normalize(light.pos - pos);
            float3 N = read_imagef(BN, samp_n, pos).xyz;
            float3 C = (float3)(.28, .36, .41);
            bg = dot(L, N) * C * Li * 0.8f;
            break;
        }

        float4 Tsamp = read_imagef(T, samp_n, pos);
        float rho = Tsamp.y;
        if (rho > RHO_EPS) {
            tx *= 1.0f - rho * ds * absorption;
            if (tx < TX_EPS) break;

            // incident light from light source (attenuated)
            float3 Li = trace_to_light(T, Spec, &light, pos);

            // blackbody radiation
            float4 bb = getBlackbody(Spec, Tsamp.x);
            float3 Le = bb.xyz * bb.w * 0.7f;

            Lo += (Li + Le * tx) * rho * ds;
        }

        pos += dir;

        // terminate if out-of-bounds
        if (pos.x < 0.0f || pos.x > 1.0f
         || pos.y < 0.0f || pos.y > 1.0f
         || pos.z < 0.0f || pos.z > 1.0f) {
            break;
        }
    }

    float3 color = Lo + tx * bg;

    uint4 rgba = {convert_uint3(color*255), 255};
    write_imageui(img, (int2)(imgPos.x, cam.size.y-1-imgPos.y), rgba);
}


void __kernel render_slice(
    const struct Camera cam,
    const struct Light light,
    __read_only image3d_t T,
    __read_only image3d_t B,
    __read_only image3d_t BN,
    __read_only image2d_t Spec,
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
    // float4 bb = getBlackbody(Spec, samp.x);
    // color.xyz = convert_uint3(255 * bb.xyz * bb.w * 0.1f);

    float r = noise3f((float3)(fpos*.25f, 1)) * 0.5f + 0.5f;
    color.xyz = convert_uint3((float3)(r)*255);

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
    return C1 * pown(wl, -5) / (exp(C2 / (wl * t)) - 1.0f) * 1e-9;
}

void __kernel gen_blackbody(
    __global float3 *CIE,
    __write_only image2d_t Spec)
{
    int pos = get_global_id(0);
    float temp = tMax * (float)(pos) / get_global_size(0);

    // avoid NaNs
    if (temp < 500.0f) {
        write_imagef(Spec, (int2)(pos, 0), 0);
        return;
    }

    const int wl_min = 380;
    const int wl_max = 780;

    // integrate blackbody spectrum against CIE color matching function
    float3 xyz = 0;
    float wl = wl_min;
    for (int i = 0; i < 81; i++) {
        xyz += CIE[i] * planck((float) wl, temp);
        wl += 5.0f;
    }

    // XYZ to sRGB conversion matrix
    const float3 sRGB[3] = {
        { 3.2404542, -1.5371385, -0.4985314},
        {-0.9692660,  1.8760108,  0.0415560},
        { 0.0556434, -0.2040259,  1.0572252}
    };
    float3 rgb = {dot(sRGB[0], xyz), dot(sRGB[1], xyz), dot(sRGB[2], xyz)};

    // ensure all components >= 0
    rgb += -min(min(0.0f, rgb.x), min(rgb.y, rgb.z));
    // make brightest component = 1
    float rgbMax = max(max(rgb.x, rgb.y), rgb.z);
    if (rgbMax > 0.0f) rgb /= rgbMax;

    write_imagef(Spec, (int2)(pos, 0), (float4)(rgb, sqrt(length(xyz))));
}


inline uint4 ixu(image3d_t img, int3 c) {
    return read_imageui(img, samp_i, c);
}

void __kernel gen_normals(
    __read_only image3d_t B,
    __write_only image3d_t BN)
{
    int3 pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    if (ixu(B, pos).x != 1) {
        write_imagef(BN, pos, (float4)(1));
        return;
    }


    float3 fdx = convert_float3(dx),
           fdy = convert_float3(dy),
           fdz = convert_float3(dz);

    float3 n = 0;
    n += ixu(B, pos+dx).x == 0 ?  fdx : 0;
    n += ixu(B, pos-dx).x == 0 ? -fdx : 0;
    n += ixu(B, pos+dy).x == 0 ?  fdy : 0;
    n += ixu(B, pos-dy).x == 0 ? -fdy : 0;
    n += ixu(B, pos+dz).x == 0 ?  fdz : 0;
    n += ixu(B, pos-dz).x == 0 ? -fdz : 0;
    n = normalize(n);
    write_imagef(BN, pos, (float4)(n, 0));
}
