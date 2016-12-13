// http://www.geeks3d.com/20100831/shader-library-noise-and-pseudo-random-number-generator-in-glsl/

int LFSR_Rand_Gen(int n) {
  n = (n << 13) ^ n;
  return (n * (n*n*15731+789221) + 1376312589) & 0x7fffffff;
}

float LFSR_Rand_Gen_f( int n ) {
  return (float)(LFSR_Rand_Gen(n));
}

float noise3f(float3 p) {
    float3 _f;
    int3 ip = convert_int3(floor(p));
    float3 u = fract(p, &_f);
    u = u*u*(3.0f-2.0f*u);

    int n = ip.x + ip.y*57 + ip.z*113;

    float res = mix(mix(mix(LFSR_Rand_Gen_f(n+(0+57*0+113*0)),
                            LFSR_Rand_Gen_f(n+(1+57*0+113*0)),u.x),
                        mix(LFSR_Rand_Gen_f(n+(0+57*1+113*0)),
                            LFSR_Rand_Gen_f(n+(1+57*1+113*0)),u.x),u.y),
                    mix(mix(LFSR_Rand_Gen_f(n+(0+57*0+113*1)),
                            LFSR_Rand_Gen_f(n+(1+57*0+113*1)),u.x),
                        mix(LFSR_Rand_Gen_f(n+(0+57*1+113*1)),
                            LFSR_Rand_Gen_f(n+(1+57*1+113*1)),u.x),u.y),u.z);

    return 1.0f - res*(1.0f / 1073741824.0f);
}
