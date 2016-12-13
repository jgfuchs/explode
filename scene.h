/* -*- C++ -*- */

#ifndef __SCENE_H__
#define __SCENE_H__

#include <fstream>
#include <vector>
#include <CL/cl.hpp>

struct SimParams {
    SimParams() :
        grid_n(128),
        nsteps(100),
        niters(30),
        dt(0.04),
        walls(true) {}

    int grid_n;
    int nsteps, niters;
    float dt;
    cl_uint walls;
};

struct Camera {
    Camera() :
        pos({0.5, 0.5, -5.0}),
        size({256, 256}) {}

    cl_float3 pos;
    cl_uint2 size;
    cl_float _1, _2;    // pad to 32 bytes
} __attribute__ ((packed));

struct Light {
    Light() :
        pos({1.0, 1.0, -1.0}),
        intensity(3) {}

    cl_float3 pos;
    cl_float intensity;
    cl_float _1, _2, _3;    // pad to 32 bytes
} __attribute__ ((packed));

struct Explosion {
    Explosion() :
        pos({.5, .2, .5}),
        size(0.05),
        t0(0.2f) {}

    cl_float3 pos;
    cl_float size, t0;
    cl_float _1, _2;       // pad to 32 bytes
} __attribute__ ((packed));

// rectangular prisms only
struct Object {
    cl_float3 pos, dim;
} __attribute__ ((packed));

class Scene {
public:
    Scene(char *fname);

    // scene description
    SimParams params;
    Camera cam;
    Light light;
    Explosion explosion;
    std::vector<Object> objects;

private:
    // for parsing
    std::ifstream in;

    void parseSimParams();
    void parseCamera();
    void parseLight();
    void parseExplosion();
    void parseObject();

    std::string getToken();
    void expect(std::string s);
    int getInt();
    float getFloat();
    cl_float3 getFloat3();
};

#endif // __SCENE_H__
