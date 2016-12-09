/* -*- C++ -*- */

#ifndef __SCENE_H__
#define __SCENE_H__

#include <fstream>
#include <vector>
#include <CL/cl.hpp>

struct SimParams {
    SimParams() :
        grid_w(128), grid_h(128), grid_d(128),
        nsteps(10), dt(0.01), walls(false) {}

    int grid_w, grid_h, grid_d;
    int nsteps;
    float dt;
    cl_uint walls;
};

struct Camera {
    Camera() :
        pos({0, 0, -10}),
        center({0, 0, 0}),
        up({0, 1, 0}),
        width(600),
        height(600) {}

    cl_float3 pos, center, up;
    cl_uint width, height;
    cl_float _1, _2;
} __attribute__ ((packed));

struct Light {
    Light() : pos({1, 10, 0}), intensity(1) {}

    cl_float3 pos;
    cl_float intensity;
    cl_float _1, _2, _3;
} __attribute__ ((packed));

struct Explosion {
    Explosion() :
        pos({0, 0, 0}),
        size(1.0f),
        t0(0.0f) {}

    cl_float3 pos;
    cl_float size, t0;
} __attribute__ ((packed));

// rectangular prisms only
struct Object {
    cl_float3 pos, dims;
} __attribute__ ((packed));

class Scene {
public:
    Scene(char *fname);

    // scene description
    SimParams params;
    Camera cam;
    Light light;
    std::vector<Explosion> explosions;
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
