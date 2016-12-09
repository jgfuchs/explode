/* -*- C++ -*- */

#ifndef __SCENE_H__
#define __SCENE_H__

#include <fstream>
#include <vector>
#include <CL/cl.hpp>

struct SimParams {
    SimParams() :
        grid_w(128), grid_h(128), grid_d(128),
        nsteps(10), dt(0.01) {}

    int grid_w, grid_h, grid_d;
    int nsteps;
    float dt;
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
};

struct Light {
    Light() : pos({1, 10, 0}), intensity(1) {}

    cl_float3 pos;
    cl_float intensity;
};

struct Explosion {
    Explosion() :
        pos({0, 0, 0}),
        size(1.0f),
        t0(0.0f) {}

    cl_float3 pos;
    cl_float size, t0;
};

struct Object {

};

class Scene {
public:
    Scene(char *fname);

    // scene description
    SimParams params;
    Camera cam;
    Light light;
    std::vector<Explosion> explosions;
    std::vector<Object *> objects;

private:
    // for parsing
    std::ifstream in;

    void parseSimParams();
    void parseCamera();
    void parseLight();
    void parseObject();
    void parseExplosion();

    std::string getToken();
    void expect(std::string s);
    int getInt();
    float getFloat();
    cl_float3 getFloat3();

    void debugInfo();
};

#endif // __SCENE_H__
