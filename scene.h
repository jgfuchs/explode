/* -*- C++ -*- */

#ifndef __SCENE_H__
#define __SCENE_H__

#include <fstream>
#include <vector>
#include <vecmath.h>

struct Camera {
    Camera() :
        pos(Vector3f(0, 0, -10)),
        center(Vector3f(0, 0, 0)),
        up(Vector3f(0, 1, 0)),
        width(600),
        height(600) {}

    Vector3f pos, center, up;
    unsigned width, height;
};

struct Explosion {
    Explosion() :
        pos(Vector3f(0, 0, 0)),
        size(1.0f),
        t0(0.0f) {}

    Vector3f pos;
    float size;
    float t0;
};

struct Object {

};

class Scene {
public:
    Scene(char *fname);

    // simulation parameters
    int grid_w, grid_h, grid_d;
    int nsteps, nparticles;
    float timestep;

    // scene description
    Camera cam;
    // std::vector<Camera> cameras;
    std::vector<Explosion> explosions;
    std::vector<Object *> objects;

private:
    // for parsing
    std::ifstream in;

    void parseSimParams();
    void parseCamera();
    void parseObject();
    void parseExplosion();

    std::string getToken();
    void expect(std::string s);
    int getInt();
    float getFloat();
    Vector3f getVector3f();

    void debugInfo();
};

#endif // __SCENE_H__
