/* -*- C++ -*- */

#ifndef __SCENE_H__
#define __SCENE_H__

#include <fstream>
#include <vector>
#include <vecmath.h>

struct SimParam {
    SimParam() :
        grid(64),
        particles(1024),
        steps(100),
        timestep(0.01f) {}

    int grid;
    int particles;
    int steps;
    float timestep;
};

struct Camera {
    Camera() :
        pos(Vector3f(0, 0, -10)),
        center(Vector3f(0, 0, 0)),
        up(Vector3f(0, 1, 0)) {}

    Vector3f pos, center, up;
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

    // scene description
    SimParam param;
    std::vector<Camera> cameras;
    std::vector<Explosion> explosions;
    std::vector<Object *> objects;

private:
    // for parsing
    std::ifstream in;

    void parseSimParam();
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
