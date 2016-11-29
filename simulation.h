/* -*- C++ -*- */

#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include <CL/cl.hpp>

#include "scene.h"

class Simulation {
public:
    Simulation(Scene *sc);

    void advance();
    void render(Camera &cam);
private:
    Scene *scene;

    cl::Buffer grid_buf;

    void initOpenCL();
};

#endif // __SIMULATION_H__
