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

    cl::Context ctx;
    cl::CommandQueue clq;

    void initOpenCL();
    void createBuffers();
    
    void checkErr(int err, std::string msg);
};

#endif // __SIMULATION_H__
