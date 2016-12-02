/* -*- C++ -*- */

#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "scene.h"
#include "util.h"

class Simulation {
public:
    Simulation(Scene *sc);

    void advance();

    void render(HostImage &img);

private:
    Scene *scene;

    cl::Program program;
    cl::Context ctx;
    cl::CommandQueue clq;

    cl::Kernel kInit, kAdvectField, kAdvectScalar, kProject;
    cl::Image3D U, U_tmp,       // velocity vector field
                P, P_tmp;       // pressure scalar field

    // render targets
    cl::Image2D target;

    void initOpenCL();

    void initState();
    void advectField(cl::Image3D, cl::Image3D, cl::Image3D);
    void advectScalar(cl::Image3D, cl::Image3D, cl::Image3D);
    void project();
};

#endif // __SIMULATION_H__
