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

    void dumpProfiling();

private:
    // initialization
    void initOpenCL();
    void initProfiling();
    void initGrid();

    // fluid dynamics
    void advect(cl::Image3D&, cl::Image3D&);
    void divergence();
    void jacobi();
    void project();

    // helper functions
    void profile(int pk);

    const Scene *scene;
    const SimParams prms;
    float t;

    // OpenCL management
    cl::Program program;
    cl::Context context;
    cl::CommandQueue queue;

    cl::Kernel kInitGrid, kAdvect, kDivergence, kJacobi, kProject, kRender;
    cl::NDRange gridRange, groupRange;

    // state variables
    cl::Image3D U, U_tmp,       // velocity vector field
                T, T_tmp,       // thermo: (temperature, fuel, smoke)
                B;              // boundaries

    // intermediates
    cl::Image3D Dvg,            // divergence
                P, P_tmp;       // pressure

    cl::Image2D target;         // render target

    // profiling
    enum {INIT_GRID, ADVECT, DIVERGENCE, ZERO_P, JACOBI, PROJECT, RENDER, _LAST};
    double kernelTimes[_LAST];
    unsigned kernelCalls[_LAST];
    cl::Event event;
};

#endif // __SIMULATION_H__
