#include "simulation.h"
#include "clerror.h"
#include <iomanip>

Simulation::Simulation(Scene *sc) : scene(sc), prms(sc->params)
{
    try {
        initOpenCL();
    } catch (cl::Error err) {
        std::cerr << "OpenCL error: "  << err.what() << ": " << getCLError(err.err()) << "\n";
        exit(1);
    }

    initProfiling();
    initGrid();
}

void Simulation::advance() {
    try {
        advectField(U, U, U_tmp);
        advectField(U, U, U_tmp);
        advectField(U, U, U_tmp);
        advectField(U, U, U_tmp);
        advectField(U, U, U_tmp);
        advectField(U, U, U_tmp);
    } catch (cl::Error err) {
        std::cerr << "OpenCL error: "  << err.what() << ": " << getCLError(err.err()) << "\n";
        exit(1);
    }
}

void Simulation::render(HostImage &img) {

}

void Simulation::initOpenCL() {
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    std::cout << "OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // read & compile simulation program
    program = cl::Program(context, slurpFile("simulate.cl"));
    try {
        program.build();
    } catch (cl::Error err) {
        std::cerr << "\nOpenCL compilation log:\n" <<
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        throw err;
    }

    // load kernels from the program
    kInitGrid = cl::Kernel(program, "init_grid");
    kAdvectField = cl::Kernel(program, "advect_field");
    kAdvectScalar = cl::Kernel(program, "advect_scalar");

    int w = prms.grid_w, h = prms.grid_h, d = prms.grid_d;
    gridRange = cl::NDRange(w, h, d);
    groupRange = cl::NDRange(8, 8, 8);

    // create buffers
    U = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    U_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);

    P = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    P_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);

    // create render target
    target = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
        scene->cam.width, scene->cam.height);
}

void Simulation::initProfiling() {
    for (int i = 0; i < _LAST; i++) {
        kernelTimes[i] = 0.0f;
        kernelCalls[i] = 0;
    }
}

void Simulation::initGrid() {
    kInitGrid.setArg(0, U);
    kInitGrid.setArg(1, P);
    queue.enqueueNDRangeKernel(kInitGrid, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(INIT_GRID);
}

void Simulation::advectField(cl::Image3D &F, cl::Image3D &V, cl::Image3D &V_out) {
    kAdvectField.setArg(0, prms.dt);
    kAdvectField.setArg(1, F);
    kAdvectField.setArg(2, V);
    kAdvectField.setArg(3, V_out);
    queue.enqueueNDRangeKernel(kAdvectField, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(ADVECT_FIELD);
}

void Simulation::advectScalar(cl::Image3D &F, cl::Image3D &S, cl::Image3D &S_out) {
    kAdvectScalar.setArg(0, prms.dt);
    kAdvectScalar.setArg(1, F);
    kAdvectScalar.setArg(2, S);
    kAdvectScalar.setArg(3, S_out);
    queue.enqueueNDRangeKernel(kAdvectScalar, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(ADVECT_SCALAR);
}

void Simulation::project() {

    const int NITERATIONS = 1;
    for (int i = 0; i < NITERATIONS; i++) {
        queue.enqueueNDRangeKernel(kAdvectScalar, cl::NullRange, gridRange, groupRange, NULL, &event);
        event.wait();
        profile(PROJECT);
    }
}

void Simulation::profile(int pk) {
    cl_ulong t2 = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t3 = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    kernelTimes[pk] += (t3 - t2) * 1e-9;
    kernelCalls[pk]++;
}

void Simulation::dumpProfiling() {
    std::cout << "\nProfiling info\n"
        << "Kernel\tCalls\tTime (s)\tMean (ms)\n";
    for (int i = 0; i < _LAST; i++) {
        double t = kernelTimes[i];
        unsigned c = kernelCalls[i];
        std::cout << "  [" << i << "]\t"
            << c << "\t" << t << "\t"
            << (c ? (t / c) : 0.0) * 1e3 << "\n";
    }
}
