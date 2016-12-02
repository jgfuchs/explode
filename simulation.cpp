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
    advect(U, U_tmp);
    advect(P, P_tmp);
    advect(T, T_tmp);
}

void Simulation::render(HostImage &img) {
    int w = scene->cam.width,
        h = scene->cam.height;

    // render to target image
    kRender.setArg(0, U);
    kRender.setArg(1, P);
    kRender.setArg(2, T);
    kRender.setArg(3, target);
    queue.enqueueNDRangeKernel(kRender, cl::NullRange, cl::NDRange(w, h),
            cl::NDRange(16, 16), NULL, &event);
    event.wait();
    profile(RENDER);

    // read rendered image into host memory
    cl::size_t<3> origin;
    cl::size_t<3> region;
    region[0] = w;
    region[1] = h;
    region[2] = 1;
    queue.enqueueReadImage(target, true, origin, region, 0, 0, img.data, NULL, &event);
    event.wait();
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
    kAdvect = cl::Kernel(program, "advect");
    kRender = cl::Kernel(program, "render");

    int w = prms.grid_w, h = prms.grid_h, d = prms.grid_d;
    gridRange = cl::NDRange(w, h, d);
    groupRange = cl::NDRange(8, 8, 8);

    // create buffers
    U = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    U_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);

    P = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    P_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);

    T = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    T_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);

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
    kInitGrid.setArg(2, T);
    queue.enqueueNDRangeKernel(kInitGrid, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(INIT_GRID);
}

void Simulation::advect(cl::Image3D &in, cl::Image3D &out) {
    kAdvect.setArg(0, prms.dt);
    kAdvect.setArg(1, prms.cellSize);
    kAdvect.setArg(2, U);
    kAdvect.setArg(3, in);
    kAdvect.setArg(4, out);
    queue.enqueueNDRangeKernel(kAdvect, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(ADVECT);
}

void Simulation::project() {

    const int NITERATIONS = 1;
    for (int i = 0; i < NITERATIONS; i++) {

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
