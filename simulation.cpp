#include "simulation.h"
#include "clerror.h"
#include <iomanip>

Simulation::Simulation(Scene *sc) : scene(sc), prms(sc->params), t(0.0)
{
    try {
        initOpenCL();
    } catch (cl::Error err) {
        std::cerr << "OpenCL error: "  << err.what() << ": " << getCLError(err.err()) << "\n";
        exit(1);
    }

    initProfiling();
    initGrid();
    reaction();
}

void Simulation::advance() {
    addForces();
    // project();
    advect(U, U_tmp);
    project();

    advect(T, T_tmp);

    t += prms.dt;
}

float Simulation::getT() {
    return t;
}

void Simulation::render(HostImage &img) {
    int w = scene->cam.width,
        h = scene->cam.height;

    // render to target image
    kRender.setArg(0, U);
    kRender.setArg(1, T);
    kRender.setArg(2, target);
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
    kAddForces = cl::Kernel(program, "add_forces");
    kReaction = cl::Kernel(program, "reaction");
    kDivergence = cl::Kernel(program, "divergence");
    kJacobi = cl::Kernel(program, "jacobi");
    kProject = cl::Kernel(program, "project");
    kRender = cl::Kernel(program, "render");

    int w = prms.grid_w, h = prms.grid_h, d = prms.grid_d;
    gridRange = cl::NDRange(w, h, d);
    groupRange = cl::NDRange(8, 8, 8);

    // create buffers
    U = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    U_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    T = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    T_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    B = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), w, h, d);

    Dvg = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
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
    kInitGrid.setArg(1, T);
    kInitGrid.setArg(2, B);
    queue.enqueueNDRangeKernel(kInitGrid, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
}

void Simulation::advect(cl::Image3D &in, cl::Image3D &out) {
    kAdvect.setArg(0, prms.dt);
    kAdvect.setArg(1, U);
    kAdvect.setArg(2, in);
    kAdvect.setArg(3, out);
    queue.enqueueNDRangeKernel(kAdvect, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(ADVECT);
    std::swap(in, out);
}

void Simulation::addForces() {
    kAddForces.setArg(0, prms.dt);
    kAddForces.setArg(1, U);
    kAddForces.setArg(2, T);
    kAddForces.setArg(3, U_tmp);
    queue.enqueueNDRangeKernel(kAddForces, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(ADD_FORCES);
    std::swap(U, U_tmp);
}

void Simulation::reaction() {
    kReaction.setArg(0, prms.dt);
    kReaction.setArg(1, T);
    kReaction.setArg(2, T_tmp);
    queue.enqueueNDRangeKernel(kReaction, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(REACTION);
    std::swap(T, T_tmp);
}

void Simulation::project() {
    // compute Dvg = div(U)
    // (also zeroes out P)
    kDivergence.setArg(0, U);
    kDivergence.setArg(1, Dvg);
    kDivergence.setArg(2, P);
    queue.enqueueNDRangeKernel(kDivergence, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(DIVERGENCE);

    // solve laplace(P) = div(U) for P
    // this where we spend ~80% of GPU time
    const int NITERATIONS = 20;
    for (int i = 0; i < NITERATIONS; i++) {
        kJacobi.setArg(0, P);
        kJacobi.setArg(1, Dvg);
        kJacobi.setArg(2, P_tmp);
        queue.enqueueNDRangeKernel(kJacobi, cl::NullRange, gridRange, groupRange, NULL, &event);
        event.wait();
        profile(JACOBI);
        std::swap(P, P_tmp);
    }

    // compute new U' = U - grad(P)
    kProject.setArg(0, U);
    kProject.setArg(1, P);
    kProject.setArg(2, U_tmp);
    queue.enqueueNDRangeKernel(kProject, cl::NullRange, gridRange, groupRange, NULL, &event);
    event.wait();
    profile(PROJECT);
    std::swap(U, U_tmp);
}

void Simulation::profile(int pk) {
    cl_ulong t2 = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t3 = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    kernelTimes[pk] += (t3 - t2) * 1e-9;
    kernelCalls[pk]++;
}

void Simulation::dumpProfiling() {
    int sumC = 0;
    double sumT = 0;

    std::cout << "\n\nProfiling info:\n";
    std::cout << "Kernel\tCalls\tTime (s)\tMean (ms)\n" << std::fixed;
    for (int i = 0; i < _LAST; i++) {
        unsigned c = kernelCalls[i];
        double t = kernelTimes[i];
        double avg = (c ? (t / c) : 0.0) * 1e3;
        sumC += c;
        sumT += t;

        std::cout << "  [" << i << "]\t" << c << "\t" << t << "\t" << avg << "\n";
    }
    std::cout << "Total:\t" << sumC << "\t" << sumT << "\n";
}
