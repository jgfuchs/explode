#include <iostream>

#include "simulation.h"
#include "clerror.h"

Simulation::Simulation(Scene *sc, bool prof) :
    scene(sc), prms(sc->params), profiling(prof), t(0.0)
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
    addForces();

    project();
    advect(U, U_tmp);
    project();

    reaction();
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
    profile(RENDER);

    // read rendered image into host memory
    cl::size_t<3> origin;
    cl::size_t<3> region;
    region[0] = w;
    region[1] = h;
    region[2] = 1;
    queue.enqueueReadImage(target, true, origin, region, 0, 0, img.data);
}

void Simulation::initOpenCL() {
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    std::cout << "OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, profiling ? CL_QUEUE_PROFILING_ENABLE : 0);

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
    kCurl = cl::Kernel(program, "curl");
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

    P = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    P_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    Dvg = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    Curl = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);

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
    queue.enqueueNDRangeKernel(kInitGrid, cl::NullRange, gridRange, groupRange);
}

void Simulation::advect(cl::Image3D &in, cl::Image3D &out) {
    kAdvect.setArg(0, prms.dt);
    kAdvect.setArg(1, U);
    kAdvect.setArg(2, in);
    kAdvect.setArg(3, out);
    queue.enqueueNDRangeKernel(kAdvect, cl::NullRange, gridRange, groupRange, NULL, &event);
    profile(ADVECT);
    std::swap(in, out);
}

void Simulation::addForces() {
    // compute curl for vorticity confinement
    kCurl.setArg(0, U);
    kCurl.setArg(1, Curl);
    queue.enqueueNDRangeKernel(kCurl, cl::NullRange, gridRange, groupRange, NULL, &event);
    profile(CURL);

    kAddForces.setArg(0, prms.dt);
    kAddForces.setArg(1, U);
    kAddForces.setArg(2, T);
    kAddForces.setArg(3, Curl);
    kAddForces.setArg(4, U_tmp);
    queue.enqueueNDRangeKernel(kAddForces, cl::NullRange, gridRange, groupRange, NULL, &event);
    profile(ADD_FORCES);
    std::swap(U, U_tmp);
}

void Simulation::reaction() {
    cl_float3 p;
    p.y = 8;
    p.x = 32 + std::cos(.8*t) * 12;
    p.z = 32 + std::sin(.8*t) * 12;

    kReaction.setArg(0, prms.dt);
    kReaction.setArg(1, T);
    kReaction.setArg(2, T_tmp);
    kReaction.setArg(3, p);
    queue.enqueueNDRangeKernel(kReaction, cl::NullRange, gridRange, groupRange, NULL, &event);
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
    profile(DIVERGENCE);

    // solve laplace(P) = div(U) for P
    // this where we spend ~3/4 of GPU time
    const int NITERATIONS = 20;
    for (int i = 0; i < NITERATIONS; i++) {
        kJacobi.setArg(0, P);
        kJacobi.setArg(1, Dvg);
        kJacobi.setArg(2, P_tmp);
        queue.enqueueNDRangeKernel(kJacobi, cl::NullRange, gridRange, groupRange, NULL, &event);
        profile(JACOBI);
        std::swap(P, P_tmp);
    }

    // compute new U' = U - grad(P)
    kProject.setArg(0, U);
    kProject.setArg(1, P);
    kProject.setArg(2, U_tmp);
    queue.enqueueNDRangeKernel(kProject, cl::NullRange, gridRange, groupRange, NULL, &event);
    profile(PROJECT);
    std::swap(U, U_tmp);
}

void Simulation::profile(int pk) {
    if (profiling) {
        event.wait();

        cl_ulong t2 = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong t3 = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        kernelTimes[pk] += (t3 - t2) * 1e-9;
        kernelCalls[pk]++;
    }
}

void Simulation::dumpProfiling() {
    if (profiling) {
        static const std::string kernelNames[_LAST] = {"advect", "curl",
            "addForces", "reaction", "divergence", "jacobi", "project", "render"};

        std::cout << "\n\nProfiling info:\n";
        printl("Kernel");
        printr("Calls", 8);
        printr("Time (s)");
        printr("Mean (ms)");
        std::cout << std::setprecision(3) << std::fixed << std::endl;

        int sumC = 0;
        double sumT = 0;
        for (int i = 0; i < _LAST; i++) {
            unsigned c = kernelCalls[i];
            double t = kernelTimes[i];
            double avg = (c ? (t / c) : 0.0) * 1e3;
            sumC += c;
            sumT += t;

            printl(' ' + kernelNames[i]);
            printr(c, 8);
            printr(t);
            printr(avg);
            std::cout << std::endl;
        }
        printl("Total:");
        printr(sumC, 8);
        printr(sumT);
        std::cout << "\n";
    }
}
