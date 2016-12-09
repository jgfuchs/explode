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
    // fluid flow
    setBounds();

    addForces();
    project();
    advect();
    project();

    // fire & dangerous things
    reaction();

    t += prms.dt;
}

float Simulation::getT() {
    return t;
}

void Simulation::render(HostImage &img) {
    int w = scene->cam.width,
        h = scene->cam.height;

    // render to target image
    kRender.setArg(0, scene->cam);
    kRender.setArg(1, scene->light);
    kRender.setArg(2, U);
    kRender.setArg(3, T);
    kRender.setArg(4, B);
    kRender.setArg(5, target);
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
    kSetBounds = cl::Kernel(program, "set_bounds");
    kRender = cl::Kernel(program, "render");

    int w = prms.grid_w, h = prms.grid_h, d = prms.grid_d;

    // create buffers
    U = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    U_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    T = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    T_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    B = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), w, h, d);

    P = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    P_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    Dvg = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    Dvg_tmp = cl::Image3D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
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
    cl_uint nobjs = scene->objects.size();
    auto objs = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(Object) * nobjs, (void *)scene->objects.data());

    kInitGrid.setArg(0, prms.walls);
    kInitGrid.setArg(1, nobjs);
    kInitGrid.setArg(2, objs);
    kInitGrid.setArg(3, U);
    kInitGrid.setArg(4, T);
    kInitGrid.setArg(5, B);
    enqueueGrid(kInitGrid);
    profile(INIT_GRID);
}

void Simulation::advect() {
    kAdvect.setArg(0, prms.dt);
    kAdvect.setArg(1, U);
    kAdvect.setArg(2, T);
    kAdvect.setArg(3, U_tmp);
    kAdvect.setArg(4, T_tmp);
    enqueueGrid(kAdvect);
    profile(ADVECT);

    std::swap(U, U_tmp);
    std::swap(T, T_tmp);
}

void Simulation::addForces() {
    // compute curl for vorticity confinement
    kCurl.setArg(0, U);
    kCurl.setArg(1, Curl);
    enqueueGrid(kCurl);
    profile(CURL);

    kAddForces.setArg(0, prms.dt);
    kAddForces.setArg(1, U);
    kAddForces.setArg(2, T);
    kAddForces.setArg(3, Curl);
    kAddForces.setArg(4, U_tmp);
    enqueueGrid(kAddForces);
    profile(ADD_FORCES);
    std::swap(U, U_tmp);
}

void Simulation::reaction() {
    static bool done = false;
    cl_float4 p = {0, 0, 0, 0};
    if (t > .1 && !done) {
        done = true;

        p.y = 10;// + randf()*0.5;
        p.x = 32;//+ std::sin(2*t);
        p.z = 32; //std::sin(.8*t) * 12;
        p.w = t;
    }

    kReaction.setArg(0, prms.dt);
    kReaction.setArg(1, T);
    kReaction.setArg(2, T_tmp);
    kReaction.setArg(3, p);
    enqueueGrid(kReaction);
    profile(REACTION);
    std::swap(T, T_tmp);
}

void Simulation::project() {
    // compute Dvg = div(U)
    // (also zeroes out P)
    kDivergence.setArg(0, U);
    kDivergence.setArg(1, Dvg);
    kDivergence.setArg(2, P);
    enqueueGrid(kDivergence);
    profile(DIVERGENCE);

    // solve laplace(P) = div(U) for P
    // this where we spend ~3/4 of GPU time
    const int NITERATIONS = 20;
    for (int i = 0; i < NITERATIONS; i++) {
        kJacobi.setArg(0, P);
        kJacobi.setArg(1, Dvg);
        kJacobi.setArg(2, P_tmp);
        enqueueGrid(kJacobi);
        profile(JACOBI);
        std::swap(P, P_tmp);
    }

    // compute new U' = U - grad(P)
    kProject.setArg(0, U);
    kProject.setArg(1, P);
    kProject.setArg(2, U_tmp);
    enqueueGrid(kProject);
    profile(PROJECT);
    std::swap(U, U_tmp);
}

void Simulation::setBounds() {
    kSetBounds.setArg(0, B);
    kSetBounds.setArg(1, U);
    kSetBounds.setArg(2, T);
    kSetBounds.setArg(3, U_tmp);
    kSetBounds.setArg(4, T_tmp);
    enqueueGrid(kSetBounds);
    profile(SET_BOUNDS);

    std::swap(U, U_tmp);
    std::swap(T, T_tmp);
}

inline void Simulation::enqueueGrid(cl::Kernel kernel) {
    queue.enqueueNDRangeKernel(kernel,
        cl::NullRange,          // 0 offset
        cl::NDRange(prms.grid_w, prms.grid_h, prms.grid_w), // global size
        cl::NDRange(8, 8, 8),   // local (workgroup) size
        NULL, &event);
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
    std::cout << "\n";
    if (profiling) {

        static const std::string kernelNames[_LAST] = { "initGrid", "advect",
            "curl", "addForces", "reaction", "divergence", "jacobi", "project",
            "setBounds", "render"};

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
