#include "simulation.h"
#include "clerror.h"

Simulation::Simulation(Scene *sc) : scene(sc) {
    try {
        initOpenCL();
    } catch (cl::Error err) {
        std::cerr << "OpenCL error: "  << err.what() << ": " << getCLError(err.err()) << "\n";
        exit(1);
    }

    // int x;
    // std::cin >> x;
}

void Simulation::advance() {

}

void Simulation::render(HostImage &img) {

}

void Simulation::initOpenCL() {
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    std::cout << "OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    ctx = cl::Context(device);
    clq = cl::CommandQueue(ctx, device);

    // read & compile simulation program
    program = cl::Program(ctx, slurpFile("simulate.cl"));
    try {
        program.build();
    } catch (cl::Error err) {
        std::cerr << "\nOpenCL compilation log:\n" <<
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        throw err;
    }

    // load kernels from the program
    kInit = cl::Kernel(program, "initialize");
    kAdvectField = cl::Kernel(program, "advect_field");
    kAdvectScalar = cl::Kernel(program, "advect_scalar");

    // create buffers
    int w = scene->grid_w, h = scene->grid_h, d = scene->grid_d;

    U = cl::Image3D(ctx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);
    U_tmp = cl::Image3D(ctx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), w, h, d);

    P = cl::Image3D(ctx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);
    P_tmp = cl::Image3D(ctx, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h, d);

    // create render target
    target = cl::Image2D(ctx, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
        scene->cam.width, scene->cam.height);
}
