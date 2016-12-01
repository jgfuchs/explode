#include "simulation.h"
#include "util.h"

Simulation::Simulation(Scene *sc) : scene(sc) {
    initOpenCL();
    initBuffers();
}

void Simulation::advance() {

}

void Simulation::render(Camera &cam) {

}

void Simulation::initOpenCL() {
    int err;

    cl::Platform platform = cl::Platform::getDefault(&err);
    checkErr(err, "getting default platform");
    std::cout << "OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    cl::Device device = cl::Device::getDefault(&err);
    checkErr(err, "getting default device");
    std::cout<< "OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    ctx = cl::Context(device);
    clq = cl::CommandQueue(ctx, device);

    cl::Program program(ctx, slurpFile("simulate.cl"));
    err = program.build({device});
    checkErr(err, "compiling simulate.cl:\n" + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));

    // TODO: create kernels (and kernel functors?)
}

void Simulation::initBuffers() {
    // TODO: create state buffers on device
}

void Simulation::checkErr(int err, std::string msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error " << err << " while " << msg << std::endl;
        exit(1);
    }
}
