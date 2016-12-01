#include "simulation.h"
#include "util.h"
#include "clerror.h"

Simulation::Simulation(Scene *sc) : scene(sc) {
    initOpenCL();
    initBuffers();
}

void Simulation::advance() {

}

void Simulation::render(Camera &cam) {

}

void Simulation::initOpenCL() {
    cl::Platform platform;
    cl::Device device;
    try {
        platform = cl::Platform::getDefault();
        device = cl::Device::getDefault();

        ctx = cl::Context(device);
        clq = cl::CommandQueue(ctx, device);

        std::cout << "OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << "OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    } catch (cl::Error err) {
        handleError(err);
    }

    cl::Program program;
    try {
        program = cl::Program(ctx, slurpFile("simulate.cl"));
        program.build();
    } catch (cl::Error err) {
        std::cerr << "\nOpenCL compilation log:\n" <<
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        handleError(err);
    }

    try {
        cl::Kernel k(program, "initialize");
    } catch (cl::Error err) {
        handleError(err);
    }
}

void Simulation::initBuffers() {
    // TODO: create state buffers on device
}

void Simulation::handleError(cl::Error err) {
    std::cerr << "Error: "  << err.what() << ": " << getCLError(err.err()) << "\n";
    exit(1);
}
