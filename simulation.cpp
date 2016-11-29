#include "simulation.h"
#include "util.h"

Simulation::Simulation(Scene *sc) : scene(sc) {
    initOpenCL();
}

void Simulation::advance() {

}

void Simulation::render(Camera &cam) {

}

void Simulation::initOpenCL() {
    int err;
    cl::Platform platform = cl::Platform::getDefault(&err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: couldn't get default OpenCL platform\n";
        exit(1);
    }
    std::cout << "OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0) {
        std::cerr << "Error: no GPU devices found\n";
        exit(1);
    }
    cl::Device device = devices[0];
    std::cout<< "OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Program program(context, slurpFile("simulate.cl"));
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "\nError compiling simulate.cl:\n"
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(1);
    }
}
