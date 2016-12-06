#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "scene.h"
#include "simulation.h"

void saveImage(HostImage &img, int idx) {
    std::ostringstream fname;
    fname << std::setfill('0') << std::setw(4);
    fname << "output/frame-" << idx << ".png";
    img.write(fname.str());
}

void printStatus(int i, int n, float t) {
    static const auto spaces = std::string(80, ' ');
    std::cout << "\r" << spaces << "\r"
        << "Rendering: frame " << i+1 << "/" << n << ", t=" << t;
    std::cout.flush();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scene>\n";
        return 1;
    }

    Scene scene(argv[1]);
    Simulation sim(&scene);

    std::cout << std::endl;

    HostImage img(scene.cam.width, scene.cam.height);
    for (int i = 0; i < scene.params.nsteps; i++) {
        sim.advance();
        sim.render(img);
        saveImage(img, i);

        printStatus(i, scene.params.nsteps, sim.getT());
    }

    sim.dumpProfiling();
}
