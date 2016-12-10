#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "scene.h"
#include "simulation.h"

void saveImage(HostImage &img, int idx) {
    std::stringstream fname;
    fname << "output/frame-" << std::setfill('0') << std::setw(4) << idx << ".png";
    img.write(fname.str());
}

void printStatus(int i, int n, float t) {
    const auto spaces = std::string(80, ' ');
    static bool first = true;
    if (first) {
        std::cout << std::endl << std::setprecision(2) << std::fixed;
        first = false;
    }

    std::cout << "\r" << spaces << "\r"
        << "Simulating: frame " << i+1 << "/" << n << ", t=" << t << "   ";
    std::cout.flush();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scene>\n";
        return 1;
    }

    Scene scene(argv[1]);
    Simulation sim(&scene, true);

    HostImage img(scene.cam.size.x, scene.cam.size.y);
    for (int i = 0; i < scene.params.nsteps; i++) {
        sim.advance();
        sim.render(img);
        saveImage(img, i);

        printStatus(i, scene.params.nsteps, sim.getT());
    }

    sim.dumpProfiling();
}
