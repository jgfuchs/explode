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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scene>\n";
        return 1;
    }

    Scene scene(argv[1]);
    Simulation sim(&scene);

    HostImage img(scene.cam.width, scene.cam.height);
    for (int i = 0; i < scene.params.nsteps; i++) {
        sim.advance();
        sim.render(img);
        saveImage(img, i);

        // break;
    }
    sim.dumpProfiling();
}
