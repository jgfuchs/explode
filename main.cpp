#include <iostream>
#include <fstream>

#include "scene.h"
#include "simulation.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scene>";
        return 1;
    }

    Scene scene(argv[1]);
    Simulation sim(&scene);

    return 0;

    for (int i = 0; i < scene.param.steps; i++) {
        sim.advance();
        for (auto &cam : scene.cameras) {
            sim.render(cam);
        }
    }
}
