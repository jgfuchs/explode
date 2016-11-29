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

    for (int i = 0; i < scene.params.steps; i++) {
        sim.advance();
        sim.render();
    }
    
    return 0;
}
