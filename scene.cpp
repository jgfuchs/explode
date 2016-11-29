#include <fstream>
#include <iostream>
#include <sstream>

#include "scene.h"

Scene::Scene(char *fname) : in(fname) {
    if (!in.is_open()) {
        std::cerr << "Error: couldn't open scene file '" << fname << "'\n";
        exit(1);
    }

    while (!in.eof()) {
        auto tok = getToken();
        if (tok == "SimParams") {
            parseSimParams();
        } else if (tok == "Camera") {
            parseCamera();
        } else if (tok == "Object") {
            parseObject();
        } else if (tok == "Explosion") {
            parseExplosion();
        } else if (tok == "") {
            // do nothing, not sure why this happens
        } else {
            std::cerr << "Error: unexpected token '" << tok << "'\n";
            exit(1);
        }
    }

    debugInfo();
}

void Scene::parseSimParams() {
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "grid") {
            params.grid = getInt();
        } else if (tok == "particles") {
            params.particles = getInt();
        } else if (tok == "timestep") {
            params.timestep = getFloat();
        } else if (tok == "steps") {
            params.steps = getInt();
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in SimParams\n";
            exit(1);
        }
    }
}

void Scene::parseCamera() {
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "pos") {
            cam.pos = getVector3f();
        } else if (tok == "center") {
            cam.center = getVector3f();
        } else if (tok == "up") {
            cam.up = getVector3f();
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in Camera\n";
            exit(1);
        }
    }
}

void Scene::parseObject() {
    expect("{");

    expect("}");
}

void Scene::parseExplosion() {
    Explosion ex;

    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "pos") {
            ex.pos = getVector3f();
        } else if (tok == "size") {
            ex.size = getFloat();
        } else if (tok == "t0") {
            ex.t0 = getFloat();
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in Explosion\n";
            exit(1);
        }
    }

    if (explosions.size()) {
        std::cerr << "Error: only one explosion per scene supported\n";
        exit(1);
    }
    explosions.push_back(ex);
}

std::string Scene::getToken() {
    std::string s;
    in >> s;
    if (s.size() && s[0] == '#') {
        std::string tmp;
        std::getline(in, tmp);
        in >> s;
    }
    return s;
}

void Scene::expect(std::string s) {
    auto tok = getToken();
    if (tok != s) {
        std::cerr << "Error: expected '" << s << "' but got '" << tok << "'\n";
        exit(1);
    }
}

int Scene::getInt() {
    auto tok = getToken();
    std::istringstream iss(tok);
    int i;
    iss >> i;
    if (iss.fail() || !iss.eof()) {
        std::cerr << "Error: expected int but got '" << tok << "'\n";
        exit(1);
    }
    return i;
}

float Scene::getFloat() {
    auto tok = getToken();
    std::istringstream iss(tok);
    float f;
    iss >> f;
    if (iss.fail() || !iss.eof()) {
        std::cerr << "Error: expected float but got '" << tok << "'\n";
        exit(1);
    }
    return f;
}

Vector3f Scene::getVector3f() {
    float f1 = getFloat(),
          f2 = getFloat(),
          f3 = getFloat();
    return Vector3f(f1, f2, f3);
}

void Scene::debugInfo() {
    std::cout << "Scene description:\n";

    std::cout << "  simulation: "
        << "grid=" << params.grid << " "
        << "particles=" << params.particles << " "
        << "steps=" << params.steps << " "
        << "timestep=" << params.timestep << "\n";

    std::cout << "  camera: "
        << "pos=" << cam.pos << " "
        << "center=" << cam.center << " "
        << "up=" << cam.up << "\n";

    std::cout << "  explosions:\n";
    for (int i = 0; i < explosions.size(); i++) {
        auto ex = explosions[i];
        std::cout << "    [" << i << "]  "
            << "pos=" << ex.pos << " "
            << "size=" << ex.size << " "
            << "t0=" << ex.t0 << "\n";
    }

    std::cout << "  objects:\n";
    for (int i = 0; i < objects.size(); i++) {
        auto obj = objects[i];
    }
}
