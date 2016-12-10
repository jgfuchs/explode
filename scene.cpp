#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "scene.h"

Scene::Scene(char *fname) : in(fname) {
    if (!in.is_open()) {
        std::cerr << "Error: couldn't open scene file '" << fname << "'\n";
        exit(1);
    }

    while (!in.eof()) {
        auto tok = getToken();
        if (tok == "SimParam") {
            parseSimParams();
        } else if (tok == "Camera") {
            parseCamera();
        } else if (tok == "Explosion") {
            parseExplosion();
        } else if (tok == "Light") {
            parseLight();
        } else if (tok == "Object") {
            parseObject();
        } else if (tok == "") {
            // do nothing, not sure why this ever occurs
        } else {
            std::cerr << "Error: unexpected token '" << tok << "'\n";
            exit(1);
        }
    }

    // "null" object
    objects.push_back({{-1,-1,-1}, {-1, -1, -1}});

    // sort explosions so that popping returns earliest
    std::sort(explosions.begin(), explosions.end(),
        [](Explosion a, Explosion b) { return a.t0 > b.t0;});
}

void Scene::parseSimParams() {
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "grid") {
            int n = getInt();
            if (n % 8) {
                std::cerr << "Error: grid size must be multiple of 8\n";
                exit(1);
            }
            params.grid_n = n;
        } else if (tok == "dt") {
            params.dt = getFloat();
        } else if (tok == "nsteps") {
            params.nsteps = getInt();
        } else if (tok == "walls") {
            params.walls = getInt();
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in SimParam\n";
            exit(1);
        }
    }
}

void Scene::parseCamera() {
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "pos") {
            cam.pos = getFloat3();
        } else if (tok == "size") {
            unsigned x = getInt(),
                     y = getInt();
            if (x % 16 || y % 16) {
                std::cerr << "Error: image dimensions must be multiple of 16\n";
                exit(1);
            }
            cam.size = {x, y};
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in Camera\n";
            exit(1);
        }
    }
}

void Scene::parseLight() {
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "pos") {
            light.pos = getFloat3();
        } else if (tok == "I") {
            light.intensity = getFloat();
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in Light\n";
            exit(1);
        }
    }
}

void Scene::parseExplosion() {
    Explosion ex;
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "pos") {
            ex.pos = getFloat3();
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
    // if (explosions.size()) {
    //     std::cerr << "Error: only one explosion supported\n";
    //     exit(1);
    // }
    explosions.push_back(ex);
}

void Scene::parseObject() {
    Object obj;
    expect("{");
    while (true) {
        auto tok = getToken();
        if (tok == "pos") {
            obj.pos = getFloat3();
        } else if (tok == "dim") {
            obj.dim = getFloat3();
        } else if (tok == "}") {
            break;
        } else {
            std::cerr << "Error: unexpected token '" << tok << "' in Object\n";
            exit(1);
        }
    }
    objects.push_back(obj);
}

std::string Scene::getToken() {
    std::string s;
    in >> s;
    while (s[0] == '#') {
        std::string tmp;
        std::getline(in, tmp);
        s.erase();
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

cl_float3 Scene::getFloat3() {
    float f1 = getFloat(),
          f2 = getFloat(),
          f3 = getFloat();
    return {f1, f2, f3};
}
