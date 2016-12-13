#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <random>

#include "util.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

HostImage::HostImage(int w, int h) :
    w(w), h(h),
    data(new char[w*h*4])
{
    // for (int i = 0; i < w*h; i++) {
    //     data[i*4+0] = std::rand() % 255;
    //     data[i*4+1] = std::rand() % 255;
    //     data[i*4+2] = std::rand() % 255;
    //     data[i*4+3] = 255;
    // }
    // fully random: ~650 kB
}

HostImage::~HostImage() {
    delete [] data;
}

void HostImage::write(std::string fname) {
    stbi_write_png(fname.c_str(), w, h, 4, data, 0);
}

std::string slurpFile(std::string fname) {
    std::fstream in(fname);
    if (!in.is_open()) {
        std::cerr << "Error: couldn't open file '" << fname << "'\n";
        exit(1);
    }

    std::stringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

float randf() {
   static std::mt19937 mt(time(NULL));
   static std::uniform_real_distribution<double> dist(-1.0, +1.0);
   return dist(mt);
}
