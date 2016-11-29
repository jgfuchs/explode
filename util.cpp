#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <jpeglib.h>

#include "util.h"

std::string slurpFile(char *fname) {
    std::fstream in(fname);
    if (!in.is_open()) {
        std::cerr << "Error: couldn't open file '" << fname << "'\n";
        exit(1);
    }

    std::stringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}
