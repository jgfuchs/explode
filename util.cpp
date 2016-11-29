#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <jpeglib.h>

#include "util.h"

// http://www.andrewewhite.net/wordpress/2008/09/02/very-simple-jpeg-writer-in-c-c/
void writeJPEG(char *fname) {

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
