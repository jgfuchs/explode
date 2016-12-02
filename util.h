/* -*- C++ -*- */

#ifndef __UTIL_H__
#define __UTIL_H__


// an image in host memory
class HostImage {
public:
    HostImage(int w, int h);
    ~HostImage();
    void write(std::string fname);

    int w, h;
    char *data;
};

std::string slurpFile(std::string fname);

#endif // __UTIL_H__
