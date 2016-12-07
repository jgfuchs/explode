/* -*- C++ -*- */

#ifndef __UTIL_H__
#define __UTIL_H__

#include <iomanip>
#include <iostream>

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

template<typename T>
inline void printr(T t, const int w=11) {
    std::cout << std::right << std::setw(w) << t;
}

template<typename T>
inline void printl(T t, const int w=11) {
    std::cout << std::left << std::setw(w) << t;
}

float randf();

#endif // __UTIL_H__
