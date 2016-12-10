/* -*- C++ -*- */

#ifndef __UTIL_H__
#define __UTIL_H__

#include <iomanip>
#include <iostream>
#include <chrono>

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

typedef std::chrono::high_resolution_clock::time_point TimePoint;

inline TimePoint time_now() {
    return std::chrono::high_resolution_clock::now();
}

inline double time_since(TimePoint &then) {
    auto now = time_now();
    std::chrono::duration<double> dur = now - then;
    return dur.count();
}


#endif // __UTIL_H__
