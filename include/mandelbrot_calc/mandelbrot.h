#ifndef MANDELBROT_H__
#define MANDELBROT_H__

#include "utils.h"

#ifndef _WIN32
#define TIME_MEASURE(...)                                               \
        unsigned int lo = 0, hi = 0;                                    \
        __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi));         \
        timestamp_t time_begin = ((timestamp_t)hi << 32) | lo;          \
                                                                        \
        __VA_ARGS__                                                     \
                                                                        \
        __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi));         \
        timestamp_t time_end = ((timestamp_t)hi << 32) | lo;            \
        printf("Measured time is %10llu ticks\n", time_end - time_begin);
#else
#define TIME_MEASURE(...)                                           \
    timestamp_t time_begin = __rdtsc();                             \
                                                                    \
    __VA_ARGS__                                                     \
                                                                    \
    timestamp_t time_end = __rdtsc();                               \
    printf("Measured time is %10llu ticks\n", time_end - time_begin);
#endif


typedef unsigned long long timestamp_t;

const int BLACK_COLOR_RGBA = 0 | 255;

int generatePixelColor(long long int N);
void fill_pixels_SISD       (int* pixels, double x_center, double y_center, double scale);
void fill_pixels_SIMD_manual(int* pixels, double x_center, double y_center, double scale);
void fill_pixels_SIMD       (int* pixels, double x_center, double y_center, double scale);
void fill_pixels_SIMD_multithread (int* pixels, double x_center, double y_center, double scale);
void       fill_pixels_SIMT       (int* pixels, double x_center, double y_center, double scale);

#endif // MANDELBROT_H__
