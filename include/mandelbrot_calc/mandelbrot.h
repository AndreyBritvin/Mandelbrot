#ifndef MANDELBROT_H__
#define MANDELBROT_H__

#include "utils.h"

const int BLACK_COLOR_RGBA = 0 | 255;

int generatePixelColor(long long int N);
err_code_t fill_pixels_SISD       (int* pixels, double x_center, double y_center, double scale);
err_code_t fill_pixels_SIMD_manual(int* pixels, double x_center, double y_center, double scale);
err_code_t fill_pixels_SIMD       (int* pixels, double x_center, double y_center, double scale);

#endif // MANDELBROT_H__
