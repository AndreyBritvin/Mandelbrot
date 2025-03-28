#ifndef MANDELBROT_H__
#define MANDELBROT_H__

#include "utils.h"

int generatePixelColor(int N);
err_code_t fill_pixels_SISD(int* pixels, double x_center, double y_center, double scale);
err_code_t fill_pixels_SIMD(int* pixels);

#endif // MANDELBROT_H__
