#include <cuda_runtime.h>
#include "mandelbrot.h"
#include "utils.h"
#include "stdio.h"
#include "assert.h"
#include "config.h"
#include <omp.h>

err_code_t fill_pixels_SIMT(int* pixels, double x_center, double y_center, double scale)    // single instruction single data
{
    assert(pixels);

    double R_square_max = 10;
    double dx = (double) 1 / WIDTH * scale;
    double dy = (double) 1 / WIDTH * scale;
    double Y0 = y_center / HEIGHT - scale / 2;

    for (int y_screen = 0; y_screen < HEIGHT; y_screen++, Y0 += dy)
    {
        double X0 = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen++, X0 += dx)
        {
            double X = X0;
            double Y = Y0;
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                double X_square = X * X;
                double Y_square = Y * Y;
                double XY       = X * Y;
                double R_square = X_square + Y_square;
                if (R_square > R_square_max)
                {
                    break;
                }
                X = X_square - Y_square + X0;
                Y = 2 * XY              + Y0;
            }

            // pixels[y_screen * WIDTH + x_screen] = generatePixelColor(N_count);
        }
    }

    return OK;
}