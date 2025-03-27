#include "mandelbrot.h"
#include "utils.h"
#include "stdio.h"
#include "assert.h"

// TODO: make constant for black color
// Make

int generatePixelColor(int N)
{
    if (N == 255)
    {
        return 0 | 255; // black
    }
    // if (N>250) printf("N = %d\n", N);
    short r = (N * 100) % 255;
    short g = (N * 10 % 2) * 255;
    short b = (N * 10) % 255 * 128;
    return (r << 24) | (g << 16) | (b << 8) | 255;  // Формат RGBA
}

err_code_t fill_pixels_SISD(int* pixels)    // single instruction single data
{
    assert(pixels);

    double dx = (double) 1/800;
    double dy = (double) 1/600;
    double R_square_max = 10;
    for (int y_screen = 0; y_screen < 600; y_screen++)
    {
        double X0 = -400.f * dx;
        double Y0 = -300.f * dy + dy * y_screen;

        for (int x_screen = 0; x_screen < 800; x_screen++, X0 += dx)
        {
            double X = X0;
            double Y = Y0;
            int N_count = 0;

            for (; N_count < 255; N_count++)
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

            if (N_count > 255)
            {
            printf( "-------begin-------\n"
                "x_screen = %d\n"
                "y_screen = %d\n"
                "dx = %lf\n"
                "X0 = %lf\n"
                "Y0 = %lf\n"
                "X  = %lf\n"
                "Y  = %lf\n"
                "N_count = %d\n"
                "--------end----\n\n",
                x_screen, y_screen, dx, X0, Y0, X, Y, N_count);
            }

            pixels[y_screen * 800 + x_screen] = generatePixelColor(N_count);
        }
    }
    printf("Finished calc\n");

    return OK;
}


err_code_t fill_pixels_SIMD(int* pixels)    // single instruction single data
{
    assert(pixels);

    double dx = (double) 1/800;
    double dy = (double) 1/600;
    double R_square_max = 10;
    for (int y_screen = 0; y_screen < 600; y_screen++)
    {
        double X0 = -400.f * dx;
        double Y0 = -300.f * dy + dy * y_screen;

        for (int x_screen = 0; x_screen < 800; x_screen++, X0 += dx)
        {
            double X = X0;
            double Y = Y0;
            int N_count = 0;

            for (; N_count < 255; N_count++)
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

            pixels[y_screen * 800 + x_screen] = generatePixelColor(N_count);
        }
    }
    printf("Finished calc\n");

    return OK;
}
