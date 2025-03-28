#include "mandelbrot.h"
#include "utils.h"
#include "stdio.h"
#include "assert.h"
#include "config.h"

// TODO: make constant for black color
// Make

int generatePixelColor(int N)
{
    if (N == 255)
    {
        return BLACK_COLOR_RGBA; // black
    }
    // if (N>250) printf("N = %d\n", N);
    short r = (N * 100) % 255;
    short g = (N * 10 % 2) * 255;
    short b = (N * 10) % 255 * 128;
    return (r << 24) | (g << 16) | (b << 8) | 255;  // Формат RGBA
}

err_code_t fill_pixels_SISD(int* pixels, double x_center, double y_center, double scale)    // single instruction single data
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

            if (N_count > N_EXIT_COUNT)
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

            pixels[y_screen * WIDTH + x_screen] = generatePixelColor(N_count);
        }
    }
    printf("Finished calc\n");

    return OK;
}

// Custom avx
err_code_t fill_pixels_SIMD_manual(int* pixels, double x_center, double y_center, double scale)    // single instruction multiple data
{
    assert(pixels);

    double R_square_max = 10;
    double dx = (double) 1 / WIDTH * scale;
    double dy = (double) 1 / WIDTH * scale;
    double Y0 = y_center / HEIGHT - scale / 2;

    for (int y_screen = 0; y_screen < HEIGHT; y_screen++, Y0 += dy)
    {
        double X0_initial = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen += 4, X0_initial += dx * 4)
        {
            double  X0[4] ={};
                X0[0] = X0_initial + dx * 0;
                X0[1] = X0_initial + dx * 1;
                X0[2] = X0_initial + dx * 2;
                X0[3] = X0_initial + dx * 3;
            double X[4] = {X0[0], X0[1], X0[2], X0[3]};
            double Y[4] = {Y0, Y0, Y0, Y0};
            int N_counts[4] = {};
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                double X_square[4] = {}; for (int i = 0; i < 4; i++) {X_square[i] = X[i] * X[i];} // X * X;
                double Y_square[4] = {}; for (int i = 0; i < 4; i++) {Y_square[i] = Y[i] * Y[i];} // Y * Y;
                double XY      [4] = {}; for (int i = 0; i < 4; i++) {      XY[i] = X[i] * Y[i];} // X * Y;
                double R_square[4] = {}; for (int i = 0; i < 4; i++) {R_square[i] = X_square[i] + Y_square[i];}
                                                                                 // X_square + Y_square;
                int cmp[4] = {}; for (int i = 0; i < 4; i++) {cmp[i] = (R_square[i] <= R_square_max);}
                int mask = 0;    for (int i = 0; i < 4; i++) {mask  |= cmp[i];}
                if (!mask)
                {
                    break;
                }
                for (int i = 0; i < 4; i++) {N_counts[i] += cmp[i];}
                for (int i = 0; i < 4; i++) {X[i] = X_square[i] - Y_square[i] + X0[i];}
                for (int i = 0; i < 4; i++) {Y[i] = 2 * XY[i]                 + Y0;}
                // X = X_square - Y_square + X0;
                // Y = 2 * XY              + Y0;
            }

            if (N_count > N_EXIT_COUNT)
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

            for (int i = 0; i < 4; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts[i]);}
        }
    }
    printf("Finished calc\n");

    return OK;
}
