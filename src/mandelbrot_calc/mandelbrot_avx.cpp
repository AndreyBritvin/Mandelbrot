#include "mandelbrot.h"
#include "utils.h"
#include "stdio.h"
#include "assert.h"
#include "config.h"
#include <immintrin.h>
#include <xmmintrin.h>
#include <omp.h>
#include <math.h>

int generatePixelColor(long long int N)
{
    if (N == 255)
    {
        return BLACK_COLOR_RGBA; // black
    }
    // if (N>250) printf("N = %d\n", N);
    // short r = (N % 9 * 100) % 255;
    // short g = (N * 10 % 2) * 255;
    // short b = (N * 10) % 255 * 128;
    short r = (short)(255 * pow(sin(0.16 * N), 2));
    short g = (short)(255 * pow(log(log(0.16 * N + 2)), 2));
    short b = (short)(255 * pow(log(0.16 * N + 4), 2));
    return (r << 24) | (g << 16) | (b << 8) | 255;  // Формат RGBA
}

void fill_pixels_SISD(int* pixels, double x_center, double y_center, double scale)    // single instruction single data
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

    return ;
}

// Custom avx
void fill_pixels_SIMD_manual(int* pixels, double x_center, double y_center, double scale)    // single instruction multiple data
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

            for (int i = 0; i < 4; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts[i]);}
        }
    }
    printf("Finished calc\n");

    return ;
}

#ifndef _WIN32
// Real avx
void fill_pixels_SIMD(int* pixels, double x_center, double y_center, double scale)    // single instruction multiple data
{
    assert(pixels);

    double R_square_max = 10;
    double dx = (double) 1 / WIDTH * scale;
    double dy = (double) 1 / WIDTH * scale;
    __m256d Y0 = _mm256_set1_pd(y_center / HEIGHT - scale / 2);
    alignas(32) long long int N_counts_total[4] = {}; // alignas - 32 bytes aligning for _mm256_store_si256

    for (int y_screen = 0; y_screen < HEIGHT; y_screen++, Y0 = _mm256_add_pd(Y0, _mm256_set1_pd(dy)))
    {
        double X0_initial = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen += 4, X0_initial += dx * 4)
        {
            __m256d X0 = _mm256_add_pd(_mm256_set1_pd(X0_initial),
                                       _mm256_mul_pd (_mm256_set1_pd(dx), _mm256_set_pd(3, 2, 1, 0)));
            __m256d X = X0;
            __m256d Y = Y0;
            __m256i N_counts = _mm256_setzero_si256();
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                __m256d X_square = _mm256_mul_pd(X, X);
                __m256d Y_square = _mm256_mul_pd(Y, Y);
                __m256d XY       = _mm256_mul_pd(X, Y);
                __m256d R_square = _mm256_add_pd(X_square, Y_square);
                __m256d cmp      = _mm256_cmp_pd(R_square, _mm256_set1_pd(R_square_max), _CMP_LE_OQ);
                long long int mask = _mm256_movemask_pd(cmp);
                if (!mask)
                {
                    break;
                }
                N_counts = _mm256_sub_epi64(N_counts, _mm256_castpd_si256(cmp)); // sub because cmp_pd returns 0xFFFF... on true, which is equal to -1
                X = _mm256_add_pd(_mm256_sub_pd(X_square, Y_square),   X0);
                Y = _mm256_add_pd(_mm256_add_pd(XY, XY), Y0);
            }
            _mm256_store_si256((__m256i *)&N_counts_total, N_counts);
            for (int i = 0; i < 4; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts_total[i]);}
        }
    }

    printf("Finished calc\n");

    return ;
}


// Real avx
void fill_pixels_SIMD_multithread(int* pixels, double x_center, double y_center, double scale)    // single instruction multiple data
{
    assert(pixels);
    #pragma omp parallel for
    for (int y_screen = 0; y_screen < HEIGHT; y_screen++)
    {
    double R_square_max = 10;
    double dx = (double) 1 / WIDTH * scale;
    double dy = (double) 1 / WIDTH * scale;
    __m256d Y0 = _mm256_set1_pd(y_center / HEIGHT - scale / 3.5 + dy * y_screen);
    alignas(32) long long int N_counts_total[4] = {}; // alignas - 32 bytes aligning for _mm256_store_si256
        double X0_initial = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen += 4, X0_initial += dx * 4)
        {
            __m256d X0 = _mm256_add_pd(_mm256_set1_pd(X0_initial),
                                       _mm256_mul_pd (_mm256_set1_pd(dx), _mm256_set_pd(3, 2, 1, 0)));
            __m256d X = X0;
            __m256d Y = Y0;
            __m256i N_counts = _mm256_setzero_si256();
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                __m256d X_square = _mm256_mul_pd(X, X);
                __m256d Y_square = _mm256_mul_pd(Y, Y);
                __m256d XY       = _mm256_mul_pd(X, Y);
                __m256d R_square = _mm256_add_pd(X_square, Y_square);
                __m256d cmp      = _mm256_cmp_pd(R_square, _mm256_set1_pd(R_square_max), _CMP_LE_OQ);
                long long int mask = _mm256_movemask_pd(cmp);
                if (!mask)
                {
                    break;
                }
                N_counts = _mm256_sub_epi64(N_counts, _mm256_castpd_si256(cmp)); // sub because cmp_pd returns 0xFFFF... on true, which is equal to -1
                X = _mm256_add_pd(_mm256_sub_pd(X_square, Y_square),   X0);
                Y = _mm256_add_pd(_mm256_add_pd(XY, XY), Y0);
            }
            _mm256_store_si256((__m256i *)&N_counts_total, N_counts);
            // #pragma omp critical
            {
            for (int i = 0; i < 4; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts_total[i]);}
            }
        }

        // Y0 = _mm256_add_pd(Y0, _mm256_set1_pd(dy));
    }

    // #pragma omp barrier

    printf("Finished calc, %d\n", omp_get_max_threads());

    return ;
}
#endif
