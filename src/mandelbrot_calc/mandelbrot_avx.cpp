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
    if (N == N_EXIT_COUNT)
    {
        return BLACK_COLOR_RGBA; // black
    }
    // if (N>250) printf("N = %d\n", N);
    short r = (N * 100) % 255;
    short g = (N * 10 % 2) * 255;
    short b = (N * 10) % 255 * 128;
    // short r = (short)(255 * pow(sin(0.16 * N), 2));
    // short g = (short)(255 * pow(log(log(0.16 * N + 2)), 2));
    // short b = (short)(255 * pow(log(0.16 * N + 4), 2));
    return (r << 24) | (g << 16) | (b << 8) | 255;  // Формат RGBA
}

void fill_pixels_SISD(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale)    // single instruction single data
{
    assert(pixels);

    mandel_t R_square_max = 10;
    mandel_t dx = (mandel_t) 1 / WIDTH * scale;
    mandel_t dy = (mandel_t) 1 / WIDTH * scale;
    mandel_t Y0 = y_center / HEIGHT - scale / 2;

    for (int y_screen = 0; y_screen < HEIGHT; y_screen++, Y0 += dy)
    {
        mandel_t X0 = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen++, X0 += dx)
        {
            mandel_t X = X0;
            mandel_t Y = Y0;
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                mandel_t X_square = X * X;
                mandel_t Y_square = Y * Y;
                mandel_t XY       = X * Y;
                mandel_t R_square = X_square + Y_square;
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
void fill_pixels_SIMD_manual(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale)    // single instruction multiple data
{
    assert(pixels);

    mandel_t R_square_max = 10;
    mandel_t dx = (mandel_t) 1 / WIDTH * scale;
    mandel_t dy = (mandel_t) 1 / WIDTH * scale;
    mandel_t Y0 = y_center / HEIGHT - scale / 2;

    for (int y_screen = 0; y_screen < HEIGHT; y_screen++, Y0 += dy)
    {
        mandel_t X0_initial = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen += VEC_SIZE, X0_initial += dx * VEC_SIZE)
        {
            mandel_t  X0[VEC_SIZE] ={};
            for (int i = 0; i < VEC_SIZE; i++) X0[i] = X0_initial + dx * i;

            mandel_t X[VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) X[i] = X0[i]; // {X0[0], X0[1], X0[2], X0[3]};
            mandel_t Y[VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) Y[i] = Y0;    // {Y0, Y0, Y0, Y0};
            int N_counts[VEC_SIZE] = {};
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                mandel_t X_square[VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) {X_square[i] = X[i] * X[i];} // X * X;
                mandel_t Y_square[VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) {Y_square[i] = Y[i] * Y[i];} // Y * Y;
                mandel_t XY      [VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) {      XY[i] = X[i] * Y[i];} // X * Y;
                mandel_t R_square[VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) {R_square[i] = X_square[i] + Y_square[i];}
                                                                                 // X_square + Y_square;
                int cmp[VEC_SIZE] = {}; for (int i = 0; i < VEC_SIZE; i++) {cmp[i] = (R_square[i] <= R_square_max);}
                int mask = 0;    for (int i = 0; i < VEC_SIZE; i++) {mask  |= cmp[i];}
                if (!mask)
                {
                    break;
                }
                for (int i = 0; i < VEC_SIZE; i++) {N_counts[i] += cmp[i];}
                for (int i = 0; i < VEC_SIZE; i++) {X[i] = X_square[i] - Y_square[i] + X0[i];}
                for (int i = 0; i < VEC_SIZE; i++) {Y[i] = 2 * XY[i]                 + Y0;}
                // X = X_square - Y_square + X0;
                // Y = 2 * XY              + Y0;
            }

            for (int i = 0; i < VEC_SIZE; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts[i]);}
        }
    }
    printf("Finished calc\n");

    return ;
}

#ifndef _WIN32
#if VEC_SIZE == 8
    #define VEC_OF_NUMBERS 7, 6, 5, 4, 3, 2, 1, 0
#else
    #define VEC_OF_NUMBERS 3, 2, 1, 0
#endif
// Real avx
void fill_pixels_SIMD(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale)    // single instruction multiple data
{
    assert(pixels);

    double R_square_max = 10;
    double dx = (double) 1 / WIDTH * scale;
    double dy = (double) 1 / WIDTH * scale;
    MM_t Y0 = MM256_set1_p(y_center / HEIGHT - scale / 2);
    #if VEC_SIZE == 8
    alignas(32) int32_t N_counts_total[8] = {}; // alignas - 32 bytes aligning for _mm256_store_si256
    #else
    alignas(32) long long int N_counts_total[4] = {}; // alignas - 32 bytes aligning for _mm256_store_si256
    #endif
    for (int y_screen = 0; y_screen < HEIGHT; y_screen++, Y0 = MM256_add_p(Y0, MM256_set1_p(dy)))
    {
        double X0_initial = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen += VEC_SIZE, X0_initial += dx * VEC_SIZE)
        {
            MM_t X0 = MM256_add_p(MM256_set1_p(X0_initial),
                                       MM256_mul_p (MM256_set1_p(dx), MM256_set_p(VEC_OF_NUMBERS)));
            MM_t X = X0;
            MM_t Y = Y0;
            __m256i N_counts = MM256_setzero_si256();
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                MM_t X_square = MM256_mul_p(X, X);
                MM_t Y_square = MM256_mul_p(Y, Y);
                MM_t XY       = MM256_mul_p(X, Y);
                MM_t R_square = MM256_add_p(X_square, Y_square);
                MM_t cmp      = MM256_cmp_p(R_square, MM256_set1_p(R_square_max), _CMP_LE_OQ);
                long long int mask = MM256_movemask_p(cmp);
                if (!mask)
                {
                    break;
                }
                N_counts = MM256_sub_epi(N_counts, MM256_castp(cmp)); // sub because cmp_pd returns 0xFFFF... on true, which is equal to -1
                X = MM256_add_p(MM256_sub_p(X_square, Y_square),   X0);
                Y = MM256_add_p(MM256_add_p(XY, XY), Y0);
            }
            _mm256_store_si256((__m256i *)&N_counts_total, N_counts);
            for (int i = 0; i < VEC_SIZE; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts_total[i]);}
        }
    }

    printf("Finished calc\n");

    return ;
}


// Real avx
void fill_pixels_SIMD_multithread(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale)    // single instruction multiple data
{
    assert(pixels);
    #pragma omp parallel for schedule(dynamic, 2)
    for (int y_screen = 0; y_screen < HEIGHT; y_screen++)
    {
        double R_square_max = 10;
        double dx = (double) 1 / WIDTH * scale;
        double dy = (double) 1 / WIDTH * scale;
        MM_t Y0 = MM256_set1_p(y_center / HEIGHT - scale / 3.5 + dy * y_screen);
        #if VEC_SIZE == 8
        alignas(32) int32_t N_counts_total[8] = {}; // alignas - 32 bytes aligning for _mm256_store_si256
        #else
        alignas(32) long long int N_counts_total[4] = {}; // alignas - 32 bytes aligning for _mm256_store_si256
        #endif

        double X0_initial = x_center / WIDTH - scale / 2;

        for (int x_screen = 0; x_screen < WIDTH; x_screen += VEC_SIZE, X0_initial += dx * VEC_SIZE)
        {
            MM_t X0 = MM256_add_p(MM256_set1_p(X0_initial),
                                       MM256_mul_p (MM256_set1_p(dx), MM256_set_p(VEC_OF_NUMBERS)));
            MM_t X = X0;
            MM_t Y = Y0;
            __m256i N_counts = MM256_setzero_si256();
            int N_count = 0;

            for (; N_count < N_EXIT_COUNT; N_count++)
            {
                MM_t X_square = MM256_mul_p(X, X);
                MM_t Y_square = MM256_mul_p(Y, Y);
                MM_t XY       = MM256_mul_p(X, Y);
                MM_t R_square = MM256_mul_p(X_square, Y_square);
                MM_t cmp      = MM256_cmp_p(R_square, MM256_set1_p(R_square_max), _CMP_LE_OQ);
                long long int mask = MM256_movemask_p(cmp);
                if (!mask)
                {
                    break;
                }
                N_counts = MM256_sub_epi(N_counts, MM256_castp(cmp)); // sub because cmp_pd returns 0xFFFF... on true, which is equal to -1
                X = MM256_add_p(MM256_sub_p(X_square, Y_square),   X0);
                Y = MM256_add_p(MM256_add_p(XY, XY), Y0);
            }
            _mm256_store_si256((__m256i *)&N_counts_total, N_counts);
            for (int i = 0; i < VEC_SIZE; i++) {pixels[y_screen * WIDTH + x_screen + i] = generatePixelColor(N_counts_total[i]);}
        }
    }

    // #pragma omp barrier

    printf("Finished calc, %d\n", omp_get_max_threads());

    return ;
}
#endif
