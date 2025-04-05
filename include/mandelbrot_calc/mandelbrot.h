#ifndef MANDELBROT_H__
#define MANDELBROT_H__

#include "utils.h"
#include "config.h"

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

#if TYPE_OF_MANDEL_T == MANDEL_DOUBLE
    #define MM256_set1_p(...)        _mm256_set1_pd(__VA_ARGS__)
    #define MM256_add_p(...)         _mm256_add_pd(__VA_ARGS__)
    #define MM256_mul_p(...)         _mm256_mul_pd(__VA_ARGS__)
    #define MM256_set_p(...)         _mm256_set_pd(__VA_ARGS__)
    #define MM256_setzero_si256(...) _mm256_setzero_si256(__VA_ARGS__)
    #define MM256_movemask_p(...)    _mm256_movemask_pd(__VA_ARGS__)
    #define MM256_sub_epi(...)       _mm256_sub_epi64(__VA_ARGS__)
    #define MM256_castp(...)         _mm256_castpd_si256(__VA_ARGS__)
    #define MM256_sub_p(...)         _mm256_sub_pd(__VA_ARGS__)
    #define MM256_store(...)         _mm256_store_si256(__VA_ARGS__)
    #define MM256_cmp_p(...)         _mm256_cmp_pd(__VA_ARGS__)
    #define MM_t __m256d
#else
    #define MM256_set1_p(...)        _mm256_set1_ps(__VA_ARGS__)
    #define MM256_add_p(...)         _mm256_add_ps(__VA_ARGS__)
    #define MM256_mul_p(...)         _mm256_mul_ps(__VA_ARGS__)
    #define MM256_set_p(...)         _mm256_set_ps(__VA_ARGS__)
    #define MM256_setzero_si256(...) _mm256_setzero_si256(__VA_ARGS__)
    #define MM256_movemask_p(...)    _mm256_movemask_ps(__VA_ARGS__)
    #define MM256_sub_epi(...)       _mm256_sub_epi32(__VA_ARGS__)
    #define MM256_castp(...)         _mm256_castps_si256(__VA_ARGS__)
    #define MM256_sub_p(...)         _mm256_sub_ps(__VA_ARGS__)
    #define MM245_store(...)         _mm256_store_si256(__VA_ARGS__)
    #define MM256_cmp_p(...)         _mm256_cmp_ps(__VA_ARGS__)
    #define MM_t __m256
#endif

typedef unsigned long long timestamp_t;

const int BLACK_COLOR_RGBA = 0 | 255;

int generatePixelColor(long long int N);
void fill_pixels_SISD       (int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);
void fill_pixels_SIMD_manual(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);
void fill_pixels_SIMD       (int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);
void fill_pixels_SIMD_multithread (int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);
void       fill_pixels_SIMT       (int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);

#endif // MANDELBROT_H__
