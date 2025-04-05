#include <cuda_runtime.h>

#include "utils.h"
#include <stdio.h>
#include "assert.h"
#include "config.h"
#include "mandelbrot.h"
#include <omp.h>
#include <intrin.h>


/*
__global__ - Runs on the GPU, called from the CPU or the GPU*. Executed with <<<dim3>>> arguments.
__device__ - Runs on the GPU, called from the GPU. Can be used with variabiles too.
__host__   - Runs on the CPU, called from the CPU.
*/

__global__ void mandelbrot_kernel(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);
__device__ int generatePixelColor_cuda(long long N_count);

__device__ int generatePixelColor_cuda(long long int N)
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
/*
__global__ void fill_pixels_SIMT(int* pixels, double x_center, double y_center, double scale)    // single instruction single data
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

            pixels[y_screen * WIDTH + x_screen] = generatePixelColor_cuda(N_count);
        }
    }

    return;
}*/
// __launch_bounds__(1024, 16) - incresed fps at full black screen (it almost same)
__global__ void __launch_bounds__(1024, 16) mandelbrot_kernel(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale)
{
    int base_x = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int base_y = (threadIdx.y + blockIdx.y * blockDim.y) * 4;

    mandel_t dx = (mandel_t) 1 / WIDTH * scale;
    mandel_t dy = (mandel_t) 1 / WIDTH * scale;

    // double x0 = x_center / height + (idx / height - width  / 2) * scale;
    // double y0 = y_center / height + (idy / height - height / 2) * scale;
    // double y0 = y_center / HEIGHT - scale / 2 / ((double)WIDTH / (double)HEIGHT) + dy * idy;
    for (int y_screen = 0; y_screen < 4; y_screen++, y_screen++)
    {
        for (int x_screen = 0; x_screen < 4; x_screen++)
        {
            int idx = base_x + x_screen;
            int idy = base_y + y_screen;     

            if (idx >= WIDTH || idy >= HEIGHT)
                continue;
            
            mandel_t y0 = y_center / HEIGHT - scale / 2 + dy * idy;
            mandel_t x0 = x_center / WIDTH  - scale / 2 + dx * idx;

            int N_count = 0;
            mandel_t X = x0,
                    Y = y0;
            mandel_t R_square_max = 10;

            while (N_count < N_EXIT_COUNT)
            {
                mandel_t X_square = X * X;
                mandel_t Y_square = Y * Y;
                mandel_t XY = X * Y;
                mandel_t R_square = X_square + Y_square;

                if (R_square > R_square_max)
                {
                    // printf("Exit ThrX = %3d, ThrY= %3d, BlX = %3d, BlY = %3d N_count = %3d\n"
                    //      "-------begin-------\n"
                    //     "x_screen = %d\n"
                    //     "y_screen = %d\n"
                    //     "X0 = %lf\n"
                    //     "Y0 = %lf\n"
                    //     "X  = %lf\n"
                    //     "Y  = %lf\n"
                    //     "N_count = %d\n"
                    //     "--------end----\n\n",  threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, N_count,
                    //     idx, idy, x0, y0, X, Y, N_count);
                    break;
                }

                N_count++;
                X = X_square - Y_square + x0;
                Y = 2 * XY + y0;
            }

            pixels[idy * WIDTH + idx] = generatePixelColor_cuda(N_count);
        }
    }
    // printf("ThrX = %3d, ThrY= %3d, BlX = %3d, BlY = %3d N_count = %3d\n",  threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, N_count);
    return;
}

extern "C"
__host__ void fill_pixels_SIMT_GPU(int* pixels, double x_center, double y_center, double scale)
{
    // cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

    // int *d_pixels = NULL;
    int *d_pixels = pixels;

    size_t size = WIDTH * HEIGHT * sizeof(int);
    cudaError_t err = {};
    // Выделение памяти на устройстве
    err = cudaMalloc((void**)&d_pixels, size);
    if (err != cudaSuccess)
    {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
    }

    // Запуск ядра
    // 2 файла с прозрачностью
    // Alpha blending = сложение двух файлов (кошка + теннисный стол)
    // +readme с nprefhud, как нашёл узкие места
    // *через микрофон спектр = раскраска. TXWave. Рисовка эквалайзера, затем мб рисовка палитры в зависимости от звука
    //  из-за FFT (синусы не пересчитывает + интерполяция).

    dim3 blockDim(32, 32);  // Размер блока
    dim3 gridDim((WIDTH  + 4 * blockDim.x - 1) / (4 * blockDim.x),
                 (HEIGHT + 4 * blockDim.y - 1) / (4 * blockDim.y));
    // TIME_MEASURE(
    mandelbrot_kernel<<<gridDim, blockDim>>>(d_pixels, (mandel_t) x_center, (mandel_t) y_center, (mandel_t) scale);
    // )
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Копирование данных с устройства на хост
    err = cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_pixels);

    return;
}
