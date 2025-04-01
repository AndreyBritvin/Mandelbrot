#include <cuda_runtime.h>

#include "utils.h"
#include <stdio.h>
#include "assert.h"
#include "config.h"
#include <omp.h>

/*
__global__ - Runs on the GPU, called from the CPU or the GPU*. Executed with <<<dim3>>> arguments.
__device__ - Runs on the GPU, called from the GPU. Can be used with variabiles too.
__host__   - Runs on the CPU, called from the CPU.
*/

__global__ void mandelbrot_kernel(int* pixels, double x_center, double y_center, double scale, int width, int height);
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

__global__ void mandelbrot_kernel(int* pixels, double x_center, double y_center, double scale, int width, int height) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    double dx = (double) 1 / WIDTH * scale;
    double dy = (double) 1 / WIDTH * scale;

    if (idy >= HEIGHT || idx >= WIDTH)
    {
        return;
    }
    // double x0 = x_center / height + (idx / height - width  / 2) * scale;
    // double y0 = y_center / height + (idy / height - height / 2) * scale;
    double y0 = y_center / HEIGHT - scale / 2 + dy * idy;
    double x0 = x_center / WIDTH  - scale / 2 + dx * idx;

    int N_count = 0;
    double X = x0, 
           Y = y0;
    double R_square_max = 10;

    while (N_count < N_EXIT_COUNT) 
    {
        double X_square = X * X;
        double Y_square = Y * Y;
        double XY = X * Y;
        double R_square = X_square + Y_square;

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

    pixels[idy * width + idx] = generatePixelColor_cuda(N_count);
    // printf("ThrX = %3d, ThrY= %3d, BlX = %3d, BlY = %3d N_count = %3d\n",  threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, N_count);
    return;
}

extern "C"
__host__ void fill_pixels_SIMT_GPU(int* pixels, double x_center, double y_center, double scale) 
{
    int *d_pixels = NULL;
    size_t size = WIDTH * HEIGHT * sizeof(int);
    
    // Выделение памяти на устройстве
    cudaError_t err = cudaMalloc((void**)&d_pixels, size);
    if (err != cudaSuccess) 
    {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
    }
    // printf("CUDA malloc success: %s\n", cudaGetErrorString(err));
 
    // Запуск ядра
    dim3 blockDim(32, 32);  // Размер блока
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);  // Размер сетки

    mandelbrot_kernel<<<gridDim, blockDim>>>(d_pixels, x_center, y_center, scale, WIDTH, HEIGHT);
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