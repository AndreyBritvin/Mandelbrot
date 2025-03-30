#include <cuda_runtime.h>

#include "utils.h"
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

__global__ void mandelbrot_kernel(int* pixels, double x_center, double y_center, double scale, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        // Реализация вычислений по аналогии с вашей CPU-функцией
        double x0 = x_center / width - scale / 2 + idx * scale;
        double y0 = y_center / height - scale / 2 + idy * scale;

        int N_count = 0;
        double X = x0, Y = y0;
        double R_square_max = 10;

        // Пример вычислений для мандельброта
        while (N_count < N_EXIT_COUNT) {
            double X_square = X * X;
            double Y_square = Y * Y;
            double XY = X * Y;
            double R_square = X_square + Y_square;

            if (R_square > R_square_max) {
                break;
            }

            N_count++;
            X = X_square - Y_square + x0;
            Y = 2 * XY + y0;
        }

        // Сохранение результата в массив
        pixels[idy * width + idx] = generatePixelColor_cuda(N_count);
    }
}

extern "C"
__host__ void fill_pixels_SIMT_GPU(int* pixels, double x_center, double y_center, double scale) {
    int *d_pixels;
    size_t size = WIDTH * HEIGHT * sizeof(int);
    
    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_pixels, size);

    // Запуск ядра
    dim3 blockDim(16, 16);  // Размер блока
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);  // Размер сетки

    mandelbrot_kernel<<<gridDim, blockDim>>>(d_pixels, x_center, y_center, scale, WIDTH, HEIGHT);

    // Копирование данных с устройства на хост
    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFree(d_pixels);
    
    return;
}