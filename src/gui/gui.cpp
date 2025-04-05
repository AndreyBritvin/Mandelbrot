#ifdef _WIN32
    #include <SDL.h>
    #include <windows.h>  // Для QueryPerformanceCounter()
#else
    #include <SDL2/SDL.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gui.h"
#include "config.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>

int init_sdl(color_setter_t set_pixels_color)
{
    assert(set_pixels_color);

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window*   window   = SDL_CreateWindow("Mandelbrot set", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_Texture*  texture  = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    GLuint gl_texture = {};
    glGenTextures(1, &gl_texture);
    glBindTexture(GL_TEXTURE_2D, gl_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) {
        printf("OpenGL texture error: 0x%X\n", gl_error);
        return -1;
    }
    cudaError_t cuda_status = {};
    cudaGraphicsResource* cuda_tex_resource = NULL;
    cuda_status = cudaGraphicsGLRegisterImage(&cuda_tex_resource, gl_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (cuda_status != cudaSuccess) 
    {
        printf("CUDA failed: %s\n", cudaGetErrorString(cuda_status));
        return -1;
    }
        // cudaGraphicsResource* cuda_texture_resource = {};
    // cudaGraphicsRegisterResource(&cuda_texture_resource, texture, cudaGraphicsRegisterFlagsNone);
    
    // Заполнение пикселей
    void*   pixels      = NULL;
    int     pitch       = 0;

    bool running = true;
    SDL_Event event = {};

    double scale = default_scale;
    double X_center = 0;
    double Y_center = 0;
    // printf("Begining loop render\n");
    // struct timespec start = {}, now = {};

#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER now;
    QueryPerformanceFrequency(&frequency); // Получаем частоту таймера (обычно 10M+)
    QueryPerformanceCounter(&start);       // Засекаем старт    int frames = 0;
    double fps = 0.0;
    int frames = 0;
#endif


    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
                {
                case SDL_QUIT:
                {
                    running = false;
                    break;
                }
                case SDL_KEYDOWN:
                {
                    printf("Scancode: 0x%02X\n", event.key.keysym.scancode);
                    switch (event.key.keysym.scancode)
                    {
                        case SDL_SCANCODE_EQUALS: // +
                        {
                            scale *= 0.9;
                            // X_center *= (double) 1 / 0.9;
                            // Y_center *= (double) 1 / 0.9;
                            break;
                        }
                        case SDL_SCANCODE_MINUS: // -
                        {
                            scale *= (double) 1 / 0.9;
                            // X_center *= 0.9;
                            // Y_center *= 0.9;
                            break;
                        }
                        case SDL_SCANCODE_UP: // up
                        {
                            Y_center -= movement_speed * scale;
                            break;
                        }
                        case SDL_SCANCODE_DOWN: // down
                        {
                            Y_center += movement_speed * scale;
                            break;
                        }

                        case SDL_SCANCODE_RIGHT: // right
                        {
                            X_center += movement_speed * scale;
                            break;
                        }
                        case SDL_SCANCODE_LEFT: // left
                        {
                            X_center -= movement_speed * scale;
                            break;
                        }
                        default:
                        {
                            break;
                        }
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
        SDL_GL_MakeCurrent(window, gl_context); // Активируем контекст перед вызовом CUDA
        if (!glGetString(GL_VERSION)) {
            printf("OpenGL not initialized! Error: %s\n", SDL_GetError());
            return -1;
        }
        printf("OpenGL context: %s\n", glGetString(GL_VERSION));

    cudaArray* cuda_array;
    cuda_status = cudaGraphicsMapResources(1, &cuda_tex_resource, 0);
    if (cuda_status != cudaSuccess) {
        printf("Map error: %s\n", cudaGetErrorString(cuda_status));
        
        // Дополнительная диагностика:
        cudaDeviceSynchronize();
        cudaError_t sync_error = cudaGetLastError();
        if (sync_error != cudaSuccess) {
            printf("Sync error: %s\n", cudaGetErrorString(sync_error));
        }
        return -1;
    }

    cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_tex_resource, 0, 0);
    if (cuda_status != cudaSuccess) 
    {
        printf("CUDA failed 3: %s\n", cudaGetErrorString(cuda_status));
        return -1;
    }

    set_pixels_color((int*) cuda_array, X_center, Y_center, scale);
    printf("After kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();

    // 7. Разрегистрируем ресурс
    cuda_status = cudaGraphicsUnmapResources(1, &cuda_tex_resource);
    if (cuda_status != cudaSuccess) {
        printf("UnMap error: %s\n", cudaGetErrorString(cuda_status));
    }
    // 8. Рендерим текстуру через OpenGL
    SDL_GL_SwapWindow(window);


        // SDL_LockTexture(texture, NULL, &pixels, &pitch);
        // set_pixels_color((int*) pixels, X_center, Y_center, scale);
        // SDL_UnlockTexture(texture);
#ifdef _WIN32
        frames++;
        QueryPerformanceCounter(&now);
        double elapsed_time = (double)(now.QuadPart - start.QuadPart) / frequency.QuadPart;

        if (elapsed_time >= 1.0) {
            SDL_RendererInfo info = {};
            SDL_GetRendererInfo(renderer, &info);
            printf("Renderer backend: %s\n", info.name);

            fps = frames / elapsed_time;
            printf("FPS: %.2f\n", fps);

            frames = 0;
            QueryPerformanceCounter(&start); // Сброс таймера
        }
#endif

        // SDL_RenderClear(renderer);
        // SDL_RenderCopy(renderer, texture, NULL, NULL);
        // SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
