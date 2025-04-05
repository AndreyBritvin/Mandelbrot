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

int init_sdl(color_setter_t set_pixels_color)
{
    assert(set_pixels_color);

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window*   window   = SDL_CreateWindow("Mandelbrot set", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_Texture*  texture  = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    // Заполнение пикселей
    void*   pixels      = NULL;
    int     pitch       = 0;

    bool running = true;
    SDL_Event event = {};

    double scale = default_scale;
    double X_center = 0;
    double Y_center = 0;
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

        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        set_pixels_color((int*) pixels, (mandel_t) X_center, (mandel_t) Y_center, (mandel_t) scale);
        SDL_UnlockTexture(texture);
#ifdef _WIN32
        frames++;
        QueryPerformanceCounter(&now);
        double elapsed_time = (double)(now.QuadPart - start.QuadPart) / frequency.QuadPart;

        if (elapsed_time >= 1.0) {
            fps = frames / elapsed_time;
            printf("FPS: %.2f\n", fps);

            frames = 0;
            QueryPerformanceCounter(&start); // Сброс таймера
        }
#endif

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
