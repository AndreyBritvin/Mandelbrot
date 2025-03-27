#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_mixer.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gui.h"
#include "utils.h"

const int WIDTH = 800;
const int HEIGHT = 600;

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

    SDL_LockTexture(texture, NULL, &pixels, &pitch);
    set_pixels_color((int*) pixels);
    SDL_UnlockTexture(texture);

    bool running = true;
    SDL_Event event = {};
    while (running) {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
        }

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
