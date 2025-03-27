#include "mandelbrot.h"
#include "gui.h"

int generatePixelColor(int x, int y) {
    short r = x % 255;
    short g = y % 255;
    short b = (x * y) % 255;
    return (r << 24) | (g << 16) | (b << 8) | 255;  // Формат RGBA
}

err_code_t fill_pixels(int* pixels) {
    for (int y = 0; y < 600; y++) {
        for (int x = 0; x < 800; x++) {
            pixels[y * 800 + x] = generatePixelColor(x, y);
        }
    }

    return OK;
}

int main()
{
    init_sdl(fill_pixels);

    return 0;
}
