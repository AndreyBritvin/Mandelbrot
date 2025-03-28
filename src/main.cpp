#include "mandelbrot.h"
#include "gui.h"
#include "stdio.h"

int main()
{
    // TODO: make arguments parsing

    init_sdl(fill_pixels_SIMD_multithread);

    return 0;
}
