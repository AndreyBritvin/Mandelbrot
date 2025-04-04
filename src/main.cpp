#include "mandelbrot.h"
#include "gui.h"
#ifndef _WIN32
    #include <getopt.h>
#else
    #include "getopt_win.h"
#endif
#include <string.h>
#include <stdlib.h>
#include "config.h"
#include <stdio.h>
#ifdef _WIN32
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#define PRINT_ERROR(...) fprintf(stderr, __VA_ARGS__)

extern "C" void fill_pixels_SIMT_GPU(int* pixels, double x_center, double y_center, double scale);

int main(int argc, char *argv[])
{
    static struct option long_options[] = {
        {"mode",        required_argument, 0, 'm'},
        {"func",        required_argument, 0, 'f'},
        {"test_count",  required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    char *mode       = NULL;
    char *func_name  = NULL;
    int test_count   = 1;
    int option_index = 0;
    int c            = 0;

    while ((c = getopt_long(argc, argv, "m:f:t:", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'm':
                mode = optarg;
                break;
            case 'f':
                func_name = optarg;
                break;
            case 't':
                test_count = atoi(optarg);
                break;
            default:
                PRINT_ERROR("Unknown option\n");
                return EXIT_FAILURE;
        }
    }

    if (mode == NULL)
    {
        PRINT_ERROR("Mode is required\n");
        return EXIT_FAILURE;
    }

    if (strcmp(mode, "graphic") == 0)
    {
#ifdef _WIN32
        init_sdl(fill_pixels_SIMT_GPU);
#else
        init_sdl(fill_pixels_SIMD_multithread);
#endif
    }
    else if (strcmp(mode, "test") == 0)
    {
        if (func_name == NULL)
        {
            PRINT_ERROR("Function name is required in test mode\n");
            return EXIT_FAILURE;
        }

        color_setter_t test_func = NULL;
        bool is_GPU = false;
        // TODO: make for to loop all instructions and finc_names
        if (strcmp(func_name, "SISD") == 0)
        {
            test_func = fill_pixels_SISD;
        }
        else if (strcmp(func_name, "SIMD_manual") == 0)
        {
            test_func = fill_pixels_SIMD_manual;
        }
#ifndef _WIN32
        else if (strcmp(func_name, "SIMD") == 0)
        {
            test_func = fill_pixels_SIMD;
        }
        else if (strcmp(func_name, "SIMDT_CPU") == 0)
        {
            test_func = fill_pixels_SIMD_multithread;
        }
#endif

        else if (strcmp(func_name, "SIMT_GPU") == 0)
        {
            is_GPU = true;
        }
        else
        {
            PRINT_ERROR("Unknown function name: %s\n", func_name);
            return EXIT_FAILURE;
        }

        int* pixels = (int *) calloc(WIDTH * HEIGHT, sizeof(int));


#ifdef _WIN32
    TIME_MEASURE
        (
        for (int i = 0; i < test_count; i++)
        {
            if (is_GPU)     fill_pixels_SIMT_GPU(pixels, 0, 0, default_scale);
            else            test_func           (pixels, 0, 0, default_scale);
        }
        )
#else
    TIME_MEASURE
        (
        for (int i = 0; i < test_count; i++)
        {
            test_func       (pixels, 0, 0, default_scale);
        }
        )
#endif

        free(pixels);
    }
    else
    {
        PRINT_ERROR("Invalid mode: %s\n", mode);
        return EXIT_FAILURE;
    }

    // init_sdl(fill_pixels_SIMT_GPU);

    return EXIT_SUCCESS;
}
