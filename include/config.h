#ifndef CONFIG_H__
#define CONFIG_H__

#define MANDEL_DOUBLE 0
#define MANDEL_FLOAT 1
#define TYPE_OF_MANDEL_T MANDEL_DOUBLE

#if TYPE_OF_MANDEL_T == MANDEL_DOUBLE
    #define VEC_SIZE 4
    typedef double mandel_t;
#else
    #define VEC_SIZE 8
    typedef float  mandel_t;
#endif

const int WIDTH = 800;
const int HEIGHT = 600;
const double movement_speed = 5.f;
const double default_scale = 3.f;
const int N_EXIT_COUNT = 255;

#endif // CONFIG_H__
