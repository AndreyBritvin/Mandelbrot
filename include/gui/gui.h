#ifndef GUI_H__
#define GUI_H__

#include "utils.h"
#include "config.h"

typedef void (*color_setter_t)(int* pixels, mandel_t x_center, mandel_t y_center, mandel_t scale);

int init_sdl(color_setter_t set_pixel_color);

#endif // GUI_H__
