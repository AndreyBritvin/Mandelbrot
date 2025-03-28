#ifndef GUI_H__
#define GUI_H__

#include "utils.h"
const double movement_speed = 5.f;

typedef err_code_t (*color_setter_t)(int* pixels, double x_center, double y_center, double scale);

int init_sdl(color_setter_t set_pixel_color);

#endif // GUI_H__
