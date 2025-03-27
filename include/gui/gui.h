#ifndef GUI_H__
#define GUI_H__

#include "utils.h"

typedef err_code_t (*color_setter_t)(int* pixels);

int init_sdl(color_setter_t set_pixel_color);

#endif // GUI_H__
