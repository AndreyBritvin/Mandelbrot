CC = g++

SDL_PATH = C:/libs/SDL2
INCLUDE = -Iinclude -Iinclude/gui -Iinclude/mandelbrot_calc -IMy_logging_system/include -I$(SDL_PATH)/include

CFLAGS = -lm -Wshadow -Winit-self -Wredundant-decls -Wcast-align -Wundef -Wfloat-equal \
         -Winline -Wunreachable-code -Wmissing-declarations -Wmissing-include-dirs \
		 -Wswitch-enum -Wswitch-default -Weffc++ -Wmain -Wextra -Wall -g -pipe -fexceptions \
		 -Wcast-qual -Wconversion -Wctor-dtor-privacy -Wempty-body -Wformat-security -Wformat=2 \
		 -Wignored-qualifiers -Wlogical-op -Wno-missing-field-initializers -Wnon-virtual-dtor \
		 -Woverloaded-virtual -Wpointer-arith -Wsign-promo -Wstack-usage=8192 -Wstrict-aliasing \
		 -Wstrict-null-sentinel -Wtype-limits -Wwrite-strings -Werror=vla -D_DEBUG -D_EJUDGE_CLIENT_SIDE

SRC_FILES     = $(wildcard src/*.cpp) $(wildcard src/gui/*.cpp) $(wildcard src/mandelbrot_calc/*.cpp)
BUILD_FILES   = $(wildcard build/*.o)

MY_LIBS = My_logging_system/log_lib.a -L$(SDL_PATH)/lib -lmingw32 -lSDL2main -lSDL2 

all: mandelbrot.exe

mandelbrot.exe:$(SRC_FILES) $(BUILD_FILES)
	@$(CC) $(CFLAGS) $(INCLUDE) $(SRC_FILES) $(MY_LIBS) -O3 -fopenmp -o mandelbrot.exe

library:
	@$(CC) $(CFLAGS) -c $(INCLUDE) $(SRC_FILES) My_logging_system/log_lib.a -o $(BUILD_FILES)

run:
	./mandelbrot.exe --mode=graphic

clean:
	rm mandelbrot.exe

count_time:
	time -p -q ./mandelbrot.exe --mode=test --func=SIMDT_CPU --test_count=1000

