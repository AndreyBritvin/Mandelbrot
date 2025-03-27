CC = g++

INCLUDE = -Iinclude -Iinclude/gui -Iinclude/mandelbrot_calc -IMy_logging_system/include

CFLAGS = -lm -D _DEBUG -ggdb3 -std=c++17 -O0 -Wall -Wextra -Weffc++ -Waggressive-loop-optimizations -Wc++14-compat -Wmissing-declarations -Wcast-align -Wcast-qual -Wchar-subscripts -Wconditionally-supported -Wconversion -Wctor-dtor-privacy -Wempty-body -Wfloat-equal -Wformat-nonliteral -Wformat-security -Wformat-signedness -Wformat=2 -Winline -Wlogical-op -Wnon-virtual-dtor -Wopenmp-simd -Woverloaded-virtual -Wpacked -Wpointer-arith -Winit-self -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=2 -Wsuggest-attribute=noreturn -Wsuggest-final-methods -Wsuggest-final-types -Wsuggest-override -Wswitch-default -Wswitch-enum -Wsync-nand -Wundef -Wunreachable-code -Wunused -Wuseless-cast -Wvariadic-macros -Wno-literal-suffix -Wno-missing-field-initializers -Wno-narrowing -Wno-old-style-cast -Wno-varargs -Wstack-protector -fcheck-new -fsized-deallocation -fstack-protector -fstrict-overflow -flto-odr-type-merging -fno-omit-frame-pointer -Wlarger-than=8192 -Wstack-usage=8192 -pie -fPIE -Werror=vla -fsanitize=address,alignment,bool,bounds,enum,float-cast-overflow,float-divide-by-zero,integer-divide-by-zero,leak,nonnull-attribute,null,object-size,return,returns-nonnull-attribute,shift,signed-integer-overflow,undefined,unreachable,vla-bound,vptr

SRC_FILES     = $(wildcard src/*.cpp) $(wildcard src/gui/*.cpp) $(wildcard src/mandelbrot_calc/*.cpp)
BUILD_FILES   = $(wildcard build/*.o)

MY_LIBS = My_logging_system/log_lib.a

all: mandelbrot.out

mandelbrot.out:$(SRC_FILES) $(BUILD_FILES)
	@$(CC) $(CFLAGS) $(INCLUDE) $(SRC_FILES) $(MY_LIBS) -o mandelbrot.out -lSDL2 -lSDL2_ttf -lSDL2_image -lSDL2_mixer

library:
	@$(CC) $(CFLAGS) -c $(INCLUDE) $(SRC_FILES) My_logging_system/log_lib.a -o $(BUILD_FILES)

run:
	./mandelbrot.out

clean:
	rm -rf build
	mkdir build
	rm mandelbrot.out

