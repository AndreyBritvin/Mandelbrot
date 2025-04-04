# CC = cl

# SDL_PATH = C:/libs/SDL2
# CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
# INCLUDE = -Iinclude -Iinclude/gui -Iinclude/mandelbrot_calc -IMy_logging_system/include -I$(SDL_PATH)/include -I$(CUDA_PATH)/include

# CFLAGS = -lm -Wshadow -Winit-self -Wredundant-decls -Wcast-align -Wundef -Wfloat-equal \
#          -Winline -Wunreachable-code -Wmissing-declarations -Wmissing-include-dirs \
# 		 -Wswitch-enum -Wswitch-default -Weffc++ -Wmain -Wextra -Wall -g -pipe -fexceptions \
# 		 -Wcast-qual -Wconversion -Wctor-dtor-privacy -Wempty-body -Wformat-security -Wformat=2 \
# 		 -Wignored-qualifiers -Wlogical-op -Wno-missing-field-initializers -Wnon-virtual-dtor \
# 		 -Woverloaded-virtual -Wpointer-arith -Wsign-promo -Wstack-usage=8192 -Wstrict-aliasing \
# 		 -Wstrict-null-sentinel -Wtype-limits -Wwrite-strings -Werror=vla -D_DEBUG -D_EJUDGE_CLIENT_SIDE

# SRC_FILES     = $(wildcard src/*.cpp) $(wildcard src/gui/*.cpp) $(wildcard src/mandelbrot_calc/*.cpp)
# BUILD_FILES   = $(wildcard build/*.o)

# MY_LIBS = My_logging_system/log_lib.a -L$(SDL_PATH)/lib -L$(CUDA_PATH)/lib/x64 -lmingw32 -lSDL2main -lSDL2

# all: mandelbrot.exe

# mandelbrot.exe:$(SRC_FILES) $(BUILD_FILES) src/mandelbrot_calc/mandelbrot.cu
# 	nvcc -c src/mandelbrot_calc/mandelbrot.cu -o mandelbrot_cu.o -arch=sm_86 $(INCLUDE)
# 	@$(CC) $(INCLUDE) $(SRC_FILES) $(MY_LIBS) -fno-stack-protector mandelbrot_cu.o -o mandelbrot.exe -lcudart -lcudadevrt -lcuda -lcublas /O2 /D_DEBUG /D_EJUDGE_CLIENT_SIDE /W3 /EHsc /MT
# #-fopenmp

# # @$(CC) $(CFLAGS) $(INCLUDE) $(SRC_FILES) $(MY_LIBS) mandelbrot_cu.o -O3 -fopenmp -o mandelbrot.exe -lcudart -lcudadevrt

CC = cl  # Используем компилятор MSVC
LINKER = link  # Явно указываем линковщик

# Путь к CUDA
CUDA_PATH = C:/Progra~1/NVIDIA~1/CUDA/v12.8  # Используем короткое имя (8.3) для путей с пробелами

# Путь к SDL
SDL_PATH = C:/libs/SDL2

# Включаем нужные директории
INCLUDE = -Iinclude -Iinclude/gui -Iinclude/mandelbrot_calc -IMy_logging_system/include \
          -I$(SDL_PATH)/include -I"$(CUDA_PATH)/include" \
          -I"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/include" \
          -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt" \
          -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared" \
          -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um" \

# Пути к библиотекам
LIBPATHS = /LIBPATH:"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/lib/x64" \
           /LIBPATH:"C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/ucrt/x64" \
           /LIBPATH:"C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64" \
           /LIBPATH:"$(SDL_PATH)/lib" \
           /LIBPATH:"$(CUDA_PATH)/lib/x64"

# Линковка с нужными библиотеками
LIBS =   SDL2.lib SDL2main.lib  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64/cudart.lib"\
 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64/cudadevrt.lib"\
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64/cuda.lib"\
   "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64/cublas.lib"


# Флаги компиляции
CFLAGS = /O2 /GL

# Исходные файлы
SRC_FILES = $(wildcard src/*.cpp) $(wildcard src/gui/*.cpp) $(wildcard src/mandelbrot_calc/*.cpp)
OBJ_FILES = $(SRC_FILES:.cpp=.obj)

# Цель
all: mandelbrot.exe

# Компиляция всех C++ файлов
%.obj: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE) /c $< /Fo$@

# Компиляция с использованием nvcc для .cu файлов
mandelbrot_cu.obj: src/mandelbrot_calc/mandelbrot.cu
	nvcc -c src/mandelbrot_calc/mandelbrot.cu -o mandelbrot_cu.obj -arch=sm_86 $(INCLUDE) -O3

# Линковка отдельно через link.exe
mandelbrot.exe: $(OBJ_FILES) mandelbrot_cu.obj
	$(LINKER) $(LIBPATHS) $(LIBS) $(OBJ_FILES) mandelbrot_cu.obj  /OUT:mandelbrot.exe



library:
	@$(CC) $(CFLAGS) -c $(INCLUDE) $(SRC_FILES) My_logging_system/log_lib.a -o $(BUILD_FILES)

run:
	./mandelbrot.exe --mode=graphic

clean:
	rm mandelbrot.exe

count_time:
	./mandelbrot.exe --mode=test --func=SIMT_GPU --test_count=10000

