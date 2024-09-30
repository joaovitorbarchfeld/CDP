# Compiler settings
CC := g++                      # Compiler for C++
NVCC := nvcc                   # Compiler for CUDA
CFLAGS := -std=c++17 -Wall -pthread -ltbb  # Compiler flags for C++
NVCCFLAGS := -std=c++17 -arch=sm_50        # Compiler flags for CUDA

# Source files and object files for Program 1 (C++)
SRCS1 := num_primos_fila_concorrente_normal.cpp
OBJS1 := $(SRCS1:.cpp=.o)
EXECUTABLE1 := num_primos_fila_concorrente_normal

# Source files and object files for Program 2 (C++)
SRCS2 := num_primos_paralelo_normal.cpp
OBJS2 := $(SRCS2:.cpp=.o)
EXECUTABLE2 := num_primos_paralelo_normal

# Source files and object files for CUDA Programs
SRCS_CUDA1 := num_primos_fila_concorrente_cuda.cu
OBJS_CUDA1 := $(SRCS_CUDA1:.cu=.o)
EXECUTABLE_CUDA1 := num_primos_fila_concorrente_cuda

SRCS_CUDA2 := num_primos_paralelo_sequencial_cuda.cu
OBJS_CUDA2 := $(SRCS_CUDA2:.cu=.o)
EXECUTABLE_CUDA2 := num_primos_paralelo_sequencial_cuda

# Makefile rules
.PHONY: all clean

all: $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE_CUDA1) $(EXECUTABLE_CUDA2)

# C++ executable rules
$(EXECUTABLE1): $(OBJS1)
	$(CC) $(CFLAGS) $^ -o $@

$(EXECUTABLE2): $(OBJS2)
	$(CC) $(CFLAGS) $^ -o $@

# CUDA executable rules
$(EXECUTABLE_CUDA1): $(OBJS_CUDA1)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(EXECUTABLE_CUDA2): $(OBJS_CUDA2)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Compilation rules for C++ source files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation rules for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule to remove object files and executables
clean:
	rm -f $(OBJS1) $(EXECUTABLE1) $(OBJS2) $(EXECUTABLE2) \
        $(OBJS_CUDA1) $(EXECUTABLE_CUDA1) $(OBJS_CUDA2) $(EXECUTABLE_CUDA2)
