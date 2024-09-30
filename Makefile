# Compiler settings
CC := g++                  # Compiler
CFLAGS := -std=c++17 -Wall -pthread -ltbb # Compiler flags

# Source files and object files for Program 1
SRCS1 := num_primos_fila_concorrente_normal.cpp
OBJS1 := $(SRCS1:.cpp=.o)
EXECUTABLE1 := num_primos_fila_concorrente_normal

# Source files and object files for Program 2
SRCS2 := num_primos_paralelo_normal.cpp
OBJS2 := $(SRCS2:.cpp=.o)
EXECUTABLE2 := num_primos_paralelo_normal


# Makefile rules
.PHONY: all clean

all: $(EXECUTABLE1) $(EXECUTABLE2) 

$(EXECUTABLE1): $(OBJS1)
	$(CC) $(CFLAGS) $^ -o $@

$(EXECUTABLE2): $(OBJS2)
	$(CC) $(CFLAGS) $^ -o $@


%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS1) $(EXECUTABLE1) $(OBJS2) $(EXECUTABLE2) $
