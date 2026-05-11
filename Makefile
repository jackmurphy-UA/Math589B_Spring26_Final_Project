SHELL = /bin/bash

TARGET = solver
SRC = src/main.cu src/solver.cu

all: $(TARGET)

$(TARGET): $(SRC) src/solver.hpp
	if command -v nvcc >/dev/null 2>&1; then \
		nvcc -std=c++17 -O2 $(SRC) -o $(TARGET); \
	else \
		g++ -std=c++17 -O2 -x c++ src/main.cu -x c++ src/solver.cu -o $(TARGET); \
	fi

clean:
	rm -f $(TARGET)
