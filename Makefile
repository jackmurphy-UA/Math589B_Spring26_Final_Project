all: solver

solver: src/main.cu src/solver.cu src/solver.hpp ; if command -v nvcc >/dev/null 2>&1; then nvcc -std=c++17 -O2 src/main.cu src/solver.cu -o solver; else g++ -std=c++17 -O2 -x c++ src/main.cu -x c++ src/solver.cu -o solver; fi

clean: ; rm -f solver
