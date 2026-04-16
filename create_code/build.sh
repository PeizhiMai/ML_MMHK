#!/bin/bash
set -e

CXX=g++
CXXFLAGS="-std=c++17 -Wall -O3 -funroll-loops -m64 -DMKL_ILP64 -march=native"

INCLUDES="
-I/home/9pm/installs/include
-I/home/9pm/intel/oneapi/mkl/2024.2/include
"

LIBDIRS="
-L/home/9pm/installs/lib
-L/home/9pm/intel/oneapi/mkl/2024.2/lib/intel64
"

LIBS="
-lqbasis
-larpackILP64
-Wl,--no-as-needed
-lmkl_intel_ilp64
-lmkl_tbb_thread
-lmkl_core
-lpthread
-ltbb
-lm
-ldl
"

SRC=Hubbard_sym_test4x4gse_allq.cc
OUT=Hubbard_sym_test4x4gse_allq.x

echo "Compiling $SRC ..."
$CXX $CXXFLAGS $INCLUDES $SRC $LIBDIRS $LIBS -o $OUT

echo "Build finished: $OUT"

