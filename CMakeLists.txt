## ========================================================================== ##
## The MIT License (MIT)                                                      ##
##                                                                            ##
## Copyright (c) 2018 Intel Corporation                                       ##
##                                                                            ##
## Permission is hereby granted, free of charge, to any person obtaining a    ##
## copy of this software and associated documentation files (the "Software"), ##
## to deal in the Software without restriction, including without limitation  ##
## the rights to use, copy, modify, merge, publish, distribute, sublicense,   ##
## and/or sell copies of the Software, and to permit persons to whom the      ##
## Software is furnished to do so, subject to the following conditions:       ##
##                                                                            ##
## The above copyright notice and this permission notice shall be included in ##
## in all copies or substantial portions of the Software.                     ##
##                                                                            ##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR ##
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   ##
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    ##
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER ##
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    ##
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        ##
## DEALINGS IN THE SOFTWARE.                                                  ##
## ========================================================================== ##

# CMake configuration #########################################################

cmake_minimum_required(VERSION 3.2)

project(simd_wrappers_cppcon2018)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(macros)

set(CMAKE_CXX_STANDARD 11)

setup_build_type()
configure_compiler()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Build Google Benchmark ######################################################

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE) # don't build tests
add_subdirectory(external/benchmark)

# Build tsimd (just get CMake interface target) ###############################

#add_subdirectory(external/tsimd)

# Build benchmarks ############################################################

if (WIN32)
  # Assume SSE4.2 on Windows
  set(FLAGS_SSE2  "/D__SSE__ /D__SSE2__")
  set(FLAGS_SSE42 "${FLAGS_SSE2} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__SSE4_2__")
  set(SIMD_COMPILE_OPTIONS ${FLAGS_SSE42})
else ()
  # Compile the best ISA you can using the host CPU's features
  set(SIMD_COMPILE_OPTIONS -march=native -fno-tree-vectorize)
endif ()

# saxpy

add_executable(saxpy saxpy.cpp)

target_link_libraries(saxpy benchmark)
target_include_directories(saxpy PRIVATE external/tsimd)
target_compile_options(saxpy PRIVATE ${SIMD_COMPILE_OPTIONS})

# saxpy_trig

add_executable(saxpy_trig saxpy_trig.cpp)

target_link_libraries(saxpy_trig benchmark)
target_include_directories(saxpy_trig PRIVATE external/tsimd)
target_compile_options(saxpy_trig PRIVATE ${SIMD_COMPILE_OPTIONS})

# mandelbrot

add_executable(mandelbrot mandelbrot.cpp)

target_link_libraries(mandelbrot benchmark)
target_include_directories(mandelbrot PRIVATE external/tsimd)
target_compile_options(mandelbrot PRIVATE ${SIMD_COMPILE_OPTIONS})
