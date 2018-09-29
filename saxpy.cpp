// ========================================================================== //
// The MIT License (MIT)                                                      //
//                                                                            //
// Copyright (c) 2018 Intel Corporation                                       //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// in all copies or substantial portions of the Software.                     //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
// ========================================================================== //

#include "benchmark/benchmark.h"
#include "tsimd/tsimd.h"

// scalar version /////////////////////////////////////////////////////////////

namespace scalar {

  void saxpy(float a, int n, float x[], float y[], float out[])
  {
    for (int i = 0; i < n; ++i) {
      const float xi = x[i];
      const float yi = y[i];
      const float result = a * xi + yi;
      out[i] = result;
    }
  }

}  // namespace scalar

// openmp version /////////////////////////////////////////////////////////////

namespace openmp {

  void saxpy(float a, int n, float x[], float y[], float out[])
  {
    #pragma omp for simd
    for (int i = 0; i < n; ++i) {
      const float xi = x[i];
      const float yi = y[i];
      const float result = a * xi + yi;
      out[i] = result;
    }
  }

}  // namespace openmp

// tsimd version //////////////////////////////////////////////////////////////

namespace tsimd {

  void saxpy(float a, int n, float x[], float y[], float out[])
  {
    for (int i = 0; i < n; i += vfloat::static_size) {
      const vfloat xi = load<vfloat>(&x[i]);
      const vfloat yi = load<vfloat>(&y[i]);
      const vfloat result = a * xi + yi;
      store(result, &out[i]);
    }
  }

} // namespace tsimd

///////////////////////////////////////////////////////////////////////////////
// Define benchmarks
///////////////////////////////////////////////////////////////////////////////

namespace params {
  const unsigned int size = 1024*1024;
  alignas(64) std::array<float, size> x;
  alignas(64) std::array<float, size> y;
  alignas(64) std::array<float, size> out;
  const float a = 5.f;
} // namespace params

static void saxpy_scalar(benchmark::State& state)
{
  for (auto _ : state) {
    scalar::saxpy(params::a,
                  params::size,
                  params::x.data(),
                  params::y.data(),
                  params::out.data());
  }
}

BENCHMARK(saxpy_scalar)->Unit(benchmark::kMicrosecond);

static void saxpy_openmp(benchmark::State& state)
{
  for (auto _ : state) {
    openmp::saxpy(params::a,
                  params::size,
                  params::x.data(),
                  params::y.data(),
                  params::out.data());
  }
}

BENCHMARK(saxpy_openmp)->Unit(benchmark::kMicrosecond);

static void saxpy_tsimd(benchmark::State& state)
{
  for (auto _ : state) {
    tsimd::saxpy(params::a,
                 params::size,
                 params::x.data(),
                 params::y.data(),
                 params::out.data());
  }
}

BENCHMARK(saxpy_tsimd)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();

