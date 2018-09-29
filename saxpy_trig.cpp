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

#define EXTRA_TANGENT_ITERATIONS 10

// scalar version /////////////////////////////////////////////////////////////

namespace scalar {

  void saxpy_trig(float a, int n, float x[], float y[], float out[])
  {
    for (int i = 0; i < n; ++i) {
      const float xi = x[i];
      const float yi = y[i];
      float result = std::tan(a * std::sin(xi) + std::cos(yi));

#if EXTRA_TANGENT_ITERATIONS > 0
      // optionally inject extra expensive compute
      for (int j = 0; j < EXTRA_TANGENT_ITERATIONS; ++j)
        result = std::tan(result);
#endif

      if (result > 1.f)
        out[i] = result;
    }
  }

}  // namespace scalar

// openmp version /////////////////////////////////////////////////////////////

namespace openmp {

  void saxpy_trig(float a, int n, float x[], float y[], float out[])
  {
    #pragma omp for simd
    for (int i = 0; i < n; ++i) {
      const float xi = x[i];
      const float yi = y[i];
      float result = std::tan(a * std::sin(xi) + std::cos(yi));

#if EXTRA_TANGENT_ITERATIONS > 0
      // optionally inject extra expensive compute
      for (int j = 0; j < EXTRA_TANGENT_ITERATIONS; ++j)
        result = std::tan(result);
#endif

      if (result > 1.f)
        out[i] = result;
    }
  }

}  // namespace openmp

// tsimd version //////////////////////////////////////////////////////////////

namespace tsimd {

  void saxpy_trig(float a, int n, float x[], float y[], float out[])
  {
    for (int i = 0; i < n; i += vfloat::static_size) {
      const vfloat xi = load<vfloat>(&x[i]);
      const vfloat yi = load<vfloat>(&y[i]);
      vfloat result = tsimd::tan(a * tsimd::sin(xi) + tsimd::cos(yi));

#if EXTRA_TANGENT_ITERATIONS > 0
      // optionally inject extra expensive compute
      for (int j = 0; j < EXTRA_TANGENT_ITERATIONS; ++j)
        result = tsimd::tan(result);
#endif

      const vboolf write_result = result > 1.f & ((n + lane_index<vint>()) > n);
      if (any(write_result))
        store(result, &out[i], write_result);
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

  const tsimd::vfloat a_v(5.f);
} // namespace params

// specific trig function benchmarks //////////////////////////////////////////

static void sin_scalar(benchmark::State& state)
{
  float result = params::a;
  for (auto _ : state) {
    result = std::sin(result);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(sin_scalar);

static void cos_scalar(benchmark::State& state)
{
  float result = params::a;
  for (auto _ : state) {
    result = std::cos(result);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(cos_scalar);

static void tan_scalar(benchmark::State& state)
{
  float result = params::a;
  for (auto _ : state) {
    result = std::tan(result);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(tan_scalar);

static void sin_tsimd(benchmark::State& state)
{
  auto result = params::a_v;
  for (auto _ : state) {
    result = tsimd::sin(result);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(sin_tsimd);

static void cos_tsimd(benchmark::State& state)
{
  auto result = params::a_v;
  for (auto _ : state) {
    result = tsimd::cos(result);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(cos_tsimd);

static void tan_tsimd(benchmark::State& state)
{
  auto result = params::a_v;
  for (auto _ : state) {
    result = tsimd::tan(result);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(tan_tsimd);

// saxpy_trig benchmarks //////////////////////////////////////////////////////

static void saxpy_trig_scalar(benchmark::State& state)
{
  for (auto _ : state) {
    scalar::saxpy_trig(params::a,
                       params::size,
                       params::x.data(),
                       params::y.data(),
                       params::out.data());
  }
}

BENCHMARK(saxpy_trig_scalar)->Unit(benchmark::kMillisecond);

static void saxpy_trig_openmp(benchmark::State& state)
{
  for (auto _ : state) {
    openmp::saxpy_trig(params::a,
                       params::size,
                       params::x.data(),
                       params::y.data(),
                       params::out.data());
  }
}

BENCHMARK(saxpy_trig_openmp)->Unit(benchmark::kMillisecond);

static void saxpy_trig_tsimd(benchmark::State& state)
{
  for (auto _ : state) {
    tsimd::saxpy_trig(params::a,
                      params::size,
                      params::x.data(),
                      params::y.data(),
                      params::out.data());
  }
}

BENCHMARK(saxpy_trig_tsimd)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

