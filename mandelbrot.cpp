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

  inline int mandel(const float c_re, const float c_im, const int count)
  {
    float z_re = c_re, z_im = c_im;
    int i = 0;
    for (; i < count; ++i) {
      if (z_re * z_re + z_im * z_im > 4.f)
        break;

      float new_re = z_re * z_re - z_im * z_im;
      float new_im = 2.f * z_re * z_im;
      z_re         = c_re + new_re;
      z_im         = c_im + new_im;
    }

    return i;
  }

  void mandelbrot(const float x0,
                  const float y0,
                  const float x1,
                  const float y1,
                  const int width,
                  const int height,
                  const int maxIterations,
                  int output[])
  {
    const float dx = (x1 - x0) / width;
    const float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; ++i) {
        const float x = x0 + i * dx;
        const float y = y0 + j * dy;

        const int index = (j * width + i);
        output[index]   = mandel(x, y, maxIterations);
      }
    }
  }

}  // namespace scalar

// openmp version /////////////////////////////////////////////////////////////

namespace openmp {

  #pragma omp declare simd
  inline int mandel(const float c_re, const float c_im, const int count)
  {
    float z_re = c_re, z_im = c_im;
    int i = 0;
    for (; i < count; ++i) {
      if (z_re * z_re + z_im * z_im > 4.f)
        break;

      const float new_re = z_re * z_re - z_im * z_im;
      const float new_im = 2.f * z_re * z_im;
      z_re         = c_re + new_re;
      z_im         = c_im + new_im;
    }

    return i;
  }

  void mandelbrot(const float x0,
                  const float y0,
                  const float x1,
                  const float y1,
                  const int width,
                  const int height,
                  const int maxIterations,
                  int output[])
  {
    const float dx = (x1 - x0) / width;
    const float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {
      #pragma omp for simd
      for (int i = 0; i < width; ++i) {
        const float x = x0 + i * dx;
        const float y = y0 + j * dy;

        const int index = (j * width + i);
        output[index]   = mandel(x, y, maxIterations);
      }
    }
  }

}  // namespace openmp


// tsimd version //////////////////////////////////////////////////////////////

namespace tsimd {

  inline vint mandel(const vboolf _active,
                     const vfloat c_re,
                     const vfloat c_im,
                     const int maxIters)
  {
    vfloat z_re = c_re, z_im = c_im;
    vint vi(0);

    for (int i = 0; i < maxIters; ++i) {
      const auto active = _active & ((z_re * z_re + z_im * z_im) <= 4.f);
      if (tsimd::none(active))
        break;

      const vfloat new_re = z_re * z_re - z_im * z_im;
      const vfloat new_im = 2.f * z_re * z_im;

      z_re = c_re + new_re;
      z_im = c_im + new_im;

      vi = tsimd::select(active, vi + 1, vi);
    }

    return vi;
  }

  void mandelbrot(const float x0,
                  const float y0,
                  const float x1,
                  const float y1,
                  const int width,
                  const int height,
                  const int maxIters,
                  int output[])
  {
    const float dx = (x1 - x0) / width;
    const float dy = (y1 - y0) / height;

    vint laneIndex(0);
    std::iota(laneIndex.begin(), laneIndex.end(), 0);

    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i += vfloat::static_size) {
        const vfloat x(x0 + (i + laneIndex) * dx);
        const vfloat y(y0 + j * dy);

        const auto active = x < width;

        const int index   = (j * width + i);
        const auto result = mandel(active, x, y, maxIters);

        tsimd::store(result, &output[index], active);
      }
    }
  }

} // namespace tsimd

///////////////////////////////////////////////////////////////////////////////
// Define benchmarks
///////////////////////////////////////////////////////////////////////////////

namespace params {
  const unsigned int width  = 1024;
  const unsigned int height = 768;
  const float x0            = -2;
  const float x1            = 1;
  const float y0            = -1;
  const float y1            = 1;
  const int maxIters        = 256;
  alignas(64) std::array<int, width*height> buf;
} // namespace params

static void mandelbrot_scalar(benchmark::State& state)
{
  for (auto _ : state) {
    scalar::mandelbrot(params::x0, params::y0,
                       params::x1, params::y1,
                       params::width, params::height,
                       params::maxIters, params::buf.data());
  }
}

BENCHMARK(mandelbrot_scalar)->Unit(benchmark::kMillisecond);

static void mandelbrot_openmp(benchmark::State& state)
{
  for (auto _ : state) {
    openmp::mandelbrot(params::x0, params::y0,
                       params::x1, params::y1,
                       params::width, params::height,
                       params::maxIters, params::buf.data());
  }
}

BENCHMARK(mandelbrot_openmp)->Unit(benchmark::kMillisecond);

static void mandelbrot_tsimd(benchmark::State& state)
{
  for (auto _ : state) {
    tsimd::mandelbrot(params::x0, params::y0,
                      params::x1, params::y1,
                      params::width, params::height,
                      params::maxIters, params::buf.data());
  }
}

BENCHMARK(mandelbrot_tsimd)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

