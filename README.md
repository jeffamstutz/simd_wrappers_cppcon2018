# SIMD Wrappers in C++ at CppCon2018

This project builds some examples used as a demo in my (upcoming) talk at
CppCon 2018.

## Requirements

- C++11
- CMake

## Build instructions

1. Clone the projects (recursive!)

```bash
git clone --recursive http://github.com/jeffamstutz/simd_wrappers_cppcon2018
```

2. Create a build directory and ```cd``` into it

```bash
cd simd_wrappers_cppcon2018
mkdir build
cd build
```

3. Run CMake and build

```bash
cmake ..
make
```

4. Run benchmark examples

```bash
./saxpy
```

...or

```bash
./saxpy_trig
```

...or

```bash
./mandelbrot
```


