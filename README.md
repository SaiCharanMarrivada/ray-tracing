[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
# ray-tracing
A simple pure Julia implementation of the book [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Running the code
The code has no dependencies other than `StaticArrays`, which is used for various `struct`s. The ray-traced image can be generated by running the following command.
```bash
# Generate the ray-traced image
julia -t<nthreads> --optimize=3 -C native main.jl > image.ppm
```
The generated image can be viewed with any image viewer.
