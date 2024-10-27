# ray-tracing
Pure julia implementation of [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

## Running the code
The code can be run with `nthreads` by running the following command. The code uses `StaticArrays` for various `struct`s, per-thread *RNG*(Random number generator) is not implemented.
```bash
julia -t<nthreads> --optimize=3 -C native main.jl > image.ppm
convert image.ppm image.png
```
