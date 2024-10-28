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

## Implementation details
`Vector4`(alias for `SVector{4}`) is used instead of `SVector{3}`, fourth component is
ignored. All operations on `Vector4`s take less instructions compared to operations on
`Vector3`s

### Example
```julia
v = Vector4{Float64}(ones(4))
w = Vector4{Float64}(ones(4))

@code_native v .* w
```
Assembly output for element-wise product of `Vector4`s:
```asm
 	push	rbp
	mov	rbp, rsp
	mov	rax, rdi
	vmovupd	ymm0, ymmword ptr [rsi]
	vmulpd	ymm0, ymm0, ymmword ptr [rdx]
	vmovupd	ymmword ptr [rdi], ymm0
	pop	rbp
	vzeroupper
	ret

```

When `Vector3`s are used, 
```julia
v = Vector3{Float64}(ones(3))
w = Vector4{Float64}(ones(3))

@code_native v .* w
```
Assembly output for element-wise product of `Vector3`s:
```asm
        push	rbp
	mov	rbp, rsp
	mov	rax, rdi
	vmovsd	xmm0, qword ptr [rsi + 16]  
	vmulsd	xmm0, xmm0, qword ptr [rdx + 16]
	vmovupd	xmm1, xmmword ptr [rsi]
	vmulpd	xmm1, xmm1, xmmword ptr [rdx]
	vmovupd	xmmword ptr [rdi], xmm1
	vmovsd	qword ptr [rdi + 16], xmm0
	pop	rbp
	ret
```

	

