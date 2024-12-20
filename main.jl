include("ray_tracing.jl")

function main()
    world = HittableList()

    ground_material = Lambertian(Vector4(0.5, 0.5, 0.5, 0.0))
    push!(world, Sphere(Point4(0.0, -1000, 0.0, 0.0), 1000.0, ground_material))

    for a in -11:10
        for b in -11:10
            choose_material = rand()
            center = Point4(a + 0.9rand(), 0.2, b + 0.9rand(), 0.0)

            p = center - Point4(4, 0.2, 0, 0.0)
            if p ⋅ p > 0.9
                if choose_material < 0.8
                    albedo = Vector4(rand(4)) .* Vector4(rand(4))
                    material = Lambertian(albedo)
                    center2 = center + Vector4{Float64}(0, 0.5 * rand(), 0, 0)
                    push!(world, Sphere(center, center2, 0.2, material))
                elseif choose_material < 0.95
                    albedo = Vector4(0.5rand(4) .+ 0.5)
                    fuzz = 0.5rand()
                    material = Metal(albedo, fuzz)
                    push!(world, Sphere(center, 0.2, material))
                else
                    material = Dielectric(1.5)
                    push!(world, Sphere(center, 0.2, material))
                end
            end
        end
    end

    material1 = Dielectric(1.5)
    push!(world, Sphere(Point4(0.0, 1, 0, 0.0), 1.0, material1))

    material2 = Lambertian(Vector4(0.4, 0.2, 0.1, 0.0))
    push!(world, Sphere(Point4(-4, 1, 0.0, 0.0), 1.0, material2))

    material3 = Metal(Vector4(0.7, 0.6, 0.5, 0.0), 0.0)
    push!(world, Sphere(Point4(4, 1, 0.0, 0.0), 1.0, material3))

    camera = Camera(
        400,
        16.0 / 9.0;
        samples_per_pixel=100,
        max_depth=50,
        vfov=20.0,
        defocus_angle=0.6,
        focus_distance=10.0,
        look_from=Point4(13.0, 2.0, 3.0, 0.0),
        look_at=Point4(0.0, 0.0, 0.0, 0.0),
        vup=Point4(0.0, 1.0, 0.0, 0.0),
    )

    image = render(camera, world)

    # write image in ppm format
    println("P3\n")
    println(camera.image_width, ' ', camera.image_height)
    println(255)

    for j in 1:camera.image_height
        for i in 1:camera.image_width
            println(join(image[i, j], ' '))
        end
    end
end

main()
