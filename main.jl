include("ray_tracing.jl")

function main()
    world = HittableList()

    ground_material = Lambertian(Vector3(0.5, 0.5, 0.5))
    push!(world, Sphere(Point3(0.0, -1000, 0.0), 1000.0, ground_material))

    for a in -11:10
        for b in -11:10
            choose_material = rand()
            center = Point3(a + 0.9rand(), 0.2, b + 0.9rand())

            p = center - Point3(4, 0.2, 0)
            if p ⋅ p > 0.9
                if choose_material < 0.8
                    albedo = Vector3(rand(3)) .* Vector3(rand(3))
                    material = Lambertian(albedo)
                    push!(world, Sphere(center, 0.2, material))
                elseif choose_material < 0.95
                    albedo = Vector3(0.5rand(3) .+ 0.5)
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
    push!(world, Sphere(Point3(0.0, 1, 0), 1.0, material1))

    material2 = Lambertian(Vector3(0.4, 0.2, 0.1))
    push!(world, Sphere(Point3(-4, 1, 0.0), 1.0, material2))

    material3 = Metal(Vector3(0.7, 0.6, 0.5), 0.0)
    push!(world, Sphere(Point3(4, 1, 0.0), 1.0, material3))

    camera = Camera(
        1200,
        16.0 / 9.0;
        samples_per_pixel=500,
        max_depth=50,
        vfov=20.0,
        defocus_angle=0.6,
        focus_distance=10.0,
        look_from=Point3(13.0, 2.0, 3.0),
        look_at=Point3(0.0, 0.0, 0.0),
        vup=Point3(0.0, 1.0, 0.0),
    )

    render(camera, world)
end

main()
