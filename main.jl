include("ray_tracing.jl")

function main()
    world = HittableList()
    ground_material = Lambertian(Vector3(0.8, 0.8, 0))
    center_material = Lambertian(Vector3(0.1, 0.2, 0.5))
    left_material = Dielectric(1.50)
    bubble = Dielectric(1.0 / 1.50)

    right_material = Metal(Vector3(0.8, 0.6, 0.2), 1.0)

    push!(world, Sphere(Point3(0, -100.5, -1.0), 100.0, ground_material))
    push!(world, Sphere(Point3(0, 0, -1.2), 0.5, center_material))
    push!(world, Sphere(Point3(-1.0, 0, -1.0), 0.5, left_material))
    push!(world, Sphere(Point3(-1.0, 0, -1.0), 0.4, bubble))
    push!(world, Sphere(Point3(1, 0, -1.0), 0.5, right_material))

    aspect_ratio = 16.0 / 9.0
    image_width = 400
    camera = Camera(image_width, aspect_ratio, samples_per_pixel=100, max_depth=50)

    render(camera, world)
end

main()
