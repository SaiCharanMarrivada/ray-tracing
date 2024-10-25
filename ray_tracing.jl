include("utils.jl")

function main()
    world = HittableList()
    push!(world, Sphere(Point3(0, 0, -1.), 0.5))
    push!(world, Sphere(Point3(0, -100.5, -1), 100.))

    aspect_ratio = 16.0 / 9.0
    image_width = 1920
    camera = Camera(image_width, aspect_ratio, samples_per_pixel=10)
    render(camera, world)
end

main()
