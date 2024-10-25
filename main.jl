include("utils.jl")

function main()
    world = HittableList()
    push!(world, Sphere(Point3(0, 0, -1.0), 0.5))
    push!(world, Sphere(Point3(0, -100.5, -1), 100.0))

    aspect_ratio = 16.0 / 9.0
    image_width = 400
    camera = Camera(image_width, aspect_ratio, samples_per_pixel = 100, max_depth = 50)
    render(camera, world)
end


main()
