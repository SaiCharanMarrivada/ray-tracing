using StaticArrays
using LinearAlgebra

struct Interval{T}
    xmin::T
    xmax::T
end

const Universe = Interval(-Inf, Inf)
const Empty = Interval(Inf, -Inf)

function Base.:∈(x::T, interval::Interval{T}) where {T}
    return (x ≥ interval.xmin) && (x ≤ interval.xmax)
end

function surrounds(x::T, interval::Interval{T}) where {T}
    return (x > interval.xmin) && (x < interval.xmax)
end

function clamp(x::R, interval::Interval{T}) where {R,T}
    if x < interval.xmin
        return interval.xmin
    elseif x > interval.xmax
        return interval.xmax
    else
        return x
    end
end

Base.length(interval::Interval{T}) where {T} = interval.xmax - interval.xmin

const Point3{T} = SVector{3,T}
const Color = SVector{3,UInt8}
const Vector3{T} = SVector{3,T}

Color() = Color(zeros(UInt8, 3))

function reflect(v::Vector3{T}, n::Vector3{T}) where {T}
    return v - 2 * (v ⋅ n) * n
end

function random_unit_vector()
    while true
        p = Vector3(rand(), rand(), rand())
        if 1e-160 < (p ⋅ p) ≤ 1
            return normalize(p)
        end
    end
end

function random_on_hemisphere(normal::Vector3{T}) where {T}
    unitvector::Vector3 = random_unit_vector()
    return (unitvector ⋅ normal) > 0 ? unitvector : -unitvector
end

linear_to_γ(linear_component::Float64) = (linear_component > 0) ? sqrt(linear_component) : 0

function write_color(v::Vector3{T}) where {T}
    interval = Ref(Interval{T}(0.000, 0.999))
    v = linear_to_γ.(v)
    color = Color(floor.(256 * clamp.(v, interval)))
    return println(join(color, ' '))
end

function Base.getproperty(v::Vector3{T}, name::Symbol) where {T}
    if name == :x
        return v[1]
    elseif name == :y
        return v[2]
    elseif name == :z
        return v[3]
    else
        throw("Unknown property")
    end
end

struct Ray{T}
    origin::Point3{T}
    direction::Vector3{T}
end

function (ray::Ray{T})(t::T) where {T}
    return ray.origin + t * ray.direction
end

abstract type Material end

struct Lambertian <: Material
    albedo::Vector3{Float64}
end

struct Metal <: Material
    albedo::Vector3{Float64}
end

mutable struct HitRecord{T}
    p::Point3{T}
    normal::Vector3{T}
    material::Material
    t::T
    front_face::Bool
end

function HitRecord{T}() where {T}
    return HitRecord{T}(
        Point3(zeros(T, 3)...),
        Vector3(zeros(T, 3)...),
        Metal(Vector3(zeros(T, 3))),
        zero(T),
        false,
    )
end

function set_face_normal(record::HitRecord{T}, ray::Ray{T}, normal::Vector3{T}) where {T}
    record.front_face = (ray.direction ⋅ normal) < 0
    return record.normal = record.front_face ? normal : -normal
end

abstract type Hittable end

struct Sphere{T} <: Hittable where {T}
    center::Point3{T}
    radius::T
    material::Material
end

function hit(
    sphere::Sphere{T}, ray::Ray{T}, interval::Interval{T}, record::HitRecord{T}
) where {T}
    oc = sphere.center - ray.origin
    a = ray.direction ⋅ ray.direction
    h = ray.direction ⋅ oc
    c = (oc ⋅ oc) - sphere.radius^2

    Δ = h^2 - a * c
    if Δ < 0
        return false
    end

    root = (h - sqrt(Δ)) / a
    if (!surrounds(root, interval))
        root = (h + sqrt(Δ)) / a
        if (!surrounds(root, interval))
            return false
        end
    end

    record.t = root
    record.p = ray(root)
    outward_normal = (record.p - sphere.center) / sphere.radius
    record.material = sphere.material
    set_face_normal(record, ray, outward_normal)

    return true
end

const HittableList = Vector{Hittable}

function hit(
    hittable_list::HittableList, ray::Ray{T}, interval::Interval{T}, record::HitRecord{T}
) where {T}
    closest_so_far = interval.xmax
    hit_anything = false
    for object in hittable_list
        if (hit(object, ray, Interval(interval.xmin, closest_so_far), record))
            closest_so_far = record.t
            hit_anything = true
        end
    end
    return hit_anything
end

function ray_color(hittable_list::HittableList, ray::Ray{T}, depth::Int) where {T}
    if depth ≤ 0
        return Vector3(0, 0, 0)
    end

    record = HitRecord{T}()
    if (hit(hittable_list, ray, Interval(0.001, Inf), record))
        scatter_info = scatter(record.material, ray, record)
        if scatter_info.is_scattered
            return scatter_info.attenuation .*
                   ray_color(hittable_list, scatter_info.scattered, depth - 1)
        end
        # @show scatter_info.attenuation
    end
    unit_vector = normalize(ray.direction)
    a = 0.5 * (unit_vector.y + 1)

    return (1 - a) * Vector3(1, 1, 1) + a * Vector3(0.5, 0.7, 1.0)
end

function scatter(metal::Metal, ray::Ray{T}, record::HitRecord{T}) where {T}
    reflected = reflect(ray.direction, record.normal)
    scattered = Ray(record.p, reflected)
    attenuation = metal.albedo
    return (scattered=scattered, attenuation=attenuation, is_scattered=true)
end

function scatter(lambertian::Lambertian, ray::Ray{T}, record::HitRecord{T}) where {T}
    near_zero(v) = all(abs.(v) .< 1e-8)
    scattered_direction = record.normal + random_unit_vector()

    if (near_zero(scattered_direction))
        scattered_direction = record.normal
    end

    scattered = Ray(record.p, scattered_direction)
    attenuation = lambertian.albedo
    return (scattered=scattered, attenuation=attenuation, is_scattered=true)
end

struct Camera{T}
    image_height::Int
    image_width::Int
    samples_per_pixel::Int
    max_depth::Int
    aspect_ratio::Float64
    center::Point3{T}
    pixel00::Point3{T}
    Δu::Vector3{T}
    Δv::Vector3{T}
end

function Camera(
    image_width::Int, aspect_ratio::Float64; samples_per_pixel=100, max_depth=10
)
    image_height = Int(floor(image_width / aspect_ratio))
    image_height = (image_height < 1) ? 1 : image_height

    focal_length = 1.0
    viewport_height = 2.0
    viewport_width = viewport_height * (image_width / image_height)
    center = Point3(0, 0, 0.0)

    viewport_u = Vector3(viewport_width, 0, 0)
    viewport_v = Vector3(0, -viewport_height, 0)

    Δu = viewport_u / image_width
    Δv = viewport_v / image_height

    pixel00 = (
        center - Vector3(0, 0, focal_length) - (viewport_u / 2) - (viewport_v / 2) +
        (Δu / 2) +
        (Δv / 2)
    )

    return Camera(
        image_height,
        image_width,
        samples_per_pixel,
        max_depth,
        aspect_ratio,
        center,
        pixel00,
        Δu,
        Δv,
    )
end

function render(camera::Camera, world::HittableList)
    println("P3")
    println(camera.image_width, ' ', camera.image_height)
    println(255)

    for j in 0:(camera.image_height - 1)
        print(stderr, "\rScanlines remaining: ", camera.image_height - j, "   ")
        for i in 0:(camera.image_width - 1)
            pixel_color = Vector3{Float64}(0.0, 0.0, 0.0)
            for sample in 1:(camera.samples_per_pixel)
                offset_x = rand() - 0.5
                offset_y = rand() - 0.5
                pixel_sample::Vector3 =
                    camera.pixel00 + (i + offset_x) * camera.Δu + (j + offset_y) * camera.Δv
                ray = Ray(camera.center, pixel_sample - camera.center)
                pixel_color += ray_color(world, ray, camera.max_depth)
            end
            write_color(pixel_color / camera.samples_per_pixel)
        end
    end
    return print(stderr, "\rDone                       \n")
end
