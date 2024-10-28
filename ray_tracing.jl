using StaticArrays
using LinearAlgebra

const Point4{T} = SVector{4,T}
const Color = SVector{3,UInt8}
const Vector4{T} = SVector{4,T}
const Vector3{T} = SVector{3,T}
const Vector2{T} = SVector{2,T}

Color() = Color(zeros(UInt8, 3))
Vector4{T}() where {T<:AbstractFloat} = Vector4(zeros(T, 4))

function reflect(v::Vector4{T}, n::Vector4{T}) where {T}
    return v - 2 * (v ⋅ n) * n
end

function refract(uv::Vector4{T}, n::Vector4{T}, relative_η::T) where {T}
    cosθ = min(-uv ⋅ n, 1.0)
    r_out_perp = relative_η * (uv + cosθ * n)
    r_out_parallel = -sqrt(abs(1 - (r_out_perp ⋅ r_out_perp))) * n
    return r_out_perp + r_out_parallel
end

function random_unit_vector()
    while true
        p = Vector4(rand(), rand(), rand(), 0)
        if 1e-160 < (p ⋅ p) ≤ 1
            return normalize(p)
        end
    end
end

function random_on_hemisphere(normal::Vector4{T}) where {T}

    unitvector::Vector4 = random_unit_vector()
    return (unitvector ⋅ normal) > 0 ? unitvector : -unitvector
end

function random_in_unit_disk()
    while true
        p = Vector2(-1 + 2rand(), -1 + 2rand())
        if (p ⋅ p) < 1
            return p
        end
    end
end

function linear_to_γ(linear_component::T) where {T<:AbstractFloat}
    return (linear_component > 0) ? sqrt(linear_component) : 0
end

struct Ray{T<:AbstractFloat}
    origin::Point4{T}
    direction::Vector4{T}
end

(ray::Ray{T})(t::T) where {T} = ray.origin + t * ray.direction

abstract type Material end

struct Lambertian{T<:AbstractFloat} <: Material
    albedo::Vector4{T}
end

struct Metal{T<:AbstractFloat} <: Material
    albedo::Vector4{T}
    fuzz::T
end

struct Dielectric{T<:AbstractFloat} <: Material
    η::T
end

mutable struct HitRecord{T<:AbstractFloat}
    p::Point4{T}
    normal::Vector4{T}
    material::Material
    t::T
    front_face::Bool
end

function HitRecord{T}() where {T}
    return HitRecord{T}(Point4{T}(), Vector4{T}(), Metal(Vector4{T}(), 0.5), T(0), false)
end

abstract type Hittable end

struct Sphere{T<:AbstractFloat} <: Hittable where {T}
    center::Point4{T}
    radius::T
    material::Material
end

function hit(
    sphere::Sphere{T}, ray::Ray{T}, tmin::T, tmax::T, record::HitRecord{T}
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
    if (root ≤ tmin || root ≥ tmax)
        root = (h + sqrt(Δ)) / a
        if (root ≤ tmin || root ≥ tmax)
            return false
        end
    end

    record.t = root
    record.p = ray(root)
    outward_normal = (record.p - sphere.center) / sphere.radius
    record.material = sphere.material
    record.front_face = (ray.direction ⋅ outward_normal) < 0
    record.normal = record.front_face ? outward_normal : -outward_normal
    return true
end

const HittableList = Vector{Hittable}

function hit(
    hittable_list::HittableList, ray::Ray{T}, tmin::T, tmax::T, record::HitRecord{T}
) where {T}
    closest_so_far = tmax
    hit_anything = false
    for object in hittable_list
        if (hit(object, ray, tmin, closest_so_far, record))
            closest_so_far = record.t
            hit_anything = true
        end
    end
    return hit_anything
end

function ray_color(hittable_list::HittableList, ray::Ray{T}, depth::Int) where {T}
    if depth ≤ 0
        return Vector4{T}()
    end

    record = HitRecord{T}()
    if (hit(hittable_list, ray, T(0.001), T(Inf), record))
        scatter_info = scatter(record.material, ray, record)
        if scatter_info.is_scattered
            return scatter_info.attenuation .*
                   ray_color(hittable_list, scatter_info.scattered, depth - 1)
        end
    end
    unit_vector = normalize(ray.direction)
    a = 0.5 * (unit_vector.y + 1)

    return (1 - a) * Vector4{T}(1, 1, 1, 0) + a * Vector4{T}(0.5, 0.7, 1.0, 0)
end

function scatter(metal::Metal, ray::Ray{T}, record::HitRecord{T}) where {T}
    reflected = reflect(ray.direction, record.normal)
    reflected = normalize(reflected) + metal.fuzz * random_unit_vector()
    scattered = Ray(record.p, reflected)
    attenuation = metal.albedo
    is_scattered = (scattered.direction ⋅ reflected > 0)
    return (scattered=scattered, attenuation=attenuation, is_scattered=is_scattered)
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

function scatter(dielectric::Dielectric, ray::Ray{T}, record::HitRecord{T}) where {T}
    attenuation = Vector4{T}(1, 1, 1, 0)
    relative_η = record.front_face ? 1 / dielectric.η : dielectric.η
    unit_direction = normalize(ray.direction)
    cosθ = min(-unit_direction ⋅ record.normal, 1.0)
    sinθ = sqrt(1 - cosθ * cosθ)

    if (relative_η * sinθ > 1) || (reflectance(cosθ, relative_η) > rand())
        direction = reflect(unit_direction, record.normal)
    else
        direction = refract(unit_direction, record.normal, relative_η)
    end

    scattered = Ray(record.p, direction)
    return (scattered=scattered, attenuation=attenuation, is_scattered=true)
end

function reflectance(cosine::T, refractive_index::T) where {T<:AbstractFloat}
    r0 = (1 - refractive_index) / (1 + refractive_index)
    r0 = r0 * r0
    return r0 + (1 - r0) * (1 - cosine)^5
end

struct Camera{T<:AbstractFloat}
    image_height::Int
    image_width::Int
    samples_per_pixel::Int
    max_depth::Int
    vfov::T
    look_from::Point4{T}
    look_at::Point4{T}
    vup::Point4{T}
    aspect_ratio::T
    center::Point4{T}
    pixel00::Point4{T}
    Δu::Vector4{T}
    Δv::Vector4{T}
    defocus_angle::T
    focus_distance::T
    defocus_disk_u::Vector4{T}
    defocus_disk_v::Vector4{T}
end

function Camera(
    image_width::Int,
    aspect_ratio::T;
    samples_per_pixel=100,
    max_depth=10,
    vfov=20.0,
    defocus_angle=20.0,
    focus_distance=3.4,
    look_from=Point4{T}(-2, 2, 1, 0),
    look_at=Point4{T}(0, 0, -1, 0),
    vup=Vector4{T}(0, 1, 0, 0),
) where {T<:AbstractFloat}
    image_height = unsafe_trunc(Int, image_width / aspect_ratio)
    image_height = (image_height < 1) ? 1 : image_height

    θ = deg2rad(vfov)
    h = tan(θ / 2)
    viewport_height = 2 * h * focus_distance
    viewport_width = viewport_height * (image_width / image_height)
    center = look_from

    w = normalize(look_from - look_at)
    u = Vector4(normalize(vup[1:3] × w[1:3])..., T(0))
    v = Vector4((w[1:3] × u[1:3])..., T(0))

    defocus_radius = focus_distance * tan(deg2rad(defocus_angle / 2))
    defocus_disk_u = defocus_radius * u
    defocus_disk_v = defocus_radius * v

    viewport_u = viewport_width * u
    viewport_v = viewport_height * -v

    Δu = viewport_u / image_width
    Δv = viewport_v / image_height

    pixel00 = (
        center - focus_distance * w - (viewport_u / 2) - (viewport_v / 2) +
        (Δu / 2) +
        (Δv / 2)
    )

    return Camera(
        image_height,
        image_width,
        samples_per_pixel,
        max_depth,
        vfov,
        look_from,
        look_at,
        vup,
        aspect_ratio,
        center,
        pixel00,
        Δu,
        Δv,
        defocus_angle,
        focus_distance,
        defocus_disk_u,
        defocus_disk_v,
    )
end

function defocus_disk_sample(camera::Camera)
    p = random_in_unit_disk()
    return camera.center + (p[1] * camera.defocus_disk_u) + (p[2] * camera.defocus_disk_v)
end

function render(camera::Camera, world::HittableList)
    image = Matrix{Color}(undef, camera.image_width, camera.image_height)

    Threads.@threads for j in 1:(camera.image_height)
        for i in 1:(camera.image_width)
            pixel_color = Vector4{Float64}()
            for sample in 1:(camera.samples_per_pixel)
                offset_x = rand() - 0.5
                offset_y = rand() - 0.5
                pixel_sample::Vector4 =
                    camera.pixel00 + (i + offset_x) * camera.Δu + (j + offset_y) * camera.Δv
                ray_origin =
                    camera.defocus_angle ≤ 0 ? camera.center : defocus_disk_sample(camera)
                ray = Ray(ray_origin, pixel_sample - ray_origin)
                pixel_color += ray_color(world, ray, camera.max_depth)
            end
            pixel_color /= camera.samples_per_pixel
            # tranform color to γ-space
            pixel_color = linear_to_γ.(pixel_color)
            pixel_color = unsafe_trunc.(UInt8, 256 * clamp.(pixel_color, 0.000, 0.999))
            @inbounds image[i, j] = pixel_color[1:3]
        end
    end
    return image
end
