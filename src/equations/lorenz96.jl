using ComponentArrays

"""
A struct containing all the parameters of the full Lorenz '96 ODE system.
This struct can also be called as a function, in which case it acts as the function `lorenz96` with the given parameters.
"""
struct Lorenz96Params{T<:Number}
    K::Int64
    J::Int64
    f::T
    ε::T
    hx::T
    hy::T
end
(params::Lorenz96Params)(u) = lorenz96(u, params)

"""
A struct containing all the parameters of the reduced Lorenz '96 ODE system.
This struct can also be called as a function, in which case it acts as the function `lorenz96` with the given parameters.
"""
struct ReducedLorenz96Params{T<:Number}
    K::Int64
    f::T
end
ReducedLorenz96Params(params::Lorenz96Params) = ReducedLorenz96Params(params.K, params.f)
(params::ReducedLorenz96Params)(u) = lorenz96(u, params)

"""
`lorenz96(u, params::Lorenz96Params, t=nothing)`

Computes the right-hand side of the full Lorenz '96 ODE system.
"""
function lorenz96(u, params::Lorenz96Params, t=nothing)
    (; J, f, ε, hx, hy) = params

    x = u.x
    y = u.y
    xₖ₋₁ = bwd(x)
    xₖ₊₁ = fwd(x)
    xₖ₋₂ = bwd(xₖ₋₁)

    yₗ = @view y[:]
    yⱼ₋₁ₖ = bwd(yₗ)
    yⱼ₊₁ₖ = fwd(yₗ)
    yⱼ₊₂ₖ = fwd(yⱼ₊₁ₖ)

    b = hx / J .* sum(y', dims=2)[:]

    du = similar(u)
    @. du.x = xₖ₋₁ * (xₖ₊₁ - xₖ₋₂) - x + f + b
    @. du.y[:] = yⱼ₊₁ₖ * (yⱼ₋₁ₖ - yⱼ₊₂ₖ) - yₗ
    du.y .+= hy * x'
    du.y ./= ε

    du
end

"""
`lorenz96!(du, u, params::Lorenz96Params, t=nothing)`

Computes the right-hand side of the full Lorenz '96 ODE system in-place, storing the result in `du`.
"""
function lorenz96!(du, u, params::Lorenz96Params, t=nothing)
    (; J, f, ε, hx, hy) = params
    x = u.x
    y = u.y
    yₗ = @view y[:]

    dx = du.x
    dy = du.y

    tmp_x1 = similar(x)
    tmp_x2 = similar(x)
    tmp_y1 = similar(yₗ)
    tmp_y2 = similar(yₗ)

    # compute derivatives of y
    circshift!(tmp_y1, yₗ, 1)               # tmp_y1[j] = yₗ[j-1]
    circshift!(tmp_y2, yₗ, -2)              # tmp_y2[j] = yₗ[j+2]
    tmp_y1 .-= tmp_y2                       # tmp_y1[j] = yₗ[j-1] - yₗ[j+2]
    circshift!(tmp_y2, yₗ, -1)
    @. tmp_y1 = (tmp_y1 * tmp_y2 - yₗ) / ε  # tmp_y1[j] = (yₗ[j+1](yₗ[j-1] - yₗ[j+2]) - yₗ[j])/ε
    dy[:] .= tmp_y1
    dy .+= hy / ε .* x'

    circshift!(tmp_x1, x, -1)               # tmp_x1[k] = x[k+1]
    circshift!(tmp_x2, x, 2)                # tmp_x2[k] = x[k-2]
    tmp_x1 .-= tmp_x2                       # tmp_x1[k] = x[k+1] - x[k-2]
    circshift!(tmp_x2, x, 1)
    @. tmp_x1 = tmp_x2 * tmp_x1 - x + f     # tmp_x1[k] = x[k-1]*(x[k+1] - x[k-2]) - x[k] + f
    sum!(dx, y')
    @. dx = (hx / J) * dx + tmp_x1

    du
end

"""
`lorenz96(u, params::ReducedLorenz96Params, t=nothing)`

Computes the right-hand side of the reduced Lorenz '96 ODE system.
"""
function lorenz96(u, params::ReducedLorenz96Params, t=nothing)
    (; f) = params
    u₋₁ = circshift(u, 1)
    u₊₁ = circshift(u, -1)
    u₋₂ = circshift(u, 2)

    @. u₋₁ * (u₊₁ - u₋₂) - u + f
end

"""
`lorenz96!(du, u, params::ReducedLorenz96Params, t=nothing)`

Computes the right-hand side of the reduced Lorenz '96 ODE system in-place, storing the result in `du`.
"""
function lorenz96!(du, u, params::ReducedLorenz96Params, t=nothing)
    (; f) = params
    tmp = similar(u)
    circshift!(du, u, -1)
    circshift!(tmp, u, 2)
    du .-= tmp
    circshift!(tmp, u, 1)
    @. du = du * tmp - u + f

    du
end
