"""
The parameters of the discretised viscous Burgers equation.

- `ν`: viscosity parameter
- `Δx`: cell size of discretisation

This struct can also be called as a function, in which case it acts as the function `burgers_jameson` with the given parameters.
"""
struct BurgersParams{T<:Number}
    ν::T
    Δx::T
end
(params::BurgersParams)(u) = burgers_jameson(u, params)

"""
`burgers_jameson(u, p, t = nothing)`

Computes the right-hand side of the discretised Burgers equation using A. Jameson's flux scheme.
This implementation is not in-place and is not the most efficient, but is differentiable with Zygote.
"""
function burgers_jameson(u, p, t=nothing)
    (; ν, Δx) = p

    u_fwd = fwd(u)
    u_fwd_diff = u_fwd .- u
    f = (u_fwd .^ 2 + u .* u_fwd + u .^ 2) ./ 6Δx
    α = abs.(u + u_fwd) / 4Δx - u_fwd_diff / 12Δx
    (ν / Δx^2) .* diff2(u) - Δbwd(f .- α .* u_fwd_diff)
end

"""
`burgers_jameson!(du, u, p, t = nothing)`

Computes the right-hand side of the discretised Burgers equation using A. Jameson's flux scheme.
The result is placed in `du`.
This implementation is in-place and more efficient than `burgers_jameson`, but not differentiable.
"""
function burgers_jameson!(du, u, p, t=nothing)
    ν = p.ν
    Δx = p.Δx
    r1 = fwd(u)
    r2 = r1 - u

    # compute convection fluxes into r2
    @. r2 *= abs(u + r1) / 4 - r2 / 12
    @. r2 -= (r1^2 + u * r1 + u^2) / 6
    r2 ./= Δx

    # compute diffusion term into du, then add convection term
    diff2!(du, u)
    du .*= ν / Δx^2
    du .+= Δbwd(r2)

    du
end
