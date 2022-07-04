"""
A struct containing the single parameter (namely the grid spacing Δx) of the discretised Kuramoto-Sivashinsky equation.
This struct can also be called as a function, in which case it acts as the function `kuramoto_sivashinsky` with the
given parameters.
"""
struct KSParams{T<:Number}
    Δx::T
end
(params::KSParams)(u) = kuramotosivashinsky(u, params)

"""
`kuramotosivashinsky(u, p, t=nothing)`

Computes the right-hand side of the discretised Kuramoto-Sivashinsky equation.
This implementation only uses out-of-place array operations, so it is differentiable by Zygote but not the most efficient.
"""
function kuramotosivashinsky(u, p, t=nothing)
    (; Δx) = p

    u_fwd = fwd(u)
    f = @. (u_fwd^2 + u * u_fwd + u^2) / 6Δx
    uₓₓ = diff2(u) ./ Δx^2
    uₓₓₓₓ = diff2(uₓₓ) ./ Δx^2
    -Δbwd(f) .- uₓₓ .- uₓₓₓₓ
end

"""
`kuramotosivashinsky!(du, u, p, t=nothing)`

Computes the right-hand side of the discretised Kuramoto-Sivashinsky equation, placing the result in `du`. This
implementation is in-place, so it is more efficient than the out-of-place implementation but cannot be differentiated
using Zygote. Since this ODE is stiff, it is recommended to solve ODEs with this RHS using implicit methods that require
Jacobians, so generally the explicit method (`kuramotosivashinsky`) should be used.
"""
function kuramotosivashinsky!(du, u, p, t=nothing)
    (; Δx) = p
    r1 = fwd(u)

    r2 = @. -(r1^2 + u * r1 + u^2) / 6Δx # r2[j] = -(u[j+1]² + u[j+1]*u[j] + u[j]²)/6Δx
    Δbwd!(du, r2)                        # du = quadratic convection term of RHS

    diff2!(r2, u)
    r2 ./= Δx^2                          # r2 ≈ d²u/dx²
    du .-= r2
    diff2!(r1, r2)
    r1 ./= Δx^2                          # r2 ≈ d²u/dx² + d⁴u/dx⁴
    du .-= r1

    du
end
