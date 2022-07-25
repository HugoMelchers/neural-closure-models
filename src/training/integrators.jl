# Implementations of four different Runge-Kutta (RK) methods for solving ODEs.

"""
`euler(f, u, p, Δt)`

Performs a single time step of size `Δt` of the ODE `du/dt = f(u, p)` using forward Euler, a first-order accurate
1-stage ODE integrator.
"""
function euler(f, u, p, Δt)
    u .+ Δt .* f(u, p)
end
name(::typeof(euler)) = "forward euler"

"""
`rk2(f, u, p, Δt)`

Performs a single time step of size `Δt` of the ODE `du/dt = f(u, p)` using Runge-Kutta 2, a second-order accurat
 2-stage ODE integrator.
"""
function rk2(f, u, p, Δt)
    k₁ = Δt .* f(u, p)
    k₂ = Δt .* f(u .+ k₁./2, p)
    u .+ k₂
end
name(::typeof(rk2)) = "midpoint rule"

"""
`rk3(f, u, p, Δt)`

Performs a single time step of size `Δt` of the ODE `du/dt = f(u, p)` using Runge-Kutta 3, a third-order accurate
3-stage ODE integrator.
"""
function rk3(f, u, p, Δt)
    k₁ = Δt .* f(u, p)
    k₂ = Δt .* f(u .+ k₁./2, p)
    k₃ = Δt .* f(u .+ 2 .* k₂ .- k₁, p)
    @. u + (k₁ + 4k₂ + k₃) / 6
end
name(::typeof(rk3)) = "Runge-Kutta 3"

"""
`rk4(f, u, p, Δt)`

Performs a single time step of size `Δt` of the ODE `du/dt = f(u, p)` using Runge-Kutta 4, a fourth-order accurate
4-stage ODE integrator.
"""
function rk4(f, u, p, Δt)
    k₁ = Δt .* f(u, p)
    k₂ = Δt .* f(u .+ k₁./2, p)
    k₃ = Δt .* f(u .+ k₂./2, p)
    k₄ = Δt .* f(u .+ k₃, p)
    @. u + (k₁ + 2k₂ + 2k₃ + k₄) / 6
end
name(::typeof(rk4)) = "Runge-Kutta 4"

# Implementations of four Exponential Time-Differencing Runge-Kutta (ETDRK) methods. These methods are useful for ODEs
# of the from du/dt = Λu + N(u), where Λ is a stiff pointwise linear function and N is a non-stiff non-linear function.
# Λ should additionally mostly be stiff due to eigenvalues with large negative real components, so that the stiffness
# doesn't make the ODE unstable. Such ODEs are well solved by exponential integrators that integrate the linear term
# exactly (which become exponentials).
using Statistics

mean_over_first(arr) = Float32.(reshape(mean(real.(arr), dims=1), size(arr)[2:end]...))

function integral_sample_points(λ⃗, Δt; M=16)
    pi = Float32(π)
    r = exp.(1im .* pi .* ((1:M) .- 0.5f0) ./ M)
    Δt .* [x + y for y in r, x in λ⃗]
end

"""
A first-order accurate 1-stage exponential Runge-Kutta method.
"""
struct ETDRK1{UType}
    a₁::UType
    b₁::UType
end

"""
Creates a first-order accurate ETDRK scheme for the ODE du/dt = `λ⃗` .* u + (nonlinear term) with time step `Δt`.
"""
function etdrk1(λ⃗, Δt)
    z⃗ = λ⃗ .* Δt

    w⃗ = integral_sample_points(λ⃗, Δt)
    ETDRK1(
        exp.(z⃗),
        Δt .* mean_over_first(@. expm1(w⃗) / w⃗)
    )
end
function (cfg::ETDRK1)(u₀, N)
    (; a₁, b₁) = cfg
    a₁ .* u₀ .+ b₁ .* N(u₀)
end

"""
A second-order accurate 2-stage exponential Runge-Kutta method.
"""
struct ETDRK2{UType}
    a₁::UType
    b₁::UType
    b₂::UType
end

"""
Creates a second-order accurate ETDRK scheme for the ODE du/dt = `λ⃗` .* u + (nonlinear term) with time step `Δt`.
"""
function etdrk2(λ⃗, Δt)
    w⃗ = integral_sample_points(λ⃗, Δt)
    w⃗² = @. w⃗^2

    z⃗ = λ⃗ .* Δt
    ETDRK2(
        exp.(z⃗),
        Δt .* mean_over_first(@. expm1(w⃗) / w⃗),
        Δt .* mean_over_first(@. (expm1(w⃗) - w⃗) / w⃗²),
    )
end
function (cfg::ETDRK2)(u₀, N)
    (; a₁, b₁, b₂) = cfg
    N₀ = N(u₀)
    u₁ = @. a₁ .* u₀ .+ b₁ .* N₀
    N₁ = N(u₁)
    @. u₁ + b₂ * (N₁ - N₀)
end

"""
A third-order accurate 3-stage exponential Runge-Kutta method.
"""
struct ETDRK3{UType}
    a₁::UType
    a₂::UType
    b₁::UType
    b₂::UType
    b₃₁::UType
    b₃₂::UType
    b₃₃::UType
end

"""
Creates a third-order accurate ETDRK scheme for the ODE du/dt = `λ⃗` .* u + (nonlinear term) with time step `Δt`.
"""
function etdrk3(λ⃗, Δt)
    w⃗ = integral_sample_points(λ⃗, Δt)
    w⃗² = @. w⃗^2
    w⃗³ = @. w⃗^3
    eʷ⃗ = @. exp(w⃗)

    z⃗ = λ⃗ .* Δt
    ETDRK3(
        exp.(z⃗ ./ 2),
        exp.(z⃗),
        Δt .* mean_over_first(@. expm1(w⃗ / 2) / w⃗),
        Δt .* mean_over_first(@. expm1(w⃗) / w⃗),
        Δt .* mean_over_first(@. (-4 - w⃗ + eʷ⃗ * (4 - 3w⃗ + w⃗²)) / w⃗³),
        Δt .* mean_over_first(@. 4 * (2 + w⃗ + eʷ⃗ * (w⃗ - 2)) / w⃗³),
        Δt .* mean_over_first(@. (-4 - 3w⃗ - w⃗² + eʷ⃗ * (4 - w⃗)) / w⃗³),
    )
end
function (cfg::ETDRK3)(u₀, N)
    (; a₁, a₂, b₁, b₂, b₃₁, b₃₂, b₃₃) = cfg
    N₀ = N(u₀)
    u₁ = @. a₁ * u₀ + b₁ * N₀
    N₁ = N(u₁)
    u₂ = @. a₂ * u₀ + b₂ * (2N₁ - N₀)
    N₂ = N(u₂)
    @. a₂ .* u₀ + b₃₁ * N₀ + b₃₂ * N₁ + b₃₃ * N₂
end

"""
A fourth-order accurate 4-stage exponential Runge-Kutta method.
"""
struct ETDRK4{UType}
    a₁::UType
    a₂::UType
    b₁::UType
    b₄₁::UType
    b₄₂::UType
    b₄₃::UType
end

"""
Creates a fourth-order accurate ETDRK scheme for the ODE du/dt = `λ⃗` .* u + (nonlinear term) with time step `Δt`.
"""
function etdrk4(λ⃗, Δt)
    z⃗ = λ⃗ .* Δt
    w⃗ = integral_sample_points(λ⃗, Δt)
    w⃗² = @. w⃗^2
    w⃗³ = @. w⃗^3
    eʷ⃗ = @. exp(w⃗)


    ETDRK4(
        exp.(z⃗ ./ 2),
        exp.(z⃗),
        Δt .* mean_over_first(@. expm1(w⃗ / 2) / w⃗),
        Δt .* mean_over_first(@. (-4 - w⃗ + eʷ⃗ * (4 - 3w⃗ + w⃗²)) / w⃗³),
        Δt .* mean_over_first(@. (4 + 2w⃗ + eʷ⃗ * (-4 + 2w⃗)) / w⃗³),
        Δt .* mean_over_first(@. (-4 - 3w⃗ - w⃗² + eʷ⃗ * (4 - w⃗)) / w⃗³),
    )
end

function (cfg::ETDRK4)(u₀, N)
    (; a₁, a₂, b₁, b₄₁, b₄₂, b₄₃) = cfg
    N₀ = N(u₀)
    u₁ = @. a₁ * u₀ + b₁ * N₀
    N₁ = N(u₁)
    u₂ = @. a₁ * u₀ + b₁ * N₁
    N₂ = N(u₂)
    u₃ = @. a₁ * u₁ + b₁ * (2N₂ - N₀)
    N₃ = N(u₃)
    @. a₂ * u₀ + b₄₁ * N₀ + b₄₂ * (N₁ + N₂) + b₄₃ * N₃
end
