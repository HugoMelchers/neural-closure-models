# Test that the manually defined integrators for ODEs produce the same results as those from DifferentialEquations.jl.
# This is done on a simple test problem, namely the Lotka-Volterra system which is solved for about one cycle.

function create_my_solution(my_integrator, f, u, Δt, N)
    result = zeros(length(u), N + 1)
    result[:, 1] .= u[:]
    for i in 2:(N + 1)
        u = my_integrator(f, u, Δt)
        result[:, i] .= u[:]
    end
    result
end

function sol_to_array(sol)
    Nt = length(sol.t)
    Nu = length(sol.u[1])
    result = zeros(Nu, Nt)
    for i in 1:Nt
        result[:, i] .= sol.u[i][:]
    end
    result
end

function create_solutions(my_integrator, diff_eq_alg)
    # initial condition and right-hand side for Lotka-Volterra equations
    u0 = [3.0, 1.0]
    f = u -> [2u[1]/3 - 4u[1] * u[2]/3, u[1]*u[2] - u[2]]

    # size and number of time steps
    Δt = 0.01
    N = 1000
    T = N * Δt
    ts = range(0.0, T, length=N + 1)
    result1 = create_my_solution(my_integrator, f, u0, Δt, N)
    prob = ODEProblem(
        (u, _p, _t) -> f(u),
        u0,
        (0, T),
        nothing
    )
    sol = solve(prob; alg=diff_eq_alg, adaptive=false, dt = Δt, saveat = ts)
    result2 = sol_to_array(sol)
    (result1, result2)
end

function max_error(my_integrator, diff_eq_alg)
    (result1, result2) = create_solutions(my_integrator, diff_eq_alg)
    maximum(abs, result1 .- result2)
end

function test_integrators()
    @assert max_error(euler,    Euler()) < sqrt(eps())
    @assert max_error(  rk2, Midpoint()) < sqrt(eps())
    # no DifferentialEquations equivalent to rk3
    @assert max_error(  rk4,      RK4()) < sqrt(eps())
    @assert max_error(tsit5,    Tsit5()) < sqrt(eps())
end
