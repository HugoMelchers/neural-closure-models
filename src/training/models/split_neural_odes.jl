"""
A struct representing a neural closure model ODE problem of the form du/dt = f(u) + nn(u). Gradients are computed using
the interpolating adjoint method, where both the forward and adjoint ODEs can be solved using a Split ODE solver.
"""
struct SplitNeuralODE{F, Inner, TS, ArgsFW, ArgsBW}
    f::F
    inner::Inner
    ts::TS
    kwargs_fw::ArgsFW
    kwargs_bw::ArgsBW
end

"""
A struct representing the adjoint ODE of the split neural ODE `ode`. This struct is used to perform backpropagation
if the adjoint ODE method for gradient computation is used.
"""
struct SplitNeuralODEAdjoint{ODE, Sol, Θ}
    ode::ODE
    sol::Sol
    ϑ::Θ
end
(adj::SplitNeuralODEAdjoint)(∇) = begin
    tstops = adj.ode.ts[1:end-1]

    p = (f = adj.ode.f, nn = adj.ode.nn, ϑ = adj.ϑ, u = adj.sol)

    update_grads_condition(u, t, integrator) = t ∈ tstops
    function update_grads_affect!(integrator)
        t = integrator.t
        i = findfirst(tstops .== t)
        integrator.u.a .+= @view ∇[:, :, :, i]
    end

    prob_backward = SplitODEProblem(
        _adjoint_ode_term1!,
        _adjoint_ode_term2!,
        ComponentArray(a=∇[:, :, :, end], b=zero(adj.ϑ)),
        (adj.ode.ts[end], adj.ode.ts[1]),
        p,
        callback = DiscreteCallback(update_grads_condition, update_grads_affect!, save_positions = (true, true))
    )
    sol_backward = solve(prob_backward, saveat = [zero(eltype(adj.ode.ts))]; tstops, adj.ode.kwargs_bw...)
    (nothing, sol_backward.u[end].a, sol_backward.u[end].b)
end


(ode::SplitNeuralODE)(u₀, ϑ) = begin
    prob = SplitODEProblem(
        (u, p, t) -> ode.f(u),
        ode.inner,
        u₀,
        extrema(ode.ts),
        ϑ
    )
    sol = solve(prob, saveat = ode.ts; ode.kwargs_fw...)
    Array(sol)
end

Zygote.@adjoint (ode::SplitNeuralODE)(u₀, ϑ) = begin
    prob = SplitODEProblem(
        (u, p, t) -> ode.f(u),
        ode.inner,
        u₀,
        extrema(ode.ts),
        ϑ
    )
    sol = solve(prob; ode.kwargs_fw...)
    # todo: this is not good for performance, should either collect arrays in loop, or even better in function with manual adjoint
    cat([sol(t) for t in ode.ts]..., dims = 4), SplitNeuralODEAdjoint(ode, sol, ϑ)
end

function _adjoint_ode_term1!(dab, ab, p, t)
    (; f, u) = p
    uₜ = u(t)
    (_, pb) = Zygote.pullback(f, uₜ)
    grads = pb(-ab.a)
    # special case if f is chosen to be constant or zero for pure neural ODEs
    dab.a .= ifelse(grads[1] === nothing, 0.0f0, grads[1])
    dab.b .= 0
end

function _adjoint_ode_term2!(dab, ab, p, t)
    (nn, ϑ, u) = (p.nn, p.ϑ, p.u)
    uₜ = u(t)
    (_, pb) = Zygote.pullback(nn, uₜ, ϑ, t)
    grads = pb(-ab.a)
    dab.a .= grads[1]
    dab.b .= grads[2]
end
