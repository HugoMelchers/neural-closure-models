"""
A neural ODE, defined by `inner`, that has been discretised. Given an input `u(t)`, the resulting model returns output
`u(t + Δt)` by taking one time step of size `Δt` of the neural ODE `du/dt = inner(u)`. The `integrator` should be one
of `euler`, `rk2`, `rk3`, `rk4`. The resulting model should be wrapped in a `DiscreteModel`, so that it can be given a
name and so that the `train_discrete!` function can be used to train the model.
"""
struct DiscretisedODE{Inner, I, T}
    inner::Inner
    integrator::I
    Δt::T
end
neuralnetwork(model::DiscretisedODE) = neuralnetwork(model.inner)
function (model::DiscretisedODE)(u)
    (; inner, integrator, Δt) = model
    integrator(inner, u, Δt)
end

function discretise(model::ContinuousModel, integrator, Δt)
    DiscreteModel(
        DiscretisedODE(model.inner, integrator, Δt),
        name(model) * ", discretised with " * name(integrator)
    )
end
