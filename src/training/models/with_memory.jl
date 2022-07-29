"""
An ANODEModel solves the Augmented Neural ODE (ANODE) given by

`d/dt [u h] = [f(u) -λ*h] + inner([u h])`

where:

- `inner`: the inner neural network or model containing a neural network that represents the closure and latent parts
    of the ANODE model
- `Nₕ`: the number of columns in the latent vector `h`
- `f`: the non-closure part of the ODE for `u`. This function is called as `f(u)`, i.e. without additional parameters
- `integrator`: the ODE integrator used. Should be one of `euler`, `rk2`, `rk3`, `rk4`
- `Δt`: the time step used to solve the ODE
- `λ`: the decay rate of the memory term `h`
"""
struct ANODEModel{Inner, F, Integrator, TNum}
    inner::Inner
    Nₕ::Int64
    f::F
    integrator::Integrator
    Δt::TNum
    λ::TNum
end
neuralnetwork(model::ANODEModel) = neuralnetwork(model.inner)
name(model::ANODEModel) = "ANODE model, Nₓ×$(model.Nₕ) latent variables"

function _derivative(model::ANODEModel, xh)
    (; inner, Nₕ, f, integrator, Δt, λ) = model
    x = @view xh[:, 1:1, :]
    h = @view xh[:, 2:end, :]
    dxh2 = inner(xh)
    dxh1 = if f === nothing
        hcat(zero(x), -λ.*h)
    else
        hcat(f(x), -λ.*h)
    end
    dxh1 .+ dxh2
end

function (model::ANODEModel)(x)
    (; inner, Nₕ, f, integrator, Δt) = model
    integrator(
        (u, p) -> _derivative(model, u), x, nothing, Δt
    )
end

function warmup(model::ANODEModel, X::AbstractArray{Float32, 3}; inith = :zero)
    Nₕ = model.Nₕ
    Nₓ, Nᵣ, Nₚ = size(X)

    x₀ = X[:, 1:1, :]
    h₀ = if inith == :zero
        zeros(Float32, Nₓ, Nₕ, Nₚ)
    elseif inith == :randn
        randn(Float32, Nₓ, Nₕ, Nₚ)
    end
    v = hcat(x₀, h₀)
    for iᵣ ∈ 2:Nᵣ
        ṽ = model(v)
        uᵢ = X[:, iᵣ:iᵣ, :]
        v = hcat(uᵢ, ṽ[:, 2:end, :])
    end
    v
end

function _predict(model::ANODEModel, v₀::AbstractArray{Float32, 3}, Nₜ)
    Nₓ, Nₕ, Nₚ = size(v₀)
    Nₕ -= 1
    W = zeros(Float32, Nₓ, 0, Nₚ)

    vᵢ = v₀
    for i ∈ 1:Nₜ
        vᵢ = model(vᵢ)
        uᵢ = vᵢ[:, 1:1, :]
        W = hcat(W, uᵢ)
    end
    W
end

"""
`predict(model::ANODEModel, u0, t⃗, cfg)`
"""
function predict(model::ANODEModel, u0::AbstractArray{Float32, 3}, t⃗, cfg)
    inith = if cfg === nothing
        :zero
    else
        cfg.inith
    end
    uh = warmup(model, u0; inith)
    _predict(model, uh, length(t⃗))
end

"""
A `DiscreteDelayModel` is a model that performs a step of `Δt` as

`u(t + Δt) = model(u(t), u(t - Δt), u(t - 2Δt), ..., u(t - NₕΔt))`
"""
struct DiscreteDelayModel{Inner, TNum <: Number, Predictor}
    inner::Inner
    Nₕ::Int64
    Δt::TNum
    predictor::Predictor
end
(model::DiscreteDelayModel)(x) = model.inner(x)
neuralnetwork(model::DiscreteDelayModel) = neuralnetwork(model.inner)
name(model::DiscreteDelayModel) = "windowmodel(history=$(model.Nₕ), extrapolation=$(name(model.predictor)))"

function _predict(model::DiscreteDelayModel, v₀::AbstractArray{Float32, 3}, Nₜ)
    Nₓ, ~, Nₚ = size(v₀)
    W::Array{Float32, 3} = zeros(Float32, Nₓ, 0, Nₚ)
    v::Array{Float32, 3} = v₀
    for iₜ ∈ 1:Nₜ
        u::Array{Float32, 3} = model.predictor(v, model.Δt, model(v))
        W = hcat(W, reshape(u, Nₓ, 1, Nₚ))
        v = hcat(v[:, 2:end, :], u)
    end
    W
end

"""
`predict(model::DiscreteDelayModel, u0, t⃗, cfg)`
"""
function predict(model::DiscreteDelayModel, u0::AbstractArray{Float32, 3}, t⃗, _cfg)
    v₀ = u0[:, (end-model.Nₕ):end, :]
    _predict(model, v₀, length(t⃗))
end

struct NoExtrapolator end
(::NoExtrapolator)(v::AbstractArray{Float32, 3}, Δt, r) = r
name(::NoExtrapolator) = "none"

struct RKPredictor{F, P, I}
    f::F
    p::P
    integrator::I
end
name(rkp::RKPredictor) = "RKPredictor($(name(rkp.integrator)))"
function (rkp::RKPredictor)(v, Δt, r)
    # step of chosen integrator for du/dt = f(u), plus NN correction
    (; f, p, integrator) = rkp
    u = v[:, end:end, :]
    integrator(f, u, p, Δt) .+ Δt .* r
end

function train_memory!(
    model::Union{ANODEModel, DiscreteDelayModel},
    t⃗::AbstractVector{Float32},
    X::AbstractArray{Float32, 3},
    Y::AbstractArray{Float32, 3};
    exit_condition::ExitCondition=exitcondition(1),
    penalty::Penalty=NoPenalty(),
    validation::Validation=NoValidation(),
    batchsize=8,
    opt=ADAM(),
    loss=Flux.Losses.mse,
    verbose=true,
)
    Nₜ = size(X, 2)
    Nₚ = size(X, 3)

    starts = 1:batchsize:Nₚ
    ends = @. min(starts + batchsize - 1, Nₚ)
    trainingdata = [
        (
            view(X, :, :, a:b),
            view(Y, :, :, a:b),
        )
        for (a, b) in zip(starts, ends)
    ]

    ps = Flux.params(neuralnetwork(model))
    log1 = Float32[]
    function trainingloss(x, y)
        ŷ = predict(model, x, t⃗, nothing)
        ℓ = loss(ŷ, y)
        Zygote.ignore() do
            push!(log1, ℓ)
        end
        ℓ + penalty()
    end

    result = TrainLog(exit_condition, ps)
    epochs = if verbose
        ProgressBar(1:exit_condition.max_epochs)
    else
        1:exit_condition.max_epochs
    end
    verbose && set_description(epochs, "Training")
    for epoch in epochs
        Flux.train!(trainingloss, ps, trainingdata, opt)

        rmstrain = sqrt(sum(log1) / length(log1))
        rmstest = validation(epoch)
        empty!(log1)

        descr = update!(result, epoch, ps, rmstrain, rmstest)

        if descr === nothing
            break
        end
        if verbose
            set_multiline_postfix(epochs, "Training "*name(model)*"\n"*descr)
        end
    end
    return result
end
