"""
A machine learning model that takes some state `u(t)` as input and produces an approximation for the state `u(t + Δt)`
for some fixed time step `Δt`.
"""
struct DiscreteModel{Inner}
    inner::Inner
    name::String
end
(model::DiscreteModel)(u) = model.inner(u)
neuralnetwork(model::DiscreteModel) = neuralnetwork(model.inner)
name(model::DiscreteModel) = model.name

"""
`predict(model::DiscreteModel, u0, t⃗, cfg)`

todo before commit: docstring
"""
function predict(model::DiscreteModel, u0, t⃗, cfg::Union{Nothing,Tuple{Float32,AbstractArray{Float32,3}}}=nothing)
    (Nₓ, _, Nₚ) = size(u0)
    result = zeros(eltype(u0), Nₓ, 0, Nₚ)
    Nₜ = length(t⃗) - 1
    uᵢ = u0
    for iₜ in 1:Nₜ
        uᵢ = model(uᵢ)
        result = cat(result, uᵢ; dims=2)

        # if teacher forcing is enabled, interpolate between prediction and reference
        if cfg !== nothing && !iszero(cfg[1])
            r = cfg[1]
            ref = cfg[2][:, iₜ:iₜ, :]
            uᵢ = (1 - r) * uᵢ + r * ref
        end
    end
    result
end

"""
todo before commit: docstring
"""
function train_discrete!(
    model::DiscreteModel,
    t⃗::AbstractVector{Float32},
    solutions::AbstractArray{Float32,3};
    exit_condition::ExitCondition=exitcondition(1),
    penalty::Penalty=NoPenalty(),
    validation::Validation=NoValidation(),
    batchsize=8,
    opt=ADAM(),
    loss=Flux.Losses.mse,
    verbose::Bool=true,
    teacherforcing=0.0f0
)
    initial = @view solutions[:, 1:1, :]
    remainder = @view solutions[:, 2:end, :]
    trainingdata = Flux.DataLoader((initial, remainder); batchsize)
    ps = Flux.params(neuralnetwork(model))
    log1 = Float32[]
    function trainingloss(x, y, tf)
        ŷ = predict(model, x, t⃗, (tf, y))
        l = loss(y, ŷ)
        Zygote.ignore() do
            push!(log1, l)
        end
        l + penalty()
    end

    result = TrainLog(exit_condition, ps)
    modelname = name(model)

    iter = if verbose
        ProgressBar(1:exit_condition.max_epochs)
    else
        1:exit_condition.max_epochs
    end
    verbose && set_description(iter, "Training")
    for epoch in iter
        tf = teacherforcingatepoch(teacherforcing, epoch)

        Flux.train!((x, y) -> trainingloss(x, y, tf), ps, trainingdata, opt)

        rms_train = sqrt(sum(log1) / length(log1))
        rms_test = validation(epoch)
        empty!(log1)

        descr = update!(result, epoch, ps, rms_train, rms_test)

        if descr === nothing
            break
        end
        if verbose
            if modelname !== nothing
                set_multiline_postfix(iter, modelname * "\n" * descr)
            else
                set_multiline_postfix(iter, descr)
            end
        end
    end
    result
end
