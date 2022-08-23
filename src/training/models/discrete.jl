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
    predict(model::DiscreteModel, u0, t⃗, cfg::Union{Nothing,Tuple{Float32,AbstractArray{<:Number,3}}}=nothing)

Uses the given `model` to predict a trajectory. The initial state `u0` is the state at time `t⃗[1]`, and the output will
be a concatenation of the states at times `t⃗[2:end]`.

Arguments:

- `model`: the `DiscreteModel` to be used for making predictions
- `u0`: the initial state given to the model
- `t⃗`: a vector of time stamps. The first time stamp is that of the initial condition, subsequent time stamps are those
  at which the solution will be returned. Note that since discrete models assume a fixed time step, the time step of `t⃗`
  should always be the same for one specific model in order to obtain valid results.
- `cfg`: an optional parameter to enable teacher forcing. The default, `nothing`, disables teacher forcing. The
  alternative is to set `cfg` to a tuple (`r`, `ref`) where `r` is the teacher forcing rate (between 0 and 1) and `ref`
  is the array of reference solutions (excluding their initial states since those are equal to `u0`).
"""
function predict(model::DiscreteModel, u0, t⃗, cfg::Union{Nothing,Tuple{Float32,AbstractArray{<:Number,3}}}=nothing)
    (Nₓ, _, Nₚ) = size(u0)
    result = zeros(eltype(u0), Nₓ, 0, Nₚ)
    Nₜ = length(t⃗) - 1
    uᵢ = u0
    for iₜ in 1:Nₜ
        uᵢ = model(uᵢ)
        result = hcat(result, uᵢ)

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
    train_discrete!(
        model::DiscreteModel,
        t⃗::AbstractVector{Float32},
        solutions::AbstractArray{<:Number,3};
        exit_condition::ExitCondition=exitcondition(1),
        penalty::Penalty=NoPenalty(),
        validation::Validation=NoValidation(),
        batchsize=8,
        opt=ADAM(),
        loss=Flux.Losses.mse,
        verbose::Bool=true,
        teacherforcing=0.0f0
    )

The main function for training discrete models.

Arguments:

- `model`: the `DiscreteModel` to train
- `t⃗`: the vector of time stamps at which the solutions are saved
- `solutions`: the trajectories, i.e. training data, of the model

The following keyword arguments can be provided:

- `exit_condition`: an `ExitCondition` object containing a maximum number of epochs to train for as well as an optional
  patience parameter for early stopping. Default: 1 epoch, no early stopping
- `penalty`: an optional `Penalty` object that adds a penalty (regularisation) term to the loss function. Default: no
  penalty term
- `validation`: an optional `Validation` object that computes the model error on some validation data. Default: no
  validation
- `batchsize`: the size of batches to use during training. Defaults to 8
- `opt`: the optimiser to use. Defaults to `ADAM()`
- `loss`: the loss function that compares predicted and actual trajectories. Defaults to mean square error
- `verbose`: whether to show a progress bar during training. Defaults to true
- `teacherforcing`: whether or not to do teacher forcing. If set, should be either a constant between 0 and 1, or a
  function so that the teacher forcing constant at epoch `i` is `teacherforcing(i)`. Default: `0.0f0`, i.e. no teacher
  forcing.
"""
function train_discrete!(
    model::DiscreteModel,
    t⃗::AbstractVector{Float32},
    solutions::AbstractArray{<:Number,3};
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
