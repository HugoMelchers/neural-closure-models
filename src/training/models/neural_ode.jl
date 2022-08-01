using DifferentialEquations, DiffEqFlux

"""
A neural ODE, i.e. a machine learning model that takes some state `u(t)` as input and produces an approximation for the
time-derivative `du/dt` as output.
"""
struct ContinuousModel{Inner}
    inner::Inner
    name::String
end
(model::ContinuousModel)(u) = model.inner(u)
neuralnetwork(model::ContinuousModel) = neuralnetwork(model.inner)
name(model::ContinuousModel) = model.name

"""
    predict(model::ContinuousModel, u0, t⃗, cfg=(; alg=Tsit5()))

Computes the predicted trajectory of the neural ODE `du/dt = model(u)`, from the initial state `u0`. The solution is
saved at time points `t⃗[2:end]`, where `u0` is the state at `t⃗[1]`. Additional keyword arguments to the `solve` call
can be passed in `cfg`. For example, to solve the neural ODE with Runge-Kutta 4, one can set `cfg = (; alg=RK4())`. By
default, the neural ODE is solved using `Tsit5`.
"""
function predict(model::ContinuousModel, u0, t⃗, cfg=(; alg=Tsit5()))
    prob = ODEProblem(
        (u, _p, _t) -> model(u),
        u0,
        (t⃗[begin], t⃗[end]),
        nothing,
        saveat = t⃗
    )

    sol = solve(prob; cfg...)
    result = Array(sol)[:, 1, :, 2:end]::Array{Float32, 3}
    permutedims(result, (1, 3, 2))
end

"""
    train_continuous!(
        model::ContinuousModel,
        t⃗::AbstractVector{Float32},
        solutions::AbstractArray{Float32,3};
        exit_condition::ExitCondition=exitcondition(1),
        penalty::Penalty=NoPenalty(),
        validation::Validation=NoValidation(),
        batchsize=8,
        opt=ADAM(),
        loss=Flux.Losses.mse,
        verbose::Bool=true,
        solve_kwargs=(;)
    )

Trains the given neural ODE `model` on the 3-dimensional array `solutions`. `solutions` should be a 3-dimensional array
with dimensions ordered as variable index, time stamp, problem index.

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
- `solve_kwargs`: other keyword arguments that can be passed to the ODE solver
"""
function train_continuous!(
    model::ContinuousModel,
    t⃗::AbstractVector{Float32},
    solutions::AbstractArray{Float32,3};
    exit_condition::ExitCondition=exitcondition(1),
    penalty::Penalty=NoPenalty(),
    validation::Validation=NoValidation(),
    batchsize=8,
    opt=ADAM(),
    loss=Flux.Losses.mse,
    verbose::Bool=true,
    solve_kwargs=(;)
)
    initial = @view solutions[:, 1:1, :]
    remainder = @view solutions[:, 2:end, :]
    trainingdata = Flux.DataLoader((initial, remainder); batchsize)

    prob_nn_ode = NeuralODE(model.inner, (t⃗[begin], t⃗[end]), solve_kwargs.alg, saveat = t⃗; solve_kwargs...)
    ps_nn_ode = prob_nn_ode.p
    ps = Flux.params(ps_nn_ode,)

    function predict(x)
        W = Array(prob_nn_ode(x, ps_nn_ode))
        permutedims(W[:, 1, :, 2:end], (1, 3, 2))
    end

    log1 = Float32[]
    function trainingloss(x, y)
        ŷ = predict(x)
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
        Flux.train!(trainingloss, ps, trainingdata, opt)

        # Copy the parameters of the DiffEqFlux NeuralODE back into the `model`. This is done after each epoch of
        # training instead of just at the end, so that the `validation` is done with the up-to-date model parameters.
        set_params!(model, ps_nn_ode)

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

"""
    train_continuous_derivative!(
        model,
        solutions::AbstractArray{Float32,3},
        derivatives::AbstractArray{Float32,3};
        exit_condition::ExitCondition=exitcondition(1),
        penalty::Penalty=NoPenalty(),
        validation::Validation=NoValidation(),
        batchsize=8,
        opt=ADAM(),
        loss=Flux.Losses.mse,
        verbose::Bool=true,
    )

Trains the given `model` by derivative fitting. `solutions` and `derivatives` should be 3-dimensional arrays of equal
dimensions whose columns (i.e. solutions[:, i, j] and derivatives[:, i, j]) are input-output pairs consisting of the
state `u(t)` and its derivative `du/dt`.

The following keyword arguments can be provided:

- `exit_condition`: an `ExitCondition` object containing a maximum number of epochs to train for as well as an optional
    patience parameter for early stopping. Default: 1 epoch, no early stopping
- `penalty`: an optional `Penalty` object that adds a penalty (regularisation) term to the loss function. Default: no penalty term
- `validation`: an optional `Validation` object that computes the model error on some validation data. Default: no validation
- `batchsize`: the size of batches to use during training. Defaults to 8
- `opt`: the optimiser to use. Defaults to `ADAM()`
- `loss`: the loss function that compares predicted and actual trajectories. Defaults to mean square error
- `verbose`: whether to show a progress bar during training. Defaults to true
"""
function train_continuous_derivative!(
    model,
    solutions::AbstractArray{Float32,3},
    derivatives::AbstractArray{Float32,3};
    exit_condition::ExitCondition=exitcondition(1),
    penalty::Penalty=NoPenalty(),
    validation::Validation=NoValidation(),
    batchsize=8,
    opt=ADAM(),
    loss=Flux.Losses.mse,
    verbose::Bool=true,
)
    (Nₓ, Nₜ, Nₚ) = size(solutions)
    _solutions = reshape(solutions, (Nₓ, 1, Nₜ * Nₚ))
    _derivatives = reshape(derivatives, (Nₓ, 1, Nₜ * Nₚ))
    trainingdata = Flux.DataLoader((_solutions, _derivatives); batchsize)

    ps = Flux.params(neuralnetwork(model))

    log1 = Float32[]
    function trainingloss(x, y)
        ŷ = model(x)
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
        Flux.train!(trainingloss, ps, trainingdata, opt)

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
