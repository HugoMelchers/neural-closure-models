"""
A neural ODE in which the output is the sum of two terms: a `base` function `f(u)` and a neural closure term `nn(u)`.
Note that the function `f` can be implemented as a layer in the neural network, but this formulation is more explicit
and makes it easier to use split ODE solvers on the resulting neural ODE.
"""
struct ClosureModel{Inner, F}
    inner::Inner
    f::F
    name::String
end
(model::ClosureModel)(u) = model.inner(u) .+ model.f(u)
neuralnetwork(model::ClosureModel) = neuralnetwork(model.inner)
name(model::ClosureModel) = model.name

"""
    predict(model::ClosureModel, u0, t⃗, cfg=(;))

Computes the predicted trajectory of the neural closure model `du/dt = model(u)`, from the initial state `u0`. The
solution is saved at time points `t⃗[2:end]`, where `u0` is the state at `t⃗[1]`. Additional keyword arguments to the
`solve` call can be passed in `cfg`. By default, no ODE solver is specified meaning that the ODE solver used is chosen
by `DifferentialEquations.jl`. In practice in order to make use of the split ODE formulation one should use a split ODE
solver such as `KenCarp47`.
"""
function predict(model::ClosureModel, u0, t⃗, cfg=(;))
    prob = SplitODEProblem(
        (u, _p, _t) -> model.f(u),
        (u, _p, _t) -> model.inner(u),
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
    train_closure!(
        model::ClosureModel,
        t⃗::AbstractVector{Float32},
        solutions::AbstractArray{Float32,3};
        exit_condition::ExitCondition=exitcondition(1),
        penalty::Penalty=NoPenalty(),
        validation::Validation=NoValidation(),
        batchsize=8,
        opt=ADAM(),
        loss=Flux.Losses.mse,
        verbose::Bool=true,
        kwargs_fw=(; alg=KenCarp47(nlsolve=NLAnderson())),
        kwargs_bw=(; alg=KenCarp47(nlsolve=NLAnderson())),
    )

Trains the given neural closure model `model` on the 3-dimensional array `solutions`. `solutions` should be a 3-dimensional array
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
- `kwargs_fw`: keyword arguments to supply to the ODE `solve` call for the forward ODEs
- `kwargs_bw`: keyword arguments to supply to the ODE `solve` call for the backward (adjoint) ODEs
"""
function train_closure!(
    model::ClosureModel,
    t⃗::AbstractVector{Float32},
    solutions::AbstractArray{Float32,3};
    exit_condition::ExitCondition=exitcondition(1),
    penalty::Penalty=NoPenalty(),
    validation::Validation=NoValidation(),
    batchsize=8,
    opt=ADAM(),
    loss=Flux.Losses.mse,
    verbose::Bool=true,
    kwargs_fw=(; alg=KenCarp47(nlsolve=NLAnderson())),
    kwargs_bw=(; alg=KenCarp47(nlsolve=NLAnderson())),
)
    initial = @view solutions[:, 1:1, :]
    remainder = @view solutions[:, 2:end, :]
    trainingdata = Flux.DataLoader((initial, remainder); batchsize)

    (ps_nn_ode, re) = Flux.destructure(neuralnetwork(model))
    ps = Flux.params(ps_nn_ode,)
    dudt_nn = (u, p, _t) -> re(p)(u)
    prob_nn_ode = SplitNeuralODE(model.f, dudt_nn, t⃗, kwargs_fw, kwargs_bw)

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
