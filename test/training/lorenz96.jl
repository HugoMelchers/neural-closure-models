begin # load data
    using JLD2

    l96data = load("trainingdata/lorenz96.reduced.jld2")
    solutions = l96data["solutions"]
    derivatives = l96data["derivatives"]
    parameters = l96data["parameters"]
    ode_params = parameters.params
    f = ReducedLorenz96Params(ode_params)
    Δt = parameters.Δt
    t⃗ = parameters.t⃗
    nothing
end

begin # train discrete delay model
    nn = basic_cnn_1d([2, 4, 8, 8, 4, 2, 1], 5)
    model = DiscreteDelayModel(
        nn, 1, Δt, NoExtrapolator()
    )
    r1 = train_memory!(
        model, t⃗[41:80], solutions[:, 1:40, 1:8], solutions[:, 41:80, 1:8];
        exit_condition=ExitCondition(100, nothing)
    )
    nothing
end

begin # train ANODE model
    nn = basic_cnn_1d([2, 4, 8, 8, 4, 2], 5)
    model = ANODEModel(
        nn, 1, f, rk4, Δt, -4.0f0
    )
    r2 = train_memory!(
        model, t⃗[41:80], solutions[:, 1:40, 1:8], solutions[:, 41:80, 1:8];
        exit_condition=ExitCondition(100, nothing)
    )
    nothing
end
