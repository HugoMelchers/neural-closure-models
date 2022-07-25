begin # load data
    using JLD2

    burgersdata = load("trainingdata/burgers.reduced.jld2")
    solutions = burgersdata["solutions"]
    parameters = burgersdata["parameters"]
    f = BurgersParams(parameters.ν, parameters.Δx)

    nothing
end

begin # create model
    nn1 = create_basic_nn()
    model1 = DiscreteModel(nn1, "Basic discrete model")
    nn2 = Flux.Parallel(
        +,
        IdentityLayer(),
        Flux.Chain(
            deepcopy(nn1).layers...,
            ScaleLayer(parameters.Δt),
        )
    )
    model2 = DiscreteModel(nn2, "Basic forward Euler model")
    nothing
end

begin # train discrete models
    @info "Training discrete models"
    train_discrete!(
        model1, parameters.t⃗, solutions[:, :, 1:96];
        exit_condition=exitcondition(10, nothing),
        validation=trajectory_rmse(model1, solutions[:, :, 97:128], parameters.t⃗, nothing),
        penalty = ScaledPenalty(L2Penalty(Flux.params(neuralnetwork(model1))), 1f-4)
    )
    train_discrete!(
        model2, parameters.t⃗, solutions[:, :, 1:96];
        exit_condition=exitcondition(10, nothing),
        validation=trajectory_rmse(model2, solutions[:, :, 97:128], parameters.t⃗, nothing),
        penalty = ScaledPenalty(L2Penalty(Flux.params(neuralnetwork(model2))), 1f-4)
    )
    nothing
end

begin # train discrete models with teacher forcing
    @info "Training discrete models with teacher forcing"
    train_discrete!(
        model1, parameters.t⃗, solutions[:, :, 1:96];
        exit_condition=exitcondition(10, nothing),
        validation=trajectory_rmse(model1, solutions[:, :, 97:128], parameters.t⃗, nothing),
        penalty = ScaledPenalty(L2Penalty(Flux.params(neuralnetwork(model1))), 1f-4),
        teacherforcing=0.9f0
    )
    train_discrete!(
        model2, parameters.t⃗, solutions[:, :, 1:96];
        exit_condition=exitcondition(10, nothing),
        validation=trajectory_rmse(model2, solutions[:, :, 97:128], parameters.t⃗, nothing),
        penalty = ScaledPenalty(L2Penalty(Flux.params(neuralnetwork(model2))), 1f-4),
        teacherforcing=0.9f0
    )
    nothing
end

begin # create and train neural ODE
    @info "Training neural ODE (trajectory fitting)"
    nn3 = create_basic_nn()
    model3 = ContinuousModel(nn3, "Basic neural ODE")
    train_continuous!(
        model3, parameters.t⃗, solutions[:, :, 1:96];
        solve_kwargs = (;alg=Tsit5()),
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model3, solutions[:, :, 97:128], parameters.t⃗, (;alg=Tsit5()))
    )
    nothing
end


begin # create and train neural ODE (derivative fitting)
    @info "Training neural ODE (derivative fitting)"
    nn4 = create_basic_nn()
    model4 = ContinuousModel(nn4, "Basic neural ODE")
    train_continuous_derivative!(
        model4, solutions[:, :, 1:96], burgersdata["derivatives"][:, :, 1:96];
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model4, solutions[:, :, 97:128], parameters.t⃗, (;alg=Tsit5()))
    )
    nothing
end

begin # create and train neural closure model (with f(u) term embedded in neural network)
    nn5 = create_basic_nn()
    nn5b = Flux.Parallel(
        +,
        nn5,
        f
    )
    model5 = ContinuousModel(nn5b, "Neural closure model (embedded in NN)")
    @info "Training closure model 1 (trajectory fitting)"
    train_continuous!(
        model5, parameters.t⃗, solutions[:, :, 1:96];
        solve_kwargs = (;alg=Tsit5()),
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model5, solutions[:, :, 97:128], parameters.t⃗, (;alg=Tsit5()))
    )
    nothing
end

begin # create and train neural closure model (with f(u) term embedded in neural network)
    nn6 = create_basic_nn()
    model6 = ClosureModel(nn6, f, "Neural closure model explicitly separated")
    @info "Training closure model 2 (trajectory fitting)"
    train_closure!(
        model6, parameters.t⃗, solutions[:, :, 1:8];
        kwargs_fw = (;alg=Tsit5()),
        kwargs_bw = (;alg=Tsit5()),
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model6, solutions[:, :, 1:8], parameters.t⃗, (;alg=Tsit5()))
    )
    nothing
end

begin # create and train neural closure model (with f(u) term embedded in neural network), using derivative fitting
    @info "Training closure model 2 (derivative fitting)"
    train_continuous_derivative!(
        model6, solutions[:, :, 1:8], burgersdata["derivatives"][:, :, 1:8];
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model6, solutions[:, :, 1:8], parameters.t⃗, (;alg=Tsit5()))
    )
    nothing
end
