begin # load data
    using JLD2

    ksdata = load("trainingdata/kuramotosivashinsky.reduced.jld2")
    solutions = ksdata["solutions"]
    derivatives = ksdata["derivatives"]
    parameters = ksdata["parameters"]
    f = KSParams(parameters.Δx)
    nothing
end

begin # create neural closure model and train with derivative fitting
    nn = create_deep_wide_nn()
    model = ClosureModel(nn, f, "Neural closure model (KS)")
    alg=KenCarp47(nlsolve=NLAnderson())
    @info "Training neural closure model (derivative fitting, RMSE validation)"
    r = train_continuous_derivative!(
        model, solutions[:, 33:73, 1:8], derivatives[:, 33:73, 1:8];
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model, solutions[:, 33:73, 1:8], parameters.t⃗[33:73], (; alg))
    )
    nothing
end

begin # create neural closure model and train with SplitNeuralODE using KenCarp47 split ODE solver
    nn = create_deep_wide_nn()
    model = ClosureModel(nn, f, "Neural closure model (KS)")
    alg=KenCarp47(nlsolve=NLAnderson())
    @info "Training neural closure model (RMSE validation)"
    r = train_closure!(
        model, parameters.t⃗[33:73], solutions[:, 33:73, 1:8];
        kwargs_fw = (; alg),
        kwargs_bw = (; alg),
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model, solutions[:, 33:73, 1:8], parameters.t⃗[33:73], (; alg))
    )
    nothing
end

begin # create neural closure model and train with SplitNeuralODE using KenCarp47 split ODE solver
    nn = create_deep_wide_nn()
    model = ClosureModel(nn, f, "Neural closure model (KS)")
    alg=KenCarp47(nlsolve=NLAnderson())
    @info "Training neural closure model (VPT validation)"
    train_closure!(
        model, parameters.t⃗[33:73], solutions[:, 33:73, 1:8];
        kwargs_fw = (; alg),
        kwargs_bw = (; alg),
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_vpt(model, solutions[:, 33:73, 1:8], parameters.t⃗[33:73], (; alg))
    )
    nothing
end

begin # test different eigenvalue-based regularisation methods
    nn = add_Δfwd(create_linear_nn())
    model = ContinuousModel(nn, "Continuous Model")

    λ = 0.1f0
    Nₓ = size(solutions, 1)
    models = [deepcopy(model) for _ in 1:4]
    penalties = [
        SpectralRadiusPenalty(Nₓ, models[1]),
        SpectralAbscissaPenalty(Nₓ, models[2]),
        Abs2EigsPenalty(Nₓ, models[3]),
        Real2EigsPenalty(Nₓ, models[4]),
    ]
    alg = Tsit5()
    @info "Traing neural ODEs with spectral penalties"
    for (model, penalty) in zip(models, penalties)
        train_continuous_derivative!(
            model, solutions[:, :, 1:8], derivatives[:, :, 1:8];
            exit_condition=ExitCondition(10, nothing),
            penalty = λ*penalty,
            validation = trajectory_rmse(model, solutions[:, 33:53, 1:8], parameters.t⃗[33:53], (; alg))
        )
    end
    nothing
end

begin # train some pseudospectral closure models for KS
	nn = add_Δfwd(create_deep_wide_nn())
    N = size(solutions, 1)
    l = parameters.L
    Δt = parameters.Δt
    model = ks_etdrk_model(N, l, Δt, nn, 4, "ETDRK4 closure model")
    trainingdata = fft(solutions[:, 33:73, 1:8], 1)
    validationdata = fft(solutions[:, 33:73, :], 1)
    @info "Training pseudospectral model"
    train_discrete!(
        model, parameters.t⃗[33:73], trainingdata;
        exit_condition=ExitCondition(10, nothing),
        validation=trajectory_rmse(model, validationdata, parameters.t⃗[33:73], nothing)
    )
end
