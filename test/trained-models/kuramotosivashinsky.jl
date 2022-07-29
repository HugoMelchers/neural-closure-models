begin # load data
    using JLD2

    ksdata = load("trainingdata/kuramotosivashinsky.reduced.jld2")
    solutions = ksdata["solutions"]
    derivatives = ksdata["derivatives"]
    parameters = ksdata["parameters"]
    f = KSParams(parameters.Δx)

    nothing
end

begin # test derivative-fitting trained models
    nn = add_Δfwd(create_deep_wide_nn())
    ps = load("trained-params/ks-derivative-fitting.jld2")["params"]
    @info "Validating derivative-fitting trained neural ODE"
    model1 = ContinuousModel(clone_with_params(nn, ps[:, 1]), "")
    pred1 = predict(model1, solutions[:, 33:33, 91:100], parameters.t⃗[33:225], (; alg=Tsit5()))
    vpts1 = validpredictiontime(solutions[:, 34:225, 91:100], pred1) ./ 24

    @info "Validating derivative-fitting trained neural closure model"
    model2 = ClosureModel(clone_with_params(nn, ps[:, 2]), f, "")
    pred2 = predict(model2, solutions[:, 33:33, 91:100], parameters.t⃗[33:225], (; alg=KenCarp47(nlsolve=NLAnderson())))
    vpts2 = validpredictiontime(solutions[:, 34:225, 91:100], pred2) ./ 24
end

begin # create nn architecture used for eigenvalue-regularisation experiments
    nn = add_Δfwd(create_linear_nn())
    # saved parameters order is slightly different from the order in table 5.9 of the thesis
    order = [2, 1, 3, 4, 5, 6]
    params = jldopen("trained-params/ks-eigenvalue.jld2") do io
        io["params"][order, 1]
    end
    nns = [clone_with_params(nn, ps) for ps in params]

    models = [ContinuousModel(nn, "") for nn in nns]

    ρ⃗ = [spectral_radius_penalty(128, model)() for model in models]
    η⃗ = [spectral_abscissa_penalty(128, model)() for model in models]

    predictions = [
        predict(model, solutions[:, 33:33, 91:100], parameters.t⃗[33:225], (; alg=Tsit5()))
        for model in models
    ]

    vpts = [
        validpredictiontime(solutions[:, 34:225, 91:100], pred)
        for pred in predictions
    ]
end

begin # test models trained with trajectory fitting
    nn = add_Δfwd(create_deep_wide_nn())
    ps, descriptions = jldopen("trained-params/ks-trajectory-fitting.jld2") do file
        file["params"], file["descriptions"]
    end
    non_closure_params = ps[[3, 2], 1, 1]
    closure_params = ps[[3, 2], 1, 2]

    prototype1 = ContinuousModel(nn, "")
    prototype2 = ClosureModel(nn, f, "")

    models1 = [clone_with_params(prototype1, ps) for ps in non_closure_params]
    models2 = [clone_with_params(prototype2, ps) for ps in closure_params]

    @info "Validating non-closure models trained with trajectory fitting"
    preds1 = [
        predict(model, solutions[:, 33:33, 91:100], parameters.t⃗[33:273], (; alg=Tsit5()))
        for model in models1
    ]

    @info "Validating closure models trained with trajectory fitting"
    preds2 = [
        predict(model, solutions[:, 33:33, 91:100], parameters.t⃗[33:273], (; alg=KenCarp47(nlsolve=NLAnderson())))
        for model in models2
    ]

    vpts1 = [
        validpredictiontime(pred, solutions[:, 34:273, 91:100])
        for pred in preds1
    ]

    vpts2 = [
        validpredictiontime(pred, solutions[:, 34:273, 91:100])
        for pred in preds2
    ]

    round.(vcat((stats.(vpts1) ./ 24)...), digits=2) |> display
    round.(vcat((stats.(vpts2) ./ 24)...), digits=2) |> display
end
