begin # load data
    using JLD2

    ksdata = load("trainingdata/kuramotosivashinsky.reduced.jld2")
    solutions = ksdata["solutions"]
    derivatives = ksdata["derivatives"]
    parameters = ksdata["parameters"]
    f = KSParams(parameters.Δx)

    nothing
end

begin # create nn architecture used for eigenvalue-regularisation experiments
    nn = add_Δfwd(create_linear_nn())
    # saved parameters order is slightly different from the order in table 5.9 of the thesis
    order = [2, 1, 3, 4, 5, 6]
    params = jldopen("trained-params/ks-eigenvalue.jld2") do io
        io["params"][order, 2]
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
