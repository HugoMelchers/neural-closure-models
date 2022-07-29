begin # load data
    using JLD2

    l96data = load("trainingdata/lorenz96.reduced.jld2")
    solutions = l96data["solutions"]
    derivatives = l96data["derivatives"]
    parameters = l96data["parameters"]
    Δt = parameters.Δt
    f = ReducedLorenz96Params(parameters.params)

    nothing
end

begin # create validation data sets
    T = solutions[:, :, 91:100]
    Xtest = cat(
        T[:,   1: 40, :],
        T[:,  41: 80, :],
        T[:,  81:120, :],
        T[:, 121:160, :],
        T[:, 161:200, :],
        T[:, 201:240, :],
        T[:, 241:280, :],
        ; dims=3
    )

    Ytest = cat(
        T[:,  41:360, :],
        T[:,  81:400, :],
        T[:, 121:440, :],
        T[:, 161:480, :],
        T[:, 201:520, :],
        T[:, 241:560, :],
        T[:, 281:600, :],
        ; dims=3
    )
    ts = parameters.t⃗[1:320]
end

begin # recreate models
    loaded = load("trained-params/lorenz96.params.jld2")["bestparams"]
    layercounts = [
        # [1, 2, 4, 8, 8, 4, 2, 1],
        [2, 4, 8, 8, 4, 2, 1],
        [4, 16, 12, 8, 4, 2, 1],
        [8, 32, 16, 8, 4, 2, 1],
        [16, 64, 32, 16, 8, 4, 2, 1],
        [2, 4, 8, 8, 4, 2],
        [4, 16, 16, 12, 8, 4],
        [8, 32, 32, 16, 12, 8],
        [16, 64, 64, 32, 24, 16]
    ]

    nns = [basic_cnn_1d(layers, 5) for layers in layercounts]
    for (nn, ps) in zip(nns, loaded)
        set_params!(nn, ps)
    end
    predictor = RKPredictor((u, _p) -> f(u), nothing, rk4)
    ddmodels = [
        DiscreteDelayModel(
            nn, layers[1] - 1, Δt, predictor
        )
        for (nn, layers) in zip(nns[1:4], layercounts[1:4])
    ]
    anodemodels = [
        ANODEModel(
            nn, layers[1] - 1, f, rk4, Δt, 3.0f0
        )
        for (nn, layers) in zip(nns[5:8], layercounts[5:8])
    ]

    @info "Validating models for Lorenz96 equation"
    preds = [
        predict(
            model, Xtest, ts, nothing
        )
        for model in [ddmodels; anodemodels]
    ]
    Tlyap = 0.8f0
    round.(mean(hcat([validpredictiontime(pred, Ytest) for pred in preds]...), dims=1)[:] * Δt ./ Tlyap, digits=2) |> display
end
