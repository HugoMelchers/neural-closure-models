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

function make_table(table)
    table = table .|> (x -> "$x")
    widths = [4 + maximum(length, table[:, j]) for j in 1:size(table, 2)]
    s = ""
    for i in 1:size(table, 1)
        for j in 1:size(table, 2)
            entry = table[i, j]
            padding = " " ^ (widths[j] - length(entry))
            s *= entry * padding
        end
        s *= "\n"
    end
    return s
end

begin # create validation data sets
    T = solutions[:, :, 91:100]
    Xtest = cat(
        T[:, 1:40, :],
        T[:, 41:80, :],
        T[:, 81:120, :],
        T[:, 121:160, :],
        T[:, 161:200, :],
        T[:, 201:240, :],
        T[:, 241:280, :],
        ; dims=3
    )

    Ytest = cat(
        T[:, 41:360, :],
        T[:, 81:400, :],
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
    predictor = RKPredictor(f, rk4)
    ddmodels = [
        DiscreteDelayModel(nn, layers[1] - 1, Δt, predictor)
        for (nn, layers) in zip(nns[1:4], layercounts[1:4])
    ]
    anodemodels = [
        ANODEModel(nn, layers[1] - 1, f, rk4, Δt, 3.0f0)
        for (nn, layers) in zip(nns[5:8], layercounts[5:8])
    ]

    @info "Validating models for Lorenz96 equation"
    preds = [
        predict(model, Xtest, ts, nothing)
        for model in [ddmodels; anodemodels]
    ]
    vpts = [validpredictiontime(pred, Ytest) / 20 for pred in preds]
end

begin
    header = [
        "Model kind" "Latent space" "Min" "Avg" "Max"
    ]
    descriptions = [
        "Delay" "1"
        "Delay" "3"
        "Delay" "7"
        "Delay" "15"
        "ANODE" "1"
        "ANODE" "3"
        "ANODE" "7"
        "ANODE" "15"
    ]
    vptdata = round.([minimum.(vpts) mean.(vpts) maximum.(vpts)], digits=2)
    table = [
        header
        descriptions vptdata
    ]
    @info "Lorenz '96 results:\n" * make_table(table)
end
