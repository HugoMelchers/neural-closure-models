begin # load data
    using JLD2

    burgersdata = load("trainingdata/burgers.reduced.jld2")
    solutions = burgersdata["solutions"]
    parameters = burgersdata["parameters"]
    f = BurgersParams(parameters.ν, parameters.Δx)

    nothing
end

begin # load models trained and verify their accuracy
    @info "Loading trained models on Burgers' equation"
    params = load("trained-params/burgers.jld2")["params"]
    function addΔfwd(nn::ContinuousModel)
        ContinuousModel(add_Δfwd(nn.inner), nn.name * "1")
    end
    function make_closure(nn::ContinuousModel)
        ContinuousModel(Flux.Parallel(+, nn.inner, f), nn.name * "2")
    end
    function make_forward_euler(nn::ContinuousModel, Δt)
        DiscreteModel(
            Flux.Parallel(
                +, IdentityLayer(), append_layer(nn.inner, ScaleLayer(Δt))
            ),
            nn.name * "3"
        )
    end
    function make_nns(nn, Δt)
        nn0 = DiscreteModel(nn, "Direct")        # direct model
        nn5 = ContinuousModel(deepcopy(nn), "E") # pure neural ODE
        nn6 = addΔfwd(nn5)                       # momentum-conserving neural ODE
        (nn7, nn8) = make_closure.((nn5, nn6))   # normal and momentum-conserving neural closure models

        (nn1, nn2, nn3, nn4) = make_forward_euler.((nn5, nn6, nn7, nn8), Δt) # discrete-time versions of all previous models
        [nn0, nn1, nn2, nn3, nn4, nn5, nn6, nn7, nn8]
    end
    nn_small = create_basic_nn()
    nn_large = create_deep_wide_nn()
    Δt = 1 / 64.0f0
    nns_small = make_nns(nn_small, Δt)
    nns_large = make_nns(nn_large, Δt)
    nns = hcat(nns_small, nns_large)
    nns = [
        clone_with_params(nn, ps)
        for (nn, ps) in zip(nns, params)
    ]
    nothing
end

begin
    xtest = solutions[:, 1:1, 97:end]
    ytest = solutions[:, 2:end, 97:end]
    ts = parameters.t⃗
    cfgs = [
        ifelse(model isa DiscreteModel, nothing, (; alg=Tsit5()))
        for model in nns
    ]
    @info "Running inference on trained Burgers' models"
    predictions = [
        predict(model, xtest, ts, cfg)
        for (model, cfg) in zip(nns, cfgs)
    ]
    rmses = [rootmeansquare(pred .- ytest) for pred in predictions]
end
