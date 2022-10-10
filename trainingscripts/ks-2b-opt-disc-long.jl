begin  # Load code
    @info "Loading code"
    include("../src/neural_closure_models.jl")
end

begin # Load data
    @info "Loading training data"
    using JLD2
    ksdata = load("trainingdata/kuramotosivashinsky.reduced.jld2")
    solutions = ksdata["solutions"]
    derivatives = ksdata["derivatives"]
    parameters = ksdata["parameters"]
    (; t⃗, Δx) = parameters
    f = KSParams(Δx)
    nothing
end

begin # Set up training parameters
    @info "Setting up training"
    exit_condition = ExitCondition(100, nothing)
    opt = Flux.Optimiser(ClipNorm(1e-2), ADAM(0.001))
    Rₚ = 1:80
    Rₜ = 33:177
    trainingdata = solutions[:, Rₜ, Rₚ]
    trainingoutput = derivatives[:, Rₜ, Rₚ]
    verbose = true
    alg = KenCarp47(nlsolve=NLAnderson())
    nothing
end

begin # Create model
    nn = add_Δfwd(create_deep_wide_nn())
    model = ClosureModel(nn, f, "Kuramoto-Sivashinsky closure model, optimise-then-discretise, Nₜ=144")
    nothing
end

begin # Train model and save
    result = train_closure!(
        model, t⃗[Rₜ], trainingdata;
        exit_condition, opt, verbose, kwargs_fw = (; alg), kwargs_bw = (; alg)
    )
    jldsave("ks-2b-opt-disc-long.jld2"; result, model, opt)
end
