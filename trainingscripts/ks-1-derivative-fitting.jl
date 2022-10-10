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
    exit_condition = ExitCondition(1000, nothing)
    opt = ADAM(0.001)
    Rₚ = 1:80
    Rₜ = 33:513
    trainingdata = solutions[:, Rₜ, Rₚ]
    trainingoutput = derivatives[:, Rₜ, Rₚ]
    validationdata = solutions[:, 33:273, 81:90]
    verbose = true
    batchsize = 128
    nothing
end

begin # Create model
    nn = add_Δfwd(create_deep_wide_nn())
    model = ClosureModel(nn, f, "Kuramoto-Sivashinsky closure model, derivative fitting")
    nothing
end

begin # Train model and save
    result = train_continuous_derivative!(
        model, trainingdata, trainingoutput;
        exit_condition, opt, verbose, batchsize
    )
    jldsave("ks-1-derivative-fitting.jld2"; result, model, opt)
end
