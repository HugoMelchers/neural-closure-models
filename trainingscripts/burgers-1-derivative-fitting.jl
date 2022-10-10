begin # Load code
    @info "Loading code"
    include("../src/neural_closure_models.jl")
end

begin # Load data
    @info "Loading training data"
    using JLD2
    burgersdata = load("trainingdata/burgers.reduced.jld2")
    solutions = burgersdata["solutions"]
    derivatives = burgersdata["derivatives"]
    parameters = burgersdata["parameters"]
    (; Δt, t⃗, ν, Δx) = parameters
    f = BurgersParams(ν, Δx)
    nothing
end

begin # Set up training parameters
    @info "Setting up training"
    exit_condition = ExitCondition(10000, nothing)
    opt = ADAM(0.001)
    trainingdata = solutions[:, :, 1:96]
    trainingoutput = derivatives[:, :, 1:96]
    validationdata = solutions[:, :, 97:end]
    verbose = true
    batchsize = 64
    nothing
end

begin # Create model
    nn = add_Δfwd(create_basic_nn())
    model = ClosureModel(nn, f, "Burgers closure model, derivative fitting")
    nothing
end

begin # Train model and save
    validation = trajectory_rmse(model, validationdata, t⃗, (; alg=Tsit5()))
    result = train_continuous_derivative!(
        model, trainingdata, trainingoutput;
        exit_condition, opt, validation, verbose, batchsize
    )
    jldsave("burgers-1-derivative-fitting.jld2"; result, model, opt)
end
