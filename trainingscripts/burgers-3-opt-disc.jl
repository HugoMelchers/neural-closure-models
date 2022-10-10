begin # Load code
    @info "Loading code"
    include("../src/neural_closure_models.jl")
end

begin # Load data
    @info "Loading training data"
    using JLD2
    burgersdata = load("trainingdata/burgers.reduced.jld2")
    solutions = burgersdata["solutions"]
    parameters = burgersdata["parameters"]
    (; Δt, t⃗, ν, Δx) = parameters
    f = BurgersParams(ν, Δx)
end

begin # Set up training parameters
    @info "Setting up training"
    exit_condition = ExitCondition(20000, nothing)
    opt = ADAM(0.001)
    trainingdata = solutions[:, :, 1:96]
    validationdata = solutions[:, :, 97:end]
    verbose = true
end

begin # Create model
    nn = add_Δfwd(create_basic_nn())
    model = ClosureModel(nn, f, "Burgers closure model, optimise then discretise")
end

begin
    validation = trajectory_rmse(model, validationdata, t⃗, (; alg=Tsit5()))
    result = train_closure!(
        model, t⃗, trainingdata;
        exit_condition, opt, validation, verbose
    )
    jldsave("burgers-3-opt-disc.jld2"; result, model, opt)
end
