begin
    # Load code and data
    include("src/neural_closure_models.jl")
    burgersdata = load("trainingdata/burgers.reduced.jld2")
    solutions = burgersdata["solutions"]
    parameters = burgersdata["parameters"]
    (; Δx, Δt, x⃗, t⃗, Nₓ, L, ν) = parameters
    # Create a struct containing the relevant parameters of the equation. This struct acts as the spatially discretised
    # right-hand side, i.e. the ODE `du/dt = f(u)` is a discretisation of Burgers' equation.
    f = BurgersParams(ν, Δx)
end

begin
    # Create a neural ODE, discretised with RK4
    nn = create_deep_wide_nn()
    model = DiscreteModel(
        DiscretisedODE(nn, rk4, Δt),
        "Neural ODE, discretised with rk4"
    )
    # Run inference on the model before training
    # Note: the neural networks all expect three-dimensional arrays, with the last index representing independent
    # trajectories, so when creating a single trajectory we can't use two-dimensional arrays but must use
    # three-dimensional arrays with size 1 in the last dimension.
    prediction1 = predict(model, solutions[:, 1:1, 97:97], t⃗, nothing)[:, :, 1]
    actual = solutions[:, 2:end, 97]
    println("Before training: RMSE = $(rootmeansquare(prediction1 .- actual))")
end

begin
    # Train the model. After every epoch, compute the RMSE on a validation data set
    validator = trajectory_rmse(
        model,                    # validate by computing root-mean-square errors
        solutions[:, :, 97:128],  # on trajectories 97 through 128 of the training data
        t⃗,                        #
        nothing                   # no teacher forcing
    )
    train_discrete!(
        model, t⃗, solutions[:, :, 1:96];              # take all x-coordinates, all time stamps, and solutions 1 through 96

        # only the first three arguments are required, the remainder is optional
        exit_condition = ExitCondition(200, nothing), # 200 epochs, no early stopping
        penalty = NoPenalty(),                        # no regularisation term
        validation = validator,                       # validate as defined above
        batchsize = 8,                                # train on batches of 8 trajectories
        opt = ADAM(0.01),                             # use ADAM optimiser with learning rate 0.01 (default = 0.001)
        loss = Flux.Losses.mse,                       # use mean-square error as a loss function for training
        verbose = true,                               # print progress bars and other info during training
        teacherforcing = 0                            # no teacher forcing
    )
    # Run inference again with the model which is now trained, and plot the actual solution as well as the predictions
    # before and after training. Note that this is just an example, and to obtain truly accurate models far more
    # training epochs are required.
    prediction2 = predict(model, solutions[:, 1:1, 97:97], t⃗, nothing)[:, :, 1]

    println("After training: RMSE = $(rootmeansquare(prediction2 .- actual))")
end

begin 
    plot1 = heatmap(t⃗[2:end], x⃗, actual, xlabel="t", ylabel="x", title="Actual solution", dpi=200)
    plot2 = heatmap(t⃗[2:end], x⃗, prediction1, xlabel="t", ylabel="x", title="Model prediction before training", dpi=200)
    plot3 = heatmap(t⃗[2:end], x⃗, prediction2, xlabel="t", ylabel="x", title="Model prediction after training", dpi=200)
    
    savefig(plot1, "burgers-actual.png")
    savefig(plot2, "burgers-prediction-before.png")
    savefig(plot3, "burgers-prediction-after.png")
end
