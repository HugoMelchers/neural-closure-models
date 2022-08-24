begin
    # Load data
    using JLD2
    ksdata = load("trainingdata/kuramotosivashinsky.reduced.jld2")
    solutions = ksdata["solutions"]
    derivatives = ksdata["derivatives"]
    parameters = ksdata["parameters"]
    (; Δx, Δt, x⃗, t⃗, Nₓ, L) = parameters
    # Similar to the Burgers example, `f` computes the right-hand side of the base discretisation of the KS equation
    f = KSParams(Δx)
end

begin
    # First, create a neural network. Add the Δfwd layer to satisfy conservation of momentum.
    nn = add_Δfwd(create_deep_wide_nn())
    # Embed the neural network as a closure term inside the ETDRK4 ODE solver for the KS equation
    model = ks_etdrk_model(Nₓ, L, Δt, nn, 4, "ETDRK4 closure model")
    # Compute the model prediction before training
    # Note that the pseudospectral works in the Fourier domain, so the initial conditions (and training data) must
    # be Fourier transformed over the first axis before being passed to the model, and the model predictions must be
    # inverse Fourier-transformed over the first axis before being compared to the original data.
    actual = solutions[:, 34:273, 81]
    prediction1 = real(ifft(predict(model, fft(solutions[:, 33:33, 81:81], 1), t⃗[33:273], nothing), 1))[:, :, 1]
    println("Before training: VPT = $(validpredictiontime(actual, prediction1)/24)")
end

begin
    # Train the pseudospectral closure model
    trainingdata = fft(solutions[:, 33:63, 1:80], 1)
    # Train the model and compute the new model prediction
    train_discrete!(
        model, t⃗[33:63], trainingdata;
        exit_condition=ExitCondition(200, nothing), batchsize=8
    )
    prediction2 = real(ifft(predict(model, fft(solutions[:, 33:33, 81:81], 1), t⃗[33:273], nothing), 1))[:, :, 1]
    println("After training: VPT = $(validpredictiontime(actual, prediction2)/24)")
end

begin
    # Plot the actual solution of the PDE, as well as the prediction and prediction error before and after training
    ts = t⃗[34:273] .- t⃗[33]
    plot_args = (dpi=200, xlims=extrema(ts), ylims=(0, L))
    plot1 = heatmap(ts, x⃗, actual, xlabel="t", ylabel="x", title="Actual solution"; plot_args...)
    plot2 = vpt_heatmap(ts, x⃗, actual, prediction1, xlabel="t", ylabel="x", title="Model prediction before training"; plot_args...)
    plot3 = vpt_heatmap(ts, x⃗, actual, prediction2, xlabel="t", ylabel="x", title="Model prediction after training"; plot_args...)
    plot4 = vpt_heatmap(ts, x⃗, actual, prediction1, xlabel="t", ylabel="x", title="Model prediction error before training", diff=true; plot_args...)
    plot5 = vpt_heatmap(ts, x⃗, actual, prediction2, xlabel="t", ylabel="x", title="Model prediction error after training", diff=true; plot_args...)
    
    savefig(plot1, "ks-actual.png")
    savefig(plot2, "ks-prediction-before.png")
    savefig(plot3, "ks-prediction-after.png")
    savefig(plot4, "ks-error-before.png")
    savefig(plot5, "ks-error-after.png")
end
