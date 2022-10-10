begin  # Load code
    # Adjust this parameter to change the number of time steps used for training
    const Nₜ = 30

    @info "Loading code"
    include("../src/neural_closure_models.jl")
end

begin # Load data
    @info "Loading training data"
    using JLD2
    ksdata = load("trainingdata/kuramotosivashinsky.reduced.jld2")
    solutions = ksdata["solutions"]
    solutions_fft = fft(solutions, 1)
    parameters = ksdata["parameters"]
    (; t⃗, Δx, Δt, Nₓ, L) = parameters
    nothing
end

function make_training_data(from, Kₜ)
    (Nₓ, Nₜ, ~) = size(from)
    result = zeros(eltype(from), Nₓ, Kₜ + 1, 0)
    a = 1
    b = 1 + Kₜ
    while b <= Nₜ
        result = cat(
            result,
            from[:, a:b, :],
            dims=3,
        )
        a += Kₜ
        b += Kₜ
    end
    result
end

begin # Setting up training parameters
    @info "Setting up training"
    exit_condition = ExitCondition(5000, nothing)
    opt = ADAM(0.001)
    Rₚ = 1:80
    Rₜ = 33:153
    trainingdata = make_training_data(solutions_fft[:, Rₜ, Rₚ], Nₜ)
    verbose = true
    nothing
end

begin # Create model
    nn = add_Δfwd(create_deep_wide_nn())
    model = ks_etdrk_model(Nₓ, L, Δt, nn, 4, "Kuramoto-Sivashinsky closure model, discretise-then-optimise (ETDRK4), Nₜ=$Nₜ")
    nothing
end

begin # train
    result = train_discrete!(
        model, t⃗[33:(33+Nₜ)], trainingdata;
        exit_condition, opt, verbose
    )
    jldsave("ks-3-disc-opt-Nₜ=$Nₜ.jld2"; model, result, opt)
    nothing
end
