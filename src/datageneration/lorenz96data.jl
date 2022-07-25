using DifferentialEquations, ProgressBars, JLD2

function createfulllorenz96data()
    T = 25.6f0  # Time span to solve for
    Nₜ = 2561   # Number of snapshots to save
    Nₚ = 100    # Number of trajectories to generate

    K = 18
    J = 20
    f = 10.0f0
    ε = 0.5f0
    hx = -1.0f0
    hy = 1.0f0
    params = Lorenz96Params(K, J, f, ε, hx, hy)

    Δt = T / (Nₜ - 1)
    t⃗ = range(0, T; length=Nₜ)

    odeproblem = ODEProblem(
        lorenz96!,
        ComponentArray(
            x=zeros(Float32, K),
            y=zeros(Float32, J, K)
        ),
        (0, T),
        params;
        saveat=t⃗
    )

    isdir("trainingdata") || mkdir("trainingdata")
    jldopen("trainingdata/lorenz96.full.jld2", "w") do outfile
        outfile["parameters"] = (; T, Nₜ, Nₚ, Δt, t⃗, params)

        iter = ProgressBar(1:Nₚ)
        set_description(iter, "Solving")
        for iₚ ∈ iter
            u0 = ComponentArray(
                x=randn(Float32, K),
                y=randn(Float32, J, K)
            )

            prob = remake(odeproblem; u0)
            sol_full = solve(prob; alg=Vern7())
            outfile["solutions/$iₚ"] = Array(sol_full)
        end
    end
    nothing
end

function createreducedlorenz96data()
    Kₜ = 601
    jldopen("trainingdata/lorenz96.full.jld2") do infile
        (; T, Nₜ, Nₚ, Δt, t⃗, params) = infile["parameters"]
        K = params.K
        J = params.J
        solutions_reduced = zeros(Float32, K, Kₜ, Nₚ)
        derivatives_reduced = zeros(Float32, K, Kₜ, Nₚ)
        Δt₂ = (T - 1.6f0) / (Kₜ - 1)
        Sₜ = Int64((Nₜ - 161) / (Kₜ - 1))
        t⃗₂ = range(0, T - 1.6f0; length=Kₜ)

        jldopen("trainingdata/lorenz96.reduced.jld2", "w") do outfile
            outfile["parameters"] = (; T, Nₜ=Kₜ, Nₚ, Δt=Δt₂, t⃗=t⃗₂, params)

            iter = ProgressBar(1:Nₚ)
            set_description(iter, "Down-sampling")
            for iₚ ∈ iter
                sol_full = infile["solutions/$iₚ"][:, 160:4:end]
                sol_reduced = sol_full[1:K, :]
                solutions_reduced[:, :, iₚ] = sol_reduced

                for iₜ in 1:size(sol_reduced, 2)
                    xₜ = sol_full[1:K, iₜ]
                    yₜ = reshape(sol_full[(K+1):end, iₜ], J, K)
                    uₜ = ComponentArray(x = xₜ, y = yₜ)
                    derivatives_reduced[:, iₜ, iₚ] = lorenz96(uₜ, params).x
                end
            end
            outfile["solutions"] = solutions_reduced
            outfile["derivatives"] = derivatives_reduced
        end
    end
    nothing
end
