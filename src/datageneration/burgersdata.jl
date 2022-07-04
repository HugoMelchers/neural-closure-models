using DifferentialEquations, ProgressBars, JLD2

function createfullburgersdata()
    L = 1.0f0       # Length of spatial domain
    T = 0.5f0       # Time span to solve for
    ν = 0.0005f0    # Viscosity
    Nₓ = 4096       # Number of points in discretisation
    Nₜ = 2049       # Number of snapshots to save
    Nₚ = 128        # Number of trajectories to generate

    Δx = L / Nₓ
    Δt = T / Nₜ
    x⃗ = centeredrange(0, L; length=Nₓ)
    t⃗ = range(0, T; length=Nₜ)

    odeproblem = ODEProblem(
        burgers_jameson!,
        zeros(Float32, Nₓ),
        (0, T),
        BurgersParams(ν, Δx);
        saveat=t⃗
    )

    isdir("$(pwd())/trainingdata") || mkdir("$(pwd())/trainingdata")
    jldopen("$(pwd())/trainingdata/burgers.full.jld2", "w") do outfile
        outfile["parameters"] = (; L, T, ν, Nₓ, Nₜ, Nₚ, Δx, Δt, x⃗, t⃗)

        iter = ProgressBar(1:Nₚ)
        set_description(iter, "Solving")
        for iₚ ∈ iter
            u0 = randominitialstate(Float32, Nₓ, 10)
            prob = remake(odeproblem; u0)
            sol_full = solve(prob; alg=Tsit5())
            outfile["solutions/$iₚ"] = Array(sol_full)
        end
    end
    nothing
end

function createreducedburgersdata()
    Kₓ = 64
    Kₜ = 33
    jldopen("$(pwd())/trainingdata/burgers.full.jld2") do infile
        (; L, T, ν, Nₓ, Nₜ, Nₚ, Δx, Δt, x⃗, t⃗) = infile["parameters"]
        solutions_reduced = zeros(Float32, Kₓ, Kₜ, Nₚ)
        derivatives_reduced = zeros(Float32, Kₓ, Kₜ, Nₚ)
        Δx₂ = L / Kₓ
        Δt₂ = T / Kₜ
        Sₓ = Int64(Nₓ / Kₓ)
        Sₜ = Int64((Nₜ - 1) / (Kₜ - 1))
        x⃗₂ = centeredrange(0, L; length=Kₓ)
        t⃗₂ = range(0, T; length=Kₜ)
        derivative = BurgersParams(ν, Δx)

        jldopen("$(pwd())/trainingdata/burgers.reduced.jld2", "w") do outfile
            outfile["parameters"] = (; L, T, ν, Nₓ=Kₓ, Nₜ=Kₜ, Nₚ, Δx=Δx₂, Δt=Δt₂, x⃗=x⃗₂, t⃗=t⃗₂)

            iter = ProgressBar(1:Nₚ)
            set_description(iter, "Down-sampling")
            for iₚ ∈ iter
                sol_full = infile["solutions/$iₚ"]
                sol_reduced = blockaverage(decimate(sol_full, Sₜ), Sₓ)
                solutions_reduced[:, :, iₚ] = sol_reduced
                derivative_full = derivative(sol_full)
                derivative_reduced = blockaverage(decimate(derivative_full, Sₜ), Sₓ)
                derivatives_reduced[:, :, iₚ] = derivative_reduced
            end
            outfile["solutions"] = solutions_reduced
            outfile["derivatives"] = derivatives_reduced
        end
    end
    nothing
end
