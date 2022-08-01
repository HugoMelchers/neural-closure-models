using DifferentialEquations, ProgressBars, JLD2

function createfullksdata()
    L = 64.0f0      # Length of spatial domain
    T = 256.0f0     # Time span to solve for
    Nₓ = 1024       # Number of points in discretisation
    Nₜ = 1025       # Number of snapshots to save
    Nₚ = 100        # Number of trajectories to generate

    Δx = L / Nₓ
    Δt = T / (Nₜ - 1)
    x⃗ = centeredrange(0, L; length=Nₓ)
    t⃗ = range(0, T; length=Nₜ)

    odeproblem = ODEProblem(
        kuramotosivashinsky,
        zeros(Float32, Nₓ),
        (0, T),
        KSParams(Δx);
        saveat=t⃗
    )

    isdir("trainingdata") || mkdir("trainingdata")
    jldopen("trainingdata/kuramotosivashinsky.full.jld2", "w") do outfile
        outfile["parameters"] = (; L, T, Nₓ, Nₜ, Nₚ, Δx, Δt, x⃗, t⃗)

        iter = ProgressBar(1:Nₚ)
        set_description(iter, "Solving")
        for iₚ in iter
            u0 = randominitialstate(Float32, Nₓ, 10)
            prob = remake(odeproblem; u0)
            sol_full = solve(prob; alg=Rodas4P())
            outfile["solutions/$iₚ"] = Array(sol_full)
        end
    end
    nothing
end

function createreducedksdata()
    Kₓ = 128
    Kₜ = 513
    jldopen("trainingdata/kuramotosivashinsky.full.jld2") do infile
        (; L, T, Nₓ, Nₜ, Nₚ, Δx, Δt, x⃗, t⃗) = infile["parameters"]
        solutions_reduced = zeros(Float32, Kₓ, Kₜ, Nₚ)
        derivatives_reduced = zeros(Float32, Kₓ, Kₜ, Nₚ)
        Δx₂ = L / Kₓ
        Δt₂ = T / (Kₜ - 1)
        Sₓ = Int64(Nₓ / Kₓ)
        Sₜ = Int64((Nₜ - 1) / (Kₜ - 1))
        x⃗₂ = centeredrange(0, L; length=Kₓ)
        t⃗₂ = range(0, T; length=Kₜ)
        derivative = KSParams(Δx)

        jldopen("trainingdata/kuramotosivashinsky.reduced.jld2", "w") do outfile
            outfile["parameters"] = (; L, T, Nₓ=Kₓ, Nₜ=Kₜ, Nₚ, Δx=Δx₂, Δt=Δt₂, x⃗=x⃗₂, t⃗=t⃗₂)

            iter = ProgressBar(1:Nₚ)
            set_description(iter, "Down-sampling")
            for iₚ in iter
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
