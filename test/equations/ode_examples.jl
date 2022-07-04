using Plots, OrdinaryDiffEq

function burgers_example()
    xs = centeredrange(0.0f0, 1.0f0; length=512)
    u0 = @. 0.2sin(2π*xs) - cos(4π*xs)
    ts = range(0.0f0, 1.0f0; length=513)
    params = BurgersParams(0.0005f0, Float32(xs.step))
    prob = ODEProblem(
        burgers_jameson!,
        u0,
        (ts[begin], ts[end]),
        params,
        saveat = ts
    )
    sol = solve(prob, alg=Tsit5())
    heatmap(ts, xs, Array(sol))
end

function ks_example()
    xs = centeredrange(0.0f0, 64.0f0; length=512)
    u0 = @. 0.2sin(2π*xs/64) - cos(4π*xs/64)
    ts = range(0.0f0, 256.0f0; length=513)
    params = KSParams(Float32(xs.step))
    prob = ODEProblem(
        kuramotosivashinsky,
        u0,
        (ts[begin], ts[end]),
        params,
        saveat = ts
    )
    sol = solve(prob, alg=Rodas4P())
    heatmap(ts, xs, Array(sol))
end

function lorenz96_example()
    params = Lorenz96Params(18, 20, 10.0f0, 0.5f0, -1.0f0, 1.0f0)

    x0 = randn(Float32, params.K)
    y0 = randn(Float32, params.J, params.K)
    u0 = ComponentArray(x=x0, y=y0)
    ts = range(0.0f0, 8.0f0; length = 1001)
    prob = ODEProblem(
        lorenz96!,
        u0,
        (ts[begin], ts[end]),
        params,
        saveat = ts
    )
    sol = solve(prob, Vern7())
    plt1 = heatmap(ts, 1:params.K, Array(sol)[1:params.K, :])
    plt2 = heatmap(ts, 1:(params.K * params.J), Array(sol)[(params.K + 1):end, :])
    plot(plt1, plt2, layout = (1, 2))
end
