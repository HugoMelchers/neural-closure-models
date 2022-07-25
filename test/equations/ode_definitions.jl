"""
Test that the in-place definitions of the Burgers, Kuramoto-Sivashinsky, and Lorenz '96 ODEs compute the same derivative
as the out-of-place definitions. Note that in their current implementations, the in-place and out-of-place
implementations produce exactly the same result. However, this is not required as in general their results are only
expected to be equal up to numerical precision limits.
"""
function test_ode_definitions()
    @assert burgers_ode_error_f64() < sqrt(eps(1.0))
    @assert burgers_ode_error_f32() < sqrt(eps(1.0f0))
    @assert ks_ode_error_f64() < sqrt(eps(1.0))
    @assert ks_ode_error_f32() < sqrt(eps(1.0f0))
    @assert lorenz96full_ode_error_f64() < sqrt(eps(1.0))
    @assert lorenz96full_ode_error_f32() < sqrt(eps(1.0f0))
    @assert lorenz96reduced_ode_error_f64() < sqrt(eps(1.0))
    @assert lorenz96reduced_ode_error_f32() < sqrt(eps(1.0f0))
end

function max_error(f_in_place!, f_out_of_place, u_prototype, params; N=1000)
    errors = []
    du1 = similar(u_prototype)
    for _ in 1:N
        u = similar(u_prototype)
        u[:] = randn(eltype(u_prototype), length(u_prototype))
        f_in_place!(du1, u, params)
        du2 = f_out_of_place(u, params)
        err = maximum(abs, du1 .- du2)
        push!(errors, err)
    end
    maximum(errors)
end

function burgers_ode_error_f64()
    params = BurgersParams(0.0005, 1.0 / 64)
    u = zeros(Float64, (32, 16))
    max_error(burgers_jameson!, burgers_jameson, u, params)
end
function burgers_ode_error_f32()
    params = BurgersParams(0.0005f0, 1.0f0 / 64)
    u = zeros(Float32, (32, 16))
    max_error(burgers_jameson!, burgers_jameson, u, params)
end

function ks_ode_error_f64()
    params = KSParams(0.5)
    u = zeros(Float64, (64, 16))
    max_error(kuramotosivashinsky!, kuramotosivashinsky, u, params)
end
function ks_ode_error_f32()
    params = KSParams(0.5f0)
    u = zeros(Float32, (64, 16))
    max_error(kuramotosivashinsky!, kuramotosivashinsky, u, params)
end

function lorenz96full_ode_error_f64()
    params = Lorenz96Params(18, 20, 10.0, 0.5, -1.0, 1.0)
    u = ComponentArray(x=zeros(18), y=zeros(20, 18))
    max_error(lorenz96!, lorenz96, u, params)
end
function lorenz96full_ode_error_f32()
    params = Lorenz96Params(18, 20, 10.0f0, 0.5f0, -1.0f0, 1.0f0)
    u = ComponentArray(x=zeros(Float32, 18), y=zeros(Float32, 20, 18))
    max_error(lorenz96!, lorenz96, u, params)
end

function lorenz96reduced_ode_error_f64()
    params = ReducedLorenz96Params(18, 10.0)
    u = zeros(18)
    max_error(lorenz96!, lorenz96, u, params)
end
function lorenz96reduced_ode_error_f32()
    params = ReducedLorenz96Params(18, 10.0f0)
    u = zeros(Float32, 18)
    max_error(lorenz96!, lorenz96, u, params)
end
