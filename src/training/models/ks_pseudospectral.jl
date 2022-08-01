using FFTW

"""
A neural closure model for the Kuramoto-Sivashinsky equation that works in the pseudo-spectral domain and uses an
exponential integrator for the linear (anti-diffusion and hyper-diffusion) terms. The linear terms are represented as
pointwise multiplications in the Fourier domain. The non-linear terms (i.e. the quadratic convection term and the neural
closure term) are still computed in the physical domain, i.e. are preceded by an inverse Fourier Transform and followed
by a Fourier transform.
"""
struct KSPSClosureModel{Inner,UType,Integrator}
    inner::Inner
    g::UType
    integrator::Integrator
end
function derivative(model::KSPSClosureModel{UType}, û) where {UType}
    u = real.(ifft(û, 1))
    dûdt1 = model.g .* fft(u .^ 2, 1)          # quadratic convection term
    dûdt2 = fft(model.inner(u)::typeof(u), 1)  # neural network closure term
    dûdt1 + dûdt2
end

(model::KSPSClosureModel)(û) = model.integrator(û, u -> derivative(model, u))
neuralnetwork(model::KSPSClosureModel) = neuralnetwork(model.inner)

"""
    ks_etdrk_model(N, l, Δt, inner, order, name)

Creates a `DiscreteModel` that performs a time step by taking one time step of the pseudospectral Kuramoto-Sivashinsky
equation using an ETDRK integrator.
Parameters:
- `N`: the number of grid points of the discretisation
- `l`: the domain length
- `Δt`: the time step
- `model`: the inner neural network to use for the closure term
- `order`: which order ETDRK method to use. Should be 1, 2, 3, or 4
- `name`: the name of the resulting model
"""
function ks_etdrk_model(N, l, Δt, inner, order, name)
    pi = Float32(π)
    k = reshape(2pi / l .* Float32[0:(N/2-1); 0; (1-N/2):-1], :, 1)
    λ⃗ = @. k^2 - k^4
    g = @. -im * k / 2

    integrator_constructor = [etdrk1, etdrk2, etdrk3, etdrk4][order]
    integrator = integrator_constructor(λ⃗, Δt)

    DiscreteModel(KSPSClosureModel(inner, g, integrator), name)
end
