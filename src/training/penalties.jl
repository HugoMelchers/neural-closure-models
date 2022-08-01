using FFTW

"""
A type representing a penalty term used for regularisation when training a model. Every struct that is a subtype of this
abstract type should be callable as a function with no arguments, returning a `Float32` penalty value.

This value should of course depend on the parameters of the model that is being trained, so that the form of the penalty
influences the optimiser.
"""
abstract type Penalty end

"""
Don't add any regularisation term during training. This is the default.
"""
struct NoPenalty <: Penalty end
(::NoPenalty)() = 0

"""
Scale the penalty `inner` by the constant `λ`. This can also be constructed by defining a penalty as `λ * penalty`.
"""
struct ScaledPenalty{P} <: Penalty
    inner::P
    λ::Float32
end
(s::ScaledPenalty)() = s.λ * s.inner()

import Base:*

*(λ::Real, p::Penalty) = ScaledPenalty(p, Float32(λ))
*(p::Penalty, λ::Real) = ScaledPenalty(p, Float32(λ))

"""
    normsquared(arr)

Computes the sum of squares of an array or Flux.Params object.
"""
normsquared(arr::AbstractArray{T}) where T = sum(abs2, arr)
normsquared(ps::Flux.Params) = sum(normsquared, ps)

struct L2Penalty{PS} <: Penalty
    params::PS
end
(p::L2Penalty)() = sum(normsquared, p.params)

linearise(::typeof(Δfwd)) = Δfwd
linearise(layer::ScaleLayer) = layer
linearise(layer::IdentityLayer) = layer
linearise(layer::CyclicPadLayer) = layer
linearise(conv::Flux.Conv) = Flux.Conv(conv.weight, false, identity)

"""
    linearise(model::Flux.Chain)

A function that 'linearises' a neural network, by stripping out all nonlinearities in the convolutional layers.
The weights of the convolutional layers are preserved, but the biases and activation functions are removed.
All layers in the network that are already linear (i.e. `ScaleLayer`, `Δfwd`, and so on) are preserved.
"""
linearise(model::Flux.Chain) = Flux.Chain([linearise(layer) for layer in model.layers]...)

real2(z) = max(0.0, real(z))^2

"""
    eigenvalues(n, linmodel)

Computes the eigenvalues of a linear convolutional neural network. This is done efficiently by only passing a single
input vector to the neural network and computing the Fourier transform of the output. This method for computing the
eigenvalues only produces correct results for networks that are fully convolutional and linear, i.e. do not have biases
or activation functions.
"""
function eigenvalues(n, linmodel)
    u = reshape([1.0f0; zeros(Float32, n-1)], :, 1, 1)
    fft(linmodel(u)[:])
end

"""
A penalty term based on the spectral radius (= largest absolute eigenvalue) of the linearised model.
"""
struct SpectralRadiusPenalty{M} <: Penalty
    Nₓ::Int64
    linmodel::M
end
spectral_radius_penalty(Nₓ, model) = SpectralRadiusPenalty(Nₓ, linearise(neuralnetwork(model)))
(m::SpectralRadiusPenalty)() = maximum(abs, eigenvalues(m.Nₓ, m.linmodel))

"""
A penalty based on the spectral abscissa (= largest real component of the eigenvalues) of the linearised model.
"""
struct SpectralAbscissaPenalty{M} <: Penalty
    Nₓ::Int64
    linmodel::M
end
spectral_abscissa_penalty(Nₓ, model) = SpectralAbscissaPenalty(Nₓ, linearise(neuralnetwork(model)))
(m::SpectralAbscissaPenalty)() = maximum(real, eigenvalues(m.Nₓ, m.linmodel))

"""
A penalty based on the sum of squared absolute values of the eigenvalues of the linearised model.
"""
struct Abs2EigsPenalty{M} <: Penalty
    Nₓ::Int64
    linmodel::M
end
asb2_eigs_penalty(Nₓ, model) = Abs2EigsPenalty(Nₓ, linearise(neuralnetwork(model)))
(m::Abs2EigsPenalty)() = sum(abs2, eigenvalues(m.Nₓ, m.linmodel))

"""
A penalty based on the sum of squared positive real components of the eigenvalues of the linearised model. Note that
this penalises eigenvalues with large positive real components only, so no penalty is added for large negative real
components or large imaginary components.
"""
struct Real2EigsPenalty{M} <: Penalty
    Nₓ::Int64
    linmodel::M
end
real2_eigs_penalty(Nₓ, model) = Real2EigsPenalty(Nₓ, linearise(neuralnetwork(model)))
(m::Real2EigsPenalty)() = sum(real2, eigenvalues(m.Nₓ, m.linmodel))
