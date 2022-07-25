"""
A layer for neural networks that pads the input array over the first dimension by `K` items at either end. These items
are taken from the opposite end of the array.
For example, padding the array `[1, 2, 3, 4, 5, 6, 7]` with `K = 2` would yield `[6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2]`.
For 2-dimensional arrays, this layer also adds a singleton dimension in between the two non-singleton dimensions, which
convolutional layers will use as a `channel` index.
"""
struct CyclicPadLayer
    K::Int64
end
(c::CyclicPadLayer)(u::AbstractArray{Float32,3}) = cat(
    u[end-c.K+1:end, :, :], u, u[1:c.K, :, :],
    dims=1
)

"""
A layer that removes the middle dimension of a three-dimensional array, since this corresponds to the 'channel' index
which should be used internally in the convolutional neural network but should not be present in the output.
"""
struct ReshapeLayer end
(r::ReshapeLayer)(u) = dropdims(u; dims=2)

"""
A layer that adds the pointwise squares of the input as a second input channel to the convolutional neural network.
This is used for neural networks for the Burgers and Kuramoto-Sivashinsky equations, since those contain quadratic
terms in the original right-hand side of the PDE.
"""
struct AddSquaresLayer end
(as::AddSquaresLayer)(u::AbstractArray{Float32,3}) = cat(u, u .^ 2, dims=2)

"""
A layer that simply scales the input by the given constant. This layer could also be implemented by using `u -> r .* u`
as a layer in a call to `Flux.Chain` or `Flux.Parallel`, but such closures cannot be saved and loaded by `JLD2`.
"""
struct ScaleLayer{TFloat}
    r::TFloat
end
(sl::ScaleLayer)(u) = sl.r .* u

"""
A layer that passes its input unmodified. Used in combination with `ScaleLayer` and `Flux.Parallel` to create discrete
ML models that essentially act as forward Euler discretisations of an ODE.
"""
struct IdentityLayer end
(::IdentityLayer)(u) = u
