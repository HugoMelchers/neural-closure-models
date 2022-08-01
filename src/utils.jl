using Zygote

"""
    fwd(u)

Circularly shifts the vector `u` forward by one over the first index, i.e. `fwd(u)[i, ...] == u[i+1, ...]` for
`1 ≤ i < size(u, 1)`, and `fwd(u)[end, ...] = u[1, ...]`. For example, circshift([1, 2, 3, 4]) == [2, 3, 4, 1].
"""
fwd(u) = circshift(u, -1)

"""
    bwd(u)

Circularly shifts the vector `u` backward by one over the first index, i.e. `fwd(u)[i, ...] == u[i-1, ...]` for
`1 < i ≤ size(u, 1)`, and `fwd(u)[1, ...] = u[end, ...]`. For example, circshift([1, 2, 3, 4]) == [4, 1, 2, 3].
"""
bwd(u) = circshift(u, 1)

"""
    fwd!(v, u)

The same as `fwd(u)`, but places the result in `v`.
"""
fwd!(v, u) = circshift!(v, u, -1)

"""
    bwd!(v, u)

The same as `bwd(u)`, but places the result in `v`.
"""
bwd!(v, u) = circshift!(v, u, 1)

"""
    Δfwd!(v, u)

Computes the backward differences of `u`, storing the result in `v`. Equivalent to `v .= u .- bwd(u)`, but in-place.
"""
function Δfwd!(v, u)
    circshift!(v, u, -1)
    @. v .-= u
    nothing
end

"""
    Δbwd!(v, u)

Computes the forward differences of `u`, storing the result in `v`. Equivalent to `v .= fwd(u) .- u`, but in-place.
"""
function Δbwd!(v, u)
    circshift!(v, u, 1)
    @. v = u - v
    nothing
end

"""
    diff2!(v, u)

Computes the second differences of `u`, i.e. `v[j] = u[j-1] - 2u[j] + u[j+1]`.
"""
function diff2!(v, u)
    w = similar(u)
    Δfwd!(w, u)
    Δbwd!(v, w)
    nothing
end

"""
    Δfwd(u)

Computes the forward differences of `u`. Equivalent to `fwd(u) .- u`.
"""
function Δfwd(u)
    v = similar(u)
    Δfwd!(v, u)
    v
end

"""
    Δbwd(u)

Computes the backward differences of `u`. Equivalent to `u .- bwd(u)`.
"""
function Δbwd(u)
    v = similar(u)
    Δbwd!(v, u)
    v
end

"""
    diff2(u)

Computes the second-order finite differences of `u`, i.e. `diff(u)[i] = u[j-1] - 2u[j] + u[j+1]`.
"""
function diff2(u)
    v = similar(u)
    diff2!(v, u)
    v
end

Zygote.@adjoint Δfwd(u) = Δfwd(u), ū -> (-Δbwd(ū),)
Zygote.@adjoint Δbwd(u) = Δbwd(u), ū -> (-Δfwd(ū),)
Zygote.@adjoint diff2(u) = diff2(u), ū -> (diff2(ū),)

"""
    rootmeansquare(A; dims)

Computes the root mean square (RMS) of the given K-dimensional array `A` along the given axes `dims`. The result is
an array of dimension K, with size 1 along each dimension in `dims`. If `dims` is not given, the root mean square is
computed over all dimensions of the array, returning a scalar.
"""
function rootmeansquare(A; dims)
    num::Int64 = prod(size(A)[[dims...]])
    sumsq = sum(abs2, A; dims)
    @. sqrt(sumsq / num)
end

function rootmeansquare(A)
    num = length(A)
    sqrt(sum(abs2, A) ./ num)
end

"""
    scalederror(y, ŷ)

Given two-dimensional arrays `y` and `ŷ` of equal size MxN, computes their error relative to `y`. The result is a
one-dimensional array of size N.
"""
function scalederror(y::AbstractArray{T,2}, ŷ::AbstractArray{T,2}) where {T}
    Eavg = rootmeansquare(y; dims=(1, 2))
    errs = rootmeansquare(y .- ŷ; dims=1) ./ Eavg
    dropdims(errs; dims=1)
end

"""
    validpredictiontime(y, ŷ; f=0.4f0)

Given 3-dimensional arrays of actual trajectories `y` and predicted trajectories `ŷ`, compute for each trajectory the
first index where the prediction error relative to the 2-norm of `y` exceeds the given threshold.

Arguments:
- `y`: 3-dimensional array of actual trajectories
- `ŷ`: 3-dimensional array of predicted trajectories
- `f`: threshold relative to 2-norm of `y` (default = 0.4)

`y` and `ŷ` should be 3-dimensional arrays of equal sizes with indices ordered (`x`, timestamp, problem)
"""
function validpredictiontime(y::AbstractArray{T,3}, ŷ::AbstractArray{T,3}; f=0.4f0) where {T}
    Nₚ = size(y, 3)
    Union{Nothing, Int64}[
        validpredictiontime(y[:, :, iₚ], ŷ[:, :, iₚ]; f)
        for iₚ in 1:Nₚ
    ]
end

"""
    validpredictiontime(y, ŷ; f=0.4f0)

Given 2-dimensional arrays of an actual trajectory `y` and predicted trajectory `ŷ`, compute for this trajectory the
first index where the prediction error relative to the 2-norm of `y` exceeds the given threshold. If the error does not
exceed this threshold at any point, `nothing` is returned.

Arguments:
- `y`: 2-dimensional array of actual trajectories
- `ŷ`: 2-dimensional array of predicted trajectories
- `f`: threshold relative to 2-norm of `y` (default = 0.4)

`y` and `ŷ` should be 2-dimensional arrays of equal sizes with indices ordered (`x`, timestamp)
"""
function validpredictiontime(y::AbstractArray{T,2}, ŷ::AbstractArray{T,2}; f=0.4f0) where {T}
    scaled_errors = scalederror(y, ŷ)
    findfirst(scaled_errors .>= f)
end

"""
    centeredrange(a, b; length)

Computes the range consisting of interval midpoints when dividing the interval `[a, b]` up into `length` intervals of
equal length. For example, `centeredrange(0.0, 10.0; length=10) == 0.5:1.0:9.5`.
"""
function centeredrange(a, b; length)
    range(a, b; length=2length + 1)[2:2:end]
end
