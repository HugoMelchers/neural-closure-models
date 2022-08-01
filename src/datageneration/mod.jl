using FFTW, Revise

"""
    randominitialstate(T, N, K)

Returns a random array of length `N` with entries of type `T` whose Fourier transform only has non-zero entries for the
`K` lowest frequencies.
"""
function randominitialstate(T, N, K)
    l_fft = [zero(T); randn(Complex{T}, K); zeros(T, N - 2K - 1); randn(Complex{T}, K)]
    l = real.(ifft(l_fft))
    m = maximum(abs.(l))
    2l ./ m
end

"""
    blockaverage(data::Matrix{T}, S) where {T}

Reduce the size of `data` in the first dimension by taking the average of blocks of `S` subsequent entries. This
requires that `size(data, 1)` is an integer multiple of `S`.
"""
function blockaverage(data::Matrix{T}, S) where {T}
    (Nₓ, Nₜ) = size(data)
    data2 = reshape(data, S, Int64(Nₓ / S), Nₜ)
    data3 = sum(data2, dims=1)
    data4 = reshape(data3, Int64(Nₓ / S), Nₜ)
    data4 ./ S
end

"""
    decimate(data::Matrix{T}, S) where {T}

Reduce the size of `data` in the second dimension by only taking each `S`-th entry. This requires that
`size(data, 2) - 1` is a multiple of `S`.
"""
function decimate(data::Matrix{T}, S) where {T}
    (Nₓ, Nₜ) = size(data)

    # Compute the new number of snapshots in the training data
    # This line throws an error if `size(data, 2) - 1` is not a multiple of `S`
    Kₜ = Int64((Nₜ - 1) / S)
    data[:, 1:S:end]
end

includet("burgersdata.jl")
includet("ksdata.jl")
includet("lorenz96data.jl")

"""
    generatetrainingdata()

Generates training data for all three equations. Note that this should not be necessary since the (down-sampled)
training data is already included in the repository. Furthermore, overwriting the training data has the effect of
changing the exact accuracies of already trained models, making it more difficult to verify that the results are still
the same as earlier.
"""
function generatetrainingdata()
    createfullburgersdata()
    createreducedburgersdata()

    createfullksdata()
    createreducedburgersdata()

    createfulllorenz96data()
    createreducedlorenz96data()
end
