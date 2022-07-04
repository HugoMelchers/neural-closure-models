using FFTW

"""Returns a random band-limited array with entries between -2 and 2."""
function randominitialstate(T, N, K)
    l_fft = [zero(T); randn(Complex{T}, K); zeros(T, N - 2K - 1); randn(Complex{T}, K)]
    l = real.(ifft(l_fft))
    m = maximum(abs.(l))
    2l ./ m
end

"Reduce the size of `data` in the first dimension by taking the average of blocks of `S` subsequent entries. This requires that `size(data, 1)` is an integer multiple of `S`"
function blockaverage(data::Matrix{T}, S) where {T}
    (Nₓ, Nₜ) = size(data)
    data2 = reshape(data, S, Int64(Nₓ / S), Nₜ)
    data3 = sum(data2, dims=1)
    data4 = reshape(data3, Int64(Nₓ / S), Nₜ)
    data4 ./ S
end

"Reduce the size of `data` in the second dimension by only taking each `S`-th entry. This requires that `size(data, 2) - 1` is a multiple of `S`"
function decimate(data::Matrix{T}, S) where {T}
    (Nₓ, Nₜ) = size(data)
    Kₜ = Int64((Nₜ - 1) / S)
    data[:, 1:S:end]
end
