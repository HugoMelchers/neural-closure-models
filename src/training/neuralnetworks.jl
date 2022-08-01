"""
    create_basic_nn(K=9)

Creates the small 57-parameter CNN used in the experiments of Chapter 4.
"""
function create_basic_nn(K=9)
    prepend_layer(
        basic_cnn_1d([2, 2, 1], K),
        AddSquaresLayer()
    )
end

"""
    create_deep_wide_nn(K=5)

Creates the 'large' 533-parameter CNN used in the experiments of Chapters 4, 5, and 6.
"""
function create_deep_wide_nn(K=5)
    prepend_layer(
        basic_cnn_1d([2, 4, 6, 6, 4, 2, 1], K),
        AddSquaresLayer()
    )
end

"""
    create_linear_nn(K=5)

Creates the 853-parameter CNN used in the Lipschitz-regularisation tests of Chapter 5.
"""
function create_linear_nn(K=5)
    basic_cnn_1d([1, 2, 4, 6, 8, 6, 4, 2, 1], K)
end

"""
    append_layer(nn, layer)

Appends a new layer to the given neural network. The result is a network whose output for input `u` equals
`u |> nn |> layer`. For networks that are already a `Flux.Chain`, this simply creates a new `Chain` with one more
layer. For `Flux.Parallel`s, this creates a Flux.Chain that concatenates the given network and the new layer.
"""
append_layer(nn::Flux.Chain, layer) = Flux.Chain(nn.layers..., layer)
append_layer(nn::Flux.Parallel, layer) = Flux.Chain(nn, layer)

"""
    prepend_layer(nn, layer)

Prepends a new layer to the given neural network. The result is a network whose output for input `u` equals
`u |> layer |> nn`. This creates a new `Flux` model, similar to `append_layer`.
"""
prepend_layer(nn::Flux.Chain, layer) = Flux.Chain(layer, nn.layers...)
prepend_layer(nn::Flux.Parallel, layer) = Flux.Chain(layer, nn)

"""
    add_Δfwd(nn)

Appends the `Δfwd` layer to the given neural network, so that its outputs always sum to zero.
This is used in ML models to enforce conservation of momentum.
"""
add_Δfwd(nn) = append_layer(nn, Δfwd)

"""
    basic_cnn_1d(channels, K, σ = tanh)

Creates a convolutional neural network with the given `channels`, kernel width `K`, and the given pointwise activation
function. `K` should be an odd integer. The input of the neural network will first be padded by circularly extending it,
so that the output array will be of the same dimensions as the input. The last convolutional layer of the network will
not use an activation function (i.e. σ = identity).
"""
function basic_cnn_1d(channels, K, σ = tanh)
    halfwidth = Int64((K - 1) / 2)
    Nchannels = length(channels) - 1
    paddinglayer = CyclicPadLayer(halfwidth*Nchannels)
    internalconvolutionlayers = [
        Flux.Conv((K,), c₁ => c₂, σ)
        for (c₁, c₂) in zip(channels[1:end-2], channels[2:end-1])
    ]
    lastconvolutionlayer = Flux.Conv((K,), channels[end-1] => channels[end])

    Flux.Chain(
        paddinglayer,
        internalconvolutionlayers...,
        lastconvolutionlayer
    )
end
