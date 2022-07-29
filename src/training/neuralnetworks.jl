"""
Creates the small 57-parameter CNN used in the experiments of Chapter 4.

todo: replace this implementation by one that calls `basic_cnn_1d`
"""
function create_basic_nn(K=9)
    Chain(
        CyclicPadLayer(K - 1),
        AddSquaresLayer(),
        Conv((K,), 2 => 2, tanh),
        Conv((K,), 2 => 1),
    )
end

"""
Creates the 'large' 533-parameter CNN used in the experiments of Chapters 4, 5, and 6.

todo: replace this implementation by one that calls `basic_cnn_1d`
"""
function create_deep_wide_nn(K=5)
    Chain(
        CyclicPadLayer(3(K - 1)),
        AddSquaresLayer(),
        Conv((K,), 2 => 4, tanh),
        Conv((K,), 4 => 6, tanh),
        Conv((K,), 6 => 6, tanh),
        Conv((K,), 6 => 4, tanh),
        Conv((K,), 4 => 2, tanh),
        Conv((K,), 2 => 1),
    )
end

"""
Creates the 853-parameter CNN used in the Lipschitz-regularisation tests of Chapter 5.

todo: replace this implementation by one that calls `basic_cnn_1d`
"""
function create_linear_nn(K=5)
    Chain(
        CyclicPadLayer(4(K - 1)),
        Conv((K,), 1 => 2, tanh),
        Conv((K,), 2 => 4, tanh),
        Conv((K,), 4 => 6, tanh),
        Conv((K,), 6 => 8, tanh),
        Conv((K,), 8 => 6, tanh),
        Conv((K,), 6 => 4, tanh),
        Conv((K,), 4 => 2, tanh),
        Conv((K,), 2 => 1),
    )
end

append_layer(nn::Flux.Chain, layer) = Flux.Chain(nn.layers..., layer)
append_layer(nn::Flux.Parallel, layer) = Flux.Chain(nn, layer)

push_layer(nn::Flux.Chain, layer) = Flux.Chain(layer, nn.layers...)
push_layer(nn::Flux.Parallel, layer) = Flux.Chain(layer, nn)

"""
Appends the `Δfwd` layer to the given neural network, so that its outputs always sum to zero.
This is used in ML models to enforce conservation of momentum.
"""
add_Δfwd(nn) = append_layer(nn, Δfwd)

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
