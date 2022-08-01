using Plots

"""
    diffmap(D; kwargs...)

Similar to `heatmap`, but sets the color gradient and scale so that 0 corresponds to white, negative values to blue, and
positive values to red. This is useful when making a heatmap of a difference, since then larger absolute differences are
shown in darker colours, and red/blue distinguishes positive and negative differences.
"""
function diffmap(D; kwargs...)
    M = maximum(abs, D)
    heatmap(D, clims=(-M, M), c=:balance; kwargs...)
end

"""
    diffmap(xs, ys, D; kwargs...)

Similar to `heatmap`, but sets the color gradient and scale so that 0 corresponds to white, negative values to blue, and
positive values to red. This is useful when making a heatmap of a difference, since then larger absolute differences are
shown in darker colours, and red/blue distinguishes positive and negative differences.
"""
function diffmap(xs, ys, D; kwargs...)
    M = maximum(abs, D)
    heatmap(xs, ys, D, clims=(-M, M), c=:balance; kwargs...)
end

"""
    vpt_heatmap(ts, xs, y, ŷ; f=0.4f0, diff=false, kwargs...)

Create a heatmap of a predicted trajectory or a diffmap of a prediction error, with a vertical line indicating the VPT.

Arguments:
- `ts`, `xs`: the x- and y- axes, respectively
- `y`, `ŷ`: the actual and predicted trajectories
- `f`: the threshold used to compute the VPT (default = 0.4)
- `diff`: whether or not to plot a diffmap instead of a heatmap
- `kwargs...`: additional keyword arguments passed on to `heatmap`
"""
function vpt_heatmap(ts, xs, y, ŷ; f=0.4f0, diff=false, kwargs...)
    vpt = ts[validpredictiontime(y, ŷ; f)]
    plt = if diff
        diffmap(ts, xs, ŷ .- y; kwargs...)
    else
        heatmap(ts, xs, ŷ; kwargs...)
    end
    plot!(plt, [vpt, vpt], [xs[1], xs[end]]; c=:black, label=nothing)
    plt
end
