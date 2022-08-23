"""
A type representing a way to compute validation error of a model. Every struct that is a subtype of `Validation` should
be callable with the current epoch number as the single argument. It should then return

- `nothing` if no validation is done, which should only be the case for `NoValidation`
- `missing` if validation is done but not for this epoch, for example if validation is only done every 10 epochs
- the actual validation error as a `Float32`

The validation error does not have to be a type of (root-) mean-square error, but lower errors have to 'better' than
higher errors for early stopping to work correctly.
"""
abstract type Validation end

struct NoValidation <: Validation end
(::NoValidation)(_) = nothing

"""
A struct that evaluates the given model on the given trajectories and returns the root mean square error between the 
predicted and actual trajectories. 
"""
struct TrajectoryRMSE{M,T<:Number,T1<:AbstractArray{T,3},T2<:AbstractVector{Float32},T3<:AbstractArray{T,3},CFG} <: Validation
    model::M
    initial::T1
    t⃗::T2
    remainder::T3
    cfg::CFG
end
function (vl::TrajectoryRMSE)(_)
    ŷ = predict(vl.model, vl.initial, vl.t⃗, vl.cfg)
    rootmeansquare(vl.remainder .- ŷ)
end

"""
    trajectory_rmse(model, data, t⃗, cfg)

Returns a `TrajectoryRMSE` object that can be called to compute the root mean square error between the actual
trajectories in `data`, and the predicted trajectories predicted by the `model`.
"""
trajectory_rmse(model, data, t⃗, cfg) = TrajectoryRMSE(model, data[:, 1:1, :], t⃗, data[:, 2:end, :], cfg)

"""
A struct that evalutes the given model on the given trajectories and returns the mean Valid Prediction Time over all
test trajectories. Note that the result is negated, so that the output still satisfies that lower values are better.
"""
struct TrajectoryVPT{M,T<:Number,T1<:AbstractArray{T,3},T2<:AbstractVector{Float32},T3<:AbstractArray{T,3},CFG} <: Validation
    model::M
    initial::T1
    t⃗::T2
    remainder::T3
    cfg::CFG
end
function (vl::TrajectoryVPT)(_)
    ŷ = predict(vl.model, vl.initial, vl.t⃗, vl.cfg)
    vpt_indices = validpredictiontime(vl.remainder, ŷ)

    # replace `nothing` entries in the output (meaning that the error threshold is never exceeded) by the maximum index
    vpt_indices[vpt_indices.===nothing] .= size(vl.remainder, 2)
    vpts = [vl.t⃗[i+1] - vl.t⃗[begin] for i in vpt_indices]
    -sum(vpts) / length(vpts)
end

"""
    trajectory_vpt(model, data, t⃗, cfg)

Returns a `TrajectoryVPT` object that can be called to compute the average valid prediction time of the `model`
predictions and the actual predictions in `data`.
"""
trajectory_vpt(model, data, t⃗, cfg) = TrajectoryVPT(model, data[:, 1:1, :], t⃗, data[:, 2:end, :], cfg)
