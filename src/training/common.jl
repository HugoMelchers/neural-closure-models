using Flux

neuralnetwork(nn::Flux.Chain) = nn
neuralnetwork(nn::Flux.Parallel) = nn

"""
The exit condition for a training procecure. The first field is the maximum number of epochs to train for. The second is
the 'patience' in the early stopping procedure. If the validation set error does not reach a new lowest value for
`patience` consecutive epochs, then training is aborted.
"""
struct ExitCondition
    max_epochs::Int64
    patience::Union{Nothing, Int64}
end
exitcondition(nr_epochs, patience=nothing) = ExitCondition(nr_epochs, patience)

teacherforcingatepoch(c::Real, _) = Float32(c)
teacherforcingatepoch(f::Function, epoch) = Float32(f(epoch))
teacherforcingatepoch(arr::AbstractArray{<:Real}, i) = Float32(arr[i])

"""
Contains the results of a training procedure, including:

- The exit condition (max number of epochs and optional patience parameter for early stopping)
- The errors on the training and validation data sets for each epoch. The validation error is not necessarily computed
  every epoch, so `error_validate` may be a shorter array than `error_train`
- The best parameters of the model so far, including the epoch at which they were found and the corresponding error value
- The total number of epochs trained so far
- The current status (`:running`, `:too_long_without_improvement`, `:yielded_nan`, or `:max_epochs_reached`)
"""
mutable struct TrainLog
    exit_condition::ExitCondition
    error_train::Vector{Float32}
    error_validate::Vector{Float32}
    best_epoch::Int64
    best_params::Flux.Params
    best_error::Float32
    nr_epochs::Int64
    return_code::Symbol
end

function update!(
        result::TrainLog,
        epoch, params, error_train, error_validate)

    result.nr_epochs = epoch

    prev_validate = if length(result.error_validate) == 0 Inf else result.error_validate[end] end

    error_current = error_train
    push!(result.error_train, error_train)
    if error_validate !== nothing && error_validate !== missing
        push!(result.error_validate, error_validate)
        error_current = error_validate
    elseif error_validate === missing
        error_validate = prev_validate
        error_current = prev_validate
    end

    if isnan(error_train)
        result.return_code = :yielded_nan
        @warn "Training resulted in NaN after $(epoch) epochs. Exiting."
        return nothing
    elseif error_current < result.best_error
        # new params are better apparently
        result.best_epoch = epoch
        result.best_error = error_current
        result.best_params = deepcopy(params)
    elseif result.exit_condition.patience !== nothing && result.exit_condition.patience < epoch - result.best_epoch
        # best params are too old, so exit
        result.return_code = :too_long_without_improvement
        @info "Error has not improved in $(result.exit_condition.patience) epochs. Exiting."
        return nothing
    end

    line1 = if error_validate === nothing
        "Epoch $epoch, error = $error_train (train)"
    else
        "Epoch $epoch, error = $error_train (train) vs $error_validate (test)"
    end
    line2 = "Best error so far: $(result.best_error), $(epoch - result.best_epoch) epochs ago (limit = $(result.exit_condition.patience))        "
    line1*"\n"*line2
end

function TrainLog(ec::ExitCondition, ps::Params)
    TrainLog(ec, Float32[], Float32[], 0, ps, Inf, 0, :running)
end

"""
    clone_with_params(model, params)

Given a `model` and model parameters `params`, create a copy of the model with the given parameters. `params` should
be either a `Flux.Params` object with arrays of appropriate dimensions, or a 1-dimensional array of appropriate length
containing the parameters in `flattened` form.
"""
function clone_with_params(model, params)
    model = deepcopy(model)
    set_params!(model, params)
    model
end

"""
    set_params!(model, params::Flux.Params)

Overwrites the parameters of the `model` to be equal to those in `params`. `params` should contain an appropriate number
of arrays of appropriate sizes, i.e.:

    length.(Flux.params(neuralnetwork(model))) .== length.(params)
"""
function set_params!(model, params::Flux.Params)
    params_write = Flux.params(neuralnetwork(model))

    for (arr1, arr2) in zip(params_write, params)
        arr1 .= arr2
    end
    nothing
end

"""
    set_params!(model, params::AbstractVector{Float32})

Overwrites the parameters of the `model` to be equal to those in `params`. `params` should be a vector whose length
equals the total number of parameters of `model`, i.e.:

    sum(length, Flux.params(neuralnetwork(model))) == length(params)
"""
function set_params!(model, params::AbstractVector{Float32})
    a = Flux.params(neuralnetwork(model))
    idx₂ = cumsum(map(length, a))
    idx₁ = [1; 1 .+ idx₂[1:end-1]]
    for (i, (i₁, i₂)) in enumerate(zip(idx₁, idx₂))
        a[i][:] = params[i₁:i₂]
    end
    nothing
end