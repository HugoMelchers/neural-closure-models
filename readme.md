# Neural closure models

## About this repo

This repository contains the code associated with my master thesis.
It includes code for:

- Generating training data by solving PDEs/ODEs:
    - Burgers' equation
    - The Kuramoto-Sivashinsky equation
    - The Lorenz '96 model
- Training a variety of ML models on the resulting data
    - Discrete models: `u(t + Δt) = model(u(t))`
    - Neural ODEs (NODEs): `du/dt = model(u)`
    - Neural closure models: `du/dt = f(u) + model(u)`
    - Augmented Neural ODEs (ANODEs): `d/dt [u, h] = model(u, h)`
    - Discrete delay models: `u(t + Δt) = model(u(t), u(t - Δt), u(t - 2Δt), ..., u(t - kΔt))`

## Installation

0. Before cloning this repo, install [Git LFS](https://git-lfs.github.com/).
    This way, the down-sampled training data for the neural networks (~60MB) will be included in the repo.
1. Install [Julia](https://julialang.org/downloads/).
    This software was run with Julia version 1.7.3, although other 1.7.x versions should also work.
2. Install dependencies by launching `julia` in this folder:

    ```shell
    > cd neural-closure-models
    > julia --project=.
    ```

    and installing all dependencies by running:

    ```julia
    using Pkg; Pkg.instantiate()
    include("src/neural_closure_models.jl")
    ```
