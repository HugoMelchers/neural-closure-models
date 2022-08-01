# For some reason, including all of the packages and code in this repo overrides the function `Base.sqrt`, causing later
# code to fail. This is likely a bug in one of the dependencies, and can be worked around by explicitly importing the
# function `Base.sqrt` so that libraries extend the definition rather than overriding it.
import Base.sqrt

using Revise

includet("utils.jl")
includet("integrators.jl")
includet("equations/mod.jl")
includet("plotting.jl")
includet("training/mod.jl")
