module ComputationalPlasmonics

const PROJECT_ROOT = pkgdir(ComputationalPlasmonics)

using StaticArrays
using CairoMakie
using LinearAlgebra
using Inti

include("pml.jl")
include("operators.jl")
include("makietheme.jl")

export CartesianPML, EulerPML, single_layer, adjoint_double_layer, solution, solution_full

end # module ComputationalPlasmonics
