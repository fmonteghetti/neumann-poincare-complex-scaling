module ComputationalPlasmonicsBEM

const PROJECT_ROOT = pkgdir(ComputationalPlasmonicsBEM)

using StaticArrays
using LinearAlgebra
using Inti

include("pml.jl")
include("operators.jl")

export CartesianPML, EulerPML, single_layer, adjoint_double_layer, solution, solution_full

end # module ComputationalPlasmonicsBEM
