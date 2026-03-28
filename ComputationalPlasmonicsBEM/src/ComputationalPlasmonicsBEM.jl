module ComputationalPlasmonicsBEM

const PROJECT_ROOT = pkgdir(ComputationalPlasmonicsBEM)

using StaticArrays
using LinearAlgebra
using Inti

include("pml.jl")
include("operators.jl")

export CartesianPML, EulerPML, single_layer, adjoint_double_layer, solution, solution_full

include("Num-Neumann-Poincare.jl")
export compute_NP
export compute_NP_corner
export compute_NP_ellipse_with_corner_isogeometric
export compute_NP_droplet_isogeometric
export compute_NP_delta_isogeometric
export compute_NP_ellipse_isogeometric 

end # module ComputationalPlasmonicsBEM
