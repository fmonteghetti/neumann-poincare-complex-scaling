"""
    abstract type AbstractPML

Abstract type for PMLs change-of-variables. Subtypes are expected to implement
the followign methods:
- `(τ::AbstractPML)(x)`: maps from `ℝ² → ℂ²`
- `adjugate(τ,x)`: compute the
  [`adjugate`](https://en.wikipedia.org/wiki/Adjugate_matrix) of `τ`, which
  is the "J_ɑ A_ɑ" matrix in the paper. 
"""
abstract type AbstractPML end

(τ::AbstractPML)(x) = error("functor interface must be implemented for $(typeof(τ))")
jacobian(τ::AbstractPML, x) =
    error("functor interface must be implemented for $(typeof(τ))")

"""
    struct CartesianPML <: AbstractPML

PML in Cartesian coordinates given by `x ↦ x*norm(x)^(α-1)`, `α ∈ ℂ` is the
single parameter of the PML stored as an attribute in the struct.

Calling `CartesianPML(;angle)` constructs a pml with `α = exp(im*angle)`.

See also: [`AbstractPML`](@ref)
"""
struct CartesianPML <: AbstractPML
    α::ComplexF64
    center::SVector{2,Float64}
    radius::Float64
end

CartesianPML(; angle, center = SVector(0, 0), radius) =
    CartesianPML(exp(im * angle), center, radius)

function (τ::CartesianPML)(x)
    x = x - τ.center
    r = norm(x)
    iszero(r) && (return convert(SVector{2,typeof(τ.α)}, τ.center))
    α = r > τ.radius ? one(τ.α) : τ.α
    mu = (r / τ.radius)^(α - 1)
    return x * mu + τ.center
end

function adjugate(τ::CartesianPML, x)
    x = x - τ.center
    r = norm(x)
    α = r > τ.radius ? one(τ.α) : τ.α
    mu = (r / τ.radius)^(α - 1)
    # Projection matrix
    P = @SMatrix [x[1]^2 x[1]*x[2]; x[1]*x[2] x[2]^2]
    P = (1 / r^2) * P
    return mu * α * (I + ((1 / α) - 1) * P)
end

struct EulerPML <: AbstractPML
    α::ComplexF64
end

EulerPML(; angle) = EulerPML(exp(im * angle))

(τ::EulerPML)(x) = SVector(τ.α * x[1], x[2])

function adjugate(τ::EulerPML, x)
    α = τ.α
    return @SMatrix [1 0; 0 α]
end
