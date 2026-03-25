struct SingleLayerKernel{T<:AbstractPML} <: Inti.AbstractKernel{ComplexF64}
    τ::T
end

function (SL::SingleLayerKernel)(p, q)
    τ = SL.τ
    x, y = Inti.coords(p), Inti.coords(q)
    r = τ(x) - τ(y)
    d = transpose(r) * r
    filter = !(d == 0)
    return filter * (-1 / (4π) * log(d))
end

struct SingleLayerKernel_full{T<:AbstractPML} <: Inti.AbstractKernel{ComplexF64}
    τ::T
end

function (SL::SingleLayerKernel_full)(p, q)
    τ = SL.τ
    R_alpha = SL.τ.radius
    xc = SL.τ.center
    # 'alpha' here is `1/alpha` in the article
    alpha = 1 / SL.τ.α
    x, y = Inti.coords(p), Inti.coords(q)
    rx = sqrt(transpose((x - xc)) * (x - xc))
    ry = sqrt(transpose((y - xc)) * (y - xc))
    q_alpha = transpose(τ(x) - τ(y)) * (τ(x) - τ(y))
    if (rx >= R_alpha) || (ry >= R_alpha)
        S = log(q_alpha)
    elseif ry <= rx
        qhat = q_alpha / exp((2 / alpha) * log(rx / R_alpha))
        S = (2 / alpha) * log(rx / R_alpha) + log(qhat)
    else
        qhat = q_alpha / exp((2 / alpha) * log(ry / R_alpha))
        S = (2 / alpha) * log(ry / R_alpha) + log(qhat)
    end
    return !(x == y) * (-1 / (4π)) * S
end

struct AdjointDoubleLayerKernel{T<:AbstractPML} <: Inti.AbstractKernel{ComplexF64}
    τ::T
end

function (ADL::AdjointDoubleLayerKernel)(p, q)
    τ = ADL.τ
    nx = Inti.normal(p)
    x, y = Inti.coords(p), Inti.coords(q)
    r = τ(x) - τ(y)
    q = transpose(r) * r
    filter = !(q == 0)
    adj = adjugate(τ, x)
    return filter * (-1 / (2π * q)) * transpose(r) * (adj * nx)
end

function single_layer(X, Y; pml, tol=1e-8, maxdist=-1, nq)
    G = SingleLayerKernel(pml)
    Sop = Inti.IntegralOperator(G, X, Y)
    if maxdist == -1 # use heuristic to set maxdist
        maxdist = Inti.farfield_distance(Sop; tol)
    end
    δS = Inti.adaptive_correction(Sop; tol, maxdist, nq)
    S = Matrix(Sop) + δS
    return S
end

function adjoint_double_layer(X, Y; pml, tol=1e-8, maxdist=-1, nq)
    dG = AdjointDoubleLayerKernel(pml)
    Kop = Inti.IntegralOperator(dG, X, Y)
    if maxdist == -1 # use heuristic to set maxdist
        maxdist = Inti.farfield_distance(Kop; tol)
    end
    if nq == -1 # use analytical correction
        @info "Using analytical correction."
        κ = Inti.curvature(Y)
        δK = [-κ[i] * Inti.weight(Y[i]) / 4π for i in eachindex(κ)] |> Diagonal
    else
        δK = Inti.adaptive_correction(Kop; tol, maxdist, nq)
    end
    K = Matrix(Kop) + δK
    return K
end

"""
    solution(σ,Y;pml)

Return `u=𝒮[σ]`, where `S` is a single-layer potential associated with the
`pml` over the domain `Y`.

The single-layer potential relies on a partial expression of the
complex-scaled Green function, which has discontinuities near the corner.
"""
function solution(σ, Y; pml)
    G = SingleLayerKernel(pml)
    𝒮 = Inti.IntegralPotential(G, Y)
    u = 𝒮[σ]
    return u
end

"""
    solution_full(σ,Y;pml)

Return `u=𝒮[σ]`, where `S` is a single-layer potential associated with the
`pml` over the domain `Y`.

The single-layer potential relies on the complete expression of the
complex-scaled Green function.
"""
function solution_full(σ, Y; pml)
    G = SingleLayerKernel_full(pml)
    𝒮 = Inti.IntegralPotential(G, Y)
    u = 𝒮[σ]
    return u
end
