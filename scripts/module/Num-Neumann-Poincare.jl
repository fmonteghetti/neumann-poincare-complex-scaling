#= 
Functions to assemble the Neumann-Poincaré operator and single layer potential.
=#
using LinearAlgebra
using ComputationalPlasmonics
using StaticArrays
using Inti
using Gmsh

"""
Compute Neumann-Poincaré operator on the boundary of the domain
`domain_name` defined in the gmsh mesh `mshfile`.

# Arguments
- `mshfile`: gmsh mesh file.
- `domain_name`: string of physical domains corresponding to Ω in the mesh
- `qrule`: quadrature rule ("fejer" or "gl")
- `qorder`: quadrature order, i.e. the maximum polynomial degree 
for which the quadrature is exact
    - When `qrule`="fejer", the number of points N satisfies `qorder = N-1`  
    - When `qrule`="gl", `qorder = 2N-1` and `qorder` is rounded to the next
    odd integer. 
- `qmaxdist`: distance below which a high-order Gauss-Legendre quadrature is
used. Defaults to -1, which sets this distance using a heuristic.
- `qorder_correction`: order of the high-order Gauss-Legendre quadrature. Rounded
to next odd integer. If set to `-1`, then an analytical correction is used. 
- `pml_param=1`: complex scaling parameter.
- `pml_radius=1`: radius of the scaling region.
- `pml_center=(0,0)`: coordinates of the pml center

# Returns
- `K`: Neumann-Poincaré operator (dense matrix)
- `sl_potential`: single layer potential (function)
"""
function compute_NP(
    mshfile,
    domain_name;
    qrule = "gl",
    qorder = 3,
    qmaxdist = -1,
    qorder_correction = 9,
    pml_param = 1,
    pml_radius = 1,
    pml_center = (0, 0),
)
    Inti.clear_entities!()
    msh = Inti.import_mesh(mshfile; dim = 2)
    ents = Inti.entities(msh)
    Ω = Inti.Domain(ents) do e
        any(l -> occursin(domain_name, l), Inti.labels(e))
    end
    Γ = Inti.boundary(Ω)
    Γ_msh = msh[Γ]
    if qrule == "fejer"
        qr = Inti.Fejer(; order = qorder)
    elseif qrule == "gl"
        qr = Inti.GaussLegendre(; order = qorder)
    end
    Q = Inti.Quadrature(Γ_msh, qr)
    @info minimum(q -> norm(q.coords), Q.qnodes)
    allunique(Q.qnodes) || error("Quadrature nodes are not unique")
    # alpha in julia code is 1/alpha in article
    τ = CartesianPML(;
        angle = -angle(pml_param),
        radius = pml_radius,
        center = SVector(pml_center),
    )
    qn_correction = ceil(Int, (qorder_correction + 1) / 2)
    if qorder_correction == -1
        qn_correction = -1
    end
    K = adjoint_double_layer(Q, Q; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    """
    Evaluate the complex-scaled single layer potential associated with the
    density `sigma` at points (`X`,`Y`).
    """
    function sl_potential(sigma, X, Y)
        u = solution_full(sigma, Q; pml = τ)
        fun = (x, y) -> u((x, y))
        return fun.(X, Y)
    end
    return K, sl_potential
end

"""
Compute Neumann-Poincaré operator on a corner.
"""
function compute_NP_corner(
    mshfile;
    qrule = "gl",
    qorder = 3,
    qmaxdist = -1,
    qorder_correction = 9,
    pml_param = 1,
    pml_radius = 1,
    pml_center = (0, 0),
)
    Inti.clear_entities!()
    msh = Inti.import_mesh(mshfile; dim = 2)
    ents = Inti.entities(msh)
    Ωm = Inti.Domain(ents) do e
        any(l -> occursin("omega-m", l), Inti.labels(e))
    end
    Ωd = Inti.Domain(ents) do e
        any(l -> occursin("omega-d", l), Inti.labels(e))
    end
    Γm = Inti.boundary(Ωm)
    Γd = Inti.boundary(Ωd)

    Γ_tra = Inti.Domain(Γm) do e
        any(l -> occursin("gamma-m", l), Inti.labels(e))
    end

    Γ_dir = Inti.external_boundary(Ωm ∪ Ωd)
    if qrule == "fejer"
        qr = Inti.Fejer(; order = qorder)
    elseif qrule == "gl"
        qr = Inti.GaussLegendre(; order = qorder)
    end
    Q_tra = Inti.Quadrature(msh[Γ_tra], qr)
    Q_dir = Inti.Quadrature(msh[Γ_dir], qr)
    τ = CartesianPML(;
        angle = -angle(pml_param),
        radius = pml_radius,
        center = SVector(pml_center),
    )
    ## mesh the geometry
    # 1 = transmission, 2 = dirichlet
    qn_correction = ceil(Int, (qorder_correction + 1) / 2)
    A11 =
        adjoint_double_layer(Q_tra, Q_tra; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    A12 =
        adjoint_double_layer(Q_tra, Q_dir; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    A21 = single_layer(Q_dir, Q_tra; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    A22 = single_layer(Q_dir, Q_dir; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    K = A11 - A12 * (A22 \ A21)
    return K
end

"""
    corner_grading_kress(Nd)

Return a change of variables `ϕ: [0,1] → [0,1]` that satisfies:
    - ϕ(0) = 0, ϕ(1) = 1, ϕ'(1) = 2
    - the first `Nd` (`Nd≥1`) derivatives of the transformation vanish at `x=0`

From A Nyström method for boundary integral equations in domains with corners,
Rainer Kressm, Numer. Math. 58, 145-161 (1990)
"""
function corner_grading_kress(Nd)
    P = Nd + 1
    v = x -> (1 / P - 1 / 2) * ((1 - x))^3 + 1 / P * ((x - 1)) + 1 / 2
    return x -> 2v(x)^P / (v(x)^P + v(2 - x)^P)
end

"""
    corner_grading_polynomial(Nd)

Return a change of variables `ϕ: [0,1] → [0,1]` that satisfies:
    - ϕ(0) = 0, ϕ(1) = 1, ϕ'(1) = 1
    - the first `Nd` (`Nd≥0`) derivatives of the transformation vanish at `x=0`
"""
function corner_grading_polynomial(Nd)
    P = Nd + 1
    return x -> x^P * (P - (P - 1) * x)
end

"""
Compute the Neumann-Poincaré operator for an ellipse perturbed by a corner.

The position of the corner is identified by the two angles:
    -pi < cor_jun2_theta < cor_jun1_theta < pi.

# Arguments
- `a`: ellipse major axis
- `b`: ellipse minor axis
- `xc,yc`: corner coordinates
- `cor_jun_1_theta,cor_jun2_theta`: angles identifying corner position.
- `meshsize`: mesh element size
- `corner_grading_type`: corner grading function ϕ: [0,1] → [0,1] that satisfies
ϕ(0)=0 and ϕ(1)=1. Choice: "polynomial", "kress".
- `corner_grading_regularity`: number of derivatives of ϕ that vanish at 0.
This is an integer that must satisfies:
    ≥0 if `corner_grading_type="polynomial"`
    ≥1 if `corner_grading_type="exponential". 

# Returns
- `K`: Neumann-Poincaré operator (dense matrix)
- `sl_potential`: single layer potential (function)
"""
function compute_NP_ellipse_with_corner_isogeometric(
    a,
    b,
    xc,
    yc,
    cor_jun1_theta,
    cor_jun2_theta;
    meshsize = 0.1,
    corner_grading_type = "polynomial",
    corner_grading_regularity = 1,
    qrule = "gl",
    qorder = 3,
    qmaxdist = -1,
    qorder_correction = 9,
    pml_param = 1,
)
    # Elliptic arc (counter clockwise)
    line_elliptic_arc = let a = a, b = b
        Inti.parametric_curve(cor_jun2_theta, cor_jun1_theta) do (s,)
            SVector(a * cos(s), b * sin(s))
        end
    end
    # Corner point
    Pc = SVector(xc, yc)
    # Junction points
    P1 = SVector(a * cos(cor_jun1_theta), b * sin(cor_jun1_theta))
    P2 = SVector(a * cos(cor_jun2_theta), b * sin(cor_jun2_theta))
    if corner_grading_type == "polynomial"
        μ = corner_grading_polynomial(corner_grading_regularity)
    elseif corner_grading_type == "kress"
        μ = corner_grading_kress(corner_grading_regularity)
    else
        error("Unknown corner grading")
    end
    line_bottom = let Pc = Pc, P2 = P2, μ = μ
        Inti.parametric_curve(0, 1) do (s,)
            t = μ(s)
            Pc + t * (P2 - Pc)
        end
    end
    line_top = let Pc = Pc, P1 = P1, μ = μ
        Inti.parametric_curve(0, 1) do (s,)
            t = 1 - μ(1 - s)
            P1 + t * (Pc - P1)
        end
    end
    # Create the mesh
    Γ = Inti.Domain(line_elliptic_arc, line_top, line_bottom)
    msh = Inti.meshgen(Γ; meshsize = meshsize)
    if qrule == "fejer"
        qr = Inti.Fejer(; order = qorder)
    elseif qrule == "gl"
        qr = Inti.GaussLegendre(; order = qorder)
    end
    Q = Inti.Quadrature(msh, qr)
    @info minimum(q -> norm(q.coords), Q.qnodes)
    allunique(Q.qnodes) || error("Quadrature nodes are not unique")
    # Corner PML
    τ = CartesianPML(;
        angle = -angle(pml_param),
        radius = norm(
            SVector(xc, yc) - SVector(a * cos(cor_jun1_theta), b * sin(cor_jun1_theta)),
        ),
        center = SVector(xc, yc),
    )
    qn_correction = ceil(Int, (qorder_correction + 1) / 2)
    if qorder_correction == -1
        qn_correction = -1
    end
    K = adjoint_double_layer(Q, Q; pml = τ, maxdist = qmaxdist, nq = qn_correction)

    """
    Evaluate the complex-scaled single layer potential associated with the
    density `sigma` at points (`X`,`Y`).
    """
    function sl_potential(sigma, X, Y)
        u = solution_full(sigma, Q; pml = τ)
        fun = (x, y) -> u((x, y))
        return fun.(X, Y)
    end
    return K, sl_potential
end


"""
Compute Neumann-Poincaré operator for droplet boundary defined in
Numer. Math. 58, 145-161 (1990).
"""
function compute_NP_droplet_isogeometric(;
    meshsize = 0.1,
    corner_grading_type = "kress",
    corner_grading_regularity = 1,
    qrule = "gl",
    qorder = 3,
    qmaxdist = -1,
    qorder_correction = 9,
    pml_param = 1,
    pml_radius = 1e-1,
)
    if corner_grading_type == "linear"
        μ = (s) -> s
    elseif corner_grading_type == "kress"
        p = corner_grading_regularity + 1
        v =
            (s) ->
                ((1 / p) - (1 / 2)) * (((pi - s) / pi)^3) +
                (1 / p) * ((s - pi) / pi) +
                (1 / 2)
        μ = (s) -> (2 * pi) * (v(s))^p / (v(s)^p + v(2 * pi - s)^p)
    end
    line_droplet = let μ = μ
        Inti.parametric_curve(0, 2 * pi) do (s,)
            SVector((2 / sqrt(3)) * sin(μ(s) / 2), -sin(μ(s)))
        end
    end
    # Create the mesh
    Γ = Inti.Domain(line_droplet)
    msh = Inti.meshgen(Γ; meshsize = meshsize)
    if qrule == "fejer"
        qr = Inti.Fejer(; order = qorder)
    elseif qrule == "gl"
        qr = Inti.GaussLegendre(; order = qorder)
    end
    Q = Inti.Quadrature(msh, qr)
    # Corner PML
    # Adjust PML radius to ensure conformity
    val, idx = findmin(x -> abs(x - pml_radius), norm.(msh.nodes))
    pml_radius = norm(msh.nodes[idx])
    τ = CartesianPML(;
        angle = -angle(pml_param),
        radius = pml_radius,
        center = SVector(0, 0),
    )
    qn_correction = ceil(Int, (qorder_correction + 1) / 2)
    K = adjoint_double_layer(Q, Q; pml = τ, maxdist = qmaxdist, nq = qn_correction)

    """
    Evaluate the complex-scaled single layer potential associated with the
    density `sigma` at points (`X`,`Y`).
    """
    function sl_potential(sigma, X, Y)
        u = solution_full(sigma, Q; pml = τ)
        fun = (x, y) -> u((x, y))
        return fun.(X, Y)
    end
    return K, sl_potential, pml_radius
end

"""
Compute Neumann-Poincaré operator for delta boundary adapted from
Numer. Math. 58, 145-161 (1990).
"""
function compute_NP_delta_isogeometric(
    phi;
    meshsize = 0.1,
    corner_grading_type = "kress",
    corner_grading_regularity = 1,
    qrule = "gl",
    qorder = 3,
    qmaxdist = -1,
    qorder_correction = 9,
    pml_param = 1,
    pml_radius = 1e-1,
)
    if corner_grading_type == "linear"
        μ = (s) -> s
    elseif corner_grading_type == "kress"
        p = corner_grading_regularity + 1
        v =
            (s) ->
                ((1 / p) - (1 / 2)) * (((pi - s) / pi)^3) +
                (1 / p) * ((s - pi) / pi) +
                (1 / 2)
        μ = (s) -> (2 * pi) * (v(s))^p / (v(s)^p + v(2 * pi - s)^p)
    end
    line = let μ = μ
        Inti.parametric_curve(0, 2 * pi) do (s,)
            SVector(-2 / (3 * tan(phi / 2)) * sin(3 * μ(s) / 2), -sin(μ(s)))
        end
    end
    # Create the mesh
    Γ = Inti.Domain(line)
    msh = Inti.meshgen(Γ; meshsize = meshsize)
    if qrule == "fejer"
        qr = Inti.Fejer(; order = qorder)
    elseif qrule == "gl"
        qr = Inti.GaussLegendre(; order = qorder)
    end
    Q = Inti.Quadrature(msh, qr)
    # Corner PML
    # Adjust PML radius to ensure conformity
    val, idx = findmin(x -> abs(x - pml_radius), norm.(msh.nodes))
    pml_radius = norm(msh.nodes[idx])
    τ = CartesianPML(;
        angle = -angle(pml_param),
        radius = pml_radius,
        center = SVector(0, 0),
    )
    qn_correction = ceil(Int, (qorder_correction + 1) / 2)
    K = adjoint_double_layer(Q, Q; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    # use outward normal
    K = -K
    """
    Evaluate the complex-scaled single layer potential associated with the
    density `sigma` at points (`X`,`Y`).
    """
    function sl_potential(sigma, X, Y)
        u = solution_full(sigma, Q; pml = τ)
        fun = (x, y) -> u((x, y))
        return fun.(X, Y)
    end
    return K, sl_potential, pml_radius
end

"""
Compute the Neumann-Poincaré operator for an ellipse.
"""
function compute_NP_ellipse_isogeometric(
    a,
    b;
    meshsize = 0.1,
    qrule = "gl",
    qorder = 3,
    qmaxdist = -1,
    qorder_correction = 9,
)
    line_elliptic_arc = let a = a, b = b
        Inti.parametric_curve(0, 2 * pi) do (s,)
            SVector(a * cos(s), b * sin(s))
        end
    end
    # Create the mesh
    Γ = Inti.Domain(line_elliptic_arc)
    msh = Inti.meshgen(Γ; meshsize = meshsize)
    if qrule == "fejer"
        qrule = Inti.Fejer(; order = qorder)
    elseif qrule == "gl"
        qrule = Inti.GaussLegendre(; order = qorder)
    end
    Q = Inti.Quadrature(msh, qrule)
    # Corner PML
    τ = CartesianPML(; angle = -angle(1.0), radius = 0.1)
    qn_correction = ceil(Int, (qorder_correction + 1) / 2)
    K = adjoint_double_layer(Q, Q; pml = τ, maxdist = qmaxdist, nq = qn_correction)
    """
    Evaluate the complex-scaled single layer potential associated with the
    density `sigma` at points (`X`,`Y`).
    """
    function sl_potential(sigma, X, Y)
        u = solution_full(sigma, Q; pml = τ)
        fun = (x, y) -> u((x, y))
        return fun.(X, Y)
    end
    return K, sl_potential
end
