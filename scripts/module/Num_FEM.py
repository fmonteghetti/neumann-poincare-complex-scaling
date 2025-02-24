#!/usr/bin/env python
# coding: utf-8
"""
Finite element method for the plasmonic eigenvalue problem (PEP):

            ∇·[a(x,κ)∇u] = 0, u(x) = O(1/|x|) at infinity.
"""
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
from scicomp_utils_misc import SLEPc_utils
from scicomp_utils_mesh import gmsh_utils
from scicomp_utils_dolfinx import fenicsx_utils
from computational_plasmonics import eigv_fenicsx as PEP_utils
from Num_utils import kappa_2_lambda, lambda_2_kappa
import dolfinx
import ufl
from scicomp_utils_dolfinx import fenicsx_utils

def compute_eigenvalues_Cartesian(gmshfile,DtN_order=0,quad_order=2,
                                  lambda_target=[0.2]):
    """ PEP in Cartesian coordinates without complex scaling."""
        # Retrieve physical groups
    dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2)
    phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile,1)
    phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile,2)
    phys_tags = dict()
    phys_tags['omega-m'] = phys_tag_2D['omega-m']
    phys_tags['omega-d'] = phys_tag_2D['omega-d']
    phys_tags['gamma-d'] = phys_tag_1D['gamma-d']
        # Assemble PEP
    PEP = PEP_utils.PEP_with_PML_fenicsx(dmesh,phys_tags,
                                         DtN_order=DtN_order)
    V_fem = ("CG", quad_order)
    PEP.assemble(V_fem)
        # Compute eigenvalues in several runs
    SLEPc_params = {
        'nev': 40,
        'problem_type': SLEPc.EPS.ProblemType.GNHEP,
        'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
        'tol': 1e-10,
        'max_it': 100}
    OptDB = PETSc.Options()
    OptDB["st_ksp_type"] = "preonly"
    OptDB["st_pc_type"] = "lu"
    OptDB["st_pc_factor_mat_solver_type"] = "mumps"
    eigval = list()
    for target in lambda_target:
        SLEPc_params['target'] = -lambda_2_kappa(target)
        SLEPc_params['shift'] = SLEPc_params['target']
        PEP.solve(SLEPc_params)
        eigval.append(kappa_2_lambda(-1 * np.array(PEP.eigval)))
    eigval = np.hstack(eigval)
    dof = PEP.get_size() 
    return (eigval,dof) 

def compute_eigenvalues_one_corner(gmshfile,a,b,phi,corner_pos,
                                   Euler_region_x_offset,
                                   Euler_region_y_offset,
                                   DtN_order=0,quad_order=2,
                                   pml_param=1, lambda_target=[0.2],
                                   nev=20, tol=1e-5):
    """ PEP in Cartesian coordinates for an ellipse with one corner,
    with the complex scaling region discretized in Euler coordinates."""
        # Retrieve physical groups
    dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2)
    phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile,1)
    phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile,2)
    tag_names = {
        'omega_m': ['omega-m'],
        'omega_d': ['omega-d'],
        'gamma_d': ['gamma-d'],
        'omega_m_pml': ['corner-omega-m-pml'],
        'omega_d_pml': ['corner-omega-d-pml'],
        'Euler_bottom_bnd': ['corner-bnd-bot'],
        'Euler_right_bnd': ['corner-bnd-right'],
    }
        # Periodic boundary conditions
    pbc = PEP_utils.get_pbc_ellipseNcorners_fenicsx(a,b,
        phi,corner_pos,Euler_region_x_offset,Euler_region_y_offset,
        [phys_tag_1D[s] for s in tag_names['Euler_bottom_bnd']],
        [phys_tag_1D[s] for s in tag_names['Euler_right_bnd']])
        # Physical entities
    phys_tags = dict()
    phys_tags['omega-m'] = [k for tag in tag_names['omega_m'] for k in phys_tag_2D[tag]]
    phys_tags['omega-d'] = [k for tag in tag_names['omega_d'] for k in phys_tag_2D[tag]]
    phys_tags['gamma-d'] = [k for tag in tag_names['gamma_d'] for k in phys_tag_1D[tag]]
    phys_tags['omega-m-pml'] = [k for tag in tag_names['omega_m_pml'] for k in phys_tag_2D[tag]]
    phys_tags['omega-d-pml'] = [k for tag in tag_names['omega_d_pml'] for k in phys_tag_2D[tag]]
        # Assemble PEP
    PEP = PEP_utils.PEP_with_PML_fenicsx(dmesh,phys_tags,pbc=pbc,
                                         DtN_order=DtN_order)
    V_fem = ("CG", quad_order)
    PEP.assemble(V_fem)
        # Compute eigenvalues in several runs 
    SLEPc_params = {
        'nev': nev,
        'problem_type': SLEPc.EPS.ProblemType.GNHEP,
        'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
        'tol': tol,
        'max_it': 100}
    OptDB = PETSc.Options()
    OptDB["st_ksp_type"] = "preonly"
    OptDB["st_pc_type"] = "lu"
    OptDB["st_pc_factor_mat_solver_type"] = "mumps"
    eigval = list()
    for target in lambda_target:
        SLEPc_params['target'] = -lambda_2_kappa(target)
        SLEPc_params['shift'] = SLEPc_params['target']
        PEP.solve(SLEPc_params,alpha=pml_param)
        eigval.append(kappa_2_lambda(-1 * np.array(PEP.eigval)))
    eigval = np.hstack(eigval)
    dof = PEP.get_size() 
    return (eigval,dof)

def compute_eigenvalues_corner_Cartesian(gmshfile, quad_order=2,
                                        pml_param=1, lambda_target=[0.3, 1/0.3],
                                        nev=20, tol=1e-5):
    """ PEP in Cartesian coordinates for a corner geometry with Dirichlet
    boundary condition. Complex scaling region discretized in Cartesian
    coordinates. """

        # Retrieve physical groups
    dmesh = fenicsx_utils.DolfinxMesh.init_from_gmsh(gmshfile,2)
    phys_tag_1D = gmsh_utils.getPhysicalNames(gmshfile,1)
    phys_tag_2D = gmsh_utils.getPhysicalNames(gmshfile,2)
    tags = dict()
    tags['omega-m'] = phys_tag_2D['omega-m']
    tags['omega-d'] = phys_tag_2D['omega-d']
    tags['gamma-d'] = phys_tag_1D['gamma-d']
    V_fem = ("CG",quad_order)
        # Assemble PEP as A_m = -kappa * A_d
    V = dolfinx.fem.FunctionSpace(dmesh.mesh, V_fem)
    u,v = ufl.TrialFunction(V), ufl.TestFunction(V)
    uD = dolfinx.fem.Function(V)
    uD.vector.setArray(0)
    uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, 
                        mode=PETSc.ScatterMode.FORWARD)
    bcs = fenicsx_utils.create_DirichletBC(dmesh.mesh,dmesh.facet_tags,
                                            V,[uD],tags['gamma-d'])
    def a_grad(u,v,tags,dx,A=1.0):
        a_list=[]
        for i in range(len(tags)):
                a_list.append(ufl.inner(A*ufl.grad(u), ufl.grad(v))*dx(tags[i]))
        return sum(a_list)
    x = ufl.SpatialCoordinate(dmesh.mesh)
    norms = x[0]*x[0] + x[1]*x[1]
    S = ufl.as_matrix(((x[0]*x[0]/norms,x[0]*x[1]/norms), 
                    (x[0]*x[1]/norms,x[1]*x[1]/norms)))
    adjS = ufl.as_matrix(((x[1]*x[1]/norms,-x[0]*x[1]/norms), 
                        (-x[0]*x[1]/norms,x[0]*x[0]/norms)))
    a_m = dict(); a_d = dict();
    a_m['a'] = a_grad(u,v,tags['omega-m'], dmesh.dx,A=S)
    a_d['a'] = a_grad(u,v,tags['omega-d'], dmesh.dx,A=S)
    a_m['a-inv'] = a_grad(u,v,tags['omega-m'], dmesh.dx,A=adjS)
    a_d['a-inv'] = a_grad(u,v,tags['omega-d'], dmesh.dx,A=adjS)
    diag_Am, diag_Ad = 1e4, 1e-4
    A_m_l = dict(); A_d_l =dict();
    for key in a_m:
        A_m_l[key] = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_m[key]),
                                                    bcs,diag_Am)
        A_m_l[key].assemble()
        A_d_l[key] = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_d[key]),
                                                    bcs,diag_Ad)
        A_d_l[key].assemble()
    coeff = dict()
    coeff = {'a': pml_param, 'a-inv': 1/pml_param}
    A_m = A_m_l['a'].copy()
    A_m.assemble()
    A_d = A_d_l['a'].copy()
    A_d.assemble()
    for key in A_m_l:
        A_m.axpy(coeff[key],A_m_l[key])
        A_d.axpy(coeff[key],A_d_l[key])
        # Compute eigenvalues in several runs
    SLEPc_params = {
        'nev': nev,
        'problem_type': SLEPc.EPS.ProblemType.GNHEP,
        'solver': SLEPc.EPS.Type.KRYLOVSCHUR,
        'tol': tol,
        'max_it': 1000}
    OptDB = PETSc.Options()
    OptDB["st_ksp_type"] = "preonly"
    OptDB["st_pc_type"] = "lu"
    OptDB["st_pc_factor_mat_solver_type"] = "mumps"
    eigval = list()
    for target in lambda_target:
        SLEPc_params['target'] = -lambda_2_kappa(target)
        SLEPc_params['shift'] = SLEPc_params['target']
        EPS = SLEPc_utils.solve_GEP_shiftinvert(A_m,A_d,**SLEPc_params)
        (eigval_tmp,eigvec_r_tmp,eigvec_i_tmp) = SLEPc_utils.EPS_get_spectrum(EPS)
        eigval.append(kappa_2_lambda(-1 * np.array(eigval_tmp)))
    eigval = np.hstack(eigval)
    dof = A_m.size[0] 
    return (eigval,dof) 