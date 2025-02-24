#!/usr/bin/env python
# coding: utf-8
"""
Boundary element method for the Neumann-Poincaré (NP) operator. 
"""
import os
from pathlib import Path
import dataclasses
import typing
import warnings
import numpy as np
import scipy.sparse.linalg
from Num_utils import DIR_JULIA_PKG, JULIA_LIBRARY
# Setup julia environment
import juliapkg as jlPkg
jlPkg.require_julia("1.11")
jlPkg.add("LinearAlgebra",uuid="37e2e46d-f89d-539d-b4ee-838fcccc9c8e")
jlPkg.add("StaticArrays",uuid="90137ffa-7385-5640-81b9-e52037218182")
jlPkg.add("Inti", uuid="fb74042b-437e-4c5b-88cf-d4e2beb394d5", 
          rev="bypass-adaptive",
          url="https://github.com/IntegralEquations/Inti.jl")
jlPkg.add("Gmsh",uuid="705231aa-382f-11e9-3f0c-b7cb4346fdeb")
jlPkg.add("ComputationalPlasmonics",
          uuid="2a52226b-d884-4b12-8d4c-6ad7bbac7726",
          path=DIR_JULIA_PKG,
          dev=True)
# # Load julia library
from juliacall import Main as jl
jl.include(JULIA_LIBRARY)

@dataclasses.dataclass
class EigenvalueSolverOpts():
    """ Options for eigenvalue solver."""

    """ If True, compute only some eigenvalues using Arnoldi."""
    use_Arnoldi: bool = False
    """ Tolerance. (Arnoldi only)."""
    tol: float = 1e-8
    """ Number of eigenvalues. (Arnoldi only)."""
    nev: int = 20
    """ Shift. (Arnoldi only)."""
    shift: typing.List = dataclasses.field(default_factory=lambda: [0.1])

def compute_NP_eigenvalues(gmshfile: str, opts_eigv: EigenvalueSolverOpts,
                            **kwargs_julia):
    """ Compute eigenvalues of (complex-scaled) NP operator."""
    K, sl_pot = jl.compute_NP(gmshfile, "omega-m", **kwargs_julia)
    K = np.array(K)
    (eigvals, eigvecs) = compute_eigenvalues(K, opts_eigv)
    return (eigvals, K.shape[0], eigvecs, sl_pot)

def compute_NP_eigenvalues_corner(gmshfile: str, opts_eigv: EigenvalueSolverOpts,
                                  **kwargs_julia):
    """ Compute eigenvalues of NP operator for a corner with Dirichlet boundary
    condition. """
    K = jl.compute_NP_corner(gmshfile, **kwargs_julia) 
    K = np.array(K)
    (eigvals, _) = compute_eigenvalues(K, opts_eigv)
    return (eigvals, K.shape[0])

def compute_NP_eigenvalues_ellipse_with_corner_isogeometric(a,b,xc,yc,
                            cor_jun1_theta, cor_jun2_theta,
                            opts_eigv: EigenvalueSolverOpts,
                            **kwargs_julia):
    """ Compute some eigenvalues of the NP for an ellipse perturbed by one 
    corner. The geometry is represented exactly.
    """
    K, sl_pot = jl.compute_NP_ellipse_with_corner_isogeometric(a,b,xc,yc,
                        cor_jun1_theta, cor_jun2_theta, **kwargs_julia)
    K = np.array(K)
    (eigvals, eigvecs) = compute_eigenvalues(K, opts_eigv)
    return (eigvals, K.shape[0], eigvecs, sl_pot)

def compute_NP_eigenvalues_droplet_isogeometric(opts_eigv: EigenvalueSolverOpts,
                                                                **kwargs_julia):
    """ Compute eigenvalues of the NP for a droplet shape. The
    geometry is represented exactly. The given PML radius is adjusted to
    ensure conformity.
    """
    phi = 2*np.pi/3
    K, sl_pot, pml_radius = jl.compute_NP_droplet_isogeometric(**kwargs_julia)
    K = np.array(K)
    (eigval, eigvecs) = compute_eigenvalues(K, opts_eigv)
    return (eigval, K.shape[0], phi,eigvecs, sl_pot, pml_radius)

def compute_NP_eigenvalues_delta_isogeometric(phi,
                               opts_eigv: EigenvalueSolverOpts, **kwargs_julia):
    """ Compute eigenvalues of the NP for a delta shape with angle pi<phi<2*pi.
    The geometry is represented exactly. The given PML radius is adjusted
    to ensure conformity.
    """
    K, sl_pot, pml_radius = jl.compute_NP_delta_isogeometric(phi, **kwargs_julia)
    K = np.array(K)
    (eigval, eigvecs) = compute_eigenvalues(K, opts_eigv)
    return (eigval, K.shape[0], eigvecs, sl_pot, pml_radius)

def compute_NP_eigenvalues_ellipse_isogeometric(a,b, 
                               opts_eigv: EigenvalueSolverOpts, **kwargs_julia):
    """ Compute some eigenvalues of the NP for an ellipse. The geometry is 
    represented exactly.
    """
    K = jl.compute_NP_ellipse_isogeometric(a,b,**kwargs_julia)[0]
    K = np.array(K)
    (eigvals, _) = compute_eigenvalues(K, opts_eigv)
    return (eigvals, K.shape[0]) 

def compute_eigenvalues(A : np.ndarray, opts : EigenvalueSolverOpts):
    """ Compute of all or some eigenvalues of matrix A.

    Returns
    -------
    (eigvals, eigvecs)
        Eigenvalues and eigenvectors
    """
    if opts.use_Arnoldi:
        eigvals_l = list()
        eigvecs_l = list()
        for shift in opts.shift: 
            try: 
                (w, v) = scipy.sparse.linalg.eigs(A,k=opts.nev,
                                                    sigma=shift,tol=opts.tol,
                                                    return_eigenvectors=True) 
                eigvals_l.append(w)
                eigvecs_l.append(v)
            except scipy.sparse.linalg.ArpackNoConvergence as e:
                # Get converged values
                eigvals_l.append(e.eigenvalues)
                eigvecs_l.append(e.eigenvectors)
        eigvals = np.hstack(eigvals_l)
        eigvecs = np.hstack(eigvecs_l)
    else:
        if A.shape[0]>1000:
            warnings.warn("Computing all eigenvectors of a large dense matrix.")
        eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs 