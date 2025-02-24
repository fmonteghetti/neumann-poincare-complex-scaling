#!/usr/bin/env python
# coding: utf-8
"""
Misc functions that do not involve BEM or FEM. 
"""
import os
import sys
from pathlib import Path
import warnings
import numpy as np
from numba import jit
try:
    from scicomp_utils_mesh import gmsh_utils
except ImportError:
    warnings.warn("scicomp_utils_mesh not available.")
# Define path
    # Mesh
DIR_MESH = str(Path(__file__).parents[1]/'mesh')
    # Image folder
DIR_ARTICLE_IMG = str(Path(__file__).parents[1]) 
    # Matplotlib style
DIR_MPLSTYLE = Path(__file__).parents[1]/'plotstyle'
MPLSTYLE_ARTICLE = [str(DIR_MPLSTYLE/'plot_style.mplstyle'),
                    str(DIR_MPLSTYLE/'plot_article_a4.mplstyle')]
MPLSTYLE_VSCODE = [str(DIR_MPLSTYLE/'plot_style.mplstyle'),
                   str(DIR_MPLSTYLE/'plot_vscode.mplstyle')]
    # path to computational plasmonics package
DIR_JULIA_PKG = str(Path(__file__).parents[2]/'bem')
JULIA_LIBRARY = str(Path(__file__).parents[0]/'Num-Neumann-Poincare.jl')

def kappa_2_lambda(kappa):
    return 0.5*(kappa+1)/(kappa-1)

def lambda_2_kappa(l):
    return (2*l+1)/(2*l-1)

if 'scicomp_utils_mesh.gmsh_utils' in sys.modules:
    def build_mesh(geofile,geo_param={},gmsh_param={},gdim=2):
        """ Build mesh file from geometry file. """
        gmshfile = os.path.splitext(geofile)[0]+'.msh'
        gmsh_utils.generate_mesh_cli(geofile,gmshfile,gdim,parameters=geo_param,
                                    **gmsh_param)
        return gmshfile

def compute_slope(dof,error,N=3):
    """ Compute sloped based on last N points. """
    x = dof[-N:]
    y = error[-N:] 
    slope = np.polyfit(np.log(x),np.log(y),1)[0]
    return slope 

def compute_exact_eigenvalues_ellipse(aspect_ratio,N=10):
    """ Compute first N exact eigenvalues for ellipse (aspect ratio in (0,1))."""
    n = [n for n in range(-N,N+1) if n!=0]
    return (np.sign(n)/2) * np.exp(-2 * np.abs(n) * np.arctanh(aspect_ratio))

def compute_exact_spectrum_corner(angle,pml_parameter=1,
                                  eta=np.logspace(-2,2,num=int(1e4))):
    """ Compute the exact essential spectrum associated with a corner. """
    psi = lambda eta,phi: np.sinh(eta*(np.pi-phi))/np.sinh(eta*np.pi)
    s = psi(pml_parameter*eta,angle)/2
    return np.concatenate((s,-np.flip(s)))

def corner_mesh_grading_linear(l,factor=1.0):
    """ Compute element size at corner using a linear refinment. """
    l_corner = l / factor
    return l_corner

def corner_mesh_grading_power(l,power=1.0, l_reference=1.0):
    """ Compute element size at corner using a power law. """
    l_corner = l_reference * (l / l_reference)**power
    return l_corner

def filter_eigenvalues_ellipse(eigvals,aspect_ratio=0.5,N=10):
    """ Keep eigenvalues closest to the largest N eigenvalues of the ellipse."""
    eigvals_ellipse = compute_exact_eigenvalues_ellipse(aspect_ratio,N=N)
    eigvals_ellipse = np.flip(np.sort(eigvals_ellipse))
    eigvals_f = np.zeros_like(eigvals_ellipse,dtype=complex)
    for (i,eigv_ellipse) in enumerate(eigvals_ellipse):
        idx = np.argmin(np.abs(eigv_ellipse-eigvals))
        eigvals_f[i] = eigvals[idx] 
    return eigvals_f

@jit(nopython=True)
def filter_distance(x,y,x_target,y_target,dist=1e-1):
    """ Filter points that are at a distance less that dist from target. """
    mask = np.ones_like(x,dtype=np.bool8) 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            distance = np.sqrt((x[i,j] - x_target)**2 +  (y[i,j] - y_target)**2)
            if np.min(distance) < dist:
                mask[i,j] = False 
    return mask