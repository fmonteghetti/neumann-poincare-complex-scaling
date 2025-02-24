#!/usr/bin/env python
# coding: utf-8
"""
Validation: eigenvalues for a corner.

This script compares the eigenvalues computed with:

    - FEM discretization of the PEP, using a Dirichlet-to-Neumann boundary
    condition on the outer boundary.
    - Nyström discretization of the NP eigenvalue problem.

Both methods take as input the same mesh, obtained with gmsh.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]/'module'))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import Num_utils
from Num_utils import DIR_MESH, DIR_ARTICLE_IMG, \
                                      MPLSTYLE_ARTICLE, MPLSTYLE_VSCODE
import Num_BEM
import Num_FEM

    # BEM parameters
qorder = 3
qnumber = int(np.ceil((qorder+1)/2))
qorder_correction = 11
opts_eigv = Num_BEM.EigenvalueSolverOpts(use_Arnoldi=False)

    # Geometry and mesh
geofile = os.path.join(DIR_MESH,"Camembert.geo")
R = 1 # corner radius
phi = np.pi/2 # corner angle 
alpha = np.exp(1j*np.pi/4) # complex scaling
geo_param={
    'R': R,
    'phi': phi,
    'corner_x': 1, # x-coordinate of corner
    'corner_y': 2, # y-coordinate of corner
    'lc_d': R/2, # mesh size on outer boundary
    'lc_c': R/3 # mesh size at corner tip
    }
gmsh_param = {
    'binary': True,
    'order': 2,
    'meshing' : 9,
    'recombination' : 3,
    'flexible_transfinite' : False,
    'refinement': 0
}

compute_eigenvalues_fem = Num_FEM.compute_eigenvalues_corner_Cartesian
compute_exact_spectrum = Num_utils.compute_exact_spectrum_corner
def compute_eigenvalues_bem(gmshfile, opts_eigv, **kwargs_julia):
    return Num_BEM.compute_NP_eigenvalues_corner(gmshfile, opts_eigv, 
                qorder=qorder, qorder_correction=qorder_correction,
                pml_radius=R,
                pml_center=(geo_param['corner_x'], geo_param['corner_y']),
                **kwargs_julia)
#%% Plot eigenvalues obtained with BEM
geo_param['lc_d']= R/5
geo_param['lc_c']= geo_param['lc_d']/1e4
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
alpha_l = [np.exp(1j*np.pi/10), np.exp(1j*np.pi/3)]
eigval_bem_l, dof_bem_l = list(), list()
for alpha in alpha_l:
    (eigval,dof) = compute_eigenvalues_bem(gmshfile,opts_eigv, 
                                        qmaxdist=10*geo_param['lc_d'],
                                        pml_param=alpha) 
    eigval_bem_l.append(eigval)
    dof_bem_l.append(dof)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.42*paperwidth, 0.18*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, axs = plt.subplots(nrows=1,ncols=len(alpha_l),sharey=True,
                      layout='constrained', figsize=figsize)
for (i,alpha) in enumerate(alpha_l):
    ax = axs[i]
    eigval_ex = compute_exact_spectrum(phi,pml_parameter=alpha)
    ax.plot(np.real(eigval_ex),np.imag(eigval_ex),
            label=r'$\sigma_{\text{ess}}(K^\star_\alpha)$', color='C0')
    ax.plot([np.abs(np.pi-phi)/(2*np.pi),-np.abs(np.pi-phi)/(2*np.pi)],[0,0],
            linestyle='none',
            label=r'$\pm\,\lambda_c(\phi)$',marker='o',color='C0')
    ax.plot(np.real(eigval_bem_l[i]),np.imag(eigval_bem_l[i]),
            label=f'BEM, $P$={qnumber}, DoF={dof_bem_l[i]}',
                                    linestyle='none',marker='x', color='C1')
    ax.set_ylim([-0.3,0.3])
    ax.set_xlim([-0.3,0.3])
    ax.set_aspect('equal')
    ax.set_xlabel('$\Re(\lambda)$')
    ax.set_title(r'$\arg(\alpha)=\pi\,/\,$'+f'{np.pi/np.angle(alpha):.2g}'+
                 r'$,\,\phi=\pi\,/\,$'+f'{np.pi/phi:.2g}')
    if i==0:
        ax.set_ylabel('$\Im(\lambda)$')
        handles, labels = ax.get_legend_handles_labels() 
        ax.add_artist(ax.legend(loc='upper left', ncols=1,handles=handles[:2]))
        ax.add_artist(ax.legend(loc='lower left', ncols=1,handles=handles[-1:]))
    ax.grid(True)
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Corner-Eigenvalues'))
#%% Plot eigenvalues obtained with BEM and FEM
alpha_l = [np.exp(1j*np.pi/10), np.exp(1j*np.pi/5), np.exp(1j*np.pi/3)]
# Compute BEM eigenvalues
geo_param['lc_d']= R/5
geo_param['lc_c']= geo_param['lc_d']/1e4
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
eigval_bem_l, dof_bem_l = list(), list()
for alpha in alpha_l:
    (eigval,dof) = compute_eigenvalues_bem(gmshfile, opts_eigv, 
                                        qmaxdist=10*geo_param['lc_d'],
                                        pml_param=alpha)
    eigval_bem_l.append(eigval)
    dof_bem_l.append(dof)
geo_param['corner_x']=0
geo_param['corner_y']=0
# geo_param['lc_d']=R/10
# geo_param['lc_c']=geo_param['lc_d']/5
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=2)
eigval_fem_l, dof_fem_l = list(), list()
for alpha in alpha_l:
    (eigval,dof) = compute_eigenvalues_fem(gmshfile,quad_order=2,
                                           pml_param=alpha,
                                           lambda_target=[0.3, 1/0.3],
                                           nev=50, tol=1e-14)
    eigval_fem_l.append(eigval)
    dof_fem_l.append(dof)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.62*paperwidth, 0.18*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, axs = plt.subplots(nrows=1,ncols=len(alpha_l),sharey=True,
                      layout='constrained', figsize=figsize)
for (i,alpha) in enumerate(alpha_l):
    ax = axs[i]
    eigval_ex = compute_exact_spectrum(phi,pml_parameter=alpha)
    ax.plot(np.real(eigval_ex),np.imag(eigval_ex),
            label=r'$\sigma_{\text{ess}}(K^\star_\alpha)$', color='C0')
    ax.plot([np.abs(np.pi-phi)/(2*np.pi),-np.abs(np.pi-phi)/(2*np.pi)],[0,0],
            linestyle='none',
            label=r'$\pm\,\lambda_c(\phi)$',marker='o',color='C0')
    ax.plot(np.real(eigval_fem_l[i]),np.imag(eigval_fem_l[i]),
            label=f'FEM, DoF={dof_fem_l[i]}',
                                    linestyle='none',marker='o', fillstyle='none',
                                    color='C2')
    ax.plot(np.real(eigval_bem_l[i]),np.imag(eigval_bem_l[i]),
            label=f'BEM, $P$={qnumber}, DoF={dof_bem_l[i]}',
                                    linestyle='none',marker='x', color='C1')
    ax.set_ylim([-0.3,0.3])
    ax.set_xlim([-0.3,0.3])
    ax.set_aspect('equal')
    ax.set_xlabel('$\Re(\lambda)$')
    ax.set_title(r'$\arg(\alpha)=\pi\,/\,$'+f'{np.pi/np.angle(alpha):.2g}'+
                 r'$,\,\phi=\pi\,/\,$'+f'{np.pi/phi:.2g}')
    if i==0:
        ax.set_ylabel('$\Im(\lambda)$')
        handles, labels = ax.get_legend_handles_labels() 
        ax.add_artist(ax.legend(loc='upper left', ncols=1,handles=handles[:2]))
        ax.add_artist(ax.legend(loc='lower left', ncols=1,handles=handles[-2:]))
    ax.grid(True)
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Corner-Eigenvalues-FEM-BEM'))