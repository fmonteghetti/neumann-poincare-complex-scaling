#!/usr/bin/env python
# coding: utf-8
"""
Validation: eigenvalues for an ellipse.

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
import Num_FEM
import Num_BEM
from computational_plasmonics import eigv_analytical as PEP_ana

def compute_error(eigval,eigval_ex,N=1):
    """ Compute the maximum error against the first N exact eigenvalues.

    The error is computed as
            max     |λ_i - λ_{i,ex}| / |λ_{i,ex}|
            1<=i<=N
    where eigenvalues are sorted by decreasing real part: 
                 ... <= 𝕽(λ_2) <= 𝕽(λ_1) 
    """
    eigval_ex = np.sort(eigval_ex)
    eigval = np.sort(eigval)
    return np.max(np.abs(eigval[-N:]-eigval_ex[-N:])/np.abs(eigval_ex[-N:]))
    # BEM parameters
qorder = 3
qnumber = int(np.ceil((qorder+1)/2))
qorder_correction = 11
opts_eigv = Num_BEM.EigenvalueSolverOpts(use_Arnoldi=False)
    # Geometry and mesh
geofile= os.path.join(DIR_MESH,"Ellipse_Unstructured.geo")
(a, b) = (2.5, 1)
aspect_ratio = np.min([a/b,b/a])
geo_param= {
    'a_m':a, 'b_m':b, # sign-changing interface
    'a_d':2*a, 'b_d':2*a, # DtN boundary
    'N_m': 1 # Number of elements on interface Γ 
}
gmsh_param = {
    'binary': True, 'order': 2,
    'meshing' : 9, 'recombination' : 3,
    'flexible_transfinite' : False
}
compute_eigenvalues_fem = Num_FEM.compute_eigenvalues_Cartesian 
def compute_eigenvalues_bem(gmshfile, opts_eigv, **kwargs_julia):
    return Num_BEM.compute_NP_eigenvalues(gmshfile, opts_eigv, qorder=qorder,
                            qorder_correction=qorder_correction, **kwargs_julia)
#%% Plot eigenvalues of both FEM and BEM
order_npt = 5 # compute order on last order_npts   
eigval_fem_l, dof_fem_l = list(), list()
for N_m in [30, 100]:
    geo_param['N_m'] = N_m
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param)
    (eigval,dof) = compute_eigenvalues_fem(gmshfile,DtN_order=30,
                                        quad_order=2, lambda_target=[0.2,-0.2])
    eigval_fem_l.append(eigval)
    dof_fem_l.append(dof)
eigval_bem_l, dof_bem_l = list(), list()
for N_m in [20, 500]:
    geo_param['N_m'] = N_m
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
    (eigval,dof,_,_) = compute_eigenvalues_bem(gmshfile, opts_eigv, qmaxdist=10*a/N_m)
    eigval_bem_l.append(eigval)
    dof_bem_l.append(dof)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.31*paperwidth, 0.18*paperheight)
f, ax = plt.subplots(layout='constrained',figsize=figsize)
eigval_ex = Num_utils.compute_exact_eigenvalues_ellipse(aspect_ratio)
for (i,offset) in enumerate([-0.15,-0.1,0.1,0.15]):
    ax.plot(np.real(eigval_ex),np.imag(eigval_ex)+offset,
                   label=r"$\lambda^{\text{el}}$" if i==0 else '_none', 
                   linestyle='none',marker='s',fillstyle='none',
                   color='C0')
ax.set_prop_cycle(color=mpl.rcParams['axes.prop_cycle'].by_key()['color'][1:3])
marker_l = [str(1),str(2)]
for (i,offset) in enumerate([0.15,0.1]):
    ax.plot(np.real(eigval_bem_l[i]),np.imag(eigval_bem_l[i])+offset,
            label=f'BEM, DoF={dof_bem_l[i]}', linestyle='none',marker=marker_l[i])
marker_l = ['+','x']
for (i,offset) in enumerate([-0.1,-0.15]):
    ax.plot(np.real(eigval_fem_l[i]),np.imag(eigval_fem_l[i])+offset,
            label=f'FEM, DoF={dof_fem_l[i]}', linestyle='none',marker=marker_l[i])
ax.set_ylim([-0.2,0.2])
ax.set_xlim([-0.3,0.3])
ax.set_xlabel('$\Re(\lambda)$')
ax.set_ylabel('')
ax.set_yticks([-0.15,-0.1,0.1,0.15])
ax.set_yticklabels(['','','',''])
ax.set_title(f'Eigenvalues, ($a$,$b$)=({a},{b})')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0,3,4]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
          ncols=2,loc='center', bbox_to_anchor=(0.5, 0.5))
ax.grid(True)
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-Eigenvalues-FEM-BEM'))
#%% Plot convergence (FEM)
# Compute eigenvalues
N_gamma_l = [30, 40, 50, 60, 70, 80, 90, 100]
eigval_l, dof_l = list(), list()
for (i,N_gamma) in enumerate(N_gamma_l):
    geo_param['N_m'] = N_gamma
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param)
    (eigval,dof) = compute_eigenvalues_fem(gmshfile,DtN_order=30,quad_order=2, 
                                                    lambda_target=[0.2,-0.2])
    eigval_l.append(eigval)
    dof_l.append(dof)
#%% Plot error
N_error_l = [1, 2, 5]
order_npt = 4 # compute order on last order_npts   
order_bypass = [1, 2]
eigval_ex = Num_utils.compute_exact_eigenvalues_ellipse(aspect_ratio,N=30)
error_l = np.zeros((len(N_error_l),len(eigval_l)))
for (i,N_error) in enumerate(N_error_l):
    for (j,eigval) in enumerate(eigval_l):
        error_l[i,j] = compute_error(eigval,eigval_ex,N=N_error)
import matplotlib.ticker as ticker
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.31*paperwidth, 0.18*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, ax = plt.subplots(layout='constrained',figsize=figsize)
ax.set_prop_cycle(marker=['o', 's', 'x'],
                  color=mpl.rcParams['axes.prop_cycle'].by_key()['color'][:3])
for (i,N_error) in enumerate(N_error_l):
    error = error_l[i,:]
    p = Num_utils.compute_slope(np.array(dof_l),np.array(error),N=order_npt)
    if i in order_bypass:
        ax.plot(np.array(dof_l)/min(dof_l),error,fillstyle='none',
                label=f'$J=${N_error}')
    else:
        ax.plot(np.array(dof_l)/min(dof_l),error,fillstyle='none',
                label=f'$J=${N_error}, $\mathcal{{O}}(h^{{{-p:.2g}}})$')
ax.set_ylim([1e-8,5e1])
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
ax.grid(which='both')
# ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=1))
ax.set_xlabel(f'DoF / {int(min(dof_l)):d}')
ax.set_ylabel(r'$\max_{1\leq{}j\leq{}J}\,|\lambda^h_j-\lambda_j^{\text{el}}|\,/\,|\lambda_j^{\text{el}}|$')
ax.set_title(f'FEM Error, ($a$,$b$)=({a},{b})')
ax.legend()
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-Convergence-FEM'))
#%% Plot convergence (BEM)
# Compute eigenvalues
gmsh_param['order'] = 2
N_gamma_l = [20, 50, 100, 125, 150, 175, 200, 250, 300, 400]
eigval_l, dof_l = list(), list()
for N_gamma in N_gamma_l:
    geo_param['N_m'] = N_gamma
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
    (eigval,dof,_,_) = compute_eigenvalues_bem(gmshfile, opts_eigv, qmaxdist=10*a/N_gamma) 
    eigval_l.append(eigval)
    dof_l.append(dof)
#%% Plot error
N_error_l = [1, 2, 5]
order_npt = 4   
eigval_ex = Num_utils.compute_exact_eigenvalues_ellipse(aspect_ratio,N=30)
error_l = np.zeros((len(N_error_l),len(eigval_l)))
for (i,N_error) in enumerate(N_error_l):
    for (j,eigval) in enumerate(eigval_l):
        error_l[i,j] = compute_error(eigval,eigval_ex,N=N_error)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.31*paperwidth, 0.18*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, ax = plt.subplots(layout='constrained',figsize=figsize)
ax.set_prop_cycle(marker=['o', 's', 'x'],
                  color=mpl.rcParams['axes.prop_cycle'].by_key()['color'][:3])
for (i,N_error) in enumerate(N_error_l):
    error = error_l[i,:]
    p = Num_utils.compute_slope(np.array(dof_l),np.array(error),N=order_npt)
    ax.loglog(np.array(dof_l)/min(dof_l),error,fillstyle='none',
              label=f'$J=${N_error}, $\mathcal{{O}}(h^{{{-p:2.2g}}})$')
ax.set_xlabel(f'DoF / {int(min(dof_l)):d}')
ax.set_ylabel(r'$\max_{1\leq{}j\leq{}J}\,|\lambda_j^h-\lambda_j^{\text{el}}|\,/\,|\lambda_j^{\text{el}}|$')
ax.set_title(f'BEM error, $P$={qnumber}, ($a$,$b$)=({a},{b})')
ax.legend(loc='best',ncols=1)
ax.grid(which='both')
ax.set_ylim([1e-9,1e-1])
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-Convergence-BEM'))
#%% Plot spectrum and eigenfunctions (BEM)
geo_param['N_m'] = 100
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
(eigval,dof, eigvecs, sl_pot) = compute_eigenvalues_bem(gmshfile, opts_eigv, 
                                                qmaxdist=10*a/geo_param['N_m'])
(ev_l,ev_k,ev_n,ev_mask) = PEP_ana.ellipse_freespace_eigenvalues(a,b,N=20)
eigvals_ex = ev_l
plot_mode_idx = [-9, 3] # index of modes to plot (|n| >=1)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.64*paperwidth, 0.1*paperheight)
fig_width_ratios = [1.7, 1, 1]
# plt.style.use(MPLSTYLE_VSCODE); figsize = mpl.rcParams['figure.figsize']
fig, axs = plt.subplots(1,3, layout='constrained', 
                              gridspec_kw=dict(width_ratios=fig_width_ratios), 
                              figsize=figsize)
    # Plot spectrum
ax=axs[0]
ax.plot(np.real(eigval),np.imag(eigval),
        label=f'$\lambda^h$, $P$={qnumber}, DoF={dof}',linestyle='none',marker='+',
        color='C1')
ax.plot(eigvals_ex,0*eigvals_ex,label=rf'$\lambda^{{\text{{el}}}}$',color='C0',marker='s',linestyle='none',fillstyle='none')
ax.set_xlim([-0.25,0.25])
ax.set_ylim([-4e-1,1])
ax.grid()
ax.legend(ncols=2,loc='upper center')
ax.set_xlabel(r'$\Re(\lambda)$')
ax.set_ylabel(r'$\Im(\lambda)$')
ax.set_title(f'Eigenvalues, ($a$,$b$)=({a},{b})')
t = np.linspace(0,2*np.pi,num=int(1e2))
x_ellipse = a * np.cos(t)
y_ellipse = b * np.sin(t)
for (i,idx) in enumerate(plot_mode_idx):
    ax=axs[1+i]
    eigv_val = ev_l[ev_n==idx][0]
    eigv_idx = (np.abs(eigval - eigv_val)).argmin()
    d_x, d_y = 1.15*a, 1.65*b
    spanx = np.linspace(-d_x,d_x,num=int(1e2))
    spany = np.linspace(-d_y,d_y,num=int(1e2))
    x_, y_ = np.meshgrid(spanx, spany)
    z_ = np.array(sl_pot(eigvecs[:,eigv_idx],x_,y_),dtype=complex)
    z_ = np.real(z_)
        # do not normalize using values close to boundary
    mask = Num_utils.filter_distance(x_,y_,x_ellipse,y_ellipse,dist=a/100)
    z_ = z_/np.max(np.abs(z_[mask]))
    coll = ax.pcolormesh(x_, y_, z_ , shading='gouraud', cmap='RdBu', 
                        rasterized=True, vmin=-1, vmax=1)
    if i==1:
        fig.colorbar(coll,ax=ax,ticks=[-1,0,1],label=r'$u^h$')
    ax.plot(x_ellipse,y_ellipse,color='C0')
    ax.set_xlabel("$x$")
    if i==0:
        ax.set_ylabel("$y$")
    else:
        ax.set_yticklabels([])
    ax.set_title(fr"$\lambda^h$={eigv_val:1.1e}$\simeq\lambda_{{{idx}}}^{{\text{{el}}}}$")
    ax.set_xlim([-d_x,d_x])
    ax.set_aspect('equal')
fig.savefig(os.path.join(DIR_ARTICLE_IMG,"PEP-Ellipse"))