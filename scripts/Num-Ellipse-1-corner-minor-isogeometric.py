#!/usr/bin/env python
# coding: utf-8
"""
Test case: eigenvalues for ellipse perturbed by one corner on the minor axis.

This script uses a Nyström discretization of the NP eigenvalue problem based
on an exact representation of the geometry.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]/'module'))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from computational_plasmonics import eigv_analytical as PEP_ana
import Num_utils
from Num_utils import DIR_MESH, DIR_ARTICLE_IMG, \
                                      MPLSTYLE_ARTICLE, MPLSTYLE_VSCODE
from dataclasses import dataclass
import Num_BEM
    # BEM parameters
qorder = 3
qnumber = int(np.ceil((qorder+1)/2))
qorder_correction = 11 # use analytical correction
opts_eigv = Num_BEM.EigenvalueSolverOpts(use_Arnoldi=False, tol=1e-8,
                                         nev=10, shift=[2e-1, -2e-1, -5e-2, 5e-2])
    # Geometry
(a, b, phi) = (2.5, 1, np.pi*0.85)
corner_pos = "top"
(x_c,y_c,x_m,y_m,R) = PEP_ana.get_C1corner_ellipse(a,b,phi,pos=corner_pos)

def compute_eigenvalues_bem(opts_eigv: Num_BEM.EigenvalueSolverOpts ,**kwargs):
    cor_jun1_theta = np.arccos(np.abs(x_m)/a)
    cor_jun2_theta = np.arccos(-np.abs(x_m)/a) - 2*np.pi
    return Num_BEM.compute_NP_eigenvalues_ellipse_with_corner_isogeometric(a,b,x_c,y_c,
                        cor_jun1_theta, cor_jun2_theta, opts_eigv = opts_eigv,
                        qorder=qorder, qorder_correction=qorder_correction, 
                        **kwargs)
#%% Plot of eigenvalues for various scaling parameters
element_size = a/10
alpha_l = [np.exp(1j*np.pi*0), np.exp(1j*np.pi/10),
           np.exp(1j*np.pi/7), np.exp(1j*np.pi/5)]
opts_eigv.use_Arnoldi = False
eigval_bem_l, dof_bem_l = list(), list()
for alpha in alpha_l:
    (eigvals, dof,_,_) = compute_eigenvalues_bem(opts_eigv,
                  qmaxdist=10*element_size, meshsize=element_size,
                  corner_grading_type="polynomial", corner_grading_regularity=1, 
                  pml_param=alpha)
    eigval_bem_l.append(eigvals)
    dof_bem_l.append(dof)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.9*paperwidth, 0.35*paperheight)
annotation_arrow_length = 20 # pt
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, axs = plt.subplots(nrows=4,ncols=1,sharey=True,sharex=True,
                      layout='constrained',figsize=figsize)
for (i,alpha) in enumerate(alpha_l):
    ax = axs.ravel()[i]
    eigval_ex = Num_utils.compute_exact_spectrum_corner(phi,pml_parameter=alpha)
    ax.plot(np.real(eigval_ex),np.imag(eigval_ex),
            label=r'$\sigma_{\text{ess}}(K^\star_\alpha)$', color='C0')
    ax.plot([np.abs(np.pi-phi)/(2*np.pi),-np.abs(np.pi-phi)/(2*np.pi)],[0,0],
            linestyle='none',
            label=r'$\pm\,\lambda_c(\phi)$',marker='o',color='C0')
    eigval_ex = Num_utils.compute_exact_eigenvalues_ellipse(np.min([b/a,a/b]),N=20)
    ax.plot(np.real(eigval_ex),np.imag(eigval_ex),
            label=r'$\lambda_n^{\text{el}}$',marker='s',fillstyle='none',
            linestyle='none',color='C0')
    eigval_f = Num_utils.filter_eigenvalues_ellipse(eigval_bem_l[i],
                                                         np.min([b/a,a/b]), N=4)
    if i!=0:
        for (k,l) in np.ndenumerate(eigval_f[np.real(eigval_f)>0]):
            (sign,valign) = (1,'bottom')
            ax.annotate(
                r'$\lambda^h_{val}$'.replace('val',f'{k[0]+1}'),
                xy=(np.real(l), np.imag(l)),
                xytext=(0, sign*annotation_arrow_length), textcoords="offset points",
                arrowprops=dict(arrowstyle="->",color="C1"),
                color='C1',clip_on=True,
                horizontalalignment='center',verticalalignment=valign,
            )
        for (k,l) in np.ndenumerate(np.sort(eigval_f[np.real(eigval_f)<0])):
            (sign,valign) = (-1,'top')
            ax.annotate(
                r'$\lambda^h_{val}$'.replace('val',f'{-(k[0]+1)}'),
                xy=(np.real(l), np.imag(l)),
                xytext=(0, sign*annotation_arrow_length), textcoords="offset points",
                arrowprops=dict(arrowstyle="->",color="C1"),
                color='C1',clip_on=True,
                horizontalalignment='center',verticalalignment=valign,
            )
    ax.plot(np.real(eigval_bem_l[i]),np.imag(eigval_bem_l[i]),
            label=f'BEM, $P$={qnumber}, DoF={dof_bem_l[i]}',linestyle='none',marker='x',
            color='C1')
        # Set title in a text box
    title=r'$\arg(\alpha)=0$'
    if np.angle(alpha)!=0:
        title = r'$\arg(\alpha)=\pi\,/\,$'+f'{np.pi/np.angle(alpha):.2g}' 
    ax.text(0.05,0.9,title,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='top', bbox={'facecolor':'white'})
    if i==len(alpha_l)-1: # xlabel on last axes only 
        ax.set_xlabel('$\Re(\lambda)$')
    ax.set_xlim([-0.2,0.2])
    ax.set_ylabel('$\Im(\lambda)$')
    ax.set_aspect('equal')
    ax.grid(True)
    # Legend inside first axis
ax = axs.ravel()[0]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles,loc='lower center',ncols=4)
    # Legend at figure level
f.suptitle(f'Eigenvalues, minor axis perturbation: '
           +f'$(a,b,\phi)=({{{a}}},{{{b}}},{{{phi/np.pi:.2g}}}\pi)$.')
#%% Plot self-convergence of eigenvalues for different corner gradings
alpha = np.exp(1j*np.pi/7)
element_size_l = [1/(2**N) for N in np.linspace(1,4,num=8)]
element_size_ref = 1/(2**7)
corner_grading_type_l = ["polynomial", "kress"]
corner_grading_regularity_l = [0, 3]
corner_grading_legend_l = ["Without corner grading", "With corner grading"]
opts_eigv.use_Arnoldi = True
def qmaxdist(element_size): return R/2
@dataclass
class SpectralComputation:
    """Stores output of a spectrum computation."""
    eigval_l: list
    dof_l: list
# Compute eigenvalues
bem_comp_l = list() # computed values
bem_ref_l = list() # reference values
for (idx, _) in enumerate(corner_grading_regularity_l):
    bem_comp_l.append(SpectralComputation([],[]))
    bem_ref_l.append(SpectralComputation([],[]))
    for (i,element_size) in enumerate(element_size_l):
        (eigvals, dof,_,_) = compute_eigenvalues_bem(opts_eigv,
                        meshsize=element_size,  qmaxdist = qmaxdist(element_size),
                        corner_grading_type=corner_grading_type_l[idx],
                        corner_grading_regularity=corner_grading_regularity_l[idx],
                        pml_param=alpha)
        bem_comp_l[-1].eigval_l.append(eigvals)
        bem_comp_l[-1].dof_l.append(dof)
    # Compute reference value
    (eigvals, dof,_,_) = compute_eigenvalues_bem(opts_eigv, 
                        meshsize=element_size_ref, qmaxdist = qmaxdist(element_size),
                        corner_grading_type=corner_grading_type_l[idx],
                        corner_grading_regularity=corner_grading_regularity_l[idx],
                        pml_param=alpha) 
    bem_ref_l[-1].eigval_l.append(eigvals)
    bem_ref_l[-1].dof_l.append(dof)
# Plot
N_track = 4 # track the first N_track eigenvalues
order_npts= len(element_size_l)
    # Parity of eigenfunctions (1: even, 2: odd)
eigvals_parity = [1, 2, 1, 2]
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.63*paperwidth, 0.3*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, axs = plt.subplots(ncols=len(corner_grading_regularity_l),sharey=True,
                      layout='constrained',figsize=figsize)
for (idx, corner_grading) in enumerate(corner_grading_regularity_l): # for each corner law
    bem = bem_comp_l[idx]
    bem_ref = bem_ref_l[idx]
    ax = axs[idx]
    ax.set_prop_cycle(marker=['o', 's', 'x','^'],
                color=mpl.rcParams['axes.prop_cycle'].by_key()['color'][:4])
    for i in range(N_track): # for each tracked eigenvalue 
        error = list()
        for (j, dof) in enumerate(bem.dof_l):
            eigval_f = Num_utils.filter_eigenvalues_ellipse(bem.eigval_l[j],
                                                aspect_ratio=np.min([b/a,a/b]),
                                                N=N_track)
            eigval_f_ref = Num_utils.filter_eigenvalues_ellipse(bem_ref.eigval_l[0],
                                                aspect_ratio=np.min([b/a,a/b]),
                                                N=N_track)
                # Maximum relative error on i-th eigenvalue
            error.append(np.abs(eigval_f[i] - eigval_f_ref[i])/np.abs(eigval_f_ref[i]))
                # Singular exponent
        eta_s = PEP_ana.compute_radial_corner_exponent_most_singular(phi,
                                      PEP_ana.np_to_contrast(eigval_f_ref[i]),
                                      pml_parameter=alpha)
        eta_s = eta_s[eigvals_parity[i]]
        print(f"Most singular η for λ_{i+1}: {eta_s:1.2g}")
        p = Num_utils.compute_slope(np.array(bem.dof_l),error,N=order_npts)
        ax.plot(np.array(bem.dof_l)/min(bem.dof_l),error,fillstyle='none',
                label=f'$\lambda^h_{{{i+1}}}, \mathcal{{O}}(h^{{{-p:2.2g}}})$')
    ax.set_ylim([1e-7,1e-1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([2,3,4, 5, 6, 7, 8],minor=True)
    ax.set_xticklabels(['2','3','4', '5', '6', '7', '8'],minor=True)
    ax.set_xlabel(f'DoF/{int(min(bem.dof_l)):d}')
    if idx==0:
        ax.set_ylabel(r'$|\lambda^h-\lambda^{\text{ref}}|\,/\,|\lambda^{\text{ref}}|$')
    ax.grid(which='both')
    ax.legend(loc='lower left')
    ax.set_title(f'{corner_grading_legend_l[idx]}\n'
                +fr'Reference at {bem_ref.dof_l[0]:d}$\simeq${bem_ref.dof_l[0]/min(bem.dof_l):2.2g}$\,\times\,${min(bem.dof_l)} DoF')
f.suptitle(fr'Isogeometric BEM error, minor axis perturbation: '
            +f'($a$,$b$,$\phi$)$=$({a},{b},{phi/np.pi:2g}$\pi$).'
            +'\n'
            +fr'$P=${qnumber}, $\arg(\alpha)=\pi/${np.pi/np.angle(alpha):.2g}.'
            )
f.savefig(os.path.join(DIR_ARTICLE_IMG,f'Num-Ellipse-1-Corner-Minor-convergence-isogeo'))
#%% Plot eigenfunctions associated with the first N_track eigenfunctions
N_track = 4 # track the first N_track eigenvalues
alpha = np.exp(1j*np.pi/7)
element_size = a/40
opts_eigv.use_Arnoldi = False
(eigvals, dof, eigvecs, sl_pot) = compute_eigenvalues_bem(opts_eigv,
                               meshsize=element_size, 
                               qmaxdist = 10 * element_size,
                               corner_grading_type="kress",
                               corner_grading_regularity=2,
                               pml_param=alpha)
eigvals_f = Num_utils.filter_eigenvalues_ellipse(eigvals,
                                    aspect_ratio=np.min([b/a,a/b]), N=N_track)
plt.style.use(MPLSTYLE_VSCODE)
figsize = mpl.rcParams['figure.figsize']
fig, axs = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True,
                        layout='constrained', figsize=figsize)
cor_jun1_theta = np.arccos(np.abs(x_m)/a)
cor_jun2_theta = np.arccos(-np.abs(x_m)/a) - 2*np.pi
t = np.linspace(cor_jun2_theta,cor_jun1_theta,num=int(1e2))
x_ellipse = np.hstack((x_c,a*np.cos(cor_jun2_theta),a*np.cos(t),x_c))
y_ellipse = np.hstack((y_c,b*np.sin(cor_jun2_theta),b*np.sin(t), y_c))
for n, ax in enumerate(fig.axes):
    eigv_idx = (np.abs(eigvals - eigvals_f[n])).argmin()
    spanx = np.linspace(-1.10*a,1.10*a,num=int(1e2))
    spany = np.linspace(-a,a,num=int(1e2))
    x_, y_ = np.meshgrid(spanx, spany)
    z_ = np.array(sl_pot(eigvecs[:,eigv_idx],x_,y_),dtype=complex)
    z_ = np.real(z_)
        # do not normalize using values close to boundary
    mask = Num_utils.filter_distance(x_,y_,x_ellipse,y_ellipse,dist=a/100)
    z_ = z_/np.max(np.abs(z_[mask]))
    coll = ax.pcolormesh(x_, y_, z_ , shading='gouraud', cmap='RdBu', 
                         rasterized=True, vmin=-1, vmax=1)
    ax.plot(x_ellipse,y_ellipse,color='C0') 
    R = np.linalg.norm(np.array([x_c,y_c]) - np.array([a*np.cos(cor_jun1_theta),b*np.sin(cor_jun1_theta)]))
    patch = mpl.patches.Circle((x_c,y_c),radius=R,
                                facecolor='none', edgecolor='C1',
                                linewidth=mpl.rcParams['lines.linewidth'])
    ax.add_artist(patch)
    ax.text(x_c,y_c-0.9*R, r'$B_\alpha$', color='C1',
                    horizontalalignment='center',verticalalignment='bottom')
    if n >= 2:
        ax.set_xlabel("$x$")
    if n % 2 ==0:
        ax.set_ylabel("$y$")
    ax.set_title(f"$\lambda^h_{n+1}$={eigvals[eigv_idx]:1.1e}")
    ax.set_xlim([-1.10*a,1.10*a])
    ax.set_ylim([-1.10*b,1.20*y_c])
    ax.set_aspect('equal')
fig.colorbar(coll,ax=axs[:,1],label=r'$\Re(u_n)$')
fig.suptitle(f'BEM eigenfunctions, minor axis perturbation: '
             + f'($a$,$b$,$\phi={{{phi/np.pi:.2g}}}\pi$).'
             + f'\n'
             + f'$P=${qnumber}, DoF$=${dof}, '
             + fr'$\arg(\alpha)=\pi/{{{np.pi/np.angle(alpha):.2g}}}$.')