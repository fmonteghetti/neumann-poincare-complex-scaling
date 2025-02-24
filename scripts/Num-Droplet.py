#!/usr/bin/env python
# coding: utf-8
"""
Test case: eigenvalues for droplet.

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
from Num_utils import DIR_ARTICLE_IMG, \
                                      MPLSTYLE_ARTICLE, MPLSTYLE_VSCODE
import Num_BEM
# BEM parameters
qorder = 3
qnumber = int(np.ceil((qorder+1)/2))
qorder_correction = 11
corner_grading_type="kress"
corner_grading_regularity=1
R_pml = 5e-1
x_c, y_c = 0, 0
opts_eigv = Num_BEM.EigenvalueSolverOpts(use_Arnoldi=False)
# Manually extracted eigenvalues of interest
eigvals_f = np.array([0.18160343579523447 - 0.0005568987991445146*1j,
                    0.0983584118186975 - 0.022660295288301015*1j,
                    0.028888198040342754 - 0.0003113620591224147*1j,
                    0.012018189869378776 - 0.0038200955049157953*1j,
                    ])
eigvals_f = np.hstack((eigvals_f,-eigvals_f))
def compute_eigenvalues_BEM(opts_eigv, **kwargs_julia):
    return Num_BEM.compute_NP_eigenvalues_droplet_isogeometric(opts_eigv,
                                    qorder=qorder,
                                    qorder_correction=qorder_correction,
                                    corner_grading_type=corner_grading_type,
                                    corner_grading_regularity=corner_grading_regularity,
                                    pml_radius=R_pml, **kwargs_julia)
#%% Plot of eigenvalues for various scaling parameters
element_size = 1/30
alpha_l = [np.exp(1j*np.pi*0), np.exp(1j*np.pi/10),
           np.exp(1j*np.pi/7), np.exp(1j*np.pi/6)]
alpha_l = [np.exp(1j*np.pi*0), np.exp(1j*np.pi/7)]
eigval_bem_l, dof_bem_l = list(), list()
for alpha in alpha_l:
    (eigvals, dof, phi,_,_,_) = compute_eigenvalues_BEM(opts_eigv,
                                meshsize=element_size, qmaxdist=10*element_size,
                                                               pml_param=alpha)
    eigval_bem_l.append(eigvals)
    dof_bem_l.append(dof)

plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.6*paperwidth, 0.20*paperheight)
annotation_arrow_length = 20 # pt
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, axs = plt.subplots(nrows=2,ncols=1,sharey=True,sharex=True,
                      layout='constrained',figsize=figsize)
for (i,alpha) in enumerate(alpha_l):
    ax = axs.ravel()[i]
    eigval_ex = Num_utils.compute_exact_spectrum_corner(phi,pml_parameter=alpha)
    ax.plot(np.real(eigval_ex),np.imag(eigval_ex),
            label=r'$\sigma_{\text{ess}}(K^\star_\alpha)$', color='C0')
    ax.plot([np.abs(np.pi-phi)/(2*np.pi),-np.abs(np.pi-phi)/(2*np.pi)],[0,0],
            linestyle='none',
            label=r'$\pm\,\lambda_c(\phi)$',marker='o',color='C0')
    if i!=0:
        for (k,l) in np.ndenumerate(eigvals_f[np.real(eigvals_f)>0]):
            (sign,valign) = (1,'bottom')
            ax.annotate(
                r'$\lambda^h_{val}$'.replace('val',f'{k[0]+1}'),
                xy=(np.real(l), np.imag(l)),
                xytext=(0, sign*annotation_arrow_length), textcoords="offset points",
                arrowprops=dict(arrowstyle="->",color="C1"),
                color='C1',clip_on=True,
                horizontalalignment='center',verticalalignment=valign,
            )
        for (k,l) in np.ndenumerate(np.sort(eigvals_f[np.real(eigvals_f)<0])):
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
    # ax.set_title(title)
    ax.text(0.05,0.9,title,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='top', bbox={'facecolor':'white'})
    if i==len(alpha_l)-1: # xlabel on last axes only 
        ax.set_xlabel('$\Re(\lambda)$')
    ax.set_xlim([-0.3,0.3])
    # ax.set_ylim([-0.1,0.1])
    ax.set_ylabel('$\Im(\lambda)$')
    ax.set_aspect('equal')
    ax.grid(True)
    # Legend inside first axis
ax = axs.ravel()[0]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles,loc='lower center',ncols=4)
# ax.legend(handles=handles,loc='upper left', bbox_to_anchor=(1,1),ncols=1)
    # Legend at figure level
f.suptitle(f'Eigenvalues, droplet.')
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Droplet-Isogeo-Eigenvalues'))
#%% Plot eigenfunctions associated with the first N_track eigenfunctions
N_track = 4 # track the first N_track eigenvalues
alpha = np.exp(1j*np.pi/7)
element_size = 1/90
(eigvals, dof, phi, eigvecs, sl_pot,R_pml_corrected) = compute_eigenvalues_BEM(opts_eigv,
                                meshsize=element_size, qmaxdist=10*element_size,
                                pml_param=alpha) 
t = np.linspace(0,2*np.pi,num=int(1e2))
x_bnd = 2/np.sqrt(3) * np.sin(t/2)
y_bnd = -np.sin(t)
plt.style.use(MPLSTYLE_VSCODE)
figsize = mpl.rcParams['figure.figsize']
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.63*paperwidth, 0.2*paperheight)
fig, axs = plt.subplots(nrows=1,ncols=4, sharex=True, sharey=True,
                        layout='constrained', figsize=figsize)
for n, ax in enumerate(fig.axes):
    eigv_idx = (np.abs(eigvals - eigvals_f[n])).argmin()
    spanx = np.linspace(-0.1,2,num=int(1e2))
    spany = np.linspace(-1.10,1.10,num=int(1e2))
    x_, y_ = np.meshgrid(spanx, spany)
    z_ = np.array(sl_pot(eigvecs[:,eigv_idx],x_,y_),dtype=complex)
    z_ = np.real(z_)
    z_ = z_/np.max(np.abs(z_))
        # do not normalize using values close to boundary
    mask = Num_utils.filter_distance(x_,y_,x_bnd,y_bnd,dist=1/100)
    z_ = z_/np.max(np.abs(z_[mask]))
    coll = ax.pcolormesh(x_, y_, z_ , shading='gouraud', cmap='RdBu', 
                         rasterized=True, vmin=-1, vmax=1)
    ax.plot(x_bnd,y_bnd,color='C0') 
    patch = mpl.patches.Circle((x_c,y_c),radius=R_pml_corrected,
                                facecolor='none', edgecolor='C1',
                                linewidth=mpl.rcParams['lines.linewidth'])
    ax.add_artist(patch)
    ax.text(x_c+1.1*R_pml_corrected,y_c, r'$B_\alpha$', color='C1',
                    horizontalalignment='left',verticalalignment='center')
    if n >= 2:
        ax.set_xlabel("$x$")
    # if n % 2 ==0:
    #     ax.set_ylabel("$y$")
    ax.set_title(f"$\lambda^h_{n+1}\simeq${eigvals[eigv_idx]:1.1e}")
    ax.set_xlim([-0.10,1.5])
    ax.set_ylim([-1.10,1.10])
    ax.set_aspect('equal')
# fig.colorbar(coll,ax=axs[:,1],label=r'$\Re(u_n)$')
fig.colorbar(coll,ax=axs[-1],label=r'$\Re(u_n)$',ticks=[-1,0,1])
fig.suptitle(f'BEM eigenfunctions, droplet. '
             + f'\n'
             + f'$P=${qnumber}, DoF$=${dof}, '
             + fr'$\arg(\alpha)=\pi/{{{np.pi/np.angle(alpha):.2g}}}$.')