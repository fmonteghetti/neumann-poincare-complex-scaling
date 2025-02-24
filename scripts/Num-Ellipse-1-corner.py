#!/usr/bin/env python
# coding: utf-8
"""
Test case: eigenvalues for ellipse perturbed by one corner on the major axis.

This script compares the eigenvalues computed with:

    - FEM discretization of the PEP, using a Dirichlet-to-Neumann boundary
    condition on the outer boundary and Euler coordinates around the corner.
    - Nyström discretization of the NP eigenvalue problem.

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
import Num_FEM
import Num_BEM
    # BEM parameters
qorder = 3
qnumber = int(np.ceil((qorder+1)/2))
qorder_correction = 11 
opts_eigv = Num_BEM.EigenvalueSolverOpts(use_Arnoldi=False, tol=1e-8,
                                         nev=10, shift=[2e-1, -2e-1, -5e-2, 5e-2])
grading_power = 4 # aggressive refinment
    # Geometry
(a, b, phi) = (2.5, 1, np.pi*0.75)
corner_pos = "left"
(x_c,y_c,x_m,y_m,R) = PEP_ana.get_C1corner_ellipse(a,b,phi,pos=corner_pos)
R_DtN = 5 # radius of DtN boundary
cor_jun1_theta = np.arccos(x_m/a)
cor_jun2_theta = -np.arccos(x_m/a)
    # Geometry and mesh (BEM: Cartesian coordinates only)
geofile=os.path.join(DIR_MESH,"Ellipse-with-1-corner_Cartesian.geo")
geo_param={
    'a_m':a,'b_m':b,
    'a_d':R_DtN,'b_d':R_DtN,
    'cor_x': x_c, 'cor_y': y_c, 
    'cor_jun1_theta': cor_jun1_theta,
    'cor_jun2_theta': cor_jun2_theta,
    'lc_m': a/10, 'cor_lc': a/20 # element size
}
gmsh_param = {
    'binary': True, 'order': 2,
    'recombination' : 3,
    'flexible_transfinite' : False
}
    # Geometry and mesh (FEM: Cartesian+Euler coordinates)
geofile_fem=os.path.join(DIR_MESH,"Ellipse-with-1-corner_Structured-v1.geo")
geo_param_fem={
    'a_m':a,'b_m':b,
    'a_d':R_DtN,'b_d':R_DtN, 
    'phi1':phi,
}
gmsh_param_fem = {
    'save_and_exit': True, 'binary': True,
    'order' : 2, 'meshing' : -1,
    'recombination' : -1, 'flexible_transfinite' : True
}
def compute_eigenvalues_BEM(gmshfile, opts_eigv, **kwargs_julia):
    return Num_BEM.compute_NP_eigenvalues(gmshfile, opts_eigv,
                            qorder=qorder, qorder_correction=qorder_correction,
                           pml_radius=R, pml_center=(x_c,y_c), **kwargs_julia)
#%% Plot of eigenvalues for various scaling parameters
element_size = a/60
opts_eigv.use_Arnoldi = False
geo_param['lc_m'] = element_size
geo_param['cor_lc'] = Num_utils.corner_mesh_grading_power(element_size,
                                                           power=grading_power) 
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
alpha_l = [np.exp(1j*np.pi*0), np.exp(1j*np.pi/10),
           np.exp(1j*np.pi/7), np.exp(1j*np.pi/5)]
eigval_bem_l, dof_bem_l = list(), list()
for alpha in alpha_l:
    (eigval,dof, _, _) = compute_eigenvalues_BEM(gmshfile, opts_eigv,
                            qmaxdist = 10*element_size, pml_param=alpha)
    eigval_bem_l.append(eigval)
    dof_bem_l.append(dof)
#%%
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.9*paperwidth, 0.35*paperheight)
annotation_arrow_length = 15 # pt
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
            label=r'$\lambda^{\text{el}}$',marker='s',fillstyle='none',
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
            label=f'BEM, $P=${qnumber}, DoF={dof_bem_l[i]}',
            linestyle='none',marker='x', color='C1')
        # Set title in a text box
    title=r'$\arg(\alpha)=0$'
    if np.angle(alpha)!=0:
        title = r'$\arg(\alpha)=\pi\,/\,$'+f'{np.pi/np.angle(alpha):.2g}' 
    ax.text(0.05,0.9,title,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='top', bbox={'facecolor':'white'})
    if i==len(alpha_l)-1: # xlabel on last axes only 
        ax.set_xlabel('$\Re(\lambda)$')
    ax.set_xlim([-0.3,0.3])
    ax.set_ylabel('$\Im(\lambda)$')
    ax.set_aspect('equal')
    ax.grid(True)
    # Legend inside first axis
ax = axs.ravel()[0]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles,loc='lower center',ncols=4)
    # Legend at figure level
# f.legend(handles, labels, loc='outside upper right',ncols=1)
f.suptitle(f"Eigenvalues, major axis perturbation: $(a,b,\phi)=$"
           +fr"({a},{b},{phi/np.pi:.2g}$\pi$).")
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-1-Corner-Eigenvalues'))
#%% Plot of eigenvalues obtained with FEM and BEM
alpha = np.exp(1j*np.pi/7)
opts_eigv.use_Arnoldi = False
element_size = a/60
    # BEM computation
geo_param['lc_m'] = element_size
geo_param['cor_lc'] = Num_utils.corner_mesh_grading_power(element_size,
                                                            power=grading_power) 
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
(eigval_bem,dof_bem, _, _) = compute_eigenvalues_BEM(gmshfile, opts_eigv,
                            qmaxdist=10*element_size, pml_param=alpha)
#%%   # FEM computation
    # Additional parameters for FEM mesh to control outer/inner layers
c = np.sqrt(a**2-b**2)
R_DtN_fem = 1.3*a # radius of DtN boundary
geo_param_fem['a_d'] = R_DtN_fem
geo_param_fem['b_d'] = R_DtN_fem
geo_param_fem['a_o'] = 1.0179*a
geo_param_fem['b_o'] = np.sqrt(geo_param_fem['a_o']**2-c**2)
geo_param_fem['a_i'] = c*np.cosh(2*np.arccosh(a/c) - np.arccosh(geo_param_fem['a_o']/c))
geo_param_fem['b_i'] = np.sqrt(geo_param_fem['a_i']**2-c**2)
geo_param_fem.update({
    'R1_TR': 1e-15, # R_TR/R radius of truncated region,
    'x_ofst_1':-1.40*R_DtN_fem, 'y_ofst_1': np.pi-R_DtN_fem,
    'Nmu': 1, # element in structured layer (1 recommended)
    'Nint': 120, # nodes along sign-changing interface
    'GeomProgint': 1.0, # geometric progression (larger than 1)
    'Nz': 100, # element in z-direction in corner region
    'CharLengthMin_adim': 1, # Characteristic mesh size (ratio)
    'CharLengthMax_adim': 1.5,
    'GenerateQuadMesh': 1
})
gmshfile = Num_utils.build_mesh(geofile_fem,geo_param_fem,gmsh_param_fem,
                                gdim=2)
(eigval_fem,dof_fem) = Num_FEM.compute_eigenvalues_one_corner(gmshfile,
                                        a,b,[phi],[corner_pos],
                                        [geo_param_fem['x_ofst_1']],
                                        [geo_param_fem['y_ofst_1']],
                                        DtN_order=30, quad_order=2,
                                        pml_param=alpha,
                                        lambda_target=[0.2,-0.2],
                                        nev=200, tol=1e-4)
#%%
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.55*paperwidth, 0.25*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
f, ax = plt.subplots(layout='constrained',figsize=figsize)
eigval_ex = Num_utils.compute_exact_spectrum_corner(phi,pml_parameter=alpha)
ax.plot(np.real(eigval_ex),np.imag(eigval_ex),
        label=r'$\sigma_{\text{ess}}(K^\star_\alpha)$', color='C0')
ax.plot([np.abs(np.pi-phi)/(2*np.pi),-np.abs(np.pi-phi)/(2*np.pi)],[0,0],
        linestyle='none',
        label=r'$\pm\,\lambda_c(\phi)$',marker='o',color='C0')
ax.plot(np.real(eigval_fem),np.imag(eigval_fem),color='C2',
            label=f'FEM mixed coords, DoF={dof_fem}',linestyle='none',
            marker='o',fillstyle='none')
ax.plot(np.real(eigval_bem),np.imag(eigval_bem),
            label=f'BEM, $P=${qnumber}, DoF={dof_bem}',linestyle='none',marker='x',
            fillstyle='none',color='C1')
        # Set title in a text box
title = r'$\arg(\alpha)=\pi\,/\,$'+f'{np.pi/np.angle(alpha):.2g}' 
ax.text(0.05,0.9,title,
        transform=ax.transAxes, horizontalalignment='left',
        verticalalignment='top', bbox={'facecolor':'white'})
ax.set_title(f"Eigenvalues, major axis perturbation: "
            +f"$(a,b,\phi)=$({a},{b},{phi/np.pi:.2g}$\pi$), "
            +fr"$\arg(\alpha)=\pi/${np.pi/np.angle(alpha):.2g}.")
ax.set_xlabel('$\Re(\lambda)$')
ax.set_ylabel('$\Im(\lambda)$')
ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.13,0.06])
ax.set_aspect('equal')
ax.grid(which='major')
ax.legend(ncols=2,loc='lower center')
f.savefig(os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-1-Corner-Eigenvalues-FEM-BEM'))
#%% Plot eigenfunctions associated with the first N_track eigenfunctions
N_track = 4 # track the first N_track eigenvalues
alpha = np.exp(1j*np.pi/7)
opts_eigv.use_Arnoldi = False
element_size = a/60
geo_param['lc_m'] = element_size
geo_param['cor_lc'] = Num_utils.corner_mesh_grading_power(element_size,
                                                            power=grading_power) 
gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=1)
(eigvals, dof, eigvecs, sl_pot) = compute_eigenvalues_BEM(gmshfile, opts_eigv,
                                      qmaxdist=10*element_size, pml_param=alpha)
eigvals_f = Num_utils.filter_eigenvalues_ellipse(eigvals,
                                    aspect_ratio=np.min([b/a,a/b]),
                                    N=N_track)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.6*paperwidth, 0.25*paperheight)
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
fig, axs = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
                        layout='constrained', figsize=figsize)
for n in range(0,N_track):
    ax = fig.axes[n]
    eigv_idx = (np.abs(eigvals - eigvals_f[n])).argmin()
    spanx = np.linspace(-1.20*a,1.20*a,num=int(1e2))
    spany = np.linspace(-a,a,num=int(1e2))
    x_, y_ = np.meshgrid(spanx, spany)
    z_ = np.array(sl_pot(eigvecs[:,eigv_idx],x_,y_),dtype=complex)
    z_ = np.real(z_)
    z_ = z_/np.max(np.abs(z_))
    coll = ax.pcolormesh(x_, y_, z_ , shading='gouraud', cmap='RdBu', 
                         rasterized=True, vmin=-1, vmax=1)
    t = np.linspace(cor_jun2_theta,cor_jun1_theta,num=int(1e2))
    ax.plot(a*np.cos(t),b*np.sin(t),color='C0')
    ax.plot([x_c,a*np.cos(cor_jun1_theta)],[y_c,b*np.sin(cor_jun1_theta)],color='C0')
    ax.plot([x_c,a*np.cos(cor_jun2_theta)],[y_c,b*np.sin(cor_jun2_theta)],color='C0')
    R = np.linalg.norm(np.array([x_c,y_c]) - np.array([a*np.cos(cor_jun1_theta),
                                                       b*np.sin(cor_jun1_theta)]))
    patch = mpl.patches.Circle((x_c,y_c),radius=R,
                                facecolor='none', edgecolor='C1',
                                linewidth=mpl.rcParams['lines.linewidth'])
    ax.add_artist(patch)
    ax.text(-a+2*R,y_c, r'$B_\alpha$', color='C1',
                    horizontalalignment='left',verticalalignment='center')
    ax.set_xlabel("$x$")
    if n % 2 ==0:
        ax.set_ylabel("$y$")
    ax.set_title(f"$\lambda^h_{n+1}\simeq${eigvals[eigv_idx]:1.1e}"+
                 fr"$\simeq\lambda^{{\text{{el}}}}_{{{n+1}}}$")
    ax.set_xlim([1.10*x_c,1.10*a])
    ax.set_ylim([-1.10*b,1.10*b])
    ax.set_aspect('equal')
fig.colorbar(coll,ax=axs[:,1],label=r'$\Re(u_n)$')
fig.suptitle(f'BEM eigenfunctions, major axis perturbation: '
             + f'($a$,$b$,$\phi={{{phi/np.pi:.2g}}}\pi$).'
             + '\n'
             + f'$P=${qnumber}, DoF$=${dof}, '
             + fr'$\arg(\alpha)=\pi/{{{np.pi/np.angle(alpha):.2g}}}$.')
fig.savefig(os.path.join(DIR_ARTICLE_IMG,f'Num-Ellipse-1-Corner-major-eigenfunctions'))