#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot analytical solutions of the Plasmonic Eigenvalue Problem for an Ellipticexact/PEP-Ellipse.pyal
obstacle.
"""
#%% Basic definitions
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

a, b = 2.5, 1 # ellipse semi-axes
nev = 20 # number of eigenvalues
plot_mode_idx = [-5,10] # index of modes to plot (|n| >=1)
plot_eigv_leg = 'none' # eigenvalue plot legend 'none'/'x'/'y'/'xy'
plot_eigv_highlight = False # highlight plotted mode
    # --
(ev_l,ev_k,ev_n,ev_mask) = PEP_ana.ellipse_freespace_eigenvalues(a,b,N=nev)
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
if plot_eigv_leg=='none':
    l = ev_l
    ax.plot(l,0*l,color='C0',marker='o',linestyle='none',fillstyle='none')
elif plot_eigv_leg=='x':
    l, ls = ev_l[ev_mask['x-even']], 'x-even'
    ax.plot(l,0*l,label=ls,color='C0',marker='o',linestyle='none',fillstyle='none')
    l, ls = ev_l[ev_mask['x-odd']], 'x-odd'
    ax.plot(l,0*l,label=ls,color='C1',marker='s',linestyle='none',fillstyle='none')
elif plot_eigv_leg=='y':
    l, ls = ev_l[ev_mask['y-even']], 'y-even'
    ax.plot(l,0*l,label=ls,color='C0',marker='o',linestyle='none',fillstyle='none')
    l, ls = ev_l[ev_mask['y-odd']], 'y-odd'
    ax.plot(l,0*l,label=ls,color='C1',marker='s',linestyle='none',fillstyle='none')
elif plot_eigv_leg=='xy':
    l, ls = ev_l[ev_mask['x-even']*ev_mask['y-even']], 'x-even/y-even'
    ax.plot(l,0*l,label=ls,color='C0',marker='o',linestyle='none',fillstyle='none')
    l, ls = ev_l[ev_mask['x-even']*ev_mask['y-odd']], 'x-even/y-odd'
    ax.plot(l,0*l,label=ls,color='C0',marker='o',linestyle='none')
    l, ls = ev_l[ev_mask['x-odd']*ev_mask['y-odd']], 'x-odd/y-odd'
    ax.plot(l,0*l,label=ls,color='C1',marker='s',linestyle='none',fillstyle='none')
    l, ls = ev_l[ev_mask['x-odd']*ev_mask['y-even']], 'x-odd/y-even'
    ax.plot(l,0*l,label=ls,color='C1',marker='s',linestyle='none')
    # Highlight plotted mode
if plot_eigv_highlight:
    import itertools
    marker = itertools.cycle(('+', 'x', '.', '*')) 
    for idx in plot_mode_idx:
        ax.plot(ev_l[ev_n==idx],0,color='C2',marker=next(marker),linestyle='none',label=r'$n=$'+f'{idx}')
    # Highlight eigenvalue 1 and -1
leg_ymargin = -2e-1
ax.text(l[ev_n==1],leg_ymargin, r'$\lambda^{\text{el}}_1$', horizontalalignment='center',verticalalignment='top')
ax.text(l[ev_n==2],leg_ymargin, r'$\lambda^{\text{el}}_2$', horizontalalignment='center',verticalalignment='top')
ax.text(l[ev_n==-1],leg_ymargin, r'$\lambda^{\text{el}}_{-1}$', horizontalalignment='center',verticalalignment='top')
ax.text(l[ev_n==-2],leg_ymargin, r'$\lambda^{\text{el}}_{-2}$', horizontalalignment='center',verticalalignment='top')
ax.set_xlim([-0.3,0.3])
ax.set_ylim([-1,1])
ax.grid()
# ax.legend(ncols=3)
ax.set_xlabel(r'$\Re(\lambda)$')
ax.set_ylabel(r'$\Im(\lambda)$')
ax.set_title(r'Eigenvalues, ellipse with $(a,b)=$'+f'({a},{b})')
    # Plot mode
for (i,idx) in enumerate(plot_mode_idx):
    ax=axs[1+i]
    u = PEP_ana.ellipse_freespace_eigenfunction(a,b,n=idx)
    d_x, d_y = 1.15*a, 1.65*b
    spanx = np.linspace(-d_x,d_x,num=int(1e2))
    spany = np.linspace(-d_y,d_y,num=int(1e2))
    x_, y_ = np.meshgrid(spanx, spany)
    z_ = u(x_,y_)
    z_ = z_/np.max(np.abs(z_))
    coll = ax.pcolormesh(x_, y_, z_ , shading='gouraud', cmap='RdBu',
                         rasterized=True,vmin=-1,vmax=1)
    if i==1:
        fig.colorbar(coll,ax=ax,ticks=[-1,0,1],label=r'$u_n$')
        # Ellipse
    t = np.linspace(-np.pi,np.pi,100)
    ax.plot(a*np.cos(t),b*np.sin(t),linestyle='--',color='k')
    ax.set_xlabel(r"$x$")
    if i==0:
        ax.set_ylabel(r"$y$")
    else:
        ax.set_yticklabels([])
    ax.set_title(r"$\lambda^{\text{el}}_{val}=$".replace("val",f"{idx}")+f"{ev_l[ev_n==idx][0]:1.1e}")
    ax.set_xlim([-d_x,d_x])
    ax.set_aspect('equal')
fig.savefig(os.path.join(DIR_ARTICLE_IMG,"PEP-Ellipse"))