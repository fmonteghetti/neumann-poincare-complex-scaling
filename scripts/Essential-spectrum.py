#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Influence of scaling parameter on critical interval and essential spectrum.
"""
#%%
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]/'module'))
from Num_utils import DIR_ARTICLE_IMG, \
                                      MPLSTYLE_ARTICLE, MPLSTYLE_VSCODE

psi = lambda eta,phi: np.sinh(eta*(np.pi-phi))/np.sinh(eta*np.pi)

def crit_interval(eta,phi,alpha=1):
    s = psi(alpha*eta,phi)
    kappa = (s+1)/(s-1)
    return np.concatenate((kappa,1/np.flip(kappa)))

def ess_spectrum(eta,phi,alpha=1):
    s = psi(alpha*eta,phi)/2
    return np.concatenate((s,-np.flip(s)))

def uncovered_regions(eta,phi,alpha):
    # Return boundary of the two uncovered regions (in lambda).
    s1 = psi(alpha*eta,phi)/2
    s2 = -s1
        # first crossing of real axis
    s1_idx_1 = np.argmax(np.imag(s1)>0)
    s2_idx_1 = np.argmax((np.imag(s2)<0))
         # second crossing of real axis
    s1_idx_2= s1_idx_1 + np.argmax(np.imag(s1[s1_idx_1:-1])<0)
    s2_idx_2 = s2_idx_1 + np.argmax((np.imag(s2[s2_idx_1:-1])>0))
        # boundary of uncovered regions
    bnd1 = np.concatenate((s1[0:s1_idx_1],s2[s2_idx_2:s2_idx_1:-1]))
    bnd2 = np.concatenate((s2[0:s2_idx_1],s1[s1_idx_2:s1_idx_1:-1]))
    return (bnd1,bnd2)

angle_l = np.array([0,np.pi/10,np.pi/3.1,np.pi/2.5]) # argument of scaling parameter alpha (rad)
alpha_l = np.exp(1j*angle_l)
phi = np.pi/2 # corner angle (rad)
eta = np.linspace(1e-5,1e2,num=int(1e4))
eta = np.logspace(-2,2,num=int(1e4))

# Plot essential spectrum (lambda)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth, paperheight] = matplotlib.rcParams['figure.figsize']
figsize = (0.65*paperwidth, 0.16*paperheight)
fig, axs = plt.subplots(ncols=np.size(alpha_l),nrows=1,
                        sharex=True,sharey=True,
                        figsize=figsize)
for (i,alpha) in enumerate(alpha_l):
    ax=axs[i]
    (bnd1, bnd2) = uncovered_regions(eta,phi,alpha)
    ax.fill(np.real(bnd1),np.imag(bnd1),color='C1',alpha=0.4)
    ax.fill(np.real(bnd2),np.imag(bnd2),color='C1',alpha=0.4,
            label=r'$U_\alpha$')
    s = ess_spectrum(eta,phi,alpha=alpha)
    ax.plot(np.real(s),np.imag(s),
                    label=r"$\sigma_{\text{ess}}(K^\star_\alpha)$")
    # ax.set_title(r"$\arg(\alpha)=$"+f"{np.rad2deg(np.angle(alpha)):.2g}°")
    if np.isclose(np.imag(alpha),0):
        ax.set_title(r"$\arg(\alpha)=0$")
    else:
        ax.set_title(r"$\arg(\alpha)=\pi/$"+f"{np.pi/np.angle(alpha):.2g}")
        # ax.set_title(r"$\arg(\alpha)=\pi/val$".replace('val',
        #                                                f"{np.pi/np.angle(alpha):.2g}"))
    ax.set_xlabel(r'$\Re(\lambda)$')
    if i==0:
        ax.set_ylabel(r'$\Im(\lambda)$')
        ax.legend(loc='upper center')
        # handles, labels = ax.get_legend_handles_labels() 
        # ax.add_artist(ax.legend(loc='upper left', ncols=1,handles=handles[:1]))
        # ax.add_artist(ax.legend(loc='lower left', ncols=1,handles=handles[-1:]))
    lambda_c = np.abs((np.pi-phi)/(2*np.pi))
    ax.set_xticks([-0.5,-lambda_c,0,lambda_c,0.5])
    ax.set_xticklabels([r'$-\frac{1}{2}$',r'$-\lambda_c(\phi)$',r'$0$',
                        r'$\lambda_c(\phi)$',r'$\frac{1}{2}$'])
    ax.set_yticks([-0.5,-lambda_c,0,lambda_c,0.5])
    ax.set_yticklabels([r'$-\frac{1}{2}$',r'$-\lambda_c(\phi)$',r'$0$',
                        r'$\lambda_c(\phi)$',r'$\frac{1}{2}$'])
    ax.set_ylim(-0.6,0.6)
    ax.grid()
    ax.set_aspect("equal")
fig.suptitle(f'Corner of angle $\phi=\pi/{np.pi/phi:2.2g}$')
fig.savefig(os.path.join(DIR_ARTICLE_IMG,"Scaled-essential-spectrum-alpha"))
#%% Plot critical interval (kappa)
plt.style.use(MPLSTYLE_VSCODE)
[paperwidth, paperheight] = matplotlib.rcParams['figure.figsize']
figsize = (paperwidth, paperheight)
fig, ax = plt.subplots(figsize=figsize)
for alpha in alpha_l:
    s = crit_interval(eta,phi,alpha=alpha)
    ax.plot(np.real(s),np.imag(s),
            label=r"$\arg(\alpha)=$"+f"{np.rad2deg(np.angle(alpha)):.2g}°")
ax.legend(loc='lower left',bbox_to_anchor=(0,1),ncols=len(alpha_l))
# ax.set_aspect("equal")
ax.set_xlabel(r'$\Re(\kappa)$')
ax.set_ylabel(r'$\Im(\kappa)$')
kappa_phi = -(2*np.pi-phi)/phi
if phi<=np.pi:
    ax.set_xticks([kappa_phi,-1,1/kappa_phi,0])
    ax.set_xticklabels([r'$\kappa_{\phi}$',r'$-1$',r'$1/\kappa_\phi$',r'$0$'])
else:
    ax.set_xticks([1/kappa_phi,-1,kappa_phi,0])
    ax.set_xticklabels([r'$1/\kappa_{\phi}$',r'$-1$',r'$\kappa_\phi$',r'$0$'])
ax.grid()
# ax.set_xlim(-1+np.array([-1e-2,1e-2]))
# ax.set_ylim(np.array([-1e-2,1e-2]))