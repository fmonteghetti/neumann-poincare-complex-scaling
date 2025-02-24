#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex-scaling path in Euler and Cartesian coordinates.
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

def gamma(z,alpha=complex(1.0),z_alpha=0.0):
    """Complex-scaling path in Euler coordinates."""
    s = np.zeros_like(z,dtype=complex)
    mask = z>z_alpha
    s[mask] = z[mask]
    mask = ~mask
    s[mask] = (z[mask]-z_alpha)/alpha + z_alpha
    return s

def Phi(s,alpha=1,log=False):
    """Intermediate function Phi(s,alpha).

    Parameters
    ----------
    s : nd.array
    alpha : nd.array, optional
        Complex-scaling parameter, by default 1
    log : bool, optional
        Whether to return ln(Phi(s,alpha)), by default False
    """    
    mask = s>=1
    z = np.zeros_like(s,dtype=complex)
    if log==False:
        z[mask] = s[mask]
        mask = ~mask
        z[mask] = s[mask]**(1/alpha)
    else:
        z[mask] = np.log(s[mask])
        mask = ~mask
        z[mask] = (1/alpha)*np.log(s[mask])
    return z

def tau_1(r,theta,alpha=1,R_alpha=1,xc=0):
    """x-coordinate of the complex-scaled path Gamma_alpha(r,theta)."""
    return xc + R_alpha * Phi(r/R_alpha,alpha=alpha) * np.cos(theta)
def tau_2(r,theta,alpha=1,R_alpha=1,yc=0):
    """ y-coordinate of the complex-scaled path Gamma_alpha(r,theta)."""
    return yc + R_alpha * Phi(r/R_alpha,alpha=alpha) * np.sin(theta)
def arg_tau_1(r,theta,alpha=1,R_alpha=1):
    """ Argument of tau_1(x) - x_c. """
    mask = r <= R_alpha
    s = np.zeros_like(r,dtype=complex)
    s[mask] = np.imag(1/alpha) * np.log(r[mask]/R_alpha)
    mask = ~mask
    s[mask] = 0
    mask = np.abs(theta)>np.pi/2
    s[mask] += np.pi
    return s

#%% Path in Euler coordinates

R = 1
R_alpha = 0.5 # radius of scaling region
alpha = np.exp(1j*np.pi/5)
z_alpha = np.log(R_alpha/R)
z_plot = np.linspace(-6,1,int(1e2))

plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = matplotlib.rcParams['figure.figsize']
figsize = (0.3*paperwidth, 0.3*paperheight)
f, ax = plt.subplots(layout='constrained',figsize=figsize)
ax.axvline(x=z_alpha,linestyle='--',color='C0')
ax.axhline(y=0,linestyle='--',color='C1',label=r'$\alpha=1$')
s = gamma(z_plot,alpha=alpha,z_alpha=z_alpha)
ax.plot(np.real(s), np.imag(s),color='C2',label=r'$\alpha\neq1, \arg(\alpha)>0$')
R_angle = 2
ax.add_patch(matplotlib.patches.Arc([z_alpha,0],2*R_angle,2*R_angle,
                                    theta1=180-np.rad2deg(np.angle(alpha)),
                                    theta2=180,linestyle='dotted'))
R_text = 1.10*R_angle
theta_text = np.pi-np.angle(alpha)/2
x_text = [z_alpha+R_text*np.cos(theta_text),R_text*np.sin(theta_text)]
# ax.plot(x_text[0],x_text[1],marker='o')
ax.text(x_text[0], x_text[1], r'$\arg(\alpha)$',horizontalalignment='right')

ax.set_xlabel(r'$\Re(\gamma_{\alpha}(z))$')
ax.set_ylabel(r'$\Im(\gamma_{\alpha}(z))$')
ax.set_xticks([z_alpha,0])
ax.set_xticklabels([r'$z_\alpha$','0'])
ax.set_yticks([0])
ax.set_yticklabels(['0'])
#ax.grid(True)
ax.set_aspect('equal') # orthonormal axis
ax.set_xlim([np.min(np.real(s)),np.max(np.real(s))])
ax.legend()
f.savefig("Complex-scaling-path-Euler-coord.pdf")

#%% Path in Cartesian coordinates
R = 1.5
R_alpha = 1 # radius of scaling region
x_r = 10*R # ||x-x_c||
x_theta = 0*np.pi/5 # angle of x
alpha = np.exp(1j*np.pi/3)
r = np.logspace(-8,np.log(x_r),num=int(1e4))
x_c = [0,0]

plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = matplotlib.rcParams['figure.figsize']
figsize = (0.65*paperwidth, 0.1*paperheight)
fig, axs = plt.subplots(ncols=2,nrows=1,layout='constrained',
                        figsize=figsize,
                        gridspec_kw=dict(width_ratios=[1.0, 2.5]))
ax = axs[0]
t = np.linspace(0,2*np.pi)
ax.plot(x_c[0]+R_alpha*np.cos(t),x_c[1]+R_alpha*np.sin(t),color='k',
        linestyle='--',label=r'$|\gamma_1-x_c|=R_\alpha$')
ax.plot(x_c[0],x_c[1],marker='.',color='k',label=r'$x_c$',linestyle='none')
ax.plot(x_c[0]+x_r*np.cos(x_theta),0,marker='o',color='k')
ax.plot(x_c[0]+np.max(R))
s = tau_1(r,x_theta,alpha=alpha,R_alpha=R_alpha,xc=x_c[0])
ax.plot(np.real(s),np.imag(s),color='C1',label=r'$\alpha=e^{i\pi/val}$'.replace('val',f'{np.pi/np.angle(alpha):2.2g}'))
s = tau_1(r,x_theta,alpha=1.0,R_alpha=R_alpha,xc=x_c[0])
ax.plot(np.real(s),np.imag(s),label=r'$\alpha=1$',linestyle=':',color='C0')
ax.set_xlabel(r'$\Re(\gamma_1)$')
ax.set_ylabel(r'$\Im(\gamma_1)$')
ax.set_xlim(R*np.array([-1,1]))
ax.set_ylim(R*np.array([-1,1]))
ax.set_xticks(np.array([-R_alpha,0,R_alpha]))
ax.set_xticklabels([r'$x_c-R_\alpha$',r'$x_c$',r'$x_c+R_\alpha$'])
ax.set_yticks(np.array([-R_alpha,0,R_alpha]))
ax.set_yticklabels([r'$-R_\alpha$',r'$0$',r'$R_\alpha$'])
ax.grid(True)
ax.set_aspect('equal') # orthonormal axis
ax = axs[1]
r = np.logspace(-8,np.log(R),num=int(1e4))
# ax.axvline(np.log(R_alpha/R),color='k')
s = tau_1(r,x_theta,alpha=alpha,R_alpha=R_alpha,xc=x_c[0])
ax.plot(np.log(r/R_alpha),np.angle(s),color='C1')
s = tau_1(r,x_theta,alpha=1.0,R_alpha=R_alpha,xc=x_c[0])
ax.plot(np.log(r/R_alpha),np.angle(s),linestyle=':',color='C0')
ax.set_xlabel(r'$\ln(r/R_{\alpha})$')
ax.set_xlim([np.min(np.log(r/R_alpha)),np.max(np.log(r/R_alpha))])
ax.set_ylabel(r'$\arg(\gamma_1-x_c)$')
ax.set_ylim([-np.pi,np.pi])
ax.set_yticks(-np.pi*np.array([-1,-1/2,0,1/2,1]))
ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$'])
# ax.set_xlim([np.min(r),np.max(r)])
ax.grid(which='both')
handles, labels = axs[0].get_legend_handles_labels() 
fig.legend(loc='outside right upper',ncols=1,handles=handles)
fig.savefig(os.path.join(DIR_ARTICLE_IMG,"Complex-scaling-path-Cart-coord"))