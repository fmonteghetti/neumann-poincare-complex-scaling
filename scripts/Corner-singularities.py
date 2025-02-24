#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot strongly-oscillating singularities in Cartesian and Euler coordinates.
"""
#%%
import os
import sys
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]/'module'))
from Num_utils import DIR_ARTICLE_IMG, \
                                      MPLSTYLE_ARTICLE, MPLSTYLE_VSCODE

# Corner geometry
phi = np.deg2rad(90) # corner angle
R = 1.0 # corner radius
R_alpha = R/2 # scaling radius
R_min, R_max = 0.2*R_alpha, 1.1*R # Euler coordinates plot

# Corner singularity
eta = 10.0 # radial exponent
alpha = np.exp(0j*pi/4) # scaling parameter

# Plot resolutions
nx = 3e2  # Cartesian plot
nz, ntheta = 1e2, 2e2 # Euler plot

def singularity(r,theta,phi=np.pi/2,eta=1.0):
    """ Symmetric corner singularity, as given in (22) of "On the use of PML at
    corners for scattering problems with sign-changing coefficients".
    """
    u = r**(1j*eta)
    mask = np.abs(theta)<phi/2 # inside negative particle
    u[mask] *= np.cosh(eta*theta[mask])/np.cosh(eta*phi/2)
    mask = theta>=phi/2
    u[mask] *= np.cosh(eta*(np.pi-theta[mask]))/np.cosh(eta*(np.pi-phi/2))
    mask = theta<=-phi/2
    u[mask] *= np.cosh(eta*(np.pi+theta[mask]))/np.cosh(eta*(np.pi-phi/2))
    return u
u_an = lambda r,theta, eta: singularity(r,theta,phi=phi,eta=eta)
u_an_Euler = lambda z,theta, eta: singularity(R*np.exp(z),theta,phi=phi,eta=eta)

def singularity_scaled(r,theta,eta=1.0,alpha=1.0,R_alpha=1.0):
    u = r**(1j*eta)
    mask = r<R_alpha
    u[mask] = R_alpha**(1j*eta) * (r[mask]/R_alpha)**(1j*eta/alpha)
    return u

# Singularity in Cartesian coordinates
span = np.linspace(-R,R,num=int(nx))
x1_, x2_ = np.meshgrid(span, span)
r = np.sqrt(x1_**2+x2_**2)
theta = np.angle(x1_+1j*x2_)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.3*paperwidth, 0.16*paperheight)
fig, ax = plt.subplots(layout='constrained',figsize=figsize)
    # corner singularity
Z = u_an(r,theta,eta/alpha)
Z = np.ma.array(Z,mask=np.isnan(Z))
Z=np.real(Z)/np.max(np.abs(np.real(Z)))
coll = ax.pcolormesh(x1_, x2_, Z , shading='gouraud', cmap='RdBu',
                     rasterized=True,vmin=-1,vmax=1)
circ = mpl.patches.Wedge((0,0),R,theta1=0,theta2=360,transform=ax.transData)
coll.set_clip_path(circ)
# ax.axis('on')
ax.set_xlim([-R,R])
ax.set_ylim([-R,R])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_xticks([-R,-R_alpha,0,R_alpha,R])
ax.set_xticklabels([r'$-R$',r'$-R_\alpha$','0',r'$R_\alpha$',r'$R$'])
ax.set_yticks([-R,-R_alpha,0,R_alpha,R])
ax.set_yticklabels([r'$-R$',r'$-R_\alpha$','0',r'$R_\alpha$',r'$R$'])
fig.colorbar(coll,ax=ax,ticks=[-1,0,1],label=r'$\Re(\mathfrak{s})$')
    # sign-changing interface
ax.set_aspect('equal') # orthonormal axis
pa = mpl.lines.Line2D([R*np.cos(-phi/2),0,R*np.cos(phi/2)],
                      [R*np.sin(-phi/2),0,R*np.sin(phi/2)],
                      color='k')
ax.add_artist(pa)
ax.text(R*np.cos(phi/2),R*np.sin(phi/2), r'$\Gamma$',
                    horizontalalignment='left',verticalalignment='bottom')
ax.text(0.9*R, 0, r'$\Omega$',
                    horizontalalignment='right')
ax.text(0,0.9*R, r'$\Omega^c$',
                    verticalalignment='top')
    # Normal vector
nv_rpos, nv_length = 0.8*R, 0.25*R
pa = mpl.patches.Arrow(nv_rpos*np.cos(phi/2),nv_rpos*np.sin(phi/2),
                       nv_length*(-np.sin(phi/2)),
                       nv_length*(np.cos(phi/2)),
                       width=0.1,
                       facecolor='k')
ax.add_patch(pa)
ax.text(nv_rpos*np.cos(phi/2)+nv_length*(-np.sin(phi/2)),
        nv_rpos*np.sin(phi/2)+nv_length*(np.cos(phi/2)), 
        r'$\nu$',
        horizontalalignment='right')
    # Corner region {r=R}
pa = mpl.patches.Circle((0,0),R,
                        edgecolor='k',
                        facecolor='none',
                        linestyle='dotted',
                        linewidth=mpl.rcParams['lines.linewidth'])
ax.add_patch(pa)
ax.text(-R/np.sqrt(2),R/np.sqrt(2), r'$r=R $',
                    horizontalalignment='center')
    # Scaling region {r=R_alpha}
pa = mpl.patches.Circle((0,0),R_alpha,
                        edgecolor='k',
                        facecolor='none',
                        linestyle='--',
                        linewidth=mpl.rcParams['lines.linewidth'])
ax.add_patch(pa)
ax.text(-R_alpha/np.sqrt(2),R_alpha/np.sqrt(2), r'$r=R_\alpha$',
                    horizontalalignment='center')
    # corner
ax.plot(0,0,marker='o',color='k')
ax.text(0, -0.05*R, r'$x_c$',horizontalalignment='center',verticalalignment='top')
    # Polar basis
pb_angle = -phi/4
pb_rpos = 0.7*nv_rpos
pa = mpl.patches.Arrow(pb_rpos*np.cos(pb_angle),pb_rpos*np.sin(pb_angle),
                       nv_length*(np.cos(pb_angle)),
                       nv_length*(np.sin(pb_angle)),
                       width=0.1,
                       facecolor='k')
ax.add_patch(pa)
ax.text(pb_rpos*np.cos(pb_angle)+nv_length*(np.cos(pb_angle)),
        pb_rpos*np.sin(pb_angle)+nv_length*(np.sin(pb_angle)), 
        r'$\hat{e}_r$',
        verticalalignment='top')
pa = mpl.patches.Arrow(pb_rpos*np.cos(pb_angle),pb_rpos*np.sin(pb_angle),
                       nv_length*(-np.sin(pb_angle)),
                       nv_length*(np.cos(pb_angle)),
                       width=0.1,
                       facecolor='k')
ax.add_patch(pa)
ax.text(pb_rpos*np.cos(pb_angle)+nv_length*(-np.sin(pb_angle)),
        pb_rpos*np.sin(pb_angle)+nv_length*(np.cos(pb_angle)), 
        r'$\hat{e}_\theta$',
        horizontalalignment='right', verticalalignment='bottom')
    # current point
# ax.plot([0,pb_rpos*np.cos(pb_angle)],[0,pb_rpos*np.sin(pb_angle)],
#                                                         color='k',
#                                                         linestyle='dashdot')
ax.plot(pb_rpos*np.cos(pb_angle),pb_rpos*np.sin(pb_angle),marker='o',color='k')
ax.text(pb_rpos*np.cos(pb_angle),pb_rpos*np.sin(pb_angle)-R/50, r'$x$',
                        horizontalalignment='right', verticalalignment='top')
fig.savefig(os.path.join(DIR_ARTICLE_IMG,"Corner-singularity-Cartesian"))
#%% Singularity in Euler coordinates
z_min, z_alpha, z_max = np.log(R_min/R), np.log(R_alpha/R), np.log(R_max/R)
z_, theta_ = np.meshgrid(np.linspace(z_min,z_max,num=int(nz)), 
                       np.linspace(-np.pi,np.pi,num=int(ntheta)))
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.31*paperwidth, 0.16*paperheight)
fig, ax = plt.subplots(layout='constrained',figsize=figsize)
Z = u_an_Euler(z_,theta_,eta/alpha)
Z = np.ma.array(Z,mask=np.isnan(Z))
Z=np.real(Z)/np.max(np.abs(np.real(Z)))
coll = ax.pcolormesh(z_, theta_, Z , shading='gouraud', cmap='RdBu',
                     rasterized=True,vmin=-1,vmax=1)
fig.colorbar(coll,ax=ax,ticks=[-1,0,1],label=r'$\Re(\mathfrak{s})$')
ax.set_yticks(np.array([-1*np.pi,-1/2*np.pi,-phi/2,0,phi/2,1/2*np.pi,1*np.pi]))
ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$-\phi/2$',r'$0$',r'$\phi/2$',r'$\pi/2$',r'$\pi$'])
ax.set_ylabel(r'$\theta$')
ax.set_xticks([np.log(R_alpha/R),0])
ax.set_xticklabels([r'$z_\alpha=\ln(R_\alpha/R)$',r'$0$'])
ax.set_xlabel(r'$z=\ln(r/R)$')
    # sign-changing interface
ax.axhline(phi/2,color='k')
ax.axhline(-phi/2,color='k')
ax.text(0.9*z_alpha,3*np.pi/4, r'$\Psi(\Omega^c)$',
                    verticalalignment='bottom')
ax.text(0.9*z_alpha,0, r'$\Psi(\Omega)$',
                    horizontalalignment='left')
ax.annotate(r'$\Psi(\Gamma)$', 
                           xy=(1.2*np.mean([z_min,z_alpha]), -phi/2), 
                           xytext=(1.0*np.mean([z_min,z_alpha]),0),
                           color='w',ha='center',va='center',
                           arrowprops=dict(arrowstyle="->",
                                           facecolor='black'))
ax.annotate(r'$\Psi(\Gamma)$', 
                           xy=(1.2*np.mean([z_min,z_alpha]), phi/2), 
                           xytext=(1.0*np.mean([z_min,z_alpha]),0),
                           ha='center',va='center',
                           arrowprops=dict(arrowstyle="->",
                                           facecolor='black'))

    # Normal vector
nv_zpos, nv_length = np.log(nv_rpos/R), np.pi/4
pa = mpl.patches.Arrow(nv_zpos,phi/2,
                       0,
                       nv_length,
                       width=0.1,
                       facecolor='k')
ax.add_patch(pa)
ax.text(nv_zpos,
        phi/2+nv_length, 
        r'$\nu$',
        verticalalignment='bottom')
    # Corner region
pa = mpl.lines.Line2D([0,0],
                      [-np.pi,np.pi],
                      color='k',
                      linestyle='dotted')
ax.add_artist(pa)
    # Scaling region
pa = mpl.lines.Line2D([np.log(R_alpha/R),np.log(R_alpha/R)],
                      [-np.pi,np.pi],
                      color='k',
                      linestyle='--')
ax.add_artist(pa)
    # Polar basis
pb_rpos_z = np.log(pb_rpos/R)
# pa = mpl.patches.Arrow(pb_rpos_z,pb_angle,
#                        0.5*nv_length*(1.0),
#                        nv_length*(0.0),
#                        width=0.1,
#                        facecolor='k')
# ax.add_patch(pa)
# ax.text(pb_rpos_z+nv_length*(1.0),
#         pb_angle+nv_length*(0), 
#         r'$\hat{e}_r$',
#         verticalalignment='top')
# pa = mpl.patches.Arrow(pb_rpos_z,pb_angle,
#                        nv_length*(0),
#                        nv_length*(1),
#                        width=0.1,
#                        facecolor='k')
# ax.add_patch(pa)
# ax.text(pb_rpos_z+nv_length*(0),
#         pb_angle+nv_length*(1), 
#         r'$\hat{e}_\theta$',
#         horizontalalignment='right')
    # current point
ax.plot(pb_rpos_z,pb_angle,marker='o',color='k')
ax.text(pb_rpos_z+R/50,pb_angle, r'$x$',
                        horizontalalignment='left', verticalalignment='top')

fig.savefig(os.path.join(DIR_ARTICLE_IMG,"Corner-singularity-Euler"))


#%% Plot along a radius in both Polar and Euler coordinates
alpha = np.exp(1j*pi/3) # scaling parameter
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.64*paperwidth, 0.1*paperheight)
fig, ax = plt.subplots(ncols=2,nrows=1,sharey=True,layout='constrained',
                       figsize=figsize)
r = np.logspace(-5,np.log10(R),num=5*int(nx))
# Polar
ax[0].axvline(R_alpha,linestyle='--',color='k')
Z = np.real(singularity_scaled(r,phi/2,eta=eta,alpha=1.0,R_alpha=R_alpha))
Z = Z/np.max(np.abs(Z))
ax[0].plot(r,Z,linestyle=':',label=r'$\alpha=1$')
Z = np.real(singularity_scaled(r,phi/2,eta=eta,alpha=alpha,R_alpha=R_alpha))
Z = Z/np.max(np.abs(Z))
ax[0].plot(r,Z,label=r'$\alpha=\exp(i\pi/val)$'.replace('val',
                                                       f"{np.pi/np.angle(alpha):2.2g}"))
ax[0].set_xticks([0,R_alpha,R])
ax[0].set_xticklabels(['0',r'$R_\alpha$',r'$R$'])
ax[0].set_yticks([-1,0,1])
ax[0].set_xlabel(r'$r$')
ax[0].set_ylabel(r'$\Re(\mathfrak{s})$')
ax[0].set_xlim(left=0,right=1)
ax[0].set_ylim(bottom=-1,top=1)
fig.legend(loc='outside upper center',ncols=2)
# Euler
r = np.logspace(np.log10(R_alpha/10),np.log10(R),num=5*int(nx))
z = np.log(r/R)
ax[1].axvline(z_alpha,linestyle='--',color='k')
Z = np.real(singularity_scaled(r,phi/2,eta=eta,alpha=1.0,R_alpha=R_alpha))
Z = Z/np.max(np.abs(Z))
ax[1].plot(z,Z,linestyle=':',label=r'$\arg(\alpha)=0$')
Z = np.real(singularity_scaled(r,phi/2,eta=eta,alpha=alpha,R_alpha=R_alpha))
Z = Z/np.max(np.abs(Z))
ax[1].plot(z,Z,label=r'$\arg(\alpha)=$'+f'{np.rad2deg(np.angle(alpha))}°')
ax[1].set_xticks([np.log(R_alpha/R),0])
ax[1].set_xticklabels([r'$z_\alpha$',r'$0$'])
ax[1].set_xlabel(r'$z=\ln(r/R)$')
ax[1].set_xlim([np.min(z),np.max(z)])
fig.savefig(os.path.join(DIR_ARTICLE_IMG,"Corner-singularity-1D"))