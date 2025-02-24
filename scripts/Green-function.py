""" 
Visualization of the complex-scaled Green's function.

This script illustrates the multivaluation of the naive expression of the
complex-scaled Green's function.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]/'module'))
from Num_utils import DIR_ARTICLE_IMG, MPLSTYLE_ARTICLE, MPLSTYLE_VSCODE
MPLSTYLE_BEAMER = [str(Path(__file__).parents[0]/'base.mplstyle'),
                   str(Path(__file__).parents[0]/'beamer.mplstyle')]
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

phi = np.deg2rad(90) # corner angle
R = 1
R_alpha = 0.8
y = 0.12 * R_alpha * np.exp(1j*np.pi/4)
y = [np.real(y),np.imag(y)]
alpha = np.exp(1j*np.pi/4)

def scaling_fun(r,alpha,R_alpha=0.5):
    """ Scaling parameter, piecewise-constant. """
    fun = np.zeros_like(r,dtype=complex)
    mask = r<R_alpha
    fun[mask] = alpha
    fun[~mask] = 1
    return fun

def gamma_alpha(x1,x2,alpha,R_alpha=0.5):
    """ Complex-scaling path [gamma_1,gamma_2]"""
    r = np.sqrt(x1**2 + x2**2)
    scaling_param = scaling_fun(r,alpha,R_alpha=R_alpha)
    val = R_alpha * np.exp((1/scaling_param)*np.log(r/R_alpha))
    return [val * x1/r, val * x2/r]

def q_alpha(x1,x2,y1,y2,alpha,R_alpha=0.5):
    """ Complex-scaled quadratic form q_alpha(x,y)."""
    (gamma_x_1, gamma_x_2) = gamma_alpha(x1,x2,alpha,R_alpha=R_alpha)
    (gamma_y_1, gamma_y_2) = gamma_alpha(y1,y2,alpha,R_alpha=R_alpha)
    return (gamma_x_1-gamma_y_1)**2 + (gamma_x_2-gamma_y_2)**2 

def Green_naive(x1,x2,y1,y2,alpha,R_alpha=0.5):
    """ Naive expression of the Green function."""
    q = q_alpha(x1,x2,y1,y2,alpha,R_alpha=R_alpha)
    return (-1/(4*np.pi)) * np.log(q) 

def q_alpha_hat(x1,x2,y1,y2,alpha,R_alpha=0.5):
    """ Complex-scaled quadratic form q_alpha(x,y) with factorization."""
    q = q_alpha(x1,x2,y1,y2,alpha,R_alpha=R_alpha)
    rx = np.sqrt(x1**2+x2**2)
    scaling_param = scaling_fun(rx,alpha,R_alpha=R_alpha)
    q = q / (np.exp((2/scaling_param) * np.log(rx/R_alpha)))
    return q 

def Green_function(x1,x2,y1,y2,alpha,R_alpha=0.5):
    """ Actual Green function."""
    rx = np.sqrt(x1**2+x2**2)
    ry = np.sqrt(y1**2+y2**2)
    val = np.zeros_like(rx,dtype=complex)
    # x and y are not both in the scaling region 
    mask1 = (rx>=R_alpha) + (ry>=R_alpha)
    q = q_alpha(x1,x2,y1,y2,alpha,R_alpha=R_alpha)
    val[mask1] = np.log(q)[mask1]
    # x and y are both in the scaling region
    mask2 = (ry<=rx) * (~mask1)
    qhat = q_alpha_hat(x1,x2,y1,y2,alpha,R_alpha=R_alpha)
    tmp = (2/alpha) * np.log(rx/R_alpha) + np.log(qhat)
    val[mask2] = tmp[mask2]  
    mask3 = (rx<ry) * (~mask1)
    qhat = q_alpha_hat(y1,y2,x1,x2,alpha,R_alpha=R_alpha)
    tmp = (2/alpha) * np.log(ry/R_alpha) + np.log(qhat)
    val[mask3] = tmp[mask3] 
    return (-1/(4*np.pi)) * val
def func(x,y): return x[0]

def plot_modulus_and_argument(fun2plot,cmap_arg,name):
    """ Plot modulus and argument of f(.,y). """
    plt.style.use(MPLSTYLE_VSCODE)
    figsize = mpl.rcParams['figure.figsize']
    plt.style.use(MPLSTYLE_ARTICLE)
    [paperwidth,paperheight] = mpl.rcParams['figure.figsize']
    figsize = (0.62*paperwidth, 0.25*paperheight)
    fig, axs = plt.subplots(nrows=2,ncols=2,layout='constrained',figsize=figsize)
    # Zoomed-out plot
    ax=axs[0,0]
    nx = 1e3  # Cartesian plot
    span = np.linspace(-1.5*R_alpha,1.5*R_alpha,num=int(nx))
    x1, x2 = np.meshgrid(span, span)
    r = np.sqrt(x1**2+x2**2)
    theta = np.angle(x1+1j*x2)
    Z = fun2plot(x1,x2,y[0],y[1],alpha,R_alpha=R_alpha)
    Z = np.angle(Z)
    coll = ax.pcolormesh(x1, x2, Z , shading='auto', cmap=cmap_arg,
                        rasterized=True,vmin=-np.pi,vmax=np.pi)
    # place colorbar in inset axis
    axins = ax.inset_axes([1.1, 0, 0.1, 1])
    cbar = fig.colorbar(coll,cax=axins,label='Argument')
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    # y-value
    ax.plot(y[0],y[1], marker='o',color='C2',fillstyle='none',linestyle='none')
    ax.text(2.5*y[0],y[1], r'$\boldsymbol{y}$',
                        color='C2',
                        horizontalalignment='left',verticalalignment='bottom')
    # Scaling region {r=R_alpha}
    pa = mpl.patches.Circle((0,0),R_alpha,
                            edgecolor='C1',
                            facecolor='none',
                            alpha=1,
                            linestyle='--',
                            fill=True,
                            linewidth=mpl.rcParams['lines.linewidth'])
    ax.add_patch(pa)
    ax.text(-R_alpha/np.sqrt(2),R_alpha/np.sqrt(2), r'$B_\alpha$',
                        color='C1',
                        horizontalalignment='left',verticalalignment='top')
    ax.set_xlim([-R,R])
    ax.set_ylim([-R,R])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(name+r'$(\cdot,\boldsymbol{y})$')
    ax.set_xticks([-R_alpha,0,R_alpha])
    ax.set_xticklabels([r'$-R_\alpha$','0',r'$R_\alpha$'])
    ax.set_yticks([-R_alpha,0,R_alpha])
    ax.set_yticklabels([r'$-R_\alpha$','0',r'$R_\alpha$'])
    ax.set_aspect('equal') # orthonormal axis
    # ax.legend(loc='upper right')
    # Zoom around xc
    ax = axs[0,1]
    nx = 1e3  # Cartesian plot
    R_zoom = 0.18 * R_alpha
    x1, x2 = np.meshgrid(np.linspace(-R_zoom,R_zoom,num=int(nx)),
                        np.linspace(-R_zoom,R_zoom,num=int(nx)))
    r = np.sqrt(x1**2+x2**2)
    theta = np.angle(x1+1j*x2)
    Z = fun2plot(x1,x2,y[0],y[1],alpha,R_alpha=R_alpha)
    Z = np.angle(Z)
    coll = ax.pcolormesh(x1, x2, Z , shading='auto', cmap=cmap_arg,
                        rasterized=True,vmin=-np.pi,vmax=np.pi)
    # place colorbar in inset axis
    axins = ax.inset_axes([1.1, 0, 0.1, 1])
    cbar = fig.colorbar(coll,cax=axins,label='Argument')
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    # line between corner and y
    line_fun = lambda s: s * np.array(y)
    ax.plot([line_fun(-1)[0],line_fun(1)[0]],
            [line_fun(-1)[0],line_fun(1)[1]],
            linestyle='--',
            color='C0')
    ax.text(line_fun(-0.7)[0], line_fun(-0.7)[1], r'$\mathcal{C}$',
                            verticalalignment='top', horizontalalignment='left')
    # markers along the line
    markers_pos = [1,1e-10,-1]
    markers_sym = ['o', 's', 'v']
    for (s,sym) in zip(markers_pos, markers_sym):
        ax.plot(line_fun(s)[0],line_fun(s)[1],marker=sym,color='C2',fillstyle='none')
    ax.text(1.3*y[0],y[1], r'$\boldsymbol{y}$',
                        color='C2',
                        horizontalalignment='left',verticalalignment='bottom')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(name + r'$(\cdot,\boldsymbol{y})$ (zoom)')
    ax.set_aspect('equal') # orthonormal axis
    # Modulus along (xc,y) line
    ax = axs[1,0]
    s = np.linspace(-1,1,num=int(1e2))
    z = fun2plot(s * y[0],s * y[1],y[0],y[1],alpha,R_alpha=R_alpha)
    ax.plot(s, np.abs(z))
    for (s,sym) in zip(markers_pos, markers_sym):
        marker_z = fun2plot(line_fun(s)[0],line_fun(s)[1],y[0],y[1],alpha,R_alpha=R_alpha)
        ax.plot(s,np.abs(marker_z),marker=sym,color='C2',fillstyle='none')
    ax.set_title(name+r'$(\cdot,\boldsymbol{y})$ along $\mathcal{C}$')
    ax.set_xlabel('Abscissa s')
    ax.set_ylabel('Modulus')
    ax.grid()
    # Argument along (xc, y) line
    ax = axs[1,1]
    s = np.linspace(-1,1,num=int(1e2))
    z = fun2plot(s * y[0],s * y[1],y[0],y[1],alpha,R_alpha=R_alpha)
    ax.plot(s, np.angle(z))
    for (s,sym) in zip(markers_pos, markers_sym):
        marker_z = fun2plot(line_fun(s)[0],line_fun(s)[1],y[0],y[1],alpha,R_alpha=R_alpha)
        ax.plot(s,np.angle(marker_z),marker=sym,color='C2',fillstyle='none')
    ax.set_title(name+r'$(\cdot,\boldsymbol{y})$ along $\mathcal{C}$')
    ax.set_xlabel('Abscissa s')
    ax.set_ylabel('Argument')
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    ax.grid()
    fig.suptitle(fr"Plot of $q_\alpha(\cdot,\boldsymbol{{y}})$ in scaling region $B_\alpha$. "
                 +fr"$\alpha=\exp(i\pi/{{{np.pi/np.angle(alpha):g}}})$"
                 +fr", $R_\alpha={{{R_alpha}}}$"
                 +fr", $|\boldsymbol{{y}}|={{{np.linalg.norm(y)/R_alpha}}}R_\alpha$.")
    # fig.suptitle(r'$x_c=(0,0)$, , '.replace('val',f'{}')+\
                # r'$|y|=val\,R_\alpha$'.replace('val',f'{np.linalg.norm(y)/R_alpha}')
                # +r', $R_\alpha=val$'.replace('val',f'{R_alpha}'))
    return fig
#%% Complex-scaled quadratic form
fun2plot = q_alpha
cmap_arg = 'twilight'
name = r'$q_\alpha$'
fig = plot_modulus_and_argument(fun2plot,cmap_arg,name)
fig.savefig(os.path.join(DIR_ARTICLE_IMG,'Complex-plot-scaled-argument'))
#%% Naive Green function
fun2plot = Green_naive
cmap_arg = 'twilight_shifted'
name = r'$\mathfrak{G}_\alpha$'
fig = plot_modulus_and_argument(fun2plot,cmap_arg,name)
#%% Actual Green function
fun2plot = Green_function
cmap_arg = 'twilight_shifted'
name = r'$G_\alpha$'
fig = plot_modulus_and_argument(fun2plot,cmap_arg,name)
#%% Comparison of the argument of naive and actual Green function
s = np.linspace(-1,1-1e-16,num=int(1e2))
line_fun = lambda s: s * np.array(y)
markers_pos = [s[-1],1e-10,s[0]]
markers_sym = ['o', 's', 'v']
# plt.style.use(MPLSTYLE_VSCODE)
# figsize = mpl.rcParams['figure.figsize']
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.62*paperwidth, 0.15*paperheight)
fig, axs = plt.subplots(ncols=2,layout="constrained",figsize=figsize)
# Modulus along (xc,y) line
ax = axs[0]
# Naive Green function
fun2plot = Green_naive
z = fun2plot(s * y[0],s * y[1],y[0],y[1],alpha,R_alpha=R_alpha)
ax.plot(s, np.real(z),label=r"$\mathfrak{G}_\alpha(\cdot,y)$")
# Actual Green function
fun2plot = Green_function
z = fun2plot(s * y[0],s * y[1],y[0],y[1],alpha,R_alpha=R_alpha)
ax.plot(s, np.real(z),label=r"$G_\alpha(\cdot,y)$",linestyle='--')
# Markers along C
for (ss,sym) in zip(markers_pos, markers_sym):
    marker_z = fun2plot(line_fun(ss)[0],line_fun(ss)[1],y[0],y[1],alpha,R_alpha=R_alpha)
    ax.plot(ss,np.real(marker_z),marker=sym,color='C2',fillstyle='none')
ax.set_xlabel('Abscissa s')
ax.set_ylabel('Real part')
ax.set_ylim([0,1])
ax.grid()
# Argument along (xc, y) line
ax = axs[1]
# Naive Green function
fun2plot = Green_naive
z = fun2plot(s * y[0],s * y[1],y[0],y[1],alpha,R_alpha=R_alpha)
ax.plot(s, np.imag(z),label=r"$\mathfrak{G}_\alpha(\cdot,\boldsymbol{y})$")
# Actual Green function
fun2plot = Green_function
z = fun2plot(s * y[0],s * y[1],y[0],y[1],alpha,R_alpha=R_alpha)
ax.plot(s, np.imag(z),label=r"$G_\alpha(\cdot,\boldsymbol{y})$",linestyle='dashed')
for (s,sym) in zip(markers_pos, markers_sym):
    marker_z = fun2plot(line_fun(s)[0],line_fun(s)[1],y[0],y[1],alpha,R_alpha=R_alpha)
    ax.plot(s,np.imag(marker_z),marker=sym,color='C2',fillstyle='none')
ax.set_xlabel('Abscissa s')
ax.set_ylabel('Imaginary part')
ax.legend(bbox_to_anchor=(1, 1),loc='upper left')
ax.grid()
fig.suptitle(fr"Comparison of $\mathfrak{{G}}_\alpha(\cdot,\boldsymbol{{y}})$ and $G_\alpha(\cdot,\boldsymbol{{y}})$ along the curve $\mathcal{{C}}$")
fig.savefig(os.path.join(DIR_ARTICLE_IMG,'Complex-plot-Green-comparison'))