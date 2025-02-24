# coding: utf-8
"""
Validation: eigenvalues for an ellipse, isogeometric BEM.
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

def compute_error(eigval,eigval_ex,N=1):
    """ Compute the maximum error against the first N exact eigenvalues.

    The error is computed as
            max     |λ_i - λ_{i,ex}|
            1<=i<=N
    where eigenvalues are sorted by decreasing real part: 
                 ... <= 𝕽(λ_2) <= 𝕽(λ_1) 
    """
    eigval_ex = np.sort(eigval_ex)
    eigval = np.sort(eigval)
    return np.max(np.abs(eigval[-N:]-eigval_ex[-N:]))

    # Geometry and mesh
geofile= os.path.join(DIR_MESH,"Ellipse_Unstructured.geo")
(a, b) = (2.5, 1)
aspect_ratio = np.min([a/b,b/a])
qorder = 3
opts_eigv = Num_BEM.EigenvalueSolverOpts(use_Arnoldi=False)
#%% Plot convergence rate for isogeometric BEM
element_size_l = [a/8,a/10, a/15, a/16, a/17, a/18, a/19, a/20]
eigval_l, dof_l = list(), list()
for element_size in element_size_l:
    (eigval,dof) = Num_BEM.compute_NP_eigenvalues_ellipse_isogeometric(a,b,
                                  opts_eigv,
                                  meshsize=element_size,
                                  qorder=qorder,qmaxdist=4*element_size) 
    eigval_l.append(eigval)
    dof_l.append(dof)
# Plot error
N_error_l = [8]
eigval_ex = Num_utils.compute_exact_eigenvalues_ellipse(aspect_ratio,N=30)
error_l = np.zeros((len(N_error_l),len(eigval_l)))
for (i,N_error) in enumerate(N_error_l):
    for (j,eigval) in enumerate(eigval_l):
        error_l[i,j] = compute_error(eigval,eigval_ex,N=N_error)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.31*paperwidth, 0.18*paperheight)
plt.style.use(MPLSTYLE_VSCODE)
figsize = mpl.rcParams['figure.figsize']
f, ax = plt.subplots(layout='constrained',figsize=figsize)
ax.set_prop_cycle(marker=['o', 's', 'x'],
                  color=mpl.rcParams['axes.prop_cycle'].by_key()['color'][:3])
for (i,N_error) in enumerate(N_error_l):
    error = error_l[i,:]
    p = Num_utils.compute_slope(np.array(dof_l),np.array(error),N=len(element_size_l))
    ax.loglog(np.array(dof_l)/min(dof_l),error,fillstyle='none',
              label=f'$J=${N_error}, $\mathcal{{O}}(h^{{{p:2.2g}}})$')
ax.set_xlabel(f'DoF / {int(min(dof_l)):d}')
ax.set_ylabel(r'$\max_{1\leq{}j\leq{}J}\,|\lambda_j^h-\lambda_j^{\text{el}}|$')
ax.set_title(f'BEM error, $P$={int(np.ceil((qorder+1)/2))}, ($a$,$b$)=({a},{b})')
ax.legend()
ax.grid(which='both')