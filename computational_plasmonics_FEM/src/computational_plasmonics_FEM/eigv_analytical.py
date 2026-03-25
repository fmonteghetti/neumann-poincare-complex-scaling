#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytical functions for solving the plasmonic eigenvalue problem (PEP).

"""

import numpy as np
from scipy.optimize import newton
import fractions
import os

def np_to_contrast(l):
    """ Convert Neumann-Poincaré eigenvalue in (-1/2,1/2) to contrast in (-∞,0)."""
    return (2*l+1)/(2*l-1)

def contrast_to_np(kappa):
    """ Convert contrast in (-∞,0) to Neumann-Poincaré eigenvalue in (-1/2,1/2)."""
    return 0.5 * (kappa+1)/(kappa-1)

def ellipse_freespace_eigenvalues(a,b,N):
    """ Return first N eigenvalues of ellipse (a,b) with a>b, alongside boolean masks giving eigenfunction parity w.r.t x and y axis."""
    r = np.min(b/a)
    n=np.concatenate([np.arange(-N,0),np.arange(1,N+1)])
        # eigenvalue of NP operator on ellipse (|n| >=1)
    l_ev = 0.5*np.sign(n) * np.exp(-2*np.abs(n)*np.arctanh(r))
    mask_pos = n>0 # positive integers
    mask_even = (np.mod(n,2)==0) # even integers
    mask = dict()
    mask["x-even"] = ~mask_pos
    mask["x-odd"] = ~mask["x-even"]
    mask["y-even"] = (mask_pos * (~mask_even)) + ((~mask_pos) * (mask_even)) 
    mask["y-odd"] = ~mask["y-even"]
    return (l_ev,np_to_contrast(l_ev),n,mask)

def ellipse_freespace_eigenfunction(a,b,n=1):
    """ Return n-th (|n| >=1) eigenfunction of ellipse (a,b) with a>b. """
    r = np.min(b/a)
    c = np.sqrt(np.max(a**2-b**2))
    mu_1 = np.arctanh(r)
        # Elliptic coordinates (mu,phi)
        # Cartesian coordinate (x,y)
    mu = lambda x,y: np.real(np.arccosh((x+1j*y)/c)) 
    phi= lambda x,y: np.imag(np.arccosh((x+1j*y)/c))
    def psi(mu,n,f): # mu-dependency of eigenfunctions
        n = np.abs(n)
        idx_int = mu<=mu_1 # exterior
        val = np.zeros_like(mu)
        val[idx_int] = np.exp(-n*mu_1)*f(n*mu[idx_int])/f(n*mu_1)
        val[~idx_int] = np.exp(-n*mu[~idx_int])
        return val
    if n>=1: # mode odd w.r.t x-axis
        u = lambda x,y: np.sin(n*phi(x,y))*psi(mu(x,y),n,np.sinh)
    else: # mode even w.r.t x-axis
        u = lambda x,y: np.cos(n*phi(x,y))*psi(mu(x,y),n,np.cosh)
    return u

def corner_essential_spectrum(phi,N=100,alpha=1):
    """ Return essential spectrum of straight corner of angle ϕ in (0,2π)\{π},
     alongside boolean masks giving singularities parity w.r.t corner axis."""
    psi = lambda eta: np.sinh(eta*(np.pi-phi))/np.sinh(eta*np.pi)
    eta = np.logspace(-2,2,num=N)
    l_ev = -psi(alpha*eta)/2 # even singularities
    l_ev = np.concatenate((l_ev,-l_ev)) # odd singularities
    kappa = np_to_contrast(l_ev) 
        # parity of strongly-oscillating singularities w.r.t to corner axis
    mask = dict()
    mask["even"] = np.zeros_like(l_ev,dtype=bool)
    mask["even"][0:N] = True
    mask["odd"] = ~mask["even"]
    return (l_ev,kappa,mask)

def ellipse_eigenvalues(a_m,b_m,a_d,b_d,N):
    """
    Exact eigenvalues for an elliptical particle (Omega_m) embedded in a confocal
    elliptic domain (Omega_d).

    Parameters
    ----------
    a_m : float
        Major semi-axis of Omega_m.
    b_m : float
        Minor semi-axis of Omega_m.
    a_d : float
        Major semi-axis of Omega_d (possibly numpy.inf).
    b_d : float
        Minor semi-axis of Omega_d.
    N : int
        Number of eigenvalues.

    Returns
    -------
    kappa_ex_ev : numpy.ndarray
        Eigenvalues associated with eigenfunctions even w.r.t the major axis.
    kappa_ex_odd : numpy.ndarray
        Eigenvalues associated with eigenfunctions odd w.r.t the major axis.

    Remark
    -------
    
    The formula used for the eigenvalues assumes that the ellipses Omega_m
    and Omega_d are confocal, see Prop. 31 of 10.1016/j.jcp.2021.110433.
    
    """    
    c = np.sqrt(a_m**2-b_m**2) # a_m > b_m since it is the major axis
    mu_m = np.arccosh(a_m/c)
    mu_d = np.arccosh(a_d/c)    
    n = np.r_[1:N]
    if a_d == np.inf:
        n_ev = n # 0 is not an eigenvalue
    else:
        n_ev = np.concatenate(( [0], n )) # add 0 for even eigenvalues
    kappa_ex_ev = -np.tanh(n_ev*(mu_d-mu_m))*np.tanh(n_ev*mu_m)
    kappa_ex_odd = -np.tanh(n*(mu_d-mu_m))/np.tanh(n*mu_m)
    return (kappa_ex_ev,kappa_ex_odd)

def ellipse_parameters(case_name):
    """
    Return value of ellipse parameters (a_m,b_m,a_d,b_d) corresponding to
    case_name.
    """
    # JCP2021 = 10.1016/j.jcp.2021.110433
    if case_name == "JCP2021_figure_9":
        a_d =3; b_d = 2
        c = np.sqrt(a_d**2-b_d**2)
        a_m =1.10*c; b_m = np.sqrt(a_m**2-c**2)
    elif case_name == "JCP2021_figure_10":
        a_m =2.5; b_m = 1; c = np.sqrt(a_m**2-b_m**2)
        a_d =3; b_d = np.sqrt(a_d**2-c**2)        
    elif case_name == "JCP2021_figure_12":
        mu_m = 0.5; mu_d = 1.25;
        a_m = 2.5; b_m = 1; c = np.sqrt(a_m**2-b_m**2) # arbitrary
        a_d = a_m * np.cosh(mu_d)/np.cosh(mu_m);
        b_d = np.sqrt(a_d**2-c**2)
    elif case_name == "JCP2021_case_A":
        (a_m,b_m,a_d,b_d) = perturbed_ellipse_parameters(case_name)[0:4]
    elif case_name == "JCP2021_case_B":
        (a_m,b_m,a_d,b_d) = perturbed_ellipse_parameters(case_name)[0:4]
    else:
        raise ValueError("Unknown case name.")
     
    return (a_m,b_m,a_d,b_d)

def perturbed_ellipse_parameters(case_name):
    """
    Return geometrical values describing an ellipse perturbed by corners.

    Parameters
    ----------
    case_name : str
        Reference name.

    Returns
    -------
    a_m, b_m : float
        Semi-axes of ellipse m.
    a_d, b_d : TYPE
        Semi-axes of ellipse d.
    x_c, y_c : list(float)
        Coordinates of each corner.
    x_m, y_m : list(float)
        Coordinates of junction point for each corner.
    phi : list(float)
        Angle for each corner.
    corner_pos : list(str)
        Position of each corner.
        
    """
        
    if case_name == "JCP2021_case_A":
        a_m =2.5; b_m = 1
        c = np.sqrt(a_m**2-b_m**2)
        phi = [0.75*np.pi]
        corner_pos = ["left"]
        (x_c,y_c,x_m,y_m,R) = get_C1corner_ellipse(a_m,b_m,phi[0],pos=corner_pos[0])
        a_d =np.abs(x_c) + 1.5*R; b_d = np.sqrt(a_d**2-c**2)
        x_c=[x_c]; y_c=[y_c]; x_m=[x_m]; y_m=[y_m]; R=[R]
    elif case_name == "JCP2021_case_B":
        a_m =2.5; b_m = 1
        c = np.sqrt(a_m**2-b_m**2)
        phi = [0.63*np.pi]
        corner_pos = ["left"]
        (x_c,y_c,x_m,y_m,R) = get_C1corner_ellipse(a_m,b_m,phi[0],pos=corner_pos[0])
        a_d =np.abs(x_c) + 1.5*R; b_d = np.sqrt(a_d**2-c**2)
        x_c=[x_c]; y_c=[y_c]; x_m=[x_m]; y_m=[y_m]; R=[R]

    return (a_m,b_m,a_d,b_d,x_c,y_c,x_m,y_m,phi,corner_pos)


def get_C1corner_ellipse(a,b,phi,pos="left"):
    """
    Compute geometrical parameters of the C^1 corner perturbation with angle phi
    in (0,pi), for an ellipse of semi-axes (a,b).

    Parameters
    ----------
    a, b : float
        Ellipse semi-axes.
    phi : float
        Corner angle in (0,pi).
    pos : string, optional
        Corner position ("left","top","right").

    """
     
    def ellipse_get_junction_point(a,b,phi):
       """ Return point (x,y) with slope tan(phi/2) with y>0. """
       x_j = np.sqrt(np.tan(phi/2)**2+(b/a)**2);
       x_j = -a * np.tan(phi/2) / x_j;
       y_j = b*np.sqrt(1-(x_j/a)**2);
       return (x_j,y_j)
           
    if (pos=="left"):
            # top junction point (y_j>0)
        (x_j,y_j) = ellipse_get_junction_point(a,b,phi)
            # abscissa of top point of corner
        x_c = x_j - y_j/np.tan(phi/2);
        y_c = 0;
    elif (pos=="top"):
        psi = np.pi - phi
        (x_j,y_j) = ellipse_get_junction_point(a,b,psi)
            # abscissa of top point of corner
        x_c = 0
        y_c = y_j - x_j*np.tan(psi/2);        
    elif (pos=="right"):
            # top junction point (y_j>0)
        (x_j,y_j) = ellipse_get_junction_point(a,b,-phi)
            # abscissa of top point of corner
        x_c = x_j + y_j/np.tan(phi/2);
        y_c = 0      
        # radius of corner circle
    R = np.sqrt((x_c-x_j)**2+(y_c-y_j)**2); 
    return (x_c,y_c,x_j,y_j,R)

def DtN_Laplace_circle(n,m):
    """ Expression of the DtN kernel for Laplace's equation on a circle. 
    
    The kernel is written as: (Givoli 1992, Numerical Methods for Problem in
    Infinite Domains, (49))
   
        k(x,y) = Σ_n Σ_m ɑ_{n} * k_{n,m}(x) * k_{n,m}(y),

    where ɑ_{n} = - n/(π*R^2), 
    
        k_{n,0}(x) =  cos(n*(θ(x))), and  k_{n,1}(x) =  sin(n*(θ(x))).

    Parameters
    ----------
    n: int
        Order (n>=1).
    m: int
        Decomposition index (m∈{0,1}).

    Returns
    -------
    alpha: float
        Constant scalar factor ɑ_{n,m}.
    k: function x-> float
        Function k_{n,m}.
    """
    alpha = -n/np.pi
        # ufl.atan_2(x[1],x[0]) not supported with complex PETSc scalar
    theta = lambda x: np.arctan2(x[1],x[0])
    r = lambda x: np.sqrt(x[0]**2 + x[1]**2)
    if n<=0:
        raise ValueError("Order n must be nonnegative.")
    if m==0:
        k = lambda x:  np.cos(n*theta(x)) / r(x)
    elif m==1:
        k = lambda x:  np.sin(n*theta(x)) / r(x)
    else:
        raise ValueError("Index m must be in {0,1}.") 
    return (alpha,k) 


def compute_radial_corner_exponents(phi,kappa,tol=1e-10):
    """ Compute the radial exponents η for a corner of angle phi and a contrast
    kappa.

    The exponents are computed by solving the dispersion relation

        sinh(eta*pi) = +/- beta * sinh(eta*(pi-phi)),

    where '+' (resp. '-') is associated with odd (resp. even) solutions and
    beta = (kappa-1)/(kappa+1).

    The dispersion relation is solved with an analytical method that assumes
    that phi/pi is rational, see Prop. 9 of 
    "Complex-scaling method for the complex plasmonic resonances of planar
    subwavelength particles with corners"  doi:10.1016/j.jcp.2021.110433

    Parameters
    ----------
    phi: float
        Corner angle in (0,2*pi)
    kappa: complex
        Value of contrast.
    tol: float
        Tolerance for residual.
    """
    def get_polynomial(p,q,beta):
        """ Construct polynomial P_beta of degree 2*(q-1)."""
        P = np.zeros(2*(q-1)+1,dtype='complex')
        P[::2] = 1
        r = np.abs(q-p)
        idx = [2*k + q - r for k in range(0,r)]
        P[idx] = beta * np.sign(q-p)
        return np.flip(P) # lowest index = highest degree
    z2beta = lambda z: (z-1)/(z+1)
    beta = z2beta(kappa)
    # Approximate phi = (p/q)*pi
    phi_frac = fractions.Fraction(phi/np.pi).limit_denominator()
    (p,q) = phi_frac.numerator, phi_frac.denominator
    if np.abs(phi-(p/q)*np.pi)/np.abs(phi)>1e-10:
        raise ValueError("Could not approximate phi accurately enough.")
    eta_even = np.roots(get_polynomial(p,q,beta))
    eta_odd = np.roots(get_polynomial(p,q,-beta))    
    eta_even = q/np.pi * np.log(eta_even)
    eta_odd = q/np.pi * np.log(eta_odd)
    # add translated roots
    add_trans = lambda eta, N: np.reshape(eta + 1j*2*q*np.c_[-N:N+1],-1)
    N = 4*q
    eta_even,eta_odd  = add_trans(eta_even,N), add_trans(eta_odd,N)
    # add imaginary roots
    eta_im = 1j*q*np.r_[-N:N+1]
    eta_even, eta_odd = np.hstack((eta_even,eta_im)), np.hstack((eta_odd,eta_im))
    # check residual
    def f_even(eta):
        return np.sinh(eta*np.pi) + beta*np.sinh(eta*(np.pi-phi))
    def f_odd(eta):
        return np.sinh(eta*np.pi) - beta*np.sinh(eta*(np.pi-phi))
    eta_even=eta_even[np.abs(f_even(eta_even))<tol]
    eta_odd=eta_odd[np.abs(f_odd(eta_odd))<tol]
    return eta_even, eta_odd

def compute_radial_corner_exponents_alt(phi,kappa,tol=1e-10):
    """ Compute the radial exponents η for a corner of angle phi and a contrast
    kappa.

    The difference with 'compute_radial_corner_exponents' is that here we
    use a Newton iteration to solve the dispersion relation. 
    """
    max_real_part = 10
    max_imag_part = 100
    step_real_part = 0.5
    step_imag_part = 1.0

    def f_even(eta,l,phi):
        return 2*l*np.sinh(eta*np.pi) + np.sinh(eta*(np.pi-phi))

    def f_even_grad(eta,l,phi):
        return 2*l*np.pi*np.cosh(eta*np.pi) + (np.pi-phi)*np.cosh(eta*(np.pi-phi))

    def f_odd(eta,l,phi):
        return 2*l*np.sinh(eta*np.pi) - np.sinh(eta*(np.pi-phi))

    def f_odd_grad(eta,l,phi):
        return 2*l*np.pi*np.cosh(eta*np.pi) - (np.pi-phi)*np.cosh(eta*(np.pi-phi))

    x = np.linspace(0, max_real_part, 
                    num=np.int(max_real_part/step_real_part))
    y = np.linspace(-max_imag_part, max_imag_part, 
                    num=np.int(2*max_imag_part/step_imag_part))
    xv, yv = np.meshgrid(x, y)
    eta0 = xv + 1j*yv 
    l = contrast_to_np(kappa)
    # Solve even dispersion relation
    (eta_even, converged, zeroder) = newton(lambda x: f_even(x,l,phi),
                x0=eta0.flatten(), fprime=lambda x: f_even_grad(x,l,phi),
                tol=tol,rtol=tol, full_output=True)
    eta_even = eta_even[converged]
    eta_even = np.hstack((eta_even,-eta_even))
    # Solve odd dispersion relation
    (eta_odd, converged, zeroder) = newton(lambda x: f_odd(x,l,phi),
                x0=eta0.flatten(), fprime=lambda x: f_odd_grad(x,l,phi),
                tol=tol,rtol=tol, full_output=True)
    eta_odd = eta_odd[converged]
    eta_odd = np.hstack((eta_odd,-eta_odd))
    return eta_even, eta_odd

def compute_radial_corner_exponent_most_singular(phi,kappa,pml_parameter=1.0):
    """ Compute the most singular radial exponent η 

                η_s =     argmax      Im(η),
                      Im(η)<=0, η!=0

        where η solves the corner dispersion relation. See 
        'compute_radial_corner_exponents' for background.

        Parameters
        ----------
        phi: float
            corner angle.
        kappa: complex
            Value of contrast.
        pml_parameter: complex
            Value of pml parameter, default to 1.0.

        Returns
        -------
        eta_s: complex
            Most singular exponent.
        eta_s_even, eta_s_odd: complex, complex
            Most singular exponent associated with even/odd solution.
       """
    (eta_even,eta_odd) = compute_radial_corner_exponents(phi,kappa)
    eta_even = eta_even/pml_parameter
    eta_odd = eta_odd/pml_parameter
    # Remove 0
    filter = lambda x: np.abs(x)>1e-15
    eta_even = eta_even[filter(eta_even)]
    eta_odd = eta_odd[filter(eta_odd)]
    # Keep Im<=0
    filter = lambda x: np.imag(x) < 1e-20
    eta_even = eta_even[filter(eta_even)]
    eta_odd = eta_odd[filter(eta_odd)]
    # Most singular exponent
    eta_s_even = eta_even[np.argmax(np.imag(eta_even))]
    eta_s_odd = eta_odd[np.argmax(np.imag(eta_odd))]
    eta = [eta_s_even,eta_s_odd]
    eta_s = eta[np.argmax(np.imag(eta))] 
    return eta_s, eta_s_even, eta_s_odd 