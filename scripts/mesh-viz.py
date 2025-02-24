"""
Mesh visualization with pyvista.
"""
import os
import sys
import warnings
from pathlib import Path
import pyvista as pv
pv.set_plot_theme('document')
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
sys.path.append(str(Path(__file__).parents[0]/'module'))
import Num_utils
from Num_utils import DIR_ARTICLE_IMG, DIR_MESH, MPLSTYLE_ARTICLE
try:
    from computational_plasmonics import eigv_analytical as PEP_ana
except ImportError:
    warnings.warn("computational_plasmonics.eigv_analytical not available.")
#%% Ellipse
a_m, b_m = 2.5, 1.0 # sign-changing interface
R_DtN = 4 # radius of DtN boundary
N_m = 40 # number of element on N
geofile=os.path.join(DIR_MESH,"Ellipse_Unstructured.geo")
gmsh_param = {
    'binary': True,
    'order' : 2,
    'meshing' : 9,
    'recombination' : 3,
    'flexible_transfinite' : False
}
geo_param={
    'a_m':a_m,'b_m':b_m,
    'a_d':R_DtN,'b_d':R_DtN,
    'N_m': N_m,
}
try:
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param)
except AttributeError:
    warnings.warn("mesh not built")
mesh = pv.read(gmshfile)
# Adjust physical domains
tag_omega = 3
tag_omega_ext = 4
mask_omega = mesh['gmsh:physical']==tag_omega
mask_omega_ext = mesh['gmsh:physical']==tag_omega_ext
mesh['gmsh:physical'][:] = 0
mesh['gmsh:physical'][mask_omega] = 1
mesh['gmsh:physical'][mask_omega_ext] = 0
pl = pv.Plotter()
pl.enable_anti_aliasing('msaa', multi_samples=8)
pl.add_mesh(mesh,
            style='wireframe',
            scalars='gmsh:physical',
            show_scalar_bar=False,
            cmap=['C0','C1'])
pl.camera.zoom('tight') 
pl.camera.focal_point = (0.0, 0.0, pl.camera.focal_point[2])
pl.camera.position = (0.0, 0.0, pl.camera.position[2])
pl.camera.zoom(2)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.3*paperwidth, 0.13*paperheight)
dpi = 300
figsize = (int(figsize[0]*dpi),int(figsize[1]*dpi))
imgfile = os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-mesh.png')
pl.screenshot(imgfile,
              window_size=figsize,
              return_img=False)  
im = Image.open(imgfile)
im.save(imgfile,dpi=(dpi,dpi))
display(im)
#%% Corner mesh
geofile = os.path.join(DIR_MESH,"Camembert.geo")
gmshfile = os.path.splitext(geofile)[0]+".msh"
R = 1 # corner radius
phi = np.pi/2 # corner angle 
geo_param={
    'R': R,
    'phi': phi,
    'corner_x': 0, # x-coordinate of corner
    'corner_y': 0, # y-coordinate of corner
    'lc_d': R/6, # mesh size on outer boundary
    'lc_c': (R/6)/10 # mesh size at corner tip
    }
gmsh_param = {
    'binary': True,
    'order': 2,
    'meshing' : 9,
    'recombination' : 3,
    'flexible_transfinite' : False,
    'refinement': 0
}
try:
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=2)
except AttributeError:
    warnings.warn("mesh not built")
mesh = pv.read(gmshfile)
# Adjust physical domains
tag_omega = 3
tag_omega_ext = 4
mask_omega = mesh['gmsh:physical']==tag_omega
mask_omega_ext = mesh['gmsh:physical']==tag_omega_ext
mesh['gmsh:physical'][:] = 0
mesh['gmsh:physical'][mask_omega] = 1
mesh['gmsh:physical'][mask_omega_ext] = 0
pl = pv.Plotter()
pl.enable_anti_aliasing('msaa', multi_samples=8)
pl.add_mesh(mesh,
            style='wireframe',
            scalars='gmsh:physical',
            show_scalar_bar=False,
            cmap=['C0','C1'])
pl.camera.zoom('tight') 
pl.camera.focal_point = (0.0, 0.0, pl.camera.focal_point[2])
pl.camera.position = (0.0, 0.0, pl.camera.position[2])
pl.camera.zoom(1)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.2*paperwidth, 0.13*paperheight)
dpi = 300
figsize = (int(figsize[0]*dpi),int(figsize[1]*dpi))
imgfile = os.path.join(DIR_ARTICLE_IMG,'Num-Corner-mesh.png')
pl.screenshot(imgfile,
              window_size=figsize,
              return_img=False)  
im = Image.open(imgfile)
im.save(imgfile,dpi=(dpi,dpi))
display(im)
#%% Ellipse with corner on major axis
geofile=os.path.join(DIR_MESH,"Ellipse-with-1-corner_Cartesian.geo")
gmshfile = os.path.splitext(geofile)[0]+".msh"
try:
    # Geometry
    (a, b, phi) = (2.5, 1, np.pi*0.75)
    corner_pos = "left"
    (x_c,y_c,x_m,y_m,R) = PEP_ana.get_C1corner_ellipse(a,b,phi,pos=corner_pos)
    R_DtN = 5 # radius of DtN boundary
    cor_jun1_theta = np.arccos(x_m/a)
    cor_jun2_theta = -np.arccos(x_m/a)
        # Geometry and mesh (BEM: Cartesian coordinates only)
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
    grading_power = 2
    element_size = a/10
    geo_param['lc_m'] = element_size
    geo_param['cor_lc'] = Num_utils.corner_mesh_grading_power(element_size,
                                                            power=grading_power) 
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=2)
except:
    warnings.warn("mesh not build")
mesh = pv.read(gmshfile)
tag_omega = 3
tag_omega_ext = 4
mask_omega = mesh['gmsh:physical']==tag_omega
mask_omega_ext = mesh['gmsh:physical']==tag_omega_ext
mesh['gmsh:physical'][:] = 0
mesh['gmsh:physical'][mask_omega] = 1
mesh['gmsh:physical'][mask_omega_ext] = 0
pl = pv.Plotter()
pl.add_mesh(mesh,style='wireframe',
            scalars='gmsh:physical',
            show_scalar_bar=False,
            cmap=['C0','C1'])
pl.camera.zoom('tight') 
# pl.camera.focal_point = (x_c, y_c, pl.camera.focal_point[2])
# pl.camera.position = (x_c, y_c, pl.camera.position[2])
pl.camera.focal_point = (0, 0, pl.camera.focal_point[2])
pl.camera.position = (0, 0, pl.camera.position[2])
pl.camera.zoom(3.7)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.3*paperwidth, 0.1*paperheight)
dpi = 300
figsize = (int(figsize[0]*dpi),int(figsize[1]*dpi))
imgfile = os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-1-Corner-Mesh.png')
pl.screenshot(imgfile,
              window_size=figsize,
              return_img=False)  
im = Image.open(imgfile)
im.save(imgfile,dpi=(dpi,dpi))
display(im)
#%% Ellipse with corner on minor axis 
geofile=os.path.join(DIR_MESH,"Ellipse-with-1-corner_Cartesian.geo")
gmshfile = os.path.splitext(geofile)[0]+".msh"
try:
        # Geometry
    (a, b, phi) = (2.5, 1, np.pi*0.85)
    corner_pos = "top"
    (x_c,y_c,x_m,y_m,R) = PEP_ana.get_C1corner_ellipse(a,b,phi,pos=corner_pos)
    R_DtN = 5 # radius of DtN boundary
    cor_jun1_theta = np.arccos(np.abs(x_m)/a)
    cor_jun2_theta = np.arccos(-np.abs(x_m)/a)
        # Geometry and mesh (BEM: Cartesian coordinates only)
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
    grading_power = 2
    element_size = a/10
    geo_param['lc_m'] = element_size
    geo_param['cor_lc'] = Num_utils.corner_mesh_grading_power(element_size,
                                                            power=grading_power) 
    gmshfile = Num_utils.build_mesh(geofile,geo_param,gmsh_param,gdim=2)
except:
    warnings.warn("Mesh not built.")
mesh = pv.read(gmshfile)
tag_omega = 3
tag_omega_ext = 4
mask_omega = mesh['gmsh:physical']==tag_omega
mask_omega_ext = mesh['gmsh:physical']==tag_omega_ext
mesh['gmsh:physical'][:] = 0
mesh['gmsh:physical'][mask_omega] = 1
mesh['gmsh:physical'][mask_omega_ext] = 0
pl = pv.Plotter()
pl.add_mesh(mesh,style='wireframe',
            scalars='gmsh:physical',
            show_scalar_bar=False,
            cmap=['C0','C1'])
pl.camera.zoom('tight') 
pl.camera.focal_point = (0, 0, pl.camera.focal_point[2])
pl.camera.position = (0, 0, pl.camera.position[2])
pl.camera.zoom(3.7)
plt.style.use(MPLSTYLE_ARTICLE)
[paperwidth,paperheight] = mpl.rcParams['figure.figsize']
figsize = (0.3*paperwidth, 0.1*paperheight)
dpi = 300
figsize = (int(figsize[0]*dpi),int(figsize[1]*dpi))
imgfile = os.path.join(DIR_ARTICLE_IMG,'Num-Ellipse-1-Corner-Minor-Mesh.png')
pl.screenshot(imgfile,
              window_size=figsize,
              return_img=False)  
im = Image.open(imgfile)
im.save(imgfile,dpi=(dpi,dpi))
display(im)