'''
This is to measure the horizontal/vertical scatter on dnu-numax diagram, 
without manipulating anything else (for example to see as a function of
mass/metallicity).

'''


rootpath = "/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import sys
sys.path.append(rootpath) 
from lib.histdist import model4
from lib.wrapper import sharpness_fit
import os

# read in unperturbed mass and radius, with edges
tnu_samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy")
mr_edges_obs = np.load(rootpath+"sample/obs/mr_edge_samples.npy")
tck_obs = np.load(rootpath+"sample/obs/mr_spline_tck.npy", allow_pickle=True)
# read in unperturbed mass and radius, with edges
tnu_samples_pdv = np.load(rootpath+"sample/padova_oversampling/tnu_samples.npy")
mr_edges_pdv = np.load(rootpath+"sample/padova_oversampling/mr_edge_samples.npy")
tck_pdv = np.load(rootpath+"sample/padova_oversampling/mr_spline_tck.npy", allow_pickle=True)

distance = "horizontal"
diagram = "mr"
hist_model = model4()

xperturb = np.arange(0.00, 0.15, 0.005) # np.arange(0.00, 0.06, 0.11)#
yperturb = np.arange(0.00, 0.05, 1.0) # np.arange(0., 0.05, 0.1)
montecarlo = 120


# trial 1: no binning
xobs, yobs = tnu_samples_obs[:,3], tnu_samples_obs[:,4]
idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
xobs, yobs = xobs[idx], yobs[idx]

xpdv, ypdv = tnu_samples_pdv[:,3], tnu_samples_pdv[:,4]
idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
xpdv, ypdv= xpdv[idx], ypdv[idx]

filepath = rootpath+"sample/sharpness/perturb_gif_mr/mass/"
if not os.path.exists(filepath): os.mkdir(filepath)

sharpness_fit(xobs, yobs, xobs, mr_edges_obs, tck_obs,
        xpdv, ypdv, xpdv, mr_edges_pdv, tck_pdv,
        diagram, distance, hist_model,
        filepath, xperturb, yperturb, montecarlo)


# trial 2: mass effect
zvalue_limits = [[0.3, 1.14, 1.47],
                [1.14, 1.47, 5.24]]
zvalue_name = "mass"

xobs, yobs, zobs = tnu_samples_obs[:,3], tnu_samples_obs[:,4], tnu_samples_obs[:,3]
idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
xobs, yobs, zobs = xobs[idx], yobs[idx], zobs[idx]

xpdv, ypdv, zpdv = tnu_samples_pdv[:,3], tnu_samples_pdv[:,4], tnu_samples_pdv[:,3]
idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
xpdv, ypdv, zpdv = xpdv[idx], ypdv[idx], zpdv[idx]

filepath = rootpath+"sample/sharpness/perturb_gif_mr/mass_mass/"
if not os.path.exists(filepath): os.mkdir(filepath)

sharpness_fit(xobs, yobs, zobs, mr_edges_obs, tck_obs,
        xpdv, ypdv, zpdv, mr_edges_pdv, tck_pdv,
        diagram, distance, hist_model,
        filepath, xperturb, yperturb, montecarlo,
        zvalue_limits=zvalue_limits, zvalue_name=zvalue_name)


# trial 3: feh effect
zvalue_limits = [[-3.0, -0.20, 0.02],
                [-0.20, 0.02, 1.0]]
zvalue_name = "feh"

xobs, yobs, zobs = tnu_samples_obs[:,3], tnu_samples_obs[:,4], tnu_samples_obs[:,2]
idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
xobs, yobs, zobs = xobs[idx], yobs[idx], zobs[idx]

xpdv, ypdv, zpdv = tnu_samples_pdv[:,3], tnu_samples_pdv[:,4], tnu_samples_pdv[:,2]
idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
xpdv, ypdv, zpdv = xpdv[idx], ypdv[idx], zpdv[idx]

filepath = rootpath+"sample/sharpness/perturb_gif_mr/mass_feh/"
if not os.path.exists(filepath): os.mkdir(filepath)

sharpness_fit(xobs, yobs, zobs, mr_edges_obs, tck_obs,
        xpdv, ypdv, zpdv, mr_edges_pdv, tck_pdv,
        diagram, distance, hist_model,
        filepath, xperturb, yperturb, montecarlo,
        zvalue_limits=zvalue_limits, zvalue_name=zvalue_name)
