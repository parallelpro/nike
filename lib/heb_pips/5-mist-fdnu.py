'''
MCMC fit of the distributions for dnu, numax, mass, and radius.
Dnu is the corrected, using formula of Sharma et al. 2011.

'''


# rootpath = "/Users/yaguang/Onedrive/Work/nike/"
rootpath = "/mnt/c/Users/yali4742/Onedrive/Work/nike/"
#rootpath = "/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import sys
sys.path.append(rootpath) 
from lib.histdist import model_heb
from lib.wrapper import heb_llim_fit, heb_ulim_fit#sharpness_fit_rescale_mcmc
import os

diagrams = ['tnu', 'tnu', 'mr', 'mr']
distances = ['horizontal', 'vertical', 'horizontal', 'vertical']
variables = ['numax', 'dnu', 'mass', 'radius']

for i in range(4):
    # fdnu corrected sharma+2016
    obsdir = rootpath+"sample/heb/yu/"
    moddir = rootpath+"sample/heb/mist/"

    diagram, distance, var = diagrams[i], distances[i], variables[i]

    # read in unperturbed data sample
    obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
    pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()

    # We only test a subset of stars. 
    # For the rest, there seems to be a disagreement between Galaxia and Kepler.
    idx = (obs["mass"]<=1.9) & (obs["mass"]>=0.8)
    for key in obs.keys():
        obs[key] = obs[key][idx]

    idx = (pdv["mass"]<=1.9) & (pdv["mass"]>=0.8)
    for key in pdv.keys():
        pdv[key] = pdv[key][idx]       


    # read in edges
    if diagram == 'tnu':
        edges_obs = np.load(obsdir+"tnu_edge_samples.npy")
        tck_obs, tp_obs = np.load(obsdir+"nike_spline_tck.npy", allow_pickle=True)
        edges_pdv = np.load(moddir+"tnu_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"nike_spline_tck.npy", allow_pickle=True)

        if distance == 'horizontal': # numax
            # to exclude those points which lies below the edge (so no horizontal distance).
            idx = obs["dnu"]>=np.min(edges_obs[:,1])
            for key in obs.keys():
                obs[key] = obs[key][idx]
            idx = pdv["dnu"]>=np.min(edges_pdv[:,1])
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]
        if distance == 'vertical': # dnu
            # to exclude those points which lies left to the edge (so no vertical distance).
            idx = obs["numax"]>=np.min(edges_obs[:,0])
            for key in obs.keys():
                obs[key] = obs[key][idx]
            idx = pdv["numax"]>=np.min(edges_pdv[:,0])
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        xobs, yobs = obs["numax"], obs["dnu"]
        e_xobs, e_yobs = obs["e_numax"]/obs["numax"], obs["e_dnu"]/obs["dnu"]
        xpdv, ypdv = pdv["numax"], pdv["dnu"]

    if diagram == 'mr':
        edges_obs = np.load(obsdir+"mr_edge_samples.npy")
        tck_obs, tp_obs = np.load(obsdir+"mr_spline_tck.npy", allow_pickle=True)
        edges_pdv = np.load(moddir+"mr_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"mr_spline_tck.npy", allow_pickle=True)

        if distance == 'horizontal': # mass
            # to exclude those points which lies below the edge (so no horizontal distance).
            idx = obs["radius"] <= np.max(edges_obs[:,1])
            for key in obs.keys():
                obs[key] = obs[key][idx]   
            idx = pdv["radius"] <= np.max(edges_obs[:,1])
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        if distance == 'vertical': # radius
            # nothing to exclude
            pass

        xobs, yobs = obs["mass"], obs["radius"]
        e_xobs, e_yobs = obs["e_mass"]/obs["mass"], obs["e_radius"]/obs["radius"]
        xpdv, ypdv = pdv["mass"], pdv["radius"]

    hist_model = model_heb()


    # trial 1: lower limit
    filepath = rootpath+"sample/heb/sharpness/mist/"+var+"/ulim/"
    if not os.path.exists(filepath): os.mkdir(filepath)
    heb_ulim_fit(xobs, yobs, edges_obs, tck_obs, tp_obs,
        xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        diagram, distance, hist_model, filepath, ifmcmc=True, nburn=500, nsteps=1000)

    # trial 2: lower limit
    eobs = e_xobs if distance=='horizontal' else e_yobs
    filepath = rootpath+"sample/heb/sharpness/mist/"+var+"/llim/"
    if not os.path.exists(filepath): os.mkdir(filepath)
    heb_llim_fit(xobs, yobs, eobs, edges_obs, tck_obs, tp_obs,
        xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        diagram, distance, hist_model, filepath, ifmcmc=True, nburn=500, nsteps=1000)

