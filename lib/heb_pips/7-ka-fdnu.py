'''
MCMC fit of the distributions for dnu, numax, mass, and radius.
Dnu is the corrected, using formula of Sharma et al. 2011.

'''

# rootpath = "/mnt/c/Users/yali4742/Onedrive/Work/nike/"
rootpath = "/Volumes/Data/Onedrive/Work/nike/"
#rootpath = "/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import sys
sys.path.append(rootpath) 
from lib.wrapper import heb_fit
import os

# diagrams = ['tnu', 'tnu', 'mr', 'mr']
distances = ['horizontal', 'vertical', 'horizontal', 'vertical']
variables = ['numax', 'dnu', 'mass', 'radius']

# for i in range(2):
def loop(params):
    i, j = params
    # fdnu corrected sharma+2016
    obsdir = rootpath+"sample/heb/ka/"
    moddir = rootpath+"sample/heb/padova/"

    distance, var = distances[i], variables[i]

    # read in unperturbed data sample
    obs = np.load(obsdir+"ka18.npy", allow_pickle=True).tolist()
    pdv = np.load(moddir+"padova.npy", allow_pickle=True).tolist()

    # We only test a subset of stars. 
    # For the rest, there seems to be a disagreement between Galaxia and Kepler.
    idx = (obs["mass"]<=1.9) & (obs["mass"]>=0.8)
    for key in obs.keys():
        obs[key] = obs[key][idx]

    idx = (pdv["mass"]<=1.9) & (pdv["mass"]>=0.8)
    for key in pdv.keys():
        pdv[key] = pdv[key][idx]       


    # read in edges
    if var in ['dnu', 'numax']:
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

    if var in ['mass', 'radius']:
        edges_obs = np.load(obsdir+"mr_edge_samples.npy")
        tck_obs, tp_obs = np.load(obsdir+"mr_spline_tck.npy", allow_pickle=True)
        edges_pdv = np.load(moddir+"mr_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"mr_spline_tck.npy", allow_pickle=True)

        if distance == 'horizontal': # mass
            # to exclude those points which lies below the edge (so no horizontal distance).
            idx = (obs["radius"] <= np.max(edges_obs[:,1])) & (obs["mass"]<=1.25)
            for key in obs.keys():
                obs[key] = obs[key][idx]   
            idx = (pdv["radius"] <= np.max(edges_pdv[:,1])) & (pdv["mass"]<=1.25)
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        if distance == 'vertical': # radius
            # nothing to exclude
            pass

        xobs, yobs = obs["mass"], obs["radius"]
        e_xobs, e_yobs = obs["e_mass"]/obs["mass"], obs["e_radius"]/obs["radius"]
        xpdv, ypdv = pdv["mass"], pdv["radius"]


    # multiprocessing workflow
    if j==0:
        # trial 1: lower limit
        filepath = rootpath+"sample/heb/sharpness/ka18sharma16/"+var+"/ulim/"   
        if not os.path.exists(filepath): os.mkdir(filepath)
        heb_fit(xobs, yobs, edges_obs, tck_obs, tp_obs,
            xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
            var, distance, filepath)
    else:
        # trial 2: lower limit
        filepath = rootpath+"sample/heb/sharpness/ka18sharma16/"+var+"/llim/"
        if not os.path.exists(filepath): os.mkdir(filepath)

        if var in ['numax', 'mass']:
            heb_fit(xobs, yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                var, distance, filepath, xerror_sample=e_xobs)
        if var in ['dnu', 'radius']:
            heb_fit(xobs, yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                var, distance, filepath, yerror_sample=e_yobs)

# # multiprocessing workflow
from multiprocessing import Pool
with Pool(8) as p:
    # p.map(loop, [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]])
    # p.map(loop, [[0,0], [0,1]])
    p.map(loop, [[2,0], [2,1]]) #mass