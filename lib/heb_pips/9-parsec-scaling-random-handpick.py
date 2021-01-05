'''
MCMC fit of the distributions for dnu, numax, mass, and radius.
Dnu is directly from scaling.

'''


#rootpath = "/Users/yaguang/Onedrive/Work/nike/"
rootpath = "/mnt/c/Users/yali4742/Onedrive/Work/nike/"
#rootpath = "/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import sys
sys.path.append(rootpath) 
from lib.wrapper import heb_fit, heb_combo_fit
import os

# diagrams = ['tnu', 'tnu', 'mr', 'mr']
distances = ['horizontal', 'vertical', 'horizontal', 'vertical']
variables = ['numax', 'dnu', 'mass', 'radius']

# for i in range(2):
def loop(params):
    i, j = params
    # fdnu corrected sharma+2016
    obsdir = rootpath+"sample/heb/yu_nc/"
    moddir = rootpath+"sample/heb/padova_nc/"

    distance, var = distances[i], variables[i]

    # read in unperturbed data sample
    obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
    pdv = np.load(moddir+"padova.npy", allow_pickle=True).tolist()

    # We only test a subset of stars. 
    # For the rest, there seems to be a disagreement between Galaxia and Kepler.
    idx = (obs["mass_nc"]<=1.9) & (obs["mass_nc"]>=0.8)
    for key in obs.keys():
        obs[key] = obs[key][idx]

    idx = (pdv["mass"]<=1.9) & (pdv["mass"]>=0.8)
    for key in pdv.keys():
        pdv[key] = pdv[key][idx]       


    # read in edges
    if var in ['dnu', 'numax']:
        data = np.load(obsdir+'tnu_perturb_data.npy', allow_pickle=True).tolist()
        edges_obs_combo = [d[0] for d in data]
        tck_obs_combo = [d[1] for d in data]
        tp_obs_combo = [d[2] for d in data]

        edges_pdv = np.load(moddir+"tnu_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"nike_spline_tck.npy", allow_pickle=True)

        if distance == 'horizontal': # numax
            # to exclude those points which lies below the edge (so no horizontal distance).
            obs_combo = []
            for edges_obs in edges_obs_combo:
                idx = obs["dnu"]>=np.min(edges_obs[:,1])
                tobs = obs
                for key in tobs.keys():
                    tobs[key] = tobs[key][idx]
                obs_combo.append(tobs)

            idx = pdv["dnu_nc"]>=np.min(edges_pdv[:,1])
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        if distance == 'vertical': # dnu
            # to exclude those points which lies left to the edge (so no vertical distance).
            obs_combo = []
            for edges_obs in edges_obs_combo:
                idx = obs["numax"]>=np.min(edges_obs[:,0])
                tobs = obs
                for key in tobs.keys():
                    tobs[key] = tobs[key][idx]
                    obs_combo.append(tobs)
            
            idx = pdv["numax"]>=np.min(edges_pdv[:,0])
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        xobs_combo, yobs_combo = [obs["numax"] for obs in obs_combo], [obs["dnu"] for obs in obs_combo]
        e_xobs_combo, e_yobs_combo = [obs["e_numax"]/obs["numax"] for obs in obs_combo], [obs["e_dnu"]/obs["dnu"]  for obs in obs_combo]
        xpdv, ypdv = pdv["numax"], pdv["dnu_nc"]

    if var in ['mass', 'radius']:
        data = np.load(obsdir+'mr_perturb_data.npy', allow_pickle=True).tolist()
        edges_obs_combo = [d[0] for d in data]
        tck_obs_combo = [d[1] for d in data]
        tp_obs_combo = [d[2] for d in data]

        edges_pdv = np.load(moddir+"mr_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"mr_spline_tck.npy", allow_pickle=True)

        if distance == 'horizontal': # mass
            # to exclude those points which lies below the edge (so no horizontal distance).
            obs_combo = []
            for edges_obs in edges_obs_combo:
                idx = obs["radius_nc"] <= np.max(edges_obs[:,1])
                tobs = obs
                for key in tobs.keys():
                    tobs[key] = tobs[key][idx]   
                    obs_combo.append(tobs)

            idx = pdv["radius"] <= np.max(edges_pdv[:,1])
            for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        if distance == 'vertical': # radius
            # nothing to exclude
            obs_combo = [obs for i in range(len(edges_obs_combo))]
            pass

        xobs_combo, yobs_combo = [obs["mass_nc"] for obs in obs_combo], [obs["radius_nc"] for obs in obs_combo]
        e_xobs_combo, e_yobs_combo = [obs["e_mass_nc"]/obs["mass_nc"] for obs in obs_combo], [obs["e_radius_nc"]/obs["radius_nc"] for obs in obs_combo]
        xpdv, ypdv = pdv["mass"], pdv["radius"]


    # multiprocessing workflow
    if j==0:
        # trial 1: lower limit
        filepath = rootpath+"sample/heb/sharpness/kb95combo/"+var+"/ulim/"   
        if not os.path.exists(filepath): os.mkdir(filepath)
        heb_combo_fit(xobs_combo, yobs_combo, edges_obs_combo, tck_obs_combo, tp_obs_combo,
            xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
            var, distance, filepath)
    else:
        # trial 2: lower limit
        filepath = rootpath+"sample/heb/sharpness/kb95combo/"+var+"/llim/"
        if not os.path.exists(filepath): os.mkdir(filepath)

        if var in ['numax', 'mass']:
            heb_combo_fit(xobs_combo, yobs_combo, edges_obs_combo, tck_obs_combo, tp_obs_combo,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                var, distance, filepath, xerror_sample=e_xobs_combo)
        if var in ['dnu', 'radius']:
            heb_combo_fit(xobs_combo, yobs_combo, edges_obs_combo, tck_obs_combo, tp_obs_combo,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                var, distance, filepath, yerror_sample=e_yobs_combo)

# # multiprocessing workflow
from multiprocessing import Pool
with Pool(8) as p:
    p.map(loop, [[0,0], [0,1], [1,0],[1,1],  [2,0], [2,1], [3,0],[3,1] ])
    # p.map(loop, [[0,1], [1,1], [2,1], [3,1]])
