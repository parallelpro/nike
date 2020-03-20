'''
MCMC fit of the distributions for dnu, numax, mass, and radius.
Dnu is the corrected, using formula of Sharma et al. 2011.

'''


rootpath = "/mnt/c/Users/yali4742/Onedrive/Work/nike/"
# rootpath = "/Volumes/Data/Onedrive/Work/nike/"
#rootpath = "/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import sys
sys.path.append(rootpath) 
from lib.histdist import model_rgb
from lib.wrapper import rgb_llim_fit, rgb_ulim_fit#sharpness_fit_rescale_mcmc
import os

diagrams = ['tnu', 'tnu', 'mr', 'mr']
distances = ['vertical', 'vertical', 'horizontal', 'vertical']
variables = ['numax', 'dnu', 'mass', 'radius']

# for i in range(2):
def loop(params):
    i, j = params
    # fdnu corrected sharma+2016
    obsdir = rootpath+"sample/rgb/yu/"
    moddir = rootpath+"sample/rgb/mist/"

    diagram, distance, var = diagrams[i], distances[i], variables[i]

    # read in unperturbed data sample
    obs = np.load(obsdir+"apk18.npy", allow_pickle=True).tolist()
    pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()

    # # We only test a subset of stars. 
    # # For the rest, there seems to be a disagreement between Galaxia and Kepler.
    # idx = (obs["mass"]<=1.9) & (obs["mass"]>=0.8)
    # for key in obs.keys():
    #     obs[key] = obs[key][idx]

    # idx = (pdv["mass"]<=1.9) & (pdv["mass"]>=0.8)
    # for key in pdv.keys():
    #     pdv[key] = pdv[key][idx]       


    # read in edges
    if var == 'numax':
        bump_obs = np.load(obsdir+"numax_bump.npy")
        bump_pdv = np.load(moddir+"numax_bump.npy")
        xobs, yobs = obs["teff"], obs["numax"]
        e_xobs, e_yobs = obs["e_teff"]/obs["teff"], obs["e_numax"]/obs["numax"]
        xpdv, ypdv = pdv["teff"], pdv["numax"]

    if var == 'dnu':
        bump_obs = np.load(obsdir+"dnu_bump.npy")
        bump_pdv = np.load(moddir+"dnu_bump.npy")
        xobs, yobs = obs["teff"], obs["dnu"]
        e_xobs, e_yobs = obs["e_teff"]/obs["teff"], obs["e_dnu"]/obs["dnu"]
        xpdv, ypdv = pdv["teff"], pdv["dnu"]

    if var in ['mass', 'radius']:
        bump_obs = np.load(obsdir+"mr_bump.npy")
        bump_pdv = np.load(moddir+"mr_bump.npy")
        xobs, yobs = obs["mass"], obs["radius"]
        e_xobs, e_yobs = obs["e_mass"]/obs["mass"], obs["e_radius"]/obs["radius"]
        xpdv, ypdv = pdv["mass"], pdv["radius"]


    hist_model = model_rgb()

    # multiprocessing workflow
    if j==0:
        # trial 1: upper limit
        filepath = rootpath+"sample/rgb/sharpness/mist/"+var+"/ulim/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        rgb_ulim_fit(xobs, yobs, bump_obs, xpdv, ypdv, bump_pdv,
            var, distance, hist_model, filepath, ifmcmc=True, nburn=500, nsteps=1000)
    else:
        # trial 2: lower limit
        eobs = e_xobs if distance=='horizontal' else e_yobs
        filepath = rootpath+"sample/rgb/sharpness/mist/"+var+"/llim/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        rgb_llim_fit(xobs, yobs, eobs, bump_obs, xpdv, ypdv, bump_pdv,
            var, distance, hist_model, filepath, ifmcmc=True, nburn=500, nsteps=1000)


# # multiprocessing workflow
from multiprocessing import Pool
with Pool(8) as p:
    # p.map(loop, [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]])
    p.map(loop, [[0,1],  [1,1],  [2,1],  [3,1]])