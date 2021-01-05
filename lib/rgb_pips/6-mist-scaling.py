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
from lib.wrapper import rgb_fit
import os

# diagrams = ['tnu', 'tnu', 'mr', 'mr']
distances = ['vertical', 'vertical', 'horizontal', 'vertical']
variables = ['numax', 'dnu', 'mass', 'radius']

# for i in range(2):
def loop(params):
    i, j = params
    # fdnu corrected sharma+2016
    obsdir = rootpath+"sample/rgb/yu/"
    moddir = rootpath+"sample/rgb/mist_nc/"

    distance, var = distances[i], variables[i]

    # read in unperturbed data sample
    obs = np.load(obsdir+"apk18.npy", allow_pickle=True).tolist()
    pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()

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
        xpdv, ypdv = pdv["teff"], pdv["dnu_nc"]

    if var in ['mass', 'radius']:
        bump_obs = np.load(obsdir+"mr_bump.npy")
        bump_pdv = np.load(moddir+"mr_bump.npy")
        xobs, yobs = obs["mass_nc"], obs["radius_nc"]
        e_xobs, e_yobs = obs["e_mass_nc"]/obs["mass_nc"], obs["e_radius_nc"]/obs["radius_nc"]
        xpdv, ypdv = pdv["mass"], pdv["radius"]



    # multiprocessing workflow
    if j==0:
        # trial 1: upper limit
        filepath = rootpath+"sample/rgb/sharpness/mistkb95/"+var+"/ulim/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        rgb_fit(xobs, yobs, bump_obs, xpdv, ypdv, bump_pdv,
            var, distance, filepath)
    else:
        # trial 2: lower limit
        filepath = rootpath+"sample/rgb/sharpness/mistkb95/"+var+"/llim/"
        if not os.path.exists(filepath): os.mkdir(filepath)

        if var in ['dnu', 'numax']:
            rgb_fit(xobs, yobs, bump_obs, xpdv, ypdv, bump_pdv,
                var, distance, filepath, yerror_sample=e_yobs)#xerror_sample=e_xobs, 
        if var=='mass':
            rgb_fit(xobs, yobs, bump_obs, xpdv, ypdv, bump_pdv,
                var, distance, filepath, xerror_sample=e_xobs)
        if var=='radius':
            rgb_fit(xobs, yobs, bump_obs, xpdv, ypdv, bump_pdv,
                var, distance, filepath, yerror_sample=e_yobs)

# # multiprocessing workflow
from multiprocessing import Pool
with Pool(8) as p:
    p.map(loop, [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]])
    # p.map(loop, [[0,1],  [1,1],  [2,1],  [3,1]])