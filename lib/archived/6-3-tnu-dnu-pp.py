'''
This is just a test case to measure the shortest distance to the edge on nike diagram.

We all know it is not ideal because manipulating mass will also change its vertical position
on nike diagram, therefore the horizontal distance is not completely dependent on mass.

I only aim to improve my algorithms in this program:
1) automate the program to guess priors and initial paramters;
2) incorporate a monte-carlo trial to reasonably reduce scatter.

'''

rootpath = "/headnode2/yali4742/nike/"

import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.histdist import model3, model4, distfit, distance_to_edge
import seaborn as sns

errorbarkwargs = {"elinewidth":1, "capsize":2, "ecolor":"black"}


def wrapper(filepath):
    # read data
    data = np.load(filepath+"data_dnu.npy").tolist()
    numax_scatter = data["numax_perturb"][0]
    dnu_scatter = data["dnu_perturb"]
    slope = data["sharpness_med"][0,:]
    eslope = data["sharpness_std"][0,:]
    slope_obs = data["sharpness_obs"]

    # initiate a plot
    fig = plt.figure(figsize=(6,4))
    axes = fig.subplots(nrows=1, ncols=1, squeeze=False).reshape(-1,)
    x = dnu_scatter*100
    y = slope-slope_obs
    axes[0].errorbar(x, y, yerr=eslope, fmt="k.", **errorbarkwargs)
    axes[0].axhline(0, c="r", ls="--")
    axes[0].axis([x.min(), x.max(), -max(np.abs(y.min()), np.abs(y.max()))*1.2 , max(np.abs(y.min()), np.abs(y.max()))*1.2])

    idx = np.abs(slope-slope_obs) == np.abs(slope-slope_obs).min()
    axes[0].axvline(dnu_scatter[idx][0]*100, c="r", ls="--")
    axes[0].set_title("Fitted scatter in dnu: {:0.2f}%, scatter in numax: {:0.2f}%".format(dnu_scatter[idx][0]*100, numax_scatter*100))

    axes[0].set_xlabel("Scatter in dnu relation [%]")
    axes[0].set_ylabel("Slope(galaxia) - Slope(obs)")
    plt.tight_layout()
    plt.savefig(filepath+"slope_scatter.png")
    plt.close()


# 1 - dnu - vertical
filepath = rootpath+"sample/sharpness/perturb_gif_tnu/dnu/"
wrapper(filepath)
