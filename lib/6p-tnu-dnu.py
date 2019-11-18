'''
Post processing for analysing dnu on tnu diagram.
'''

rootpath = "/headnode2/yali4742/nike/"

import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.wrapper import plot_slope_scatter, plot_scatter_zvalue

# 2 - vertical
filepaths = [rootpath+"sample/sharpness/perturb_gif_tnu/dnu/",
            rootpath+"sample/sharpness/perturb_gif_tnu/dnu_feh/0/",
            rootpath+"sample/sharpness/perturb_gif_tnu/dnu_feh/1/",
            rootpath+"sample/sharpness/perturb_gif_tnu/dnu_feh/2/",
            rootpath+"sample/sharpness/perturb_gif_tnu/dnu_mass/0/",
            rootpath+"sample/sharpness/perturb_gif_tnu/dnu_mass/1/",
            rootpath+"sample/sharpness/perturb_gif_tnu/dnu_mass/2/"]
for filepath in filepaths:
    plot_slope_scatter(filepath, distance="vertical", diagram="tnu")

filepath = rootpath+"sample/sharpness/perturb_gif_tnu/dnu_feh/"
plot_scatter_zvalue(filepath, distance="vertical", diagram="tnu")
filepath = rootpath+"sample/sharpness/perturb_gif_tnu/dnu_mass/"
plot_scatter_zvalue(filepath, distance="vertical", diagram="tnu")
