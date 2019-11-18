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
filepaths = [rootpath+"sample/sharpness/perturb_gif_mr/radius/",
            rootpath+"sample/sharpness/perturb_gif_mr/radius_feh/0/",
            rootpath+"sample/sharpness/perturb_gif_mr/radius_feh/1/",
            rootpath+"sample/sharpness/perturb_gif_mr/radius_feh/2/",
            rootpath+"sample/sharpness/perturb_gif_mr/radius_mass/0/",
            rootpath+"sample/sharpness/perturb_gif_mr/radius_mass/1/",
            rootpath+"sample/sharpness/perturb_gif_mr/radius_mass/2/"]
for filepath in filepaths:
    plot_slope_scatter(filepath, distance="vertical", diagram="mr")

filepath = rootpath+"sample/sharpness/perturb_gif_mr/radius_feh/"
plot_scatter_zvalue(filepath, distance="vertical", diagram="mr")
filepath = rootpath+"sample/sharpness/perturb_gif_mr/radius_mass/"
plot_scatter_zvalue(filepath, distance="vertical", diagram="mr")
