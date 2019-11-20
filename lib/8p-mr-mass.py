'''
Post processing for analysing numax on tnu diagram.
'''

rootpath = "/headnode2/yali4742/nike/"

import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.wrapper import plot_slope_scatter, plot_scatter_zvalue

# 1 - horizontal
filepaths = [rootpath+"sample/sharpness/perturb_gif_mr/mass/",
            rootpath+"sample/sharpness/perturb_gif_mr/mass_feh/0/",
            rootpath+"sample/sharpness/perturb_gif_mr/mass_feh/1/",
            rootpath+"sample/sharpness/perturb_gif_mr/mass_feh/2/",
            rootpath+"sample/sharpness/perturb_gif_mr/mass_mass/0/",
            rootpath+"sample/sharpness/perturb_gif_mr/mass_mass/1/",
            rootpath+"sample/sharpness/perturb_gif_mr/mass_mass/2/"]
for filepath in filepaths:
    plot_slope_scatter(filepath, distance="horizontal", diagram="mr")

filepath = rootpath+"sample/sharpness/perturb_gif_mr/mass_feh/"
plot_scatter_zvalue(filepath, distance="horizontal", diagram="mr")
filepath = rootpath+"sample/sharpness/perturb_gif_mr/mass_mass/"
plot_scatter_zvalue(filepath, distance="horizontal", diagram="mr")
