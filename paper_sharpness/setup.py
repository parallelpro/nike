from matplotlib import rcParams
rcParams["figure.dpi"] = 100
rcParams["savefig.dpi"] = 100

import numpy as np
import corner
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from lightkurve import search_lightcurvefile
import lightkurve as lk
import seaborn as sns
import os 
from matplotlib.colors import ListedColormap
import sys

# color stuff
red = sns.xkcd_rgb["pale red"]
blue = sns.xkcd_rgb["denim blue"]
green = sns.xkcd_rgb["faded green"]
orange = sns.xkcd_rgb["amber"]
grey = sns.xkcd_rgb["greyish"]
purple = sns.xkcd_rgb["dusty purple"]
black = sns.xkcd_rgb["black"]

def cmap_diverging(n=10):
    return ListedColormap(sns.diverging_palette(220, 20, n=n))

def cmap_grey(n=10):
    return ListedColormap(sns.color_palette("Greys", n))

def blues(n=10):
    return sns.color_palette("Blues", n)

if os.name == 'nt':
    rootpath = '/Users/yali4742/'
else:
    rootpath = '/Volumes/Data/'
overleaf_path = '/Users/yaguang/Dropbox (Sydney Uni)/Apps/Overleaf/Yaguang_SharpRedGiants/figs/'
work_path = rootpath+'Onedrive/Work/nike/'
sys.path.append(work_path)

matplotlib.rcParams["font.size"] = 7.0#7.5
matplotlib.rcParams["legend.fontsize"] = 5.5#7.5
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
#plt.rc('font', family='serif')
matplotlib.rcParams['xtick.labelsize'] = 7.0#7
matplotlib.rcParams['ytick.labelsize'] = 7.0#7
matplotlib.rcParams['ytick.direction']='out'
matplotlib.rcParams['ytick.major.size']=5.0
matplotlib.rcParams['ytick.minor.size']=3.0
matplotlib.rcParams['xtick.direction']='out'
matplotlib.rcParams['xtick.major.size']=5.0
matplotlib.rcParams['xtick.minor.size']=3.0


# mnras size in pt
columnwidth = 240
textwidth = 504

def mnras_size(column="one", square=False, ratio=None):
    # Thanks Dan!
    # Parameters:
    # column: "one" or "double"
    # square: True or False
    # ratio: height/width

    inches_per_pt = 1.0/72.00              # Convert pt to inches
    golden_mean = (np.sqrt(5)-1.0)/2.0     # Most aesthetic ratio
    if (ratio is None): ratio = golden_mean
    if (column is "one"):
        fig_width_pt = columnwidth
    elif (column is "double"):
        fig_width_pt = textwidth
    else:
        raise ValueError("column should be one of ``one'' or ``double''. ")
    fig_width = fig_width_pt*inches_per_pt # Figure width in inches
    if square:
        fig_height = fig_width
    else:
        fig_height = fig_width*ratio
    return [fig_width,fig_height]

errstyle = {'capsize':2, 'ecolor':grey, 'elinewidth':1, 'capthick':1, 'linestyle':'None'}