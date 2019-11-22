'''
test whether metric H/sigma or sigma is a better metric to describe to sharpness.
calculate sigma as a function of star number (H), see if it changes.

Calculate the scatter inside radius.
'''

rootpath = "/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.histdist import model4
from lib.wrapper import sharpness_fit, reduce_samples
import os

# read in unperturbed mass and radius, with edges
tnu_samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy")
mr_edges_obs = np.load(rootpath+"sample/obs/mr_edge_samples.npy")
tck_obs = np.load(rootpath+"sample/obs/mr_spline_tck.npy", allow_pickle=True)
# read in unperturbed mass and radius, with edges
tnu_samples_pdv = np.load(rootpath+"sample/padova_oversampling/tnu_samples.npy")
mr_edges_pdv = np.load(rootpath+"sample/padova_oversampling/mr_edge_samples.npy")
tck_pdv = np.load(rootpath+"sample/padova_oversampling/mr_spline_tck.npy", allow_pickle=True)

distance = "vertical"
diagram = "mr"
hist_model = model4()

xperturb = np.arange(0.00, 0.05, 0.1)
yperturb = np.arange(0.00, 0.05, 0.002) # np.arange(0., 0.05, 0.1)
montecarlo = 120


# trial 1: no binning
xobs, yobs = tnu_samples_obs[:,3], tnu_samples_obs[:,4]
idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
xobs, yobs = xobs[idx], yobs[idx]

xpdv, ypdv = tnu_samples_pdv[:,3], tnu_samples_pdv[:,4]
idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
xpdv, ypdv= xpdv[idx], ypdv[idx]


for i in range(3):
    Ndata = xobs.shape[0]*(i+1)
    filepath = rootpath+"sample/test_metric/metric1/"+str(i)+"/"
    if not os.path.exists(filepath): os.mkdir(filepath)

    sharpness_fit(xobs, yobs, xobs, mr_edges_obs, tck_obs,
            xpdv, ypdv, xpdv, mr_edges_pdv, tck_pdv,
            diagram, distance, hist_model,
            filepath, xperturb, yperturb, montecarlo, Ndata=Ndata)

    filepath = rootpath+"sample/test_metric/metric2/"+str(i)+"/"
    if not os.path.exists(filepath): os.mkdir(filepath)

    sharpness_fit(xobs, yobs, xobs, mr_edges_obs, tck_obs,
            xpdv, ypdv, xpdv, mr_edges_pdv, tck_pdv,
            diagram, distance, hist_model,
            filepath, xperturb, yperturb, montecarlo, Ndata=Ndata)

