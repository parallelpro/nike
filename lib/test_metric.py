'''
test whether metric H/sigma or sigma is a better metric to describe to sharpness.
calculate sigma as a function of star number (H), see if it changes.

Calculate the scatter inside radius.
'''

rootpath = ""#'\\c\\Users\\yali4742\\Dropbox (Sydney Uni)\\Work\\nike'.replace("\\","/")
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import sys
# sys.path.append(rootpath) 
from lib.histdist import model4, model5
from lib.wrapper import sharpness_fit, reduce_samples
import os

# # # running:

# if __name__ == "__main__":
#         # read in unperturbed mass and radius, with edges
#         tnu_samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy")
#         mr_edges_obs = np.load(rootpath+"sample/obs/mr_edge_samples.npy")
#         tck_obs = np.load(rootpath+"sample/obs/mr_spline_tck.npy", allow_pickle=True)
#         # read in unperturbed mass and radius, with edges
#         tnu_samples_pdv = np.load(rootpath+"sample/padova_oversampling/tnu_samples.npy")
#         mr_edges_pdv = np.load(rootpath+"sample/padova_oversampling/mr_edge_samples.npy")
#         tck_pdv = np.load(rootpath+"sample/padova_oversampling/mr_spline_tck.npy", allow_pickle=True)

#         distance = "vertical"
#         diagram = "mr"
#         slope_model = model4()
#         sigma_model = model5()

#         xperturb = np.arange(0.00, 0.05, 0.1)
#         yperturb = np.arange(0.00, 0.05, 0.002) # np.arange(0., 0.05, 0.1)
#         montecarlo = 120


#         # trial 1: no binning
#         xobs, yobs = tnu_samples_obs[:,3], tnu_samples_obs[:,4]
#         idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
#         xobs, yobs = xobs[idx], yobs[idx]

#         xpdv, ypdv = tnu_samples_pdv[:,3], tnu_samples_pdv[:,4]
#         idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
#         xpdv, ypdv= xpdv[idx], ypdv[idx]


#         for i in range(3):

#                 Ndata = xobs.shape[0]*(i+1)

#                 # filepath = rootpath+"sample/test_metric/metric_slope/"+str(i)+"/"
#                 # if not os.path.exists(filepath): os.mkdir(filepath)

#                 # sharpness_fit(xobs, yobs, xobs, mr_edges_obs, tck_obs,
#                 #         xpdv, ypdv, xpdv, mr_edges_pdv, tck_pdv,
#                 #         diagram, distance, slope_model,
#                 #         filepath, xperturb, yperturb, montecarlo, Ndata=Ndata, cores=6)

#                 filepath = rootpath+"sample/test_metric/metric_sigma/"+str(i)+"/"
#                 if not os.path.exists(filepath): os.mkdir(filepath)

#                 sharpness_fit(xobs, yobs, xobs, mr_edges_obs, tck_obs,
#                         xpdv, ypdv, xpdv, mr_edges_pdv, tck_pdv,
#                         diagram, distance, sigma_model,
#                         filepath, xperturb, yperturb, montecarlo, Ndata=Ndata, cores=6)

# # # postprocessing - plot

# initiate a plot
fig = plt.figure(figsize=(10,4))
axes = fig.subplots(nrows=1, ncols=2, squeeze=False).reshape(-1,)

distance = "vertical"
diagram = "mr"
filepaths = ["sample/test_metric/metric_sigma/"+str(i)+"/" for i in range(3)]
filepaths2 = ["sample/test_metric/metric_slope/"+str(i)+"/" for i in range(3)]
tnu_samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy", allow_pickle=True)
Nobs = tnu_samples_obs.shape[0]
colors = ["red", "blue", "green"]

for i in range(len(filepaths)):
        # read data
        data = np.load(filepaths[i]+"data.npy", allow_pickle=True).tolist()
        scatter = data["yperturb"]
        cp_scatter = data["xperturb"][0]
        slope = data["sharpness_med"][0,:]
        eslope = data["sharpness_std"][0,:]
        slope_obs = data["sharpness_obs"]

        errorbarkwargs = {"elinewidth":1, "capsize":2, "ecolor":"black"}
        x, y = scatter*100, slope-slope_obs
        ey = eslope
        axes[0].errorbar(x, y, yerr=ey, fmt=".", color=colors[i], label="Nsize={:0.0f}".format(Nobs*(i+1)), **errorbarkwargs)
        axes[0].axhline(0, c="r", ls="--")

        idx = np.abs(y) == np.abs(y).min()
        axes[0].axvline(x[idx][0], c="r", ls="--")
           
        # axes[0].set_title("Scatter in R: {:0.2f}%, scatter in M: {:0.2f}%".format(x[idx][0], cp_scatter))
        axes[0].set_xlabel("Scatter in R relation [%]") 
        axes[0].set_ylabel("Sharpness(galaxia) - Sharpness(obs)")


        # read data
        data = np.load(filepaths2[i]+"data.npy", allow_pickle=True).tolist()
        scatter = data["yperturb"]
        cp_scatter = data["xperturb"][0]
        slope = data["sharpness_med"][0,:]
        eslope = data["sharpness_std"][0,:]
        slope_obs = data["sharpness_obs"]

        errorbarkwargs = {"elinewidth":1, "capsize":2, "ecolor":"black"}
        x, y = scatter*100, slope-slope_obs
        ey = eslope
        axes[1].errorbar(x, y, yerr=ey, fmt=".", color=colors[i], label="Nsize={:0.0f}".format(Nobs*(i+1)), **errorbarkwargs)
        axes[1].axhline(0, c="r", ls="--")
        idx = np.abs(y) == np.abs(y).min()
        axes[1].axvline(x[idx][0], c="r", ls="--")
        
        # axes[0].set_title("Scatter in R: {:0.2f}%, scatter in M: {:0.2f}%".format(x[idx][0], cp_scatter))
        axes[1].set_xlabel("Scatter in R relation [%]") 
        axes[1].set_ylabel("Sharpness(galaxia) - Sharpness(obs)")

axes[0].axis([0., 4.8, -0.2,  0.4])
axes[1].axis([0., 4.8, -2000, 16000.])
axes[1].set_title("Metric: H/sigma")
axes[0].set_title("Metric: sigma")
axes[1].legend()
axes[0].legend()
plt.tight_layout()
plt.savefig("sample/test_metric/test_metric.png")
plt.close()

