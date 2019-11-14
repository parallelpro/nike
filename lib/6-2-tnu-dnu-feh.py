'''
This is to measure the horizontal/vertical scatter on dnu-numax diagram, 
without manipulating anything else (for example to see as a function of
mass/metallicity).

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
from multiprocessing import Pool
import random
import os

distance = "vertical"
diagram = "tnu"
hist_model = model4()


# read in unperturbed dnu and numax, with edges
tnu_samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy")
tnu_edges_obs = np.load(rootpath+"sample/obs/tnu_edge_samples.npy")
tck_obs = np.load(rootpath+"sample/obs/spline_tck.npy", allow_pickle=True)
# read in unperturbed dnu and numax, with edges
tnu_samples_pdv = np.load(rootpath+"sample/padova/tnu_samples.npy")
tnu_edges_pdv = np.load(rootpath+"sample/padova/tnu_edge_samples.npy")
tck_pdv = np.load(rootpath+"sample/padova/spline_tck.npy", allow_pickle=True)


feh_limits = [-3.0, -0.20, 0.02, 1.0]

for ifeh in range(len(feh_limits)-1):
    feh = "{:0.0f}".format(ifeh)

    # set up observations
    # calculate observational distance
    ycut = 0.1

    xedge_obs, yedge_obs = tnu_edges_obs[:,0], tnu_edges_obs[:,1]
    idx = (xedge_obs**0.75/yedge_obs>=ycut) #& (yedge<=3.5)
    xedge_obs, yedge_obs = xedge_obs[idx], yedge_obs[idx]

    xobs, yobs, feh_obs = tnu_samples_obs[:,0], tnu_samples_obs[:,1], tnu_samples_obs[:,2]
    idx = (xobs**0.75/yobs>=ycut) & (xobs >= np.min(xedge_obs))#& (yobs<=3.5)
    idx = idx & (feh_obs >= feh_limits[ifeh]) & (feh_obs < feh_limits[ifeh+1])
    xobs, yobs, feh_obs = xobs[idx], yobs[idx], feh_obs[idx]


    hdist_obs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, diagram=diagram, distance=distance)
    obj_obs = distfit(hdist_obs, hist_model)
    obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
    obj_obs.fit()
    sharpness_obs = hist_model.sharpness(obj_obs.para_fit)
    Nobs = hdist_obs.shape[0]

    # set up models
    xedge_pdv, yedge_pdv = tnu_edges_pdv[:,0], tnu_edges_pdv[:,1]
    idx = (xedge_pdv**0.75/yedge_pdv>=ycut) #& (yedge<=3.5)
    xedge_pdv, yedge_pdv = xedge_pdv[idx], yedge_pdv[idx]

    numax, dnu, feh_pdv = tnu_samples_pdv[:,0], tnu_samples_pdv[:,1], tnu_samples_pdv[:,2]
    idx = (numax**0.75/dnu>=ycut) & (numax >= np.min(xedge_pdv)) #& (numax**0.75/dnu<=3.5)
    idx = idx & (feh_pdv >= feh_limits[ifeh]) & (feh_pdv < feh_limits[ifeh+1])
    dnu, numax, feh_pdv = dnu[idx], numax[idx], feh_pdv[idx]



    # # # customization
    # step 1: deal with numax
    filepath = rootpath+"sample/sharpness/perturb_gif_tnu/dnu_feh/dnu_"+feh+"/"
    if not os.path.exists(filepath): os.mkdir(filepath)
    output = filepath+"data_dnu_feh_"+feh
    montecarlo = 120
    numax_perturb = np.arange(0.00, 0.050, 0.1) # np.arange(0.00, 0.06, 0.11)#
    dnu_perturb = np.arange(0.00, 0.02, 0.001) #np.arange(0., 0.05, 0.1)
    # # # end of customization


    sharpness_med = np.zeros((numax_perturb.shape[0], dnu_perturb.shape[0]), dtype=float)
    sharpness_std = np.zeros((numax_perturb.shape[0], dnu_perturb.shape[0]), dtype=float)

    for inumax in range(numax_perturb.shape[0]):
        for idnu in range(dnu_perturb.shape[0]):
            print(inumax, " numax: ", numax_perturb[inumax], " , ", idnu, " dnu:", dnu_perturb[idnu])
            
            # initiate a plot
            fig = plt.figure(figsize=(12,12))
            axes = fig.subplots(nrows=2, ncols=1)
            obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Observations", "zorder":1})
            obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Observations fit", "linestyle":"--", "zorder":1})

            # add perturbations
            Ndata = numax.shape[0]
            # sharpness_pdv_mcs = np.zeros(montecarlo)

            # for imc in range(montecarlo):
            def simulation(imc):
                # print(imc, "/", montecarlo)
                fnumax = (np.array([random.gauss(0,1) for i in range(Ndata)]) * numax_perturb[inumax] + 1)
                fdnu = (np.array([random.gauss(0,1) for i in range(Ndata)]) * dnu_perturb[idnu] + 1)
                xdata, ydata = (numax*fnumax), (dnu*fdnu)
                
                dist = distance_to_edge(xdata, ydata, xedge_pdv, yedge_pdv, tck_pdv, diagram=diagram, distance=distance)
                obj = distfit(dist, hist_model, bins=obj_obs.bins)
                obj.fit()
                sharpness_pdv = hist_model.sharpness(obj.para_fit)
                return xdata, ydata, obj, sharpness_pdv

            
            pool = Pool(processes=12)
            result = pool.map(simulation, np.arange(0,montecarlo).tolist())
            xdata, ydata, obj, _ = result[0]
            sharpness_pdv_mcs = [result[i][-1] for i in range(len(result))]

            sharpness_med[inumax, idnu] = np.median(sharpness_pdv_mcs)
            sharpness_std[inumax, idnu] = np.std(sharpness_pdv_mcs)

            # obj.histy = obj.histy/(Ndata/Nobs)  
            obj.plot_hist(ax=axes[0], histkwargs={"color":"blue", "label":"Galaxia", "zorder":1})
            obj.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Galaxia fit", "zorder":1})

            
            axes[0].text(0.9, 0.6, "Galaxia slope: {:0.1f} $\pm$ {:0.1f}".format(sharpness_med[inumax, idnu], sharpness_std[inumax, idnu]), ha="right", va="top", transform=axes[0].transAxes, zorder=10)
            axes[0].text(0.9, 0.55, "Observational slope: {:0.1f}".format(sharpness_obs), ha="right", va="top", transform=axes[0].transAxes, zorder=10)
            axes[0].set_title("Scatter in dnu: {:0.2f}%, Scatter in numax: {:0.2f}%".format(dnu_perturb[idnu]*100,numax_perturb[inumax]*100))   
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
            axes[0].set_ylim(0., obj_obs.histy.max()*1.5)


            # diagram axes[1]
            axes[1].plot(xobs, yobs, "r.", ms=1)
            axes[1].plot(xdata, ydata, "b.", ms=1)
            axes[1].plot(xedge_obs, yedge_obs, "k--")
            axes[1].plot(xedge_pdv, yedge_pdv, "k-")

            axes[1].grid(True)
            axes[1].axis([10, 200, 2.0, 10.0])
            axes[1].set_xscale("log")


            plt.savefig(filepath+"{:0.0f}_numax_{:0.0f}_dnu.png".format(inumax, idnu))
            plt.close()

    # save data
    data = {"numax_perturb":numax_perturb, "dnu_perturb":dnu_perturb, "montecarlo":montecarlo,
            "sharpness_med":sharpness_med, "sharpness_std":sharpness_std, "sharpness_obs":sharpness_obs}
    np.save(output, data)





