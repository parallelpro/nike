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
from lib.histdist import model3, model4, distfit, distance_to_edge, reduce_samples
import seaborn as sns
from multiprocessing import Pool
import random
import os
from functools import partial


def simulation(imc, Ndata, xp, yp, xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, diagram, distance, hist_model, obj_obs):
    # print(imc, "/", montecarlo)
    fx = (np.array([random.gauss(0,1) for i in range(Ndata)]) * xp + 1)
    fy = (np.array([random.gauss(0,1) for i in range(Ndata)]) * yp + 1)
    xdata, ydata = (xpdv*fx), (ypdv*fy)
    
    dist = distance_to_edge(xdata, ydata, xedge_pdv, yedge_pdv, tck_pdv, diagram=diagram, distance=distance)
    obj = distfit(dist, hist_model, bins=obj_obs.bins)
    obj.fit()
    sharpness_pdv = hist_model.sharpness(obj.para_fit)
    return xdata, ydata, obj, sharpness_pdv

def sharpness_fit(xobso, yobso, zobso, edges_obs, tck_obs, 
        xpdvo, ypdvo, zpdvo, edges_pdv, tck_pdv,
        diagram, distance, hist_model,
        filepath, xperturb, yperturb, montecarlo,
        zvalue_limits=None, zvalue_name=None):
    
    print("zvalue_name:",zvalue_name)
    print("zvalue_limits:",zvalue_limits)
    print("diagram:",diagram)
    print("distance:",distance)

    if (zvalue_limits is None): 
        Nz = 1
    else:
        Nz = len(zvalue_limits[0])

    for iz in range(Nz):
        izn = "{:0.0f}".format(iz)

        # set up observations
        xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
        if not (zvalue_limits is None):
            idx = (zobso >= zvalue_limits[0][iz]) & (zobso < zvalue_limits[1][iz])
            xobs, yobs = xobso[idx], yobso[idx]
        else:
            xobs, yobs = xobso, yobso

        # calculate observational distance
        hdist_obs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, diagram=diagram, distance=distance)
        obj_obs = distfit(hdist_obs, hist_model)
        obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
        obj_obs.fit()
        sharpness_obs = hist_model.sharpness(obj_obs.para_fit)
        Nobs = hdist_obs.shape[0]

        # set up models
        xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
        if not (zvalue_limits is None):
            idx = (zpdvo >= zvalue_limits[0][iz]) & (zpdvo < zvalue_limits[1][iz]) 
            xpdv, ypdv = xpdvo[idx], ypdvo[idx]
        else:
            xpdv, ypdv = xpdvo, ypdvo

        # reduce to same number of points
        idx = reduce_samples(xpdv.shape[0], xobs.shape[0])
        xpdv, ypdv = xpdv[idx], ypdv[idx]

        # filepath
        if (zvalue_limits is None):
            tfilepath = filepath
        else:
            tfilepath = filepath+izn+"/"
        if not os.path.exists(tfilepath): os.mkdir(tfilepath)   

        sharpness_med = np.zeros((xperturb.shape[0], yperturb.shape[0]), dtype=float)
        sharpness_std = np.zeros((xperturb.shape[0], yperturb.shape[0]), dtype=float)

        for ix in range(xperturb.shape[0]):
            for iy in range(yperturb.shape[0]):
                print("{:0.0f}  xscatter: {:0.2f}%, {:0.0f}  yscatter: {:0.2f}%".format(ix, xperturb[ix]*100, iy, yperturb[iy]*100))
                
                # initiate a plot
                fig = plt.figure(figsize=(12,12))
                axes = fig.subplots(nrows=2, ncols=1)
                obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Observations", "zorder":100})
                obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Observations fit", "linestyle":"--", "zorder":100})
                
                # add scatters
                Ndata = xpdv.shape[0]
                # for imc in range(montecarlo):

                fsimulation=partial(simulation, Ndata=Ndata, xp=xperturb[ix], yp=yperturb[iy], 
                        xpdv=xpdv, ypdv=ypdv,
                        xedge_pdv=xedge_pdv, yedge_pdv=yedge_pdv, tck_pdv=tck_pdv,
                        diagram=diagram, distance=distance, hist_model=hist_model, obj_obs=obj_obs)

                pool = Pool(processes=12)
                result = pool.map(fsimulation, np.arange(0,montecarlo).tolist())
                pool.close()
                pool.join()
                xdata, ydata, obj, _ = result[0]
                sharpness_pdv_mcs = [result[i][-1] for i in range(len(result))]

                sharpness_med[ix, iy] = np.median(sharpness_pdv_mcs)
                sharpness_std[ix, iy] = np.std(sharpness_pdv_mcs)/np.sqrt(montecarlo)
 
                obj.plot_hist(ax=axes[0], histkwargs={"color":"blue", "label":"Galaxia"})
                obj.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Galaxia fit"})

                alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
                axes[0].text(0.0, 1.00, "Scatter in x: {:0.2f}%".format(xperturb[ix]*100), **alignment)
                axes[0].text(0.0, 0.95, "Scatter in y: {:0.2f}%".format(yperturb[iy]*100), **alignment)
                axes[0].text(0.0, 0.90, "Galaxia slope: {:0.1f} $\pm$ {:0.1f}".format(sharpness_med[ix, iy], sharpness_std[ix, iy]), **alignment)
                axes[0].text(0.0, 0.85, "Observational slope: {:0.1f}".format(sharpness_obs), **alignment)      
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
                if diagram=="mr":
                    axes[1].axis([0., 3., 5., 20.])
                if diagram=="tnu":
                    axes[1].axis([10, 200, 2.0, 10.0])

                plt.savefig(tfilepath+"{:0.0f}_x_{:0.0f}_y.png".format(ix, iy))
                plt.close()

        # save data
        data = {"xperturb":xperturb, "yperturb":yperturb, "montecarlo":montecarlo,
                "sharpness_med":sharpness_med, "sharpness_std":sharpness_std, "sharpness_obs":sharpness_obs,
                "diagram":diagram, "distance":distance,
                "zvalue_limits": zvalue_limits, "zvalue_name":zvalue_name}
        np.save(tfilepath+"data", data)
