
rootpath =  "/mnt/c/Users/yali4742/Dropbox (Sydney Uni)/Work/nike/"

import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.histdist import model6, distfit, distance_to_edge, reduce_samples, display_bar
import seaborn as sns
from multiprocessing import Pool
import random
import os
from functools import partial
import scipy.signal
import scipy.special
import emcee
import corner
from multiprocessing import Pool
from scipy.optimize import basinhopping


# fdnu corrected sharma+2016
obsdir = rootpath+"sample/yu/"
moddir = rootpath+"sample/padova/"

# read in unperturbed mass and radius, with edges
obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
edges_obs = np.load(obsdir+"tnu_edge_samples.npy")
tck_obs, tp_obs = np.load(obsdir+"nike_spline_tck.npy", allow_pickle=True)

# read in unperturbed mass and radius, with edges
pdv = np.load(moddir+"padova.npy", allow_pickle=True).tolist()
edges_pdv = np.load(moddir+"tnu_edge_samples.npy")
tck_pdv, tp_pdv = np.load(moddir+"nike_spline_tck.npy", allow_pickle=True)

# to exclude those points which lies below the edge (so no horizontal distance).
idx = obs["dnu"]>=np.min(edges_obs[:,1])
for key in obs.keys():
        obs[key] = obs[key][idx]

idx = pdv["dnu"]>=np.min(edges_pdv[:,1])
for key in pdv.keys():
        pdv[key] = pdv[key][idx]

distance = "horizontal"
diagram = "tnu"
hist_model = model6()

montecarlo = 120

ifmcmc=False

# trial 2: mcmc fit
xobs, yobs = obs["numax"], obs["dnu"]
e_xobs, e_yobs = obs["e_numax"], obs["e_dnu"]
e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
# idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
# xobs, yobs = xobs[idx], yobs[idx]
# e_xobs, e_yobs = e_xobs[idx], e_yobs[idx]

xpdv, ypdv = pdv["numax"], pdv["dnu"]
# idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
# xpdv, ypdv= xpdv[idx], ypdv[idx]

filepath = rootpath+"sample/sharpness/sharma16/numax/perturb_mcmc_test/"
if not os.path.exists(filepath): os.mkdir(filepath)


print("diagram:",diagram)
print("distance:",distance)

# filepath
tfilepath = filepath

# set up observations
xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
# xobs, yobs, e_xobs, e_yobs = xobso, yobso, e_xobso, e_yobso

# set up models
xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
# xpdv, ypdv = xpdvo, ypdvo


# calculate Kepler distance
hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
obj_obs = distfit(hdist_obs, hist_model)
obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
obj_obs.fit(ifmcmc=False)
obj_obs.output(tfilepath, ifmcmc=False)

# calculate Galaxia distance
hdist_pdv, xdata, ydata = distance_to_edge(xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
obj_pdv = distfit(hdist_pdv, hist_model, bins=obj_obs.bins)
obj_pdv.fit(ifmcmc=False)
obj_pdv.output(tfilepath, ifmcmc=False)


# run mcmc with ensemble sampler
ndim, nwalkers, nburn, nsteps = 2, 200, 200, 100

para_names = ["shift", "scatter"]
xc = obj_obs.para_fit[1]
para_limits = [[-5.0*1, 5.0*1], [0., 0.1]]
para_guess = [xc, 0.001]


def model(theta):#, obj_obs, xpdv, ypdv):
    # tied to model6
    weight = np.zeros(obj_obs.histx.shape, dtype=bool)
    sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
    idx = (obj_obs.histx <= x0+sigma) & (obj_obs.histx >= x0-2*sigma)
    weight[idx] = True

    # theta[0]: offset in distance
    # theta[1]: perturb

    Ndata = xpdv.shape[0]

    if (e_xobs is None):
        fx = np.zeros(Ndata)
    else:
        # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
        fx2 = np.array([random.gauss(0,1) for i in range(Ndata)]) * theta[1]
        fx = fx2
    if (e_yobs is None):
        fy = np.zeros(Ndata)
    else:
        # fy1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_yobs), Ndata) * scalar
        fy2 = np.array([random.gauss(0,1) for i in range(Ndata)]) * theta[1]
        fy = fy2

    # disturb with artificial scatter
    xdata, ydata = (xpdv + xpdv*(fx)), (ypdv + ypdv*(fy))

    hdist, xdata, ydata = distance_to_edge(xdata, ydata, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    hdist = hdist + theta[0]
    obj = distfit(hdist, hist_model, bins=obj_obs.bins)

    # normalize the number of points in the weighted region
    histy = obj.histy / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
    return histy, weight, xdata, ydata

def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
    histy, weight, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
    d, m = obj_obs.histy[weight], histy[weight]
    if m[m==0.].shape[0] != 0:
        return -np.inf 
    else:
        logfact = scipy.special.gammaln(d+1)#np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
        lnL =  np.sum(d*np.log(m)-m-logfact)
        return lnL

def minus_lnlikelihood(theta):
    return -lnlikelihood(theta)

def lnprior(theta):#, para_limits):
    for i in range(len(theta)):
        if not (para_limits[i][0] <= theta[0] <= para_limits[i][1] ):
            return -np.inf
    return 0.

def lnpost(theta):#, para_limits, obj_obs, xpdv, ypdv):
    lp = lnprior(theta)#, para_limits)
    if np.isfinite(lp):
        lnL = lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
        return lnL
    else:
        return -np.inf


if ifmcmc:
    print("enabling Ensemble sampler.")
    # pos0=[para_guess + 1.0e-7*np.random.randn(ndim) for j in range(nwalkers)]
    pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool)#, args=(para_limits, obj_obs, xpdv, ypdv))

        # # burn-in
        print("start burning in. nburn:", nburn)
        for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
                display_bar(j, nburn)
                pass
        sys.stdout.write("\n")
        pos, _, _ = result
        sampler.reset()

        # actual iteration
        print("start iterating. nsteps:", nsteps)
        for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
                display_bar(j, nsteps)
                pass
        sys.stdout.write("\n")

    # modify samples
    samples = sampler.chain[:,:,:].reshape((-1,ndim))

    # save estimation result
    # 16, 50, 84 quantiles
    result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
    para_fit = result[:,0]
    e_para_fit = (result[:,1]+result[:,2])/2.0

    # corner plot
    fig = corner.corner(samples, labels=para_names, show_titles=True)
    plt.savefig(tfilepath+"corner.png")

else:
    res = basinhopping(minus_lnlikelihood, para_guess, minimizer_kwargs={"bounds":para_limits})
    para_fit = res.x
    e_para_fit = None

# result plot
fig = plt.figure(figsize=(12,12))
axes = fig.subplots(nrows=2, ncols=1)
obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Observations", "zorder":100})
obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Observations fit", "linestyle":"--", "zorder":100})
        
obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})
# obj_pdv.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Galaxia fit"})

yfit, _, xdata, ydata = model(para_fit)#, obj_obs, xpdv, ypdv)
xfit = obj_obs.histx
axes[0].step(xfit, yfit, **{"color":"blue", "label":"Galaxia best fit"})
axes[0].plot(xfit, yfit, **{"color":"black", "label":"Galaxia best fit"})

alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
axes[0].text(0.0, 1.00, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
axes[0].text(0.0, 0.95, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment)
# axes[0].text(0.0, 0.90, "Galaxia sigma: {:0.3f} $\pm$ {:0.3f}".format(sharpness_med[ix, iy][0], sharpness_std[ix, iy][0]), **alignment)
# axes[0].text(0.0, 0.85, "Kepler sigma: {:0.3f} $\pm$ {:0.3f}".format(sharpness_obs[0], e_sharpness_obs[0]), **alignment)      
# axes[0].text(0.0, 0.80, "Galaxia slope: {:0.3f} $\pm$ {:0.3f}".format(sharpness_med[ix, iy][1], sharpness_std[ix, iy][1]), **alignment)
# axes[0].text(0.0, 0.75, "Kepler slope: {:0.3f} $\pm$ {:0.3f}".format(sharpness_obs[1], e_sharpness_obs[1]), **alignment)      
# axes[0].legend()
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
    axes[1].axis([10, 150, 2.0, 10.0])

plt.savefig(tfilepath+"result.png")
plt.close()

# save data
data = {"sample":samples, "ndim":ndim, "result":result,
        "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
        "diagram":diagram, "distance":distance, 
        "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv,
        "xdata":xdata, "ydata":ydata, "obj_obs":obj_obs, "obj_pdv":obj_pdv}

np.save(tfilepath+"data", data)