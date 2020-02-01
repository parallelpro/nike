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
from lib.histdist import distfit, distance_to_edge, reduce_samples, display_bar
import seaborn as sns
from multiprocessing import Pool
import random
import os
from functools import partial
import scipy.signal
import scipy.special
import emcee
import corner
from scipy.optimize import basinhopping, minimize


# def simulation_obs(imc, Ndata, xerr, yerr, xobs, yobs, xedge_obs, yedge_obs, tck_obs, diagram, distance, hist_model, obj_obs):
#     # print(imc, "/", montecarlo)
#     sx = (np.array([random.gauss(0,1) for i in range(Ndata)]) * xerr)
#     sy = (np.array([random.gauss(0,1) for i in range(Ndata)]) * yerr)
#     xdata, ydata = (xobs+sx), (yobs+sy)
    
#     dist = distance_to_edge(xdata, ydata, xedge_obs, yedge_obs, tck_obs, diagram=diagram, distance=distance)
#     obj = distfit(dist, hist_model, bins=obj_obs.bins)
#     obj.fit()
#     sharpness_obs = hist_model.sharpness(obj.para_fit)
#     return xdata, ydata, obj, sharpness_obs

def simulation(imc, Ndata, xp, yp, scalar, xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram, distance, hist_model, obj_obs, e_xobs, e_yobs):
    # print(imc, "/", montecarlo)
    
    # disturb with observational error distribution
    if (e_xobs is None):
        fx1 = np.zeros(Ndata)
    else:
        fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
    if (e_yobs is None):
        fy1 = np.zeros(Ndata)
    else:
        fy1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_yobs), Ndata) * scalar
    
    # disturb with artificial scatter
    fx2 = np.array([random.gauss(0,1) for i in range(Ndata)]) * xp
    fy2 = np.array([random.gauss(0,1) for i in range(Ndata)]) * yp
    xdata, ydata = (xpdv + xpdv*(fx1+fx2)), (ypdv + ypdv*(fy1+fy2))
    
    dist, xdata, ydata = distance_to_edge(xdata, ydata, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    obj = distfit(dist, hist_model, bins=obj_obs.bins)
    obj.fit()
    sharpness_pdv = hist_model.sharpness(obj.para_fit)
    return xdata, ydata, obj, sharpness_pdv


def sharpness_fit_perturb_mcmc(xobso, yobso, edges_obs, tck_obs, tp_obs,
        xpdvo, ypdvo, edges_pdv, tck_pdv, tp_pdv,
        diagram, distance, hist_model, filepath, ifmcmc=False):
    
    print("diagram:",diagram)
    print("distance:",distance)

    global model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost, obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight
    global fp
    # filepath
    tfilepath = filepath

    # set up observations
    xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
    xobs, yobs = xobso, yobso

    # set up models
    xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
    xpdv, ypdv = xpdvo, ypdvo


    # calculate Kepler distance
    hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
    obj_obs = distfit(hdist_obs, hist_model)
    obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
    obj_obs.fit(ifmcmc=False)

    # calculate Galaxia distance
    hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    obj_pdv = distfit(hdist_pdv, hist_model, bins=obj_obs.bins)
    obj_pdv.fit(ifmcmc=False)

    # run mcmc with ensemble sampler
    ndim, nwalkers, nburn, nsteps = 2, 200, 2000, 1000

    para_names = ["shift", "scatter"]
    xd = np.abs(obj_obs.para_fit[1]-obj_pdv.para_fit[1])
    para_limits = [[-3*xd, 3*xd], [0., 0.20]]
    para_guess = [0., 0.005]

    Ndata = xpdv.shape[0]

    # tied to model6
    weight = np.zeros(obj_obs.histx.shape, dtype=bool)
    sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
    idx = (obj_obs.histx >= x0-3*sigma) & (obj_obs.histx <= x0+3*sigma)
    weight[idx] = True

    if distance=="vertical":
        fy2_base = np.random.normal(size=Ndata)
        fp = ypdv*fy2_base
    else:
        # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
        # "horizontal"
        fx2_base = np.random.normal(size=Ndata)
        fp = xpdv*fx2_base
   

    def model(theta):#, obj_obs, xpdv, ypdv):

        # theta[0]: offset in distance
        # theta[1]: perturb

        # disturb with artificial scatter
        # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

        hdist = hdist_pdv + fp*theta[1]
        hdist = hdist + theta[0]
        obj = distfit(hdist, hist_model, bins=obj_obs.bins)

        # normalize the number of points in the weighted region
        if np.sum(obj.histy[weight])!=0:
            number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
        else:
            number_reduction_factor = 0.
        histy = obj.histy * number_reduction_factor
        return histy, hdist, number_reduction_factor

    def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
        histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
        d, m = obj_obs.histy[weight], histy[weight]
        if m[m==0].shape[0] == 0:
            logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
            lnL =  np.sum(d*np.log(m)-m-logdfact)
            # print(theta, lnL)
            return lnL
        else:
            # print(theta, "d")
            return -np.inf

    def minus_lnlikelihood(theta):
        return -lnlikelihood(theta)

    def lnprior(theta):#, para_limits):
        for i in range(len(theta)):
            if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
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
        print(para_guess)
        # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
        pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

        with Pool() as pool:
        # if True:
            # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
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

        # estimation result
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
        # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
        para_fit = res.x
        print(para_guess, para_fit)
        e_para_fit = None
        samples = None


    # result plot
    fig = plt.figure(figsize=(12,12))
    axes = fig.subplots(nrows=2, ncols=1)
    obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
    obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
    # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

    # calculate best fitted results
    yfit, hdist, number_reduction_factor = model(para_fit)
    Ndata = hdist.shape[0]
    idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

    if distance=="vertical":
        xdata = xpdv
        ydata = ypdv + fp*para_fit[1]
    else:
        xdata = xpdv + fp*para_fit[1]
        ydata = ypdv

    xfit, yfit = xdata[idx], ydata[idx]
    hdist_fit, xfit, yfit = distance_to_edge(xfit, yfit, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    hdist_fit = hdist_fit + para_fit[0]
    obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
    obj_fit.fit(ifmcmc=False)
    axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

    alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
    axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
    axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
    axes[0].text(0.01, 0.85, "Diagram: {:s}".format(diagram), **alignment)
    axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

    axes[0].legend()

    axes[0].grid(True)
    axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
    axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

    # diagram axes[1]
    axes[1].plot(xobs, yobs, "r.", ms=1)
    axes[1].plot(xfit, yfit, "b.", ms=1)
    axes[1].plot(xedge_obs, yedge_obs, "k--")
    axes[1].plot(xedge_pdv, yedge_pdv, "k-")
    axes[1].grid(True)
    if diagram=="mr":
        axes[1].axis([0., 3., 5., 20.])
    if diagram=="tnu":
        axes[1].axis([10, 150, 2.0, 10.0])

    # fill weighted region
    xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
    axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

    plt.savefig(tfilepath+"result.png")
    plt.close()


    # save data
    data = {"samples":samples, "ndim":ndim,
            "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
            "diagram":diagram, "distance":distance, 
            "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
            "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
            "number_reduction_factor":number_reduction_factor}

    np.save(tfilepath+"data", data)
    return



def sharpness_fit_rescale_mcmc(xobso, yobso, eobso, edges_obs, tck_obs, tp_obs,
        xpdvo, ypdvo, edges_pdv, tck_pdv, tp_pdv,
        diagram, distance, hist_model, filepath, ifmcmc=False):
    
    print("diagram:",diagram)
    print("distance:",distance)

    global model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost, obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight
    global fp
    # filepath
    tfilepath = filepath

    # set up observations
    xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
    xobs, yobs, eobs = xobso, yobso, eobso

    # set up models
    xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
    xpdv, ypdv = xpdvo, ypdvo


    # calculate Kepler distance
    hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
    obj_obs = distfit(hdist_obs, hist_model)
    obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
    obj_obs.fit(ifmcmc=False)

    # calculate Galaxia distance
    hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    obj_pdv = distfit(hdist_pdv, hist_model, bins=obj_obs.bins)
    obj_pdv.fit(ifmcmc=False)

    # run mcmc with ensemble sampler
    ndim, nwalkers, nburn, nsteps = 2, 200, 2000, 1000

    para_names = ["shift", "scale_factor"]
    xd = np.abs(obj_obs.para_fit[1]-obj_pdv.para_fit[1])
    para_limits = [[-3*xd, 3*xd], [0., 3.0]]
    para_guess = [0., 1.0]

    Ndata = xpdv.shape[0]

    # tied to model6
    weight = np.zeros(obj_obs.histx.shape, dtype=bool)
    sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
    idx = (obj_obs.histx >= x0-3*sigma) & (obj_obs.histx <= x0+3*sigma)
    weight[idx] = True

    if distance=="vertical":
        fy1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
        fp = ypdv*fy1_base
    else:
        # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
        # "horizontal"
        fx1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
        fp = xpdv*fx1_base
   

    def model(theta):#, obj_obs, xpdv, ypdv):

        # theta[0]: offset in distance
        # theta[1]: perturb

        # disturb with artificial scatter
        # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

        hdist = hdist_pdv + fp*theta[1]
        hdist = hdist + theta[0]
        obj = distfit(hdist, hist_model, bins=obj_obs.bins)

        # normalize the number of points in the weighted region
        if np.sum(obj.histy[weight])!=0:
            number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
        else:
            number_reduction_factor = 0.
        histy = obj.histy * number_reduction_factor
        return histy, hdist, number_reduction_factor

    def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
        histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
        d, m = obj_obs.histy[weight], histy[weight]
        if m[m==0].shape[0] == 0:
            logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
            lnL =  np.sum(d*np.log(m)-m-logdfact)
            # print(theta, lnL)
            return lnL
        else:
            # print(theta, "d")
            return -np.inf

    def minus_lnlikelihood(theta):
        return -lnlikelihood(theta)

    def lnprior(theta):#, para_limits):
        for i in range(len(theta)):
            if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
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
        print(para_guess)
        # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
        pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

        with Pool() as pool:
        # if True:
            # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
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

        # estimation result
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
        # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
        para_fit = res.x
        print(para_guess, para_fit)
        e_para_fit = None
        samples = None


    # result plot
    fig = plt.figure(figsize=(12,12))
    axes = fig.subplots(nrows=2, ncols=1)
    obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
    obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
    # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

    # calculate best fitted results
    yfit, hdist, number_reduction_factor = model(para_fit)
    Ndata = hdist.shape[0]
    idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

    if distance=="vertical":
        xdata = xpdv
        ydata = ypdv + fp*para_fit[1]
    else:
        xdata = xpdv + fp*para_fit[1]
        ydata = ypdv

    xfit, yfit = xdata[idx], ydata[idx]
    hdist_fit, xfit, yfit = distance_to_edge(xfit, yfit, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    hdist_fit = hdist_fit + para_fit[0]
    obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
    obj_fit.fit(ifmcmc=False)
    axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

    alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
    axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
    axes[0].text(0.01, 0.90, "Scale factor: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
    axes[0].text(0.01, 0.85, "Diagram: {:s}".format(diagram), **alignment)
    axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

    axes[0].legend()

    axes[0].grid(True)
    axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
    axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

    # diagram axes[1]
    axes[1].plot(xobs, yobs, "r.", ms=1)
    axes[1].plot(xfit, yfit, "b.", ms=1)
    axes[1].plot(xedge_obs, yedge_obs, "k--")
    axes[1].plot(xedge_pdv, yedge_pdv, "k-")
    axes[1].grid(True)
    if diagram=="mr":
        axes[1].axis([0., 3., 5., 20.])
    if diagram=="tnu":
        axes[1].axis([10, 150, 2.0, 10.0])

    # fill weighted region
    xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
    axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

    plt.savefig(tfilepath+"result.png")
    plt.close()


    # save data
    data = {"samples":samples, "ndim":ndim,
            "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
            "diagram":diagram, "distance":distance, 
            "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
            "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
            "number_reduction_factor":number_reduction_factor}

    np.save(tfilepath+"data", data)
    return



# def sharpness_fit_perturb(xobso, yobso, zobso, e_xobso, e_yobso, edges_obs, tck_obs, tp_obs,
#         xpdvo, ypdvo, zpdvo, edges_pdv, tck_pdv, tp_pdv,
#         diagram, distance, hist_model,
#         filepath, xperturb, yperturb, montecarlo,
#         zvalue_limits=None, zvalue_name=None, Ndata=None,
#         cores=10, ifmcmc=True):
    
#     print("zvalue_name:",zvalue_name)
#     print("zvalue_limits:",zvalue_limits)
#     print("diagram:",diagram)
#     print("distance:",distance)

#     if (zvalue_limits is None): 
#         Nz = 1
#     else:
#         Nz = len(zvalue_limits[0])

#     for iz in range(Nz):
#         izn = "{:0.0f}".format(iz)

#         # filepath
#         if (zvalue_limits is None):
#             tfilepath = filepath
#         else:
#             tfilepath = filepath+izn+"/"
#         if not os.path.exists(tfilepath): os.mkdir(tfilepath)   

#         # set up observations
#         xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
#         if not (zvalue_limits is None):
#             idx = (zobso >= zvalue_limits[0][iz]) & (zobso < zvalue_limits[1][iz])
#             xobs, yobs = xobso[idx], yobso[idx]
#         else:
#             xobs, yobs = xobso, yobso

#         # calculate observational distance
#         hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
#         obj_obs = distfit(hdist_obs, hist_model)
#         obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
#         obj_obs.fit(ifmcmc=ifmcmc)
#         obj_obs.output(tfilepath, ifmcmc=ifmcmc)

#         sharpness_obs = hist_model.sharpness(obj_obs.para_fit)
#         e_sharpness_obs = hist_model.e_sharpness(obj_obs.para_fit, obj_obs.e_para_fit)

#         Nobs = hdist_obs.shape[0]

#         # set up models
#         xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
#         if not (zvalue_limits is None):
#             idx = (zpdvo >= zvalue_limits[0][iz]) & (zpdvo < zvalue_limits[1][iz]) 
#             xpdv, ypdv = xpdvo[idx], ypdvo[idx]
#         else:
#             xpdv, ypdv = xpdvo, ypdvo

#         # reduce to same number of points
#         if (Ndata is None): Ndata =  Nobs
#         idx = reduce_samples(xpdv.shape[0], Ndata)
#         xpdv, ypdv = xpdv[idx], ypdv[idx]

#         sharpness_med = np.zeros((xperturb.shape[0], yperturb.shape[0]), dtype=object)
#         sharpness_std = np.zeros((xperturb.shape[0], yperturb.shape[0]), dtype=object)

#         for ix in range(xperturb.shape[0]):
#             for iy in range(yperturb.shape[0]):
#                 print("{:0.0f}  xscatter: {:0.2f}%, {:0.0f}  yscatter: {:0.2f}%".format(ix, xperturb[ix]*100, iy, yperturb[iy]*100))
                
#                 # initiate a plot
#                 fig = plt.figure(figsize=(12,12))
#                 axes = fig.subplots(nrows=2, ncols=1)
#                 obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Observations", "zorder":100})
#                 obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Observations fit", "linestyle":"--", "zorder":100})
                
#                 # add scatters
#                 Ndata = xpdv.shape[0]
#                 # for imc in range(montecarlo):

#                 fsimulation=partial(simulation, Ndata=Ndata, xp=xperturb[ix], yp=yperturb[iy], scalar=0.,
#                         xpdv=xpdv, ypdv=ypdv,
#                         xedge_pdv=xedge_pdv, yedge_pdv=yedge_pdv, tck_pdv=tck_pdv, tp_pdv=tp_pdv,
#                         diagram=diagram, distance=distance, hist_model=hist_model, obj_obs=obj_obs,
#                         e_xobs=e_xobso, e_yobs=e_yobso)

#                 pool = Pool(processes=cores)
#                 result = pool.map(fsimulation, np.arange(0,montecarlo).tolist())
#                 pool.close()
#                 pool.join()
#                 xdata, ydata, obj, _ = result[0]
#                 sharpness_pdv_mcs = np.array([result[i][-1] for i in range(len(result))])

#                 sharpness_med[ix, iy] = np.nanmedian(sharpness_pdv_mcs, axis=0)
#                 sharpness_std[ix, iy] = np.nanstd(sharpness_pdv_mcs, axis=0)/np.sqrt(montecarlo)
 
#                 obj.plot_hist(ax=axes[0], histkwargs={"color":"blue", "label":"Galaxia"})
#                 obj.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Galaxia fit"})

#                 alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
#                 axes[0].text(0.0, 1.00, "Scatter in x: {:0.2f}%".format(xperturb[ix]*100), **alignment)
#                 axes[0].text(0.0, 0.95, "Scatter in y: {:0.2f}%".format(yperturb[iy]*100), **alignment)
#                 axes[0].text(0.0, 0.90, "Galaxia sigma: {:0.3f} $\pm$ {:0.3f}".format(sharpness_med[ix, iy][0], sharpness_std[ix, iy][0]), **alignment)
#                 axes[0].text(0.0, 0.85, "Kepler sigma: {:0.3f} $\pm$ {:0.3f}".format(sharpness_obs[0], e_sharpness_obs[0]), **alignment)      
#                 axes[0].text(0.0, 0.80, "Galaxia slope: {:0.3f} $\pm$ {:0.3f}".format(sharpness_med[ix, iy][1], sharpness_std[ix, iy][1]), **alignment)
#                 axes[0].text(0.0, 0.75, "Kepler slope: {:0.3f} $\pm$ {:0.3f}".format(sharpness_obs[1], e_sharpness_obs[1]), **alignment)      
#                 # axes[0].legend()
#                 axes[0].grid(True)
#                 axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#                 axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

#                 # diagram axes[1]
#                 axes[1].plot(xobs, yobs, "r.", ms=1)
#                 axes[1].plot(xdata, ydata, "b.", ms=1)
#                 axes[1].plot(xedge_obs, yedge_obs, "k--")
#                 axes[1].plot(xedge_pdv, yedge_pdv, "k-")
#                 axes[1].grid(True)
#                 if diagram=="mr":
#                     axes[1].axis([0., 3., 5., 20.])
#                 if diagram=="tnu":
#                     axes[1].axis([10, 150, 2.0, 10.0])

#                 plt.savefig(tfilepath+"{:0.0f}_x_{:0.0f}_y.png".format(ix, iy))
#                 plt.close()

#         # save data
#         data = {"xperturb":xperturb, "yperturb":yperturb, "montecarlo":montecarlo,
#                 "sharpness_med":sharpness_med, "sharpness_std":sharpness_std,
#                 "sharpness_obs":sharpness_obs, "e_sharpness_obs":e_sharpness_obs,
#                 "diagram":diagram, "distance":distance,
#                 "zvalue_limits": zvalue_limits, "zvalue_name":zvalue_name}
#         np.save(tfilepath+"data", data)
#     return

# def sharpness_fit_rescale(xobso, yobso, zobso, e_xobso, e_yobso, edges_obs, tck_obs, tp_obs,
#         xpdvo, ypdvo, zpdvo, edges_pdv, tck_pdv, tp_pdv,
#         diagram, distance, hist_model,
#         filepath, scalar, montecarlo,
#         zvalue_limits=None, zvalue_name=None, Ndata=None,
#         cores=10, ifmcmc=True):
    
#     print("zvalue_name:",zvalue_name)
#     print("zvalue_limits:",zvalue_limits)
#     print("diagram:",diagram)
#     print("distance:",distance)

#     if (zvalue_limits is None): 
#         Nz = 1
#     else:
#         Nz = len(zvalue_limits[0])

#     for iz in range(Nz):
#         izn = "{:0.0f}".format(iz)

#         # filepath
#         if (zvalue_limits is None):
#             tfilepath = filepath
#         else:
#             tfilepath = filepath+izn+"/"
#         if not os.path.exists(tfilepath): os.mkdir(tfilepath)   

#         # set up observations
#         xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
#         if not (zvalue_limits is None):
#             idx = (zobso >= zvalue_limits[0][iz]) & (zobso < zvalue_limits[1][iz])
#             xobs, yobs = xobso[idx], yobso[idx]
#         else:
#             xobs, yobs = xobso, yobso

#         # calculate observational distance
#         hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
#         obj_obs = distfit(hdist_obs, hist_model)
#         obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
#         obj_obs.fit(ifmcmc=ifmcmc)
#         obj_obs.output(tfilepath, ifmcmc=ifmcmc)

#         sharpness_obs = hist_model.sharpness(obj_obs.para_fit)
#         e_sharpness_obs = hist_model.e_sharpness(obj_obs.para_fit, obj_obs.e_para_fit)

#         Nobs = hdist_obs.shape[0]

#         # set up models
#         xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
#         if not (zvalue_limits is None):
#             idx = (zpdvo >= zvalue_limits[0][iz]) & (zpdvo < zvalue_limits[1][iz]) 
#             xpdv, ypdv = xpdvo[idx], ypdvo[idx]
#         else:
#             xpdv, ypdv = xpdvo, ypdvo

#         # reduce to same number of points
#         if (Ndata is None): Ndata =  Nobs
#         idx = reduce_samples(xpdv.shape[0], Ndata)
#         xpdv, ypdv = xpdv[idx], ypdv[idx]

#         sharpness_med = np.zeros((scalar.shape[0]), dtype=object)
#         sharpness_std = np.zeros((scalar.shape[0]), dtype=object)

#         for iscalar in range(scalar.shape[0]):
#             print("{:0.0f}  scalar: {:0.2f}%".format(iscalar, scalar[iscalar]*100))
            
#             # initiate a plot
#             fig = plt.figure(figsize=(12,12))
#             axes = fig.subplots(nrows=2, ncols=1)
#             obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Observations", "zorder":100})
#             obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Observations fit", "linestyle":"--", "zorder":100})
            
#             # add scatters
#             Ndata = xpdv.shape[0]
#             # for imc in range(montecarlo):

#             fsimulation=partial(simulation, Ndata=Ndata, xp=0., yp=0., scalar=scalar[iscalar],
#                     xpdv=xpdv, ypdv=ypdv,
#                     xedge_pdv=xedge_pdv, yedge_pdv=yedge_pdv, tck_pdv=tck_pdv, tp_pdv=tp_pdv,
#                     diagram=diagram, distance=distance, hist_model=hist_model, obj_obs=obj_obs,
#                     e_xobs=e_xobso, e_yobs=e_yobso)

#             pool = Pool(processes=cores)
#             result = pool.map(fsimulation, np.arange(0,montecarlo).tolist())
#             pool.close()
#             pool.join()
#             xdata, ydata, obj, _ = result[0]
#             sharpness_pdv_mcs = np.array([result[i][-1] for i in range(len(result))])

#             sharpness_med[iscalar] = np.nanmedian(sharpness_pdv_mcs, axis=0)
#             sharpness_std[iscalar] = np.nanstd(sharpness_pdv_mcs, axis=0)/np.sqrt(montecarlo)

#             obj.plot_hist(ax=axes[0], histkwargs={"color":"blue", "label":"Galaxia"})
#             obj.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Galaxia fit"})

#             alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
#             axes[0].text(0.0, 1.00, "Rescale error distribution: {:0.2f}%".format(scalar[iscalar]*100), **alignment)
#             axes[0].text(0.0, 0.90, "Galaxia sigma: {:0.3f} $\pm$ {:0.3f}".format(sharpness_med[iscalar][0], sharpness_std[iscalar][0]), **alignment)
#             axes[0].text(0.0, 0.85, "Kepler sigma: {:0.3f} $\pm$ {:0.3f}".format(sharpness_obs[0], e_sharpness_obs[0]), **alignment)      
#             axes[0].text(0.0, 0.80, "Galaxia slope: {:0.3f} $\pm$ {:0.3f}".format(sharpness_med[iscalar][1], sharpness_std[iscalar][1]), **alignment)
#             axes[0].text(0.0, 0.75, "Kepler slope: {:0.3f} $\pm$ {:0.3f}".format(sharpness_obs[1], e_sharpness_obs[1]), **alignment)      

#             # axes[0].legend()
#             axes[0].grid(True)
#             axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#             axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

#             # diagram axes[1]
#             axes[1].plot(xobs, yobs, "r.", ms=1)
#             axes[1].plot(xdata, ydata, "b.", ms=1)
#             axes[1].plot(xedge_obs, yedge_obs, "k--")
#             axes[1].plot(xedge_pdv, yedge_pdv, "k-")
#             axes[1].grid(True)
#             if diagram=="mr":
#                 axes[1].axis([0., 3., 5., 20.])
#             if diagram=="tnu":
#                 axes[1].axis([10, 150, 2.0, 10.0])

#             plt.savefig(tfilepath+"{:0.0f}_scale.png".format(iscalar))
#             plt.close()

#         # save data
#         data = {"scalar":scalar, "montecarlo":montecarlo,
#                 "sharpness_med":sharpness_med, "sharpness_std":sharpness_std,
#                 "sharpness_obs":sharpness_obs, "e_sharpness_obs":e_sharpness_obs,
#                 "diagram":diagram, "distance":distance,
#                 "zvalue_limits": zvalue_limits, "zvalue_name":zvalue_name}
#         np.save(tfilepath+"data", data)
#     return

# def plot_slope_scatter(filepath, distance="horizontal", diagram="tnu"):
#     if not (distance in ["horizontal", "vertical"]):
#         raise ValueError("distance should be horizontal or vertical.")
#     if not (diagram in ["tnu", "mr"]):
#         raise ValueError("diagram should be tnu or mr.")

#     # read data
#     data = np.load(filepath+"data.npy", allow_pickle=True).tolist()
#     if distance=="horizontal":
#         cp_scatter = data["yperturb"][0]
#         scatter = data["xperturb"]
#         slope = data["sharpness_med"][:,0]
#         eslope = data["sharpness_std"][:,0]
#     else:
#         scatter = data["yperturb"]
#         cp_scatter = data["xperturb"][0]
#         slope = data["sharpness_med"][0,:]
#         eslope = data["sharpness_std"][0,:]
#     slope_obs = data["sharpness_obs"]
#     iferr = False if (data["e_sharpness_obs"] is None) else True
#     if iferr:
#         e_slope_obs = data["e_sharpness_obs"]

#     errorbarkwargs = {"elinewidth":1, "capsize":2, "ecolor":"black"}
#     x, y = scatter*100, slope-slope_obs
#     ey = eslope
#     # initiate a plot
#     fig = plt.figure(figsize=(6,4))
#     axes = fig.subplots(nrows=1, ncols=1, squeeze=False).reshape(-1,)
#     axes[0].errorbar(x, y, yerr=ey, fmt="k.", **errorbarkwargs)
#     axes[0].axhline(0, c="r", ls="--")

#     idx = np.abs(y) == np.abs(y).min()
#     xmed = x[idx][0]
#     axes[0].axvline(xmed, c="r", ls="--")
#     axes[0].axis([x.min(), x.max(), y.min(), y.max()])

#     if iferr:
#         idx = np.abs(y-e_slope_obs) == np.abs(y-e_slope_obs).min()
#         xmin = x[idx][0] 
#         idx = np.abs(y+e_slope_obs) == np.abs(y+e_slope_obs).min()
#         xmax = x[idx][0]
#         axes[0].fill_betweenx([y.min(), y.max()], xmin, xmax, color="lightgray",zorder=-100)
#         axes[0].fill_between([x.min(), x.max()], e_slope_obs, -e_slope_obs, color="lightgray",zorder=-100)

#     if diagram=="tnu":
#         if distance=="horizontal":
#             axes[0].set_title("Scatter in dnu: {:0.2f}%, scatter in numax: {:0.2f}%".format(cp_scatter, xmed))
#             axes[0].set_xlabel("Scatter in numax relation [%]")
#         if distance=="vertical":
#             axes[0].set_title("Scatter in dnu: {:0.2f}%, scatter in numax: {:0.2f}%".format(xmed, cp_scatter))
#             axes[0].set_xlabel("Scatter in dnu relation [%]")            
#     if diagram=="mr":
#         if distance=="horizontal":
#             axes[0].set_title("Scatter in R: {:0.2f}%, scatter in M: {:0.2f}%".format(cp_scatter, xmed))
#             axes[0].set_xlabel("Scatter in M relation [%]")
#         if distance=="vertical":
#             axes[0].set_title("Scatter in R: {:0.2f}%, scatter in M: {:0.2f}%".format(xmed, cp_scatter))
#             axes[0].set_xlabel("Scatter in R relation [%]") 

#     axes[0].set_ylabel("Slope(galaxia) - Slope(obs)")
#     plt.tight_layout()
#     plt.savefig(filepath+"slope_scatter.png")
#     plt.close()
#     return

def plot_scatter_zvalue(filepath, bins=3, distance="horizontal", diagram="tnu", iferr=False):
    if not (distance in ["horizontal", "vertical"]):
        raise ValueError("distance should be horizontal or vertical.")
    if not (diagram in ["tnu", "mr"]):
        raise ValueError("diagram should be tnu or mr.")

    # read zvalue and zname
    data = np.load(filepath+str(bins-1)+"/data.npy", allow_pickle=True).tolist()
    zvalue_limits = data["zvalue_limits"]
    zvalue_name = data["zvalue_name"]
    zvalue = np.array([(zvalue_limits[0][i] + zvalue_limits[1][i])/2.0 for i in range(len(zvalue_limits[0]))])
    
    # calculate scatter in each folder
    fs, e_fs = np.zeros(len(zvalue)), np.zeros(len(zvalue))
    for i in range(len(zvalue)):
        data = np.load(filepath+str(i)+"/data.npy", allow_pickle=True).tolist()
        if distance=="horizontal":
            cp_scatter = data["yperturb"][0]
            scatter = data["xperturb"]
            slope = data["sharpness_med"][:,0]
            eslope = data["sharpness_std"][:,0]
        else:
            scatter = data["yperturb"]
            cp_scatter = data["xperturb"][0]
            slope = data["sharpness_med"][0,:]
            eslope = data["sharpness_std"][0,:]
        slope_obs = data["sharpness_obs"]

        x, y = scatter*100, slope-slope_obs
        idx = np.abs(y) == np.abs(y).min()
        fs[i] = x[idx][0]

        if iferr:
            e_slope_obs = data["e_sharpness_obs"]
            idx = np.abs(y-e_slope_obs) == np.abs(y-e_slope_obs).min()
            xmin = x[idx][0]
            idx = np.abs(y+e_slope_obs) == np.abs(y+e_slope_obs).min()
            xmax = x[idx][0]
            e_fs[i] = np.abs(xmax-xmin)/2.

    # initiate a plot
    fig = plt.figure(figsize=(6,4))
    axes = fig.subplots(nrows=1, ncols=1, squeeze=False).reshape(-1,)
    axes[0].errorbar(zvalue, fs, yerr=e_fs, c="black", linestyle="--", marker=".", ms=12, capsize=2)
    # axes[0].plot(zvalue, fs, "k--")
    axes[0].set_ylim([0., fs.max()*1.1])#fs.max()+(fs.max()-fs.min())*0.2

    if diagram=="tnu":
        if distance=="horizontal":
            axes[0].set_ylabel("Fitted scatter in numax relation [%]")
        if distance=="vertical":
            axes[0].set_ylabel("Fitted scatter in dnu relation [%]")            
    if diagram=="mr":
        if distance=="horizontal":
            axes[0].set_ylabel("Fitted scatter in M relation [%]")
        if distance=="vertical":
            axes[0].set_ylabel("Fitted scatter in R relation [%]") 

    axes[0].set_xlabel(zvalue_name)
    plt.tight_layout()
    plt.savefig(filepath+"scatter_zvalue.png")
    plt.close()
    return



