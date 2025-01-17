'''
This program is to provide wrappers for comparing vertical/horizontal distributions,
either for the zero-age HeB stars or the RGB bump stars.


'''

rootpath = "/Volumes/Data/Onedrive/Work/nike/"

import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.histdist import distfit, distance_to_edge, reduce_samples, display_bar, distance_to_bump, model_heb, model_rgb
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


class Fitter:
    '''
    Fit the distribution of data (Kepler) with model (Galaxia), in the form of histogram.
    The two free parameters are an offset controlling the shift in the histogram, and a scatter to
    tweak the sharpness of the feature (RGB bump/zero-age HeB).

    Parameters
    ----------

    filepath : str
        output dir

    obs_obj : distfit class
        defined with data

    model_obj : distfit class
        defined with model

    scatter : array_like[Ndata, ]
        added to the model_obj.dist

    mask : array_like[obs_obj.histy.shape], bool
        mask on histograms

    Methods
    ----------
    model : (theta)
        returns histy, dist, normalize_factor

    fit : (ifmcmc=True, para_limits=None, para_guess=None, nburn=500, nsteps=1000)
        runs the fitting process


    '''
    
    def __init__(self, filepath, obs_obj, model_obj, scatter, mask=None,):
        self.filepath = filepath
        self._obs_obj = obs_obj
        self._model_obj = model_obj
        self._scatter = scatter

        if mask is None:
            self._mask = np.ones(obs_obj.histx.shape, dtype=bool)
        elif np.sum(mask) == 0.:
            raise ValueError('mask can not be all zero.')
        else:
            self._mask = mask


        return

    def model(self, theta):

        # theta[0]: offset in distance
        # theta[1]: perturb

        # step 1, manipulate the distribution
        dist = self._model_obj.dist + self._scatter * theta[1]
        dist += theta[0]

        # step 2, get a histogram
        bins = self._obs_obj.bins
        histy, _ = np.histogram(dist, bins=bins)

        # setp 3, normalize the number of points in the weighted region
        normalize_factor = np.sum(self._obs_obj.histy[self._mask]) / np.sum(histy[self._mask])
        if not np.isfinite(normalize_factor): normalize_factor=1.
        histy = histy * normalize_factor

        return histy, dist, normalize_factor


    def lnlikelihood(self, theta):
        histy, _, _ = self.model(theta)
        d, m = self._obs_obj.histy[self._mask], histy[self._mask]
        if m[m==0].shape[0] == 0:
            m[m==0] = 1
            logdfact = scipy.special.gammaln(d+1) 
            lnL =  np.sum(d*np.log(m)-m-logdfact)
            return lnL
        else:
            return -np.inf


    def chi2(self, theta):
        return -self.lnlikelihood(theta)


    def lnprior(self, theta):#, para_limits):
        for i in range(len(theta)):
            if not (self.para_limits[i][0] <= theta[i] <= self.para_limits[i][1] ):
                return -np.inf
        return 0.


    def lnpost(self, theta):#, para_limits, obj_obs, xpdv, ypdv):
        lp = self.lnprior(theta)#, para_limits)
        if np.isfinite(lp):
            lnL = self.lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
            return lnL
        else:
            return -np.inf


    def fit(self, ifmcmc=True, para_limits=None, para_guess=None, nburn=500, nsteps=1000):

        self.para_limits = para_limits
        self.para_guess = para_guess
        self.nburn = nburn
        self.nsteps = nsteps
        self.ifmcmc = ifmcmc

        self.para_names = ['offset', 'scatter']
        self.ndim = 2
        self.nwalkers = 200

        if ifmcmc:
            print("enabling Ensemble sampler.")
            print(para_guess)
            # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
            pos0 = [np.array([np.random.uniform(low=self.para_limits[idim][0], 
                    high=self.para_limits[idim][1]) 
                    for idim in range(self.ndim)]) 
                    for iwalker in range(self.nwalkers)]

            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnpost)
            # # burn-in
            print("start burning in. nburn:", self.nburn)
            for j, result in enumerate(sampler.sample(pos0, iterations=self.nburn, thin=10)):
                    display_bar(j, self.nburn)
                    pass
            sys.stdout.write("\n")
            pos, _, _ = result
            sampler.reset()

            # # actual iteration
            print("start iterating. nsteps:", self.nsteps)
            for j, result in enumerate(sampler.sample(pos, iterations=self.nsteps)):
                    display_bar(j, self.nsteps)
                    pass
            sys.stdout.write("\n")

            # modify samples
            self.samples = sampler.chain[:,:,:].reshape((-1,self.ndim))

            # estimation result
            # 16, 50, 84 quantiles
            result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                    zip(*np.percentile(self.samples, [16, 50, 84],axis=0)))))
            self.para_fit = result[:,0]
            self.e_para_fit = (result[:,1]+result[:,2])/2.0

            # maximum
            para_fitmax = np.zeros(self.ndim)
            for ipara in range(self.ndim):
                n, bins, _ = plt.hist(self.samples[:,ipara], bins=80)
                idx = np.where(n == n.max())[0][0]
                para_fitmax[ipara] = bins[idx:idx+1].mean()
            self.para_fitmax = para_fitmax

            # corner plot
            fig = corner.corner(self.samples, labels=self.para_names, show_titles=True, truths=para_fitmax)
            plt.savefig(self.filepath+"corner.png")
            plt.close()

        else: 
            print("enabling basinhopping.")
            res = basinhopping(self.chi2, self.para_guess, minimizer_kwargs={"bounds":self.para_limits})
            # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)

            self.para_fit = res.x 
            self.para_fitmax = res.x 
            self.e_para_fit = None
            self.samples = None
            
        return



# # #
# Zero-age core-helium-burning stars fitting wrappers

def heb_fit(xobs, yobs, edges_obs, tck_obs, tp_obs,
        xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        variable, distance, filepath, 
        xerror_sample=None, yerror_sample=None, ifmcmc=True, nburn=None, nsteps=None):
    
    print("variable:",variable)
    print("distance:",distance)
    diagram = 'tnu' if variable in ['dnu', 'numax'] else 'mr'


    # calculate Kepler distance
    hist_model = model_heb()

    hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, edges_obs[:,0], edges_obs[:,1],
                    tck_obs, tp_obs, diagram=diagram, distance=distance)
    obs = distfit(hdist_obs, hist_model)
    obs = distfit(hdist_obs, hist_model, bins=obs.bins)
    obs.fit(ifmcmc=False)


    # calculate Galaxia distance
    Ndata = xpdv.shape[0]
    if not (xerror_sample is None):
        eobs = xerror_sample
        xerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
        xpdv += xpdv*xerror_resampled

    if not (yerror_sample is None):
        eobs = yerror_sample
        yerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
        ypdv += ypdv*yerror_resampled

    hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, edges_pdv[:,0], edges_pdv[:,1],
                    tck_pdv, tp_pdv, diagram=diagram, distance=distance)
    mod = distfit(hdist_pdv, hist_model, bins=obs.bins)
    mod.fit(ifmcmc=False)


    # defind a mask
    mask = np.zeros(obs.histx.shape, dtype=bool)
    sigma, x0 = obs.para_fit[0], obs.para_fit[1]
    idx = (obs.histx >= x0-3*sigma) & (obs.histx <= x0+3*sigma)
    mask[idx] = True

    # define a scatter sample
    Ndata = xpdv.shape[0]
    if distance=="vertical":
        scatter = ypdv * np.random.normal(size=Ndata)
    else:
        scatter = xpdv * np.random.normal(size=Ndata)
 
    # set up Fitter parameters
    nburn = 500 if (nburn is None) else nburn
    nsteps = 1000 if (nsteps is None) else nsteps
    xd = np.abs(obs.histx[mask].max()-obs.histx[mask].min())#np.abs(obs.para_fit[1]-mod.para_fit[1])
    if diagram == "tnu":
        para_limits = [[-xd, xd], [0., 0.08]]
        para_guess = [0., 0.005]
    else:
        para_limits = [[-xd, xd], [0., 0.20]]
        para_guess = [0., 0.005]  

    # set up a fit class
    fitter = Fitter(filepath, obs, mod, scatter, mask=mask)
    fitter.fit(ifmcmc=ifmcmc, para_limits=para_limits, para_guess=para_guess, nburn=nburn, nsteps=nsteps)


    # result plot
    fig = plt.figure(figsize=(12,12))
    axes = fig.subplots(nrows=2, ncols=1)            

    # calculate best fitted results
    _, dist_fit, normalize_factor = fitter.model(fitter.para_fitmax)
    Ndata = dist_fit.shape[0]
    ridx = reduce_samples(Ndata, Ndata*normalize_factor)

    dist_fit = dist_fit[ridx]
    if distance=="vertical":
        xfit = xpdv[ridx]
        yfit = ypdv[ridx] + scatter[ridx]*fitter.para_fitmax[1]
    else:
        xfit = xpdv[ridx] + scatter[ridx]*fitter.para_fitmax[1]
        yfit = ypdv[ridx]

    # dist_fit += fitter.para_fitmax[0]
    ofit = distfit(dist_fit, hist_model, bins=obs.bins)

    # # axes[0]: histograms
    obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
    obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
    axes[0].step(ofit.histx, ofit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

    alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
    axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(fitter.para_fitmax[0], para_guess[0]), **alignment)
    axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(fitter.para_fitmax[1]*100, para_guess[1]*100), **alignment) 
    axes[0].text(0.01, 0.85, "Variable: {:s}".format(variable), **alignment)
    axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(obs.histx.min(), obs.histx.max())
    axes[0].set_ylim(0., obs.histy.max()*1.5)
    # fill weighted region
    xmin_, xmax_ = obs.histx[mask].min(), obs.histx[mask].max()
    axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")


    # # axes[1]: diagram
    axes[1].plot(xobs, yobs, "r.", ms=1)
    axes[1].plot(xfit, yfit, "b.", ms=1)
    axes[1].plot(edges_obs[:,0], edges_obs[:,1], "k--")
    axes[1].plot(edges_pdv[:,0], edges_pdv[:,1], "k-")
    axes[1].grid(True)
    if diagram=="mr":
        axes[1].axis([0., 3., 5., 20.])
    if diagram=="tnu":
        axes[1].axis([10, 150, 2.0, 10.0])

    plt.savefig(filepath+"result.png")
    plt.close()


    # save data
    data = {"samples":fitter.samples, "ndim":fitter.ndim,
            "para_fit":fitter.para_fit, "para_fitmax":fitter.para_fitmax,
            "e_para_fit":fitter.e_para_fit, "para_guess":fitter.para_guess,
            "variable":variable, "distance":distance, 
            "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
            "obj_obs":obs, "obj_pdv":mod, "obj_fit":ofit, "fitter":fitter,
            "normalize_factor":normalize_factor}

    np.save(filepath+"data", data)
    return




# # #
# RGB bump stars fitting wrappers

def rgb_fit(xobs, yobs, bump_obs, xpdv, ypdv, bump_pdv,
        variable, distance, filepath, 
        xerror_sample=None, yerror_sample=None, ifmcmc=True, nburn=None, nsteps=None):
    
    print("variable:",variable)
    print("distance:",distance)
    

    if variable == "numax":
        bins = np.arange(-30, 30, 2.0)
    if variable == "dnu":
        bins = np.arange(-3, 3, 0.2)
    if variable == "radius":
        bins = np.arange(-2, 2, 0.2)
    if variable == "mass":
        bins = np.arange(-0.4, 0.4, 0.05)

    # calculate Kepler distance
    hist_model = model_rgb()

    hdist_obs, xobs, yobs = distance_to_bump(xobs, yobs, bump_obs, distance=distance)
    obs = distfit(hdist_obs, hist_model, bins=bins)
    obs.fit(ifmcmc=False)


    # calculate Galaxia distance
    Ndata = xpdv.shape[0]
    sigma = obs.para_fit[0]
    if not (xerror_sample is None):
        eobs = xerror_sample[np.abs(hdist_obs) <= 3*sigma]
        xerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
        xpdv += xpdv*xerror_resampled

    if not (yerror_sample is None):
        eobs = yerror_sample[np.abs(hdist_obs) <= 3*sigma]
        yerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
        ypdv += ypdv*yerror_resampled

    hdist_pdv, xpdv, ypdv = distance_to_bump(xpdv, ypdv, bump_pdv, distance=distance)
    mod = distfit(hdist_pdv, hist_model, bins=obs.bins)
    mod.fit(ifmcmc=False)


    # defind a mask
    mask = np.zeros(obs.histx.shape, dtype=bool)
    sigma, x0 = obs.para_fit[0], obs.para_fit[1]
    idx = (obs.histx >= x0-4*sigma) & (obs.histx <= x0+4*sigma)
    mask[idx] = True

    # define a scatter sample
    Ndata = xpdv.shape[0]
    if distance=="vertical":
        scatter = ypdv * np.random.normal(size=Ndata)
    else:
        scatter = xpdv * np.random.normal(size=Ndata)
 
    # set up Fitter parameters
    nburn = 500 if (nburn is None) else nburn
    nsteps = 1000 if (nsteps is None) else nsteps
    sig = obs.para_fit[0]
    if variable in ["dnu", "numax"]:
        para_limits = [[-sig, sig], [0., 0.30]]
        para_guess = [0., 0.005]
    else:
        para_limits = [[-sig, sig], [0., 0.30]]
        para_guess = [0., 0.005]  

    # set up a fit class
    fitter = Fitter(filepath, obs, mod, scatter, mask=mask)
    fitter.fit(ifmcmc=ifmcmc, para_limits=para_limits, para_guess=para_guess, nburn=nburn, nsteps=nsteps)


    # result plot
    fig = plt.figure(figsize=(12,12))
    axes = fig.subplots(nrows=2, ncols=1)            

    # calculate best fitted results
    _, dist_fit, normalize_factor = fitter.model(fitter.para_fitmax)
    Ndata = dist_fit.shape[0]
    ridx = reduce_samples(Ndata, Ndata*normalize_factor)

    dist_fit = dist_fit[ridx]
    if distance=="vertical":
        xfit = xpdv[ridx]
        yfit = ypdv[ridx] + scatter[ridx]*fitter.para_fitmax[1]
    else:
        xfit = xpdv[ridx] + scatter[ridx]*fitter.para_fitmax[1]
        yfit = ypdv[ridx]

    # dist_fit += fitter.para_fitmax[0]
    ofit = distfit(dist_fit, hist_model, bins=obs.bins)

    # # axes[0]: histograms
    obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
    obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
    axes[0].step(ofit.histx, ofit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

    alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
    axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(fitter.para_fitmax[0], para_guess[0]), **alignment)
    axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(fitter.para_fitmax[1]*100, para_guess[1]*100), **alignment) 
    axes[0].text(0.01, 0.85, "Variable: {:s}".format(variable), **alignment)
    axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(obs.histx.min(), obs.histx.max())
    axes[0].set_ylim(0., obs.histy.max()*1.5)
    # fill weighted region
    xmin_, xmax_ = obs.histx[mask].min(), obs.histx[mask].max()
    axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")


    # # axes[1]: diagram
    axes[1].plot(xobs, yobs, "r.", ms=1)
    axes[1].plot(xfit, yfit, "b.", ms=1)
    axes[1].grid(True)
    if variable=="dnu":
        axes[1].axis([5500, 4000, 20, 1])
    if variable=="numax":
        axes[1].axis([5500, 4000, 200, 10])
    if variable in ["mass", "radius"]:
        axes[1].axis([2.6, 0.5, 1, 20])
    xlim = np.sort(np.array(list(axes[1].get_xlim())))
    xbump = np.linspace(xlim[0], xlim[1], 10)
    ybump_obs = bump_obs[0]*xbump + bump_obs[1]
    ybump_pdv = bump_pdv[0]*xbump + bump_pdv[1]
    axes[1].plot(xbump, ybump_obs, "k--")
    axes[1].plot(xbump, ybump_pdv, "k-")

    plt.savefig(filepath+"result.png")
    plt.close()


    # save data
    data = {"samples":fitter.samples, "ndim":fitter.ndim,
            "para_fit":fitter.para_fit, "para_fitmax":fitter.para_fitmax,
            "e_para_fit":fitter.e_para_fit, "para_guess":fitter.para_guess,
            "variable":variable, "distance":distance, 
            "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
            "obj_obs":obs, "obj_pdv":mod, "obj_fit":ofit, "fitter":fitter,
            "normalize_factor":normalize_factor}

    np.save(filepath+"data", data)
    return


def heb_combo_fit(xobs_combo, yobs_combo, edges_obs_combo, tck_obs_combo, tp_obs_combo,
        xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        variable, distance, filepath, 
        xerror_sample=None, yerror_sample=None):
    
    print("variable:",variable)
    print("distance:",distance)
    diagram = 'tnu' if variable in ['dnu', 'numax'] else 'mr'

    para_fit = []
    para_fitmax = []
    e_para_fit = []

    Nexp = len(edges_obs_combo)
    for iexp in range(Nexp):
        xobs, yobs = xobs_combo[iexp], yobs_combo[iexp]
        edges_obs, tck_obs, tp_obs = edges_obs_combo[iexp], tck_obs_combo[iexp], tp_obs_combo[iexp]

        # calculate Kepler distance
        hist_model = model_heb()

        hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, edges_obs[:,0], edges_obs[:,1],
                        tck_obs, tp_obs, diagram=diagram, distance=distance)
        obs = distfit(hdist_obs, hist_model)
        obs = distfit(hdist_obs, hist_model, bins=obs.bins)
        obs.fit(ifmcmc=False)

        # calculate Galaxia distance
        Ndata = xpdv.shape[0]
        if not (xerror_sample is None):
            eobs = xerror_sample[iexp]
            xerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
            xpdv += xpdv*xerror_resampled

        if not (yerror_sample is None):
            eobs = yerror_sample[iexp]
            yerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
            ypdv += ypdv*yerror_resampled

        hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, edges_pdv[:,0], edges_pdv[:,1],
                        tck_pdv, tp_pdv, diagram=diagram, distance=distance)
        mod = distfit(hdist_pdv, hist_model, bins=obs.bins)
        mod.fit(ifmcmc=False)

        # defind a mask
        mask = np.zeros(obs.histx.shape, dtype=bool)
        sigma, x0 = obs.para_fit[0], obs.para_fit[1]
        idx = (obs.histx >= x0-3*sigma) & (obs.histx <= x0+3*sigma)
        mask[idx] = True

        # define a scatter sample
        Ndata = xpdv.shape[0]
        if distance=="vertical":
            scatter = ypdv * np.random.normal(size=Ndata)
        else:
            scatter = xpdv * np.random.normal(size=Ndata)
    
        # set up Fitter parameters
        # nburn = 500 if (nburn is None) else nburn
        # nsteps = 1000 if (nsteps is None) else nsteps
        xd = np.abs(obs.histx[mask].max()-obs.histx[mask].min())#np.abs(obs.para_fit[1]-mod.para_fit[1])
        if diagram == "tnu":
            para_limits = [[-xd, xd], [0., 0.08]]
            para_guess = [0., 0.005]
        else:
            para_limits = [[-xd, xd], [0., 0.20]]
            para_guess = [0., 0.005]  

        # set up a fit class
        fitter = Fitter(filepath, obs, mod, scatter, mask=mask)
        fitter.fit(ifmcmc=True, para_limits=para_limits, para_guess=para_guess, nburn=250, nsteps=250)

        para_fit.append(fitter.para_fit)
        para_fitmax.append(fitter.para_fitmax)
        e_para_fit.append(fitter.e_para_fit)

    # save data
    data = {"para_fit":para_fit, "para_fitmax":para_fitmax,
            "e_para_fit":e_para_fit,
            "variable":variable, "distance":distance}

    np.save(filepath+"data", data)

    return



def rgb_combo_fit(xobs, yobs, bump_obs_combo, xpdv, ypdv, bump_pdv,
        variable, distance, filepath, 
        xerror_sample=None, yerror_sample=None):
    
    print("variable:",variable)
    print("distance:",distance)
    

    if variable == "numax":
        bins = np.arange(-30, 30, 2.0)
    if variable == "dnu":
        bins = np.arange(-3, 3, 0.2)
    if variable == "radius":
        bins = np.arange(-2, 2, 0.2)
    if variable == "mass":
        bins = np.arange(-0.4, 0.4, 0.05)

    para_fit = []
    para_fitmax = []
    e_para_fit = []

    Nexp = len(bump_obs_combo)
    for iexp in range(Nexp):
        bump_obs = bump_obs_combo[iexp]

        # calculate Kepler distance
        hist_model = model_rgb()

        hdist_obs, xobs, yobs = distance_to_bump(xobs, yobs, bump_obs, distance=distance)
        obs = distfit(hdist_obs, hist_model, bins=bins)
        obs.fit(ifmcmc=False)


        # calculate Galaxia distance
        Ndata = xpdv.shape[0]
        sigma = obs.para_fit[0]
        if not (xerror_sample is None):
            eobs = xerror_sample[np.abs(hdist_obs) <= 3*sigma]
            xerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
            xpdv += xpdv*xerror_resampled

        if not (yerror_sample is None):
            eobs = yerror_sample[np.abs(hdist_obs) <= 3*sigma]
            yerror_resampled = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
            ypdv += ypdv*yerror_resampled

        hdist_pdv, xpdv, ypdv = distance_to_bump(xpdv, ypdv, bump_pdv, distance=distance)
        mod = distfit(hdist_pdv, hist_model, bins=obs.bins)
        mod.fit(ifmcmc=False)


        # defind a mask
        mask = np.zeros(obs.histx.shape, dtype=bool)
        sigma, x0 = obs.para_fit[0], obs.para_fit[1]
        idx = (obs.histx >= x0-4*sigma) & (obs.histx <= x0+4*sigma)
        mask[idx] = True

        # define a scatter sample
        Ndata = xpdv.shape[0]
        if distance=="vertical":
            scatter = ypdv * np.random.normal(size=Ndata)
        else:
            scatter = xpdv * np.random.normal(size=Ndata)
    
        # set up Fitter parameters
        sig = obs.para_fit[0]
        if variable in ["dnu", "numax"]:
            para_limits = [[-sig, sig], [0., 0.30]]
            para_guess = [0., 0.005]
        else:
            para_limits = [[-sig, sig], [0., 0.30]]
            para_guess = [0., 0.005]  

        # set up a fit class
        fitter = Fitter(filepath, obs, mod, scatter, mask=mask)
        fitter.fit(ifmcmc=True, para_limits=para_limits, para_guess=para_guess, nburn=250, nsteps=250)

        para_fit.append(fitter.para_fit)
        para_fitmax.append(fitter.para_fitmax)
        e_para_fit.append(fitter.e_para_fit)

    # save data
    data = {"para_fit":para_fit, "para_fitmax":para_fitmax,
            "e_para_fit":e_para_fit,
            "variable":variable, "distance":distance}

    np.save(filepath+"data", data)
    return


# def heb_ulim_fit(xobso, yobso, edges_obs, tck_obs, tp_obs,
#         xpdvo, ypdvo, edges_pdv, tck_pdv, tp_pdv,
#         diagram, distance, hist_model, filepath, ifmcmc=False, nburn=None, nsteps=None):
    
#     print("diagram:",diagram)
#     print("distance:",distance)

#     global obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight, model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost
#     global fp
#     # filepath
#     tfilepath = filepath

#     # set up observations
#     xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
#     xobs, yobs = xobso, yobso

#     # set up models
#     xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
#     xpdv, ypdv = xpdvo, ypdvo


#     # calculate Kepler distance
#     hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
#     obj_obs = distfit(hdist_obs, hist_model)
#     obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
#     obj_obs.fit(ifmcmc=False)

#     # calculate Galaxia distance
#     hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
#     obj_pdv = distfit(hdist_pdv, hist_model, bins=obj_obs.bins)
#     obj_pdv.fit(ifmcmc=False)

#     # run mcmc with ensemble sampler
#     ndim, nwalkers = 2, 200
#     nburn = 2000 if (nburn is None) else nburn
#     nsteps = 1000 if (nsteps is None) else nsteps

#     para_names = ["shift", "scatter"]
#     xd = np.abs(obj_obs.para_fit[1]-obj_pdv.para_fit[1])
#     if diagram == "tnu":
#         para_limits = [[-3*xd, 3*xd], [0., 0.08]]
#         para_guess = [0., 0.005]
#     else:
#         para_limits = [[-3*xd, 3*xd], [0., 0.20]]
#         para_guess = [0., 0.005]  

#     Ndata = xpdv.shape[0]

#     # tied to model_heb
#     weight = np.zeros(obj_obs.histx.shape, dtype=bool)
#     sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
#     idx = (obj_obs.histx >= x0-3*sigma) & (obj_obs.histx <= x0+3*sigma)
#     weight[idx] = True

#     if distance=="vertical":
#         fy2_base = np.random.normal(size=Ndata)
#         fp = ypdv*fy2_base
#     else:
#         # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
#         # "horizontal"
#         fx2_base = np.random.normal(size=Ndata)
#         fp = xpdv*fx2_base
 

#     def model(theta):#, obj_obs, xpdv, ypdv):

#         # theta[0]: offset in distance
#         # theta[1]: perturb

#         # disturb with artificial scatter
#         # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

#         hdist = hdist_pdv + fp*theta[1]
#         hdist = hdist + theta[0]
#         obj = distfit(hdist, hist_model, bins=obj_obs.bins)

#         # normalize the number of points in the weighted region
#         if np.sum(obj.histy[weight])!=0:
#             number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
#         else:
#             number_reduction_factor = 0.
#         histy = obj.histy * number_reduction_factor
#         return histy, hdist, number_reduction_factor

#     def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
#         histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
#         d, m = obj_obs.histy[weight], histy[weight]
#         if m[m==0].shape[0] == 0:
#             logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
#             lnL =  np.sum(d*np.log(m)-m-logdfact)
#             # print(theta, lnL)
#             return lnL
#         else:
#             # print(theta, "d")
#             return -np.inf

#     def minus_lnlikelihood(theta):
#         return -lnlikelihood(theta)

#     def lnprior(theta):#, para_limits):
#         for i in range(len(theta)):
#             if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
#                 return -np.inf
#         return 0.

#     def lnpost(theta):#, para_limits, obj_obs, xpdv, ypdv):
#         lp = lnprior(theta)#, para_limits)
#         if np.isfinite(lp):
#             lnL = lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
#             return lnL
#         else:
#             return -np.inf

#     if ifmcmc:
#         print("enabling Ensemble sampler.")
#         print(para_guess)
#         # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
#         pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

#         # with Pool() as pool:
#         # if True:
#         # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)#, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
#         # # burn-in
#         print("start burning in. nburn:", nburn)
#         for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
#                 display_bar(j, nburn)
#                 pass
#         sys.stdout.write("\n")
#         pos, _, _ = result
#         sampler.reset()

#         # actual iteration
#         print("start iterating. nsteps:", nsteps)
#         for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
#                 display_bar(j, nsteps)
#                 pass
#         sys.stdout.write("\n")

#         # modify samples
#         samples = sampler.chain[:,:,:].reshape((-1,ndim))

#         # estimation result
#         # 16, 50, 84 quantiles
#         result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
#         para_fit = result[:,0]
#         e_para_fit = (result[:,1]+result[:,2])/2.0

#         # corner plot
#         fig = corner.corner(samples, labels=para_names, show_titles=True)
#         plt.savefig(tfilepath+"corner.png")
#         plt.close()

#     else: 
#         res = basinhopping(minus_lnlikelihood, para_guess, minimizer_kwargs={"bounds":para_limits})
#         # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
#         para_fit = res.x
#         print(para_guess, para_fit)
#         e_para_fit = None
#         samples = None


#     # result plot
#     fig = plt.figure(figsize=(12,12))
#     axes = fig.subplots(nrows=2, ncols=1)
#     obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
#     obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
#     # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

#     # calculate best fitted results
#     yfit, hdist, number_reduction_factor = model(para_fit)
#     Ndata = hdist.shape[0]
#     idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

#     if distance=="vertical":
#         xdata = xpdv
#         ydata = ypdv + fp*para_fit[1]
#     else:
#         xdata = xpdv + fp*para_fit[1]
#         ydata = ypdv

#     xfit, yfit = xdata[idx], ydata[idx]
#     hdist_fit, xfit, yfit = distance_to_edge(xfit, yfit, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
#     hdist_fit = hdist_fit + para_fit[0]
#     obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
#     obj_fit.fit(ifmcmc=False)
#     axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

#     alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
#     axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
#     axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
#     axes[0].text(0.01, 0.85, "Diagram: {:s}".format(diagram), **alignment)
#     axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

#     axes[0].legend()

#     axes[0].grid(True)
#     axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#     axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

#     # diagram axes[1]
#     axes[1].plot(xobs, yobs, "r.", ms=1)
#     axes[1].plot(xfit, yfit, "b.", ms=1)
#     axes[1].plot(xedge_obs, yedge_obs, "k--")
#     axes[1].plot(xedge_pdv, yedge_pdv, "k-")
#     axes[1].grid(True)
#     if diagram=="mr":
#         axes[1].axis([0., 3., 5., 20.])
#     if diagram=="tnu":
#         axes[1].axis([10, 150, 2.0, 10.0])

#     # fill weighted region
#     xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
#     axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

#     plt.savefig(tfilepath+"result.png")
#     plt.close()


#     # save data
#     data = {"samples":samples, "ndim":ndim,
#             "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
#             "diagram":diagram, "distance":distance, 
#             "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
#             "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
#             "number_reduction_factor":number_reduction_factor}

#     np.save(tfilepath+"data", data)
#     return

# def heb_llim_fit(xobso, yobso, eobso, edges_obs, tck_obs, tp_obs,
#         xpdvo, ypdvo, edges_pdv, tck_pdv, tp_pdv,
#         diagram, distance, hist_model, filepath, ifmcmc=False, nburn=None, nsteps=None):
    
#     print("diagram:",diagram)
#     print("distance:",distance)

#     global obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight, model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost
#     global fp1, fp2
#     # filepath
#     tfilepath = filepath

#     # set up observations
#     xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
#     xobs, yobs, eobs = xobso, yobso, eobso

#     # set up models
#     xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
#     xpdv, ypdv = xpdvo, ypdvo


#     # calculate Kepler distance
#     hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
#     obj_obs = distfit(hdist_obs, hist_model)
#     obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
#     obj_obs.fit(ifmcmc=False)

#     # calculate Galaxia distance
#     hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
#     obj_pdv = distfit(hdist_pdv, hist_model, bins=obj_obs.bins)
#     obj_pdv.fit(ifmcmc=False)

#     # run mcmc with ensemble sampler
#     ndim, nwalkers = 2, 200
#     nburn = 2000 if (nburn is None) else nburn
#     nsteps = 1000 if (nsteps is None) else nsteps

#     para_names = ["shift", "scatter"]
#     xd = np.abs(obj_obs.para_fit[1]-obj_pdv.para_fit[1])
#     if diagram == "tnu":
#         para_limits = [[-3*xd, 3*xd], [0., 0.08]]
#         para_guess = [0., 0.005]
#     else:
#         para_limits = [[-3*xd, 3*xd], [0., 0.20]]
#         para_guess = [0., 0.005]        

#     Ndata = xpdv.shape[0]

#     # tied to model_heb
#     weight = np.zeros(obj_obs.histx.shape, dtype=bool)
#     sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
#     idx = (obj_obs.histx >= x0-3*sigma) & (obj_obs.histx <= x0+3*sigma)
#     weight[idx] = True

#     if distance=="vertical":
#         fy1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
#         fp1 = ypdv*fy1_base
#         fy2_base = np.random.normal(size=Ndata)
#         fp2 = ypdv*fy2_base
#     else:
#         # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
#         # "horizontal"
#         fx1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
#         fp1 = xpdv*fx1_base
#         fx2_base = np.random.normal(size=Ndata)
#         fp2 = xpdv*fx2_base


#     def model(theta):#, obj_obs, xpdv, ypdv):

#         # theta[0]: offset in distance
#         # theta[1]: perturb

#         # disturb with artificial scatter
#         # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

#         hdist = hdist_pdv + fp1 + fp2*theta[1]
#         hdist = hdist + theta[0]
#         obj = distfit(hdist, hist_model, bins=obj_obs.bins)

#         # normalize the number of points in the weighted region
#         if np.sum(obj.histy[weight])!=0:
#             number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
#         else:
#             number_reduction_factor = 0.
#         histy = obj.histy * number_reduction_factor
#         return histy, hdist, number_reduction_factor

#     def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
#         histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
#         d, m = obj_obs.histy[weight], histy[weight]
#         if m[m==0].shape[0] == 0:
#             logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
#             lnL =  np.sum(d*np.log(m)-m-logdfact)
#             # print(theta, lnL)
#             return lnL
#         else:
#             # print(theta, "d")
#             return -np.inf

#     def minus_lnlikelihood(theta):
#         return -lnlikelihood(theta)

#     def lnprior(theta):#, para_limits):
#         for i in range(len(theta)):
#             if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
#                 return -np.inf
#         return 0.

#     def lnpost(theta):#, para_limits, obj_obs, xpdv, ypdv):
#         lp = lnprior(theta)#, para_limits)
#         if np.isfinite(lp):
#             lnL = lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
#             return lnL
#         else:
#             return -np.inf

#     if ifmcmc:
#         print("enabling Ensemble sampler.")
#         print(para_guess)
#         # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
#         pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

#         # with Pool() as pool:
#         # if True:
#         # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)#, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
#         # # burn-in
#         print("start burning in. nburn:", nburn)
#         for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
#                 display_bar(j, nburn)
#                 pass
#         sys.stdout.write("\n")
#         pos, _, _ = result
#         sampler.reset()

#         # actual iteration
#         print("start iterating. nsteps:", nsteps)
#         for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
#                 display_bar(j, nsteps)
#                 pass
#         sys.stdout.write("\n")

#         # modify samples
#         samples = sampler.chain[:,:,:].reshape((-1,ndim))

#         # estimation result
#         # 16, 50, 84 quantiles
#         result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
#         para_fit = result[:,0]
#         e_para_fit = (result[:,1]+result[:,2])/2.0

#         # corner plot
#         fig = corner.corner(samples, labels=para_names, show_titles=True)
#         plt.savefig(tfilepath+"corner.png")
#         plt.close()

#     else: 
#         res = basinhopping(minus_lnlikelihood, para_guess, minimizer_kwargs={"bounds":para_limits})
#         # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
#         para_fit = res.x
#         print(para_guess, para_fit)
#         e_para_fit = None
#         samples = None


#     # result plot
#     fig = plt.figure(figsize=(12,12))
#     axes = fig.subplots(nrows=2, ncols=1)
#     obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
#     obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
#     # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

#     # calculate best fitted results
#     yfit, hdist, number_reduction_factor = model(para_fit)
#     Ndata = hdist.shape[0]
#     idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

#     if distance=="vertical":
#         xdata = xpdv
#         ydata = ypdv + fp1 + fp2*para_fit[1]
#     else:
#         xdata = xpdv + fp1 + fp2*para_fit[1]
#         ydata = ypdv

#     xfit, yfit = xdata[idx], ydata[idx]
#     hdist_fit, xfit, yfit = distance_to_edge(xfit, yfit, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
#     hdist_fit = hdist_fit + para_fit[0]
#     obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
#     obj_fit.fit(ifmcmc=False)
#     axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

#     alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
#     axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
#     axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
#     axes[0].text(0.01, 0.85, "Diagram: {:s}".format(diagram), **alignment)
#     axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

#     axes[0].legend()

#     axes[0].grid(True)
#     axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#     axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

#     # diagram axes[1]
#     axes[1].plot(xobs, yobs, "r.", ms=1)
#     axes[1].plot(xfit, yfit, "b.", ms=1)
#     axes[1].plot(xedge_obs, yedge_obs, "k--")
#     axes[1].plot(xedge_pdv, yedge_pdv, "k-")
#     axes[1].grid(True)
#     if diagram=="mr":
#         axes[1].axis([0., 3., 5., 20.])
#     if diagram=="tnu":
#         axes[1].axis([10, 150, 2.0, 10.0])

#     # fill weighted region
#     xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
#     axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

#     plt.savefig(tfilepath+"result.png")
#     plt.close()


#     # save data
#     data = {"samples":samples, "ndim":ndim,
#             "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
#             "diagram":diagram, "distance":distance, 
#             "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
#             "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
#             "number_reduction_factor":number_reduction_factor}

#     np.save(tfilepath+"data", data)
#     return




# def rgb_ulim_fit(xobso, yobso, bump_obs,
#         xpdvo, ypdvo, bump_pdv, 
#         variable, distance, hist_model, filepath, ifmcmc=False, nburn=None, nsteps=None):
    
#     print("variable:",variable)
#     print("distance:",distance)

#     # global obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight, model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost
#     # global fp
#     # filepath
#     tfilepath = filepath

#     # set up observations
#     xobs, yobs = xobso, yobso

#     # set up models
#     xpdv, ypdv = xpdvo, ypdvo

#     if variable == "numax":
#         bins = np.arange(-30, 30, 2.0)
#     if variable == "dnu":
#         bins = np.arange(-3, 3, 0.2)
#     if variable == "radius":
#         bins = np.arange(-2, 2, 0.2)
#     if variable == "mass":
#         bins = np.arange(-0.4, 0.4, 0.05)

#     # calculate Kepler distance
#     hdist_obs, xobs, yobs = distance_to_bump(xobs, yobs, bump_obs, distance=distance)
#     obj_obs = distfit(hdist_obs, hist_model, bins=bins)
#     obj_obs.fit(ifmcmc=False)

#     # calculate Galaxia distance
#     hdist_pdv, xpdv, ypdv = distance_to_bump(xpdv, ypdv, bump_pdv, distance=distance)
#     obj_pdv = distfit(hdist_pdv, hist_model, bins=bins)
#     obj_pdv.fit(ifmcmc=False)

#     # run mcmc with ensemble sampler
#     ndim, nwalkers = 2, 200
#     nburn = 2000 if (nburn is None) else nburn
#     nsteps = 1000 if (nsteps is None) else nsteps

#     para_names = ["shift", "scatter"]
#     sig = obj_obs.para_fit[0]
#     if variable in ["dnu", "numax"]:
#         para_limits = [[-sig, sig], [0., 0.30]]
#         para_guess = [0., 0.005]
#     else:
#         para_limits = [[-sig, sig], [0., 0.30]]
#         para_guess = [0., 0.005]  

#     Ndata = xpdv.shape[0]

#     # tied to model_rgb
#     weight = np.zeros(obj_obs.histx.shape, dtype=bool)
#     sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
#     idx = (obj_obs.histx >= x0-4*sigma) & (obj_obs.histx <= x0+4*sigma)
#     weight[idx] = True

#     if distance=="vertical":
#         fy2_base = np.random.normal(size=Ndata)
#         fp = ypdv*fy2_base
#     else:
#         # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
#         # "horizontal"
#         fx2_base = np.random.normal(size=Ndata)
#         fp = xpdv*fx2_base
 

#     def model(theta):#, obj_obs, xpdv, ypdv):

#         # theta[0]: offset in distance
#         # theta[1]: perturb

#         # disturb with artificial scatter
#         # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

#         hdist = hdist_pdv + fp*theta[1]
#         hdist = hdist + theta[0]
#         obj = distfit(hdist, hist_model, bins=obj_obs.bins)

#         # normalize the number of points in the weighted region
#         if np.sum(obj.histy[weight])!=0:
#             number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
#         else:
#             number_reduction_factor = 0.
#         histy = obj.histy * number_reduction_factor
#         return histy, hdist, number_reduction_factor

#     def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
#         histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
#         d, m = obj_obs.histy[weight], histy[weight]
#         if m[m==0].shape[0] == 0:
#             logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
#             lnL =  np.sum(d*np.log(m)-m-logdfact)
#             # print(theta, lnL)
#             return lnL
#         else:
#             # print(theta, "d")
#             return -np.inf

#     def minus_lnlikelihood(theta):
#         return -lnlikelihood(theta)

#     def lnprior(theta):#, para_limits):
#         for i in range(len(theta)):
#             if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
#                 return -np.inf
#         return 0.

#     def lnpost(theta):#, para_limits, obj_obs, xpdv, ypdv):
#         lp = lnprior(theta)#, para_limits)
#         if np.isfinite(lp):
#             lnL = lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
#             return lnL
#         else:
#             return -np.inf

#     if ifmcmc:
#         print("enabling Ensemble sampler.")
#         print(para_guess)
#         # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
#         pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

#         # with Pool() as pool:
#         # if True:
#         # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)#, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
#         # # burn-in
#         print("start burning in. nburn:", nburn)
#         for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
#                 display_bar(j, nburn)
#                 pass
#         sys.stdout.write("\n")
#         pos, _, _ = result
#         sampler.reset()

#         # actual iteration
#         print("start iterating. nsteps:", nsteps)
#         for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
#                 display_bar(j, nsteps)
#                 pass
#         sys.stdout.write("\n")

#         # modify samples
#         samples = sampler.chain[:,:,:].reshape((-1,ndim))

#         # estimation result
#         # 16, 50, 84 quantiles
#         result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
#         para_fit = result[:,0]
#         e_para_fit = (result[:,1]+result[:,2])/2.0

#         # corner plot
#         fig = corner.corner(samples, labels=para_names, show_titles=True)
#         plt.savefig(tfilepath+"corner.png")
#         plt.close()

#     else: 
#         res = basinhopping(minus_lnlikelihood, para_guess, minimizer_kwargs={"bounds":para_limits})
#         # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
#         para_fit = res.x
#         print(para_guess, para_fit)
#         e_para_fit = None
#         samples = None


#     # result plot
#     fig = plt.figure(figsize=(12,12))
#     axes = fig.subplots(nrows=2, ncols=1)
#     obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
#     obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
#     # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

#     # calculate best fitted results
#     yfit, hdist, number_reduction_factor = model(para_fit)
#     Ndata = hdist.shape[0]
#     idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

#     if distance=="vertical":
#         xdata = xpdv
#         ydata = ypdv + fp*para_fit[1]
#     else:
#         xdata = xpdv + fp*para_fit[1]
#         ydata = ypdv

#     xfit, yfit = xdata[idx], ydata[idx]
#     hdist_fit, xfit, yfit = distance_to_bump(xfit, yfit, bump_pdv, distance=distance)
#     hdist_fit = hdist_fit + para_fit[0]
#     obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
#     obj_fit.fit(ifmcmc=False)
#     axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

#     alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
#     axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
#     axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
#     axes[0].text(0.01, 0.85, "Variable: {:s}".format(variable), **alignment)
#     axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

#     axes[0].legend()

#     axes[0].grid(True)
#     axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#     axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

#     # diagram axes[1]
#     axes[1].plot(xobs, yobs, "r.", ms=1)
#     axes[1].plot(xfit, yfit, "b.", ms=1)
#     axes[1].grid(True)


#     if variable=="dnu":
#         axes[1].axis([5500, 4000, 20, 1])
#     if variable=="numax":
#         axes[1].axis([5500, 4000, 200, 10])
#     if variable in ["mass", "radius"]:
#         axes[1].axis([2.6, 0.5, 1, 20])
#     xlim = np.sort(np.array(list(axes[1].get_xlim())))
#     xbump = np.linspace(xlim[0], xlim[1], 10)
#     ybump_obs = bump_obs[0]*xbump + bump_obs[1]
#     ybump_pdv = bump_pdv[0]*xbump + bump_pdv[1]
#     axes[1].plot(xbump, ybump_obs, "k--")
#     axes[1].plot(xbump, ybump_pdv, "k-")


#     # fill weighted region
#     xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
#     axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

#     plt.savefig(tfilepath+"result.png")
#     plt.close()


#     # save data
#     data = {"samples":samples, "ndim":ndim,
#             "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
#             "variable":variable, "distance":distance, 
#             "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
#             "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
#             "number_reduction_factor":number_reduction_factor}

#     np.save(tfilepath+"data", data)
#     return



# def rgb_llim_fit(xobso, yobso, eobso, bump_obs,
    #     xpdvo, ypdvo, bump_pdv,
    #     variable, distance, hist_model, filepath, ifmcmc=False, nburn=None, nsteps=None):
    
    # print("variable:",variable)
    # print("distance:",distance)

    # # global obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight, model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost
    # # global fp1, fp2
    # # filepath
    # tfilepath = filepath

    # # set up observations
    # xobs, yobs, eobs = xobso, yobso, eobso

    # # set up models
    # xpdv, ypdv = xpdvo, ypdvo

    # if variable == "numax":
    #     bins = np.arange(-30, 30, 2.0)
    # if variable == "dnu":
    #     bins = np.arange(-3, 3, 0.2)
    # if variable == "radius":
    #     bins = np.arange(-2, 2, 0.2)
    # if variable == "mass":
    #     bins = np.arange(-0.4, 0.4, 0.05)

    # # calculate Kepler distance
    # hdist_obs, xobs, yobs = distance_to_bump(xobs, yobs, bump_obs, distance=distance)
    # obj_obs = distfit(hdist_obs, hist_model, bins=bins)
    # obj_obs.fit(ifmcmc=False)

    # # calculate Galaxia distance
    # hdist_pdv, xpdv, ypdv = distance_to_bump(xpdv, ypdv, bump_pdv, distance=distance)
    # obj_pdv = distfit(hdist_pdv, hist_model, bins=bins)
    # obj_pdv.fit(ifmcmc=False)

    # # run mcmc with ensemble sampler
    # ndim, nwalkers = 2, 200
    # nburn = 2000 if (nburn is None) else nburn
    # nsteps = 1000 if (nsteps is None) else nsteps

    # para_names = ["shift", "scatter"]
    # sig = obj_obs.para_fit[0]
    # if variable in ["dnu", "numax"]:
    #     para_limits = [[-sig, sig], [0., 0.30]]
    #     para_guess = [0., 0.005]
    # else:
    #     para_limits = [[-sig, sig], [0., 0.30]]
    #     para_guess = [0., 0.005]      

    # Ndata = xpdv.shape[0]

    # # tied to model_heb
    # weight = np.zeros(obj_obs.histx.shape, dtype=bool)
    # sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
    # idx = (obj_obs.histx >= x0-4*sigma) & (obj_obs.histx <= x0+4*sigma)
    # weight[idx] = True

    # eobs_cut = eobs[np.abs(hdist_obs) <= 3*sigma]

    # if distance=="vertical":
    #     fy1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs_cut), Ndata)
    #     fp1 = ypdv*fy1_base
    #     # fp1 = fy1_base
    #     fy2_base = np.random.normal(size=Ndata)
    #     fp2 = ypdv*fy2_base
    # else:
    #     # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
    #     # "horizontal"
    #     fx1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs_cut), Ndata)
    #     fp1 = xpdv*fx1_base
    #     # fp1 = fx1_base
    #     fx2_base = np.random.normal(size=Ndata)
    #     fp2 = xpdv*fx2_base


    # def model(theta):#, obj_obs, xpdv, ypdv):

    #     # theta[0]: offset in distance
    #     # theta[1]: perturb

    #     # disturb with artificial scatter
    #     # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

    #     hdist = hdist_pdv + fp1 + fp2*theta[1]
    #     hdist = hdist + theta[0]
    #     obj = distfit(hdist, hist_model, bins=obj_obs.bins)

    #     # normalize the number of points in the weighted region
    #     if np.sum(obj.histy[weight])!=0:
    #         number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
    #     else:
    #         number_reduction_factor = 0.
    #     histy = obj.histy * number_reduction_factor
    #     return histy, hdist, number_reduction_factor

    # def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
    #     histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
    #     d, m = obj_obs.histy[weight], histy[weight]
    #     if m[m==0].shape[0] == 0:
    #         logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
    #         lnL =  np.sum(d*np.log(m)-m-logdfact)
    #         # print(theta, lnL)
    #         return lnL
    #     else:
    #         # print(theta, "d")
    #         return -np.inf

    # def minus_lnlikelihood(theta):
    #     return -lnlikelihood(theta)

    # def lnprior(theta):#, para_limits):
    #     for i in range(len(theta)):
    #         if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
    #             return -np.inf
    #     return 0.

    # def lnpost(theta):#, para_limits, obj_obs, xpdv, ypdv):
    #     lp = lnprior(theta)#, para_limits)
    #     if np.isfinite(lp):
    #         lnL = lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
    #         return lnL
    #     else:
    #         return -np.inf

    # if ifmcmc:
    #     print("enabling Ensemble sampler.")
    #     print(para_guess)
    #     # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
    #     pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

    #     # with Pool() as pool:
    #     # if True:
    #     # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)#, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
    #     # # burn-in
    #     print("start burning in. nburn:", nburn)
    #     for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
    #             display_bar(j, nburn)
    #             pass
    #     sys.stdout.write("\n")
    #     pos, _, _ = result
    #     sampler.reset()

    #     # actual iteration
    #     print("start iterating. nsteps:", nsteps)
    #     for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    #             display_bar(j, nsteps)
    #             pass
    #     sys.stdout.write("\n")

    #     # modify samples
    #     samples = sampler.chain[:,:,:].reshape((-1,ndim))

    #     # estimation result
    #     # 16, 50, 84 quantiles
    #     result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #             zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
    #     para_fit = result[:,0]
    #     e_para_fit = (result[:,1]+result[:,2])/2.0

    #     # corner plot
    #     fig = corner.corner(samples, labels=para_names, show_titles=True)
    #     plt.savefig(tfilepath+"corner.png")
    #     plt.close()

    # else: 
    #     res = basinhopping(minus_lnlikelihood, para_guess, minimizer_kwargs={"bounds":para_limits})
    #     # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
    #     para_fit = res.x
    #     print(para_guess, para_fit)
    #     e_para_fit = None
    #     samples = None


    # # result plot
    # fig = plt.figure(figsize=(12,12))
    # axes = fig.subplots(nrows=2, ncols=1)
    # obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
    # obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
    # # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

    # # calculate best fitted results
    # yfit, hdist, number_reduction_factor = model(para_fit)
    # Ndata = hdist.shape[0]
    # idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

    # if distance=="vertical":
    #     xdata = xpdv
    #     ydata = ypdv + fp1 + fp2*para_fit[1]
    # else:
    #     xdata = xpdv + fp1 + fp2*para_fit[1]
    #     ydata = ypdv

    # xfit, yfit = xdata[idx], ydata[idx]
    # hdist_fit, xfit, yfit = distance_to_bump(xfit, yfit, bump_pdv, distance=distance)
    # hdist_fit = hdist_fit + para_fit[0]
    # obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
    # obj_fit.fit(ifmcmc=False)
    # axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

    # alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
    # axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
    # axes[0].text(0.01, 0.90, "Scatter: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
    # axes[0].text(0.01, 0.85, "Variable: {:s}".format(variable), **alignment)
    # axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

    # axes[0].legend()

    # axes[0].grid(True)
    # axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
    # axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

    # # diagram axes[1]
    # axes[1].plot(xobs, yobs, "r.", ms=1)
    # axes[1].plot(xfit, yfit, "b.", ms=1)
    # axes[1].grid(True)

    # if variable=="dnu":
    #     axes[1].axis([5500, 4000, 20, 1])
    # if variable=="numax":
    #     axes[1].axis([5500, 4000, 200, 10])
    # if variable in ["mass", "radius"]:
    #     axes[1].axis([2.6, 0.5, 1, 20])
    # xlim = np.sort(np.array(list(axes[1].get_xlim())))
    # xbump = np.linspace(xlim[0], xlim[1], 10)
    # ybump_obs = bump_obs[0]*xbump + bump_obs[1]
    # ybump_pdv = bump_pdv[0]*xbump + bump_pdv[1]
    # axes[1].plot(xbump, ybump_obs, "k--")
    # axes[1].plot(xbump, ybump_pdv, "k-")


    # # fill weighted region
    # xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
    # axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

    # plt.savefig(tfilepath+"result.png")
    # plt.close()


    # # save data
    # data = {"samples":samples, "ndim":ndim,
    #         "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
    #         "variable":variable, "distance":distance, 
    #         "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
    #         "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
    #         "number_reduction_factor":number_reduction_factor}

    # np.save(tfilepath+"data", data)
    # return



# def sharpness_fit_rescale_mcmc(xobso, yobso, eobso, edges_obs, tck_obs, tp_obs,
#         xpdvo, ypdvo, edges_pdv, tck_pdv, tp_pdv,
#         diagram, distance, hist_model, filepath, ifmcmc=False):
    
#     print("diagram:",diagram)
#     print("distance:",distance)

#     global obj_obs, obj_pdv, xpdv, ypdv, Ndata, weight
#     #model, lnprior, lnlikelihood, minus_lnlikelihood, lnpost, 
#     global fp
#     # filepath
#     tfilepath = filepath

#     # set up observations
#     xedge_obs, yedge_obs = edges_obs[:,0], edges_obs[:,1]
#     xobs, yobs, eobs = xobso, yobso, eobso

#     # set up models
#     xedge_pdv, yedge_pdv = edges_pdv[:,0], edges_pdv[:,1]
#     xpdv, ypdv = xpdvo, ypdvo


#     # calculate Kepler distance
#     hdist_obs, xobs, yobs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, tp_obs, diagram=diagram, distance=distance)
#     obj_obs = distfit(hdist_obs, hist_model)
#     obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
#     obj_obs.fit(ifmcmc=False)

#     # calculate Galaxia distance
#     hdist_pdv, xpdv, ypdv = distance_to_edge(xpdv, ypdv, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
#     obj_pdv = distfit(hdist_pdv, hist_model, bins=obj_obs.bins)
#     obj_pdv.fit(ifmcmc=False)

#     # run mcmc with ensemble sampler
#     ndim, nwalkers, nburn, nsteps = 2, 200, 2000, 1000

#     para_names = ["shift", "scale_factor"]
#     xd = np.abs(obj_obs.para_fit[1]-obj_pdv.para_fit[1])
#     para_limits = [[-3*xd, 3*xd], [0., 3.0]]
#     para_guess = [0., 1.0]

#     Ndata = xpdv.shape[0]

#     # tied to model6
#     weight = np.zeros(obj_obs.histx.shape, dtype=bool)
#     sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
#     idx = (obj_obs.histx >= x0-3*sigma) & (obj_obs.histx <= x0+3*sigma)
#     weight[idx] = True

#     if distance=="vertical":
#         fy1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
#         fp = ypdv*fy1_base
#     else:
#         # fx1 = np.array([random.gauss(0,1) for i in range(Ndata)]) * 10.0**scipy.signal.resample(np.log10(e_xobs), Ndata) * scalar
#         # "horizontal"
#         fx1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs), Ndata)
#         fp = xpdv*fx1_base
   

#     def model(theta):#, obj_obs, xpdv, ypdv):

#         # theta[0]: offset in distance
#         # theta[1]: perturb

#         # disturb with artificial scatter
#         # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

#         hdist = hdist_pdv + fp*theta[1]
#         hdist = hdist + theta[0]
#         obj = distfit(hdist, hist_model, bins=obj_obs.bins)

#         # normalize the number of points in the weighted region
#         if np.sum(obj.histy[weight])!=0:
#             number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
#         else:
#             number_reduction_factor = 0.
#         histy = obj.histy * number_reduction_factor
#         return histy, hdist, number_reduction_factor

#     def lnlikelihood(theta):#, obj_obs, xpdv, ypdv):
#         histy, _, _ = model(theta)#, obj_obs, xpdv, ypdv)
#         d, m = obj_obs.histy[weight], histy[weight]
#         if m[m==0].shape[0] == 0:
#             logdfact = scipy.special.gammaln(d+1) #np.array([np.sum(np.log(np.arange(1,id+0.1))) for id in d])
#             lnL =  np.sum(d*np.log(m)-m-logdfact)
#             # print(theta, lnL)
#             return lnL
#         else:
#             # print(theta, "d")
#             return -np.inf

#     def minus_lnlikelihood(theta):
#         return -lnlikelihood(theta)

#     def lnprior(theta):#, para_limits):
#         for i in range(len(theta)):
#             if not (para_limits[i][0] <= theta[i] <= para_limits[i][1] ):
#                 return -np.inf
#         return 0.

#     def lnpost(theta):#, para_limits, obj_obs, xpdv, ypdv):
#         lp = lnprior(theta)#, para_limits)
#         if np.isfinite(lp):
#             lnL = lnlikelihood(theta)#, obj_obs, xpdv, ypdv)
#             return lnL
#         else:
#             return -np.inf

#     if ifmcmc:
#         print("enabling Ensemble sampler.")
#         print(para_guess)
#         # pos0=[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
#         pos0 = [np.array([np.random.uniform(low=para_limits[idim][0], high=para_limits[idim][1]) for idim in range(ndim)]) for iwalker in range(nwalkers)]

#         with Pool() as pool:
#         # if True:
#             # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost) #, pool=pool #, args=(para_limits, obj_obs, xpdv, ypdv))
#             sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool) #, args=(para_limits, obj_obs, xpdv, ypdv))
#             # # burn-in
#             print("start burning in. nburn:", nburn)
#             for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
#                     display_bar(j, nburn)
#                     pass
#             sys.stdout.write("\n")
#             pos, _, _ = result
#             sampler.reset()

#             # actual iteration
#             print("start iterating. nsteps:", nsteps)
#             for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
#                     display_bar(j, nsteps)
#                     pass
#             sys.stdout.write("\n")

#         # modify samples
#         samples = sampler.chain[:,:,:].reshape((-1,ndim))

#         # estimation result
#         # 16, 50, 84 quantiles
#         result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
#         para_fit = result[:,0]
#         e_para_fit = (result[:,1]+result[:,2])/2.0

#         # corner plot
#         fig = corner.corner(samples, labels=para_names, show_titles=True)
#         plt.savefig(tfilepath+"corner.png")

#     else: 
#         res = basinhopping(minus_lnlikelihood, para_guess, minimizer_kwargs={"bounds":para_limits})
#         # res = minimize(minus_lnlikelihood, para_guess, bounds=para_limits)
#         para_fit = res.x
#         print(para_guess, para_fit)
#         e_para_fit = None
#         samples = None


#     # result plot
#     fig = plt.figure(figsize=(12,12))
#     axes = fig.subplots(nrows=2, ncols=1)
#     obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Kepler", "zorder":100})
#     obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Kepler fit", "linestyle":"--", "zorder":100})
            
#     # obj_pdv.plot_hist(ax=axes[0], histkwargs={"color":"green", "label":"Galaxia initial model"})

#     # calculate best fitted results
#     yfit, hdist, number_reduction_factor = model(para_fit)
#     Ndata = hdist.shape[0]
#     idx = reduce_samples(Ndata, Ndata*number_reduction_factor)

#     if distance=="vertical":
#         xdata = xpdv
#         ydata = ypdv + fp*para_fit[1]
#     else:
#         xdata = xpdv + fp*para_fit[1]
#         ydata = ypdv

#     xfit, yfit = xdata[idx], ydata[idx]
#     hdist_fit, xfit, yfit = distance_to_edge(xfit, yfit, xedge_pdv, yedge_pdv, tck_pdv, tp_pdv, diagram=diagram, distance=distance)
#     hdist_fit = hdist_fit + para_fit[0]
#     obj_fit = distfit(hdist_fit, hist_model, bins=obj_obs.bins)
#     obj_fit.fit(ifmcmc=False)
#     axes[0].step(obj_fit.histx, obj_fit.histy, **{"color":"blue", "label":"Galaxia best fitted model"})

#     alignment = {"ha":"left", "va":"top", "transform":axes[0].transAxes}
#     axes[0].text(0.01, 0.95, "Offset: {:0.2f} (initial: {:0.2f})".format(para_fit[0], para_guess[0]), **alignment)
#     axes[0].text(0.01, 0.90, "Scale factor: {:0.2f}% (initial: {:0.2f}%)".format(para_fit[1]*100, para_guess[1]*100), **alignment) 
#     axes[0].text(0.01, 0.85, "Diagram: {:s}".format(diagram), **alignment)
#     axes[0].text(0.01, 0.80, "Distance: {:s}".format(distance), **alignment)

#     axes[0].legend()

#     axes[0].grid(True)
#     axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#     axes[0].set_ylim(0., obj_obs.histy.max()*1.5)

#     # diagram axes[1]
#     axes[1].plot(xobs, yobs, "r.", ms=1)
#     axes[1].plot(xfit, yfit, "b.", ms=1)
#     axes[1].plot(xedge_obs, yedge_obs, "k--")
#     axes[1].plot(xedge_pdv, yedge_pdv, "k-")
#     axes[1].grid(True)
#     if diagram=="mr":
#         axes[1].axis([0., 3., 5., 20.])
#     if diagram=="tnu":
#         axes[1].axis([10, 150, 2.0, 10.0])

#     # fill weighted region
#     xmin_, xmax_ = obj_obs.histx[weight].min(), obj_obs.histx[weight].max()
#     axes[0].fill_betweenx(axes[0].get_ylim(), [xmin_, xmin_], [xmax_, xmax_], color="lightgray")

#     plt.savefig(tfilepath+"result.png")
#     plt.close()


#     # save data
#     data = {"samples":samples, "ndim":ndim,
#             "para_fit":para_fit, "e_para_fit":e_para_fit, "para_guess":para_guess,
#             "diagram":diagram, "distance":distance, 
#             "xobs":xobs, "yobs":yobs, "xpdv":xpdv, "ypdv":ypdv, "xfit":xfit, "yfit":yfit,
#             "obj_obs":obj_obs, "obj_pdv":obj_pdv, "obj_fit":obj_fit,
#             "number_reduction_factor":number_reduction_factor}

#     np.save(tfilepath+"data", data)
#     return



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

# def plot_scatter_zvalue(filepath, bins=3, distance="horizontal", diagram="tnu", iferr=False):
#     if not (distance in ["horizontal", "vertical"]):
#         raise ValueError("distance should be horizontal or vertical.")
#     if not (diagram in ["tnu", "mr"]):
#         raise ValueError("diagram should be tnu or mr.")

#     # read zvalue and zname
#     data = np.load(filepath+str(bins-1)+"/data.npy", allow_pickle=True).tolist()
#     zvalue_limits = data["zvalue_limits"]
#     zvalue_name = data["zvalue_name"]
#     zvalue = np.array([(zvalue_limits[0][i] + zvalue_limits[1][i])/2.0 for i in range(len(zvalue_limits[0]))])
    
#     # calculate scatter in each folder
#     fs, e_fs = np.zeros(len(zvalue)), np.zeros(len(zvalue))
#     for i in range(len(zvalue)):
#         data = np.load(filepath+str(i)+"/data.npy", allow_pickle=True).tolist()
#         if distance=="horizontal":
#             cp_scatter = data["yperturb"][0]
#             scatter = data["xperturb"]
#             slope = data["sharpness_med"][:,0]
#             eslope = data["sharpness_std"][:,0]
#         else:
#             scatter = data["yperturb"]
#             cp_scatter = data["xperturb"][0]
#             slope = data["sharpness_med"][0,:]
#             eslope = data["sharpness_std"][0,:]
#         slope_obs = data["sharpness_obs"]

#         x, y = scatter*100, slope-slope_obs
#         idx = np.abs(y) == np.abs(y).min()
#         fs[i] = x[idx][0]

#         if iferr:
#             e_slope_obs = data["e_sharpness_obs"]
#             idx = np.abs(y-e_slope_obs) == np.abs(y-e_slope_obs).min()
#             xmin = x[idx][0]
#             idx = np.abs(y+e_slope_obs) == np.abs(y+e_slope_obs).min()
#             xmax = x[idx][0]
#             e_fs[i] = np.abs(xmax-xmin)/2.

#     # initiate a plot
#     fig = plt.figure(figsize=(6,4))
#     axes = fig.subplots(nrows=1, ncols=1, squeeze=False).reshape(-1,)
#     axes[0].errorbar(zvalue, fs, yerr=e_fs, c="black", linestyle="--", marker=".", ms=12, capsize=2)
#     # axes[0].plot(zvalue, fs, "k--")
#     axes[0].set_ylim([0., fs.max()*1.1])#fs.max()+(fs.max()-fs.min())*0.2

#     if diagram=="tnu":
#         if distance=="horizontal":
#             axes[0].set_ylabel("Fitted scatter in numax relation [%]")
#         if distance=="vertical":
#             axes[0].set_ylabel("Fitted scatter in dnu relation [%]")            
#     if diagram=="mr":
#         if distance=="horizontal":
#             axes[0].set_ylabel("Fitted scatter in M relation [%]")
#         if distance=="vertical":
#             axes[0].set_ylabel("Fitted scatter in R relation [%]") 

#     axes[0].set_xlabel(zvalue_name)
#     plt.tight_layout()
#     plt.savefig(filepath+"scatter_zvalue.png")
#     plt.close()
#     return



