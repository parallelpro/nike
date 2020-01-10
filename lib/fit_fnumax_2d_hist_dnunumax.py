'''
Fit fnumax and fdnu using method 2 - directly compare distributions on the nike plane.
'''

import numpy as np 
from astropy.io import ascii
import matplotlib.pyplot as plt 

import emcee
import corner
import sys
import os

import scipy
import scipy.spatial
import scipy.interpolate
import scipy.optimize


# sys.path.append("/Users/yaguang/Onedrive/Work/nike/code/")
# from fast_histogram import histogram2d

## read in data
# synthetic stars
import ebf
synp = ebf.read("sample/kepler_galaxia_mrtd5.ebf")
Nstar = synp["alpha"].shape[0]
factor = int(Nstar/16000)
idx = np.arange(0,int(Nstar/factor))*factor
for key in synp.keys():
    synp[key] = synp[key][idx]
idx = (synp["evstate"]==2) & (np.isfinite(synp["numax"]))
xpdv, ypdv, ypdvt = synp["numax"][idx], synp["numax"][idx]**0.75/synp["dnu"][idx], synp["dnu"][idx]
Npdv = xpdv.shape[0]
mass = synp["mact"][idx]
g, teff = 10.0**synp["log_grav"][idx], 10.0**synp["log_teff"][idx]


# observation stars
yu=ascii.read("sample/yu+2018.csv")
idx = (yu["Phase"]==2)
xobs, yobs, yobst = yu["numax"][idx], yu["numax"][idx]**0.75/yu["Delnu"][idx], yu["Delnu"][idx]
Nobs = xobs.shape[0]


## read in edge
# edge = np.load("sample/edge_nike.npy")
# xedge, yedge = edge[:,0], edge[:,1]
# yedget = xedge**0.75/yedge
# Nedge = xedge.shape[0]

tck = np.load("sample/spline.npy",allow_pickle=True)


class fit_1d_dnu_numax:
    def __init__(self, xobs, yobs, xmod, ymod, tck, 
                fnumax_value=[], fdnu_value=[], 
                fix_fnumax=False, fix_fdnu=False):

        self.xobs, self.yobs = xobs, yobs
        self.xmod, self.ymod = xmod, ymod
        self.tck = tck
        self.Nobs, self.Nmod = xobs.shape[0], xmod.shape[0]
        # synp["numax"][idx]**0.75/synp["dnu"][idx]
        yedge = np.linspace(np.min(xobs**0.75/yobs), np.max(xobs**0.75/yobs), 100)
        xedge = 10.0**scipy.interpolate.splev(np.log10(yedge), tck, der=0)
        self.xedge, self.yedge = xedge, xedge**0.75/yedge
        self.Nedge = xedge.shape[0]

        self.fnumax_value = fnumax_value
        self.fdnu_value = fdnu_value
        
        self.Nfdnu_thetas = 0 if fix_fdnu else len(self.fdnu_value)+1
        self.Nfnumax_thetas = 0 if fix_fnumax else len(self.fnumax_value)+1
        self.ndim = self.Nfdnu_thetas + self.Nfnumax_thetas

        self.bins = [np.arange(20, 100, 0.5), np.arange(2.0, 10.0, 0.01)]
        self.hobs, _, _ = np.histogram2d(self.xobs, self.yobs, bins=self.bins)

        return
    
    def _display_bar(self, j, nburn, width=30):
        n = int((width+1) * float(j) / nburn)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        return 0.

    def _return_2dmap_axes(self, numberOfSquareBlocks):
        # Some magic numbers for pretty axis layout.
        # stole from corner
        Kx = int(np.ceil(numberOfSquareBlocks**0.5))
        Ky = Kx if (Kx**2-numberOfSquareBlocks) < Kx else Kx-1

        factor = 2.0           # size of one side of one panel
        lbdim = 0.4 * factor   # size of left/bottom margin, default=0.2
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.30         # w/hspace size
        plotdimx = factor * Kx + factor * (Kx - 1.) * whspace
        plotdimy = factor * Ky + factor * (Ky - 1.) * whspace
        dimx = lbdim + plotdimx + trdim
        dimy = lbdim + plotdimy + trdim

        # Create a new figure if one wasn't provided.
        fig, axes = plt.subplots(Ky, Kx, figsize=(dimx, dimy), squeeze=False)

        # Format the figure.
        l = lbdim / dimx
        b = lbdim / dimy
        t = (lbdim + plotdimy) / dimy
        r = (lbdim + plotdimx) / dimx
        fig.subplots_adjust(left=l, bottom=b, right=r, top=t,
                            wspace=whspace, hspace=whspace)
        axes = np.concatenate(axes)

        return fig, axes

    def _plot_mcmc_traces(self, ndim, samples, para_names):
        fig, axes = self._return_2dmap_axes(ndim)

        for i in range(ndim):
            ax = axes[i]
            evol = samples[:,i]
            Npoints = samples.shape[0]
            ax.plot(np.arange(Npoints)/Npoints, evol, color="gray", lw=1, zorder=1)
            Nseries = int(len(evol)/15.0)
            evol_median = np.array([np.median(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
            evol_std = np.array([np.std(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
            evol_x = np.array([np.median(np.arange(Npoints)[i*Nseries:(i+1)*Nseries]/Npoints) for i in range(0,15)])
            ax.errorbar(evol_x, evol_median, yerr=evol_std, color="C0", ecolor="C0", capsize=2)
            ax.set_ylabel(para_names[i])

        for ax in axes[i+1:]:
            fig.delaxes(ax)

        return fig

    def _plot_fit_results(self, para_fit):
        dist = self.dist(self.xobs, self.yobs, self.tck)

        fdnu = 1.0
        for itheta in range(0, self.Nfdnu_thetas):
            if itheta==0:
                fdnu *= para_fit[itheta]
            else:
                fdnu *= self.fdnu_value[itheta-1]**para_fit[itheta]

        fnumax = 1.0
        for itheta in range(self.Nfdnu_thetas, self.Nfdnu_thetas+self.Nfnumax_thetas):
            if itheta==self.Nfdnu_thetas:
                fnumax *= para_fit[itheta]
            else:
                fnumax *= self.fnumax_value[itheta-1-self.Nfdnu_thetas]**para_fit[itheta]
        
        txmod, tymod = self.xmod*fnumax, self.ymod*(fnumax**0.75)/fdnu

        distp = self.dist(txmod, tymod, self.tck)

        fig = plt.figure(figsize=(12,12))
        axes = fig.subplots(nrows=2, ncols=2, squeeze=False).reshape(-1)

        ## plot distributions
        Z = np.concatenate((dist, distp))
        # axes[0].set_xlim(np.nanmin(Z), np.nanmax(Z))
        axes[0].set_xlim(-0.2, 0.4)
        bins0 = np.linspace(np.nanmin(Z), np.nanmax(Z), 1500)
        h = axes[0].hist(dist, color="red", histtype="step", label="Observations",bins=bins0, zorder=0)
        h = axes[0].hist(distp, color="black", histtype="step", label="Padova",bins=bins0, zorder=0)
        axes[0].set_xlabel("Distances to the edge on diagram 1")
        axes[0].legend()       

        ## plot diagram
        axes[1].axis([10, 200, 2.0, 10.0])
        axes[1].set_xscale("log")
        # plot observations
        axes[1].plot(self.xobs, self.yobs, "r.", markersize=2)
        axes[1].grid(which="both")
        axes[1].set_xlabel("numax")
        axes[1].set_ylabel("numax^0.75/dnu")
        # plot edge
        axes[1].plot(self.xedge, self.yedge, "b-", linewidth=2, zorder=10) 
        # plot models - padova
        axes[1].plot(txmod, tymod, "k.", markersize=2)

        ## plot 
        axes[2].hist2d(self.xobs, self.yobs, bins=self.bins)
        axes[2].plot(self.xedge, self.yedge, "w-", linewidth=2, zorder=10) 
        axes[3].hist2d(txmod, tymod, bins=self.bins)
        axes[3].plot(self.xedge, self.yedge, "w-", linewidth=2, zorder=10) 
        # axes[2].hist(, bins=self.bins)

        return fig 

    def dist(self, xdata, ydata, tck):
        # shortest distance distribution, prep for obs
        log10_xobs_edge = scipy.interpolate.splev(np.log10(ydata), tck, der=0)
        log10_xobs_edge_prime = scipy.interpolate.splev(np.log10(ydata), tck, der=1)
        dist = (log10_xobs_edge-np.log10(xdata))/(log10_xobs_edge_prime**2. + 1)**0.5
        return dist

    # def func(self, y0, tck, xd, yd):
    #     x0 = 10.0**scipy.interpolate.splev(np.log10(y0), tck)
    #     # print(xd, yd, x0, y0)
    #     return scipy.spatial.distance.euclidean([xd, yd], [x0, y0])

    # def dist(self, xdata, ydata, tck):
    #     # shortest distance distribution, prep for obs
    #     Ndata = xdata.shape[0]
    #     dist = np.zeros(Ndata)
    #     for idata in range(Ndata):
    #         xd, yd = xdata[idata], ydata[idata]
    #         print(xd, yd)
    #         output = scipy.optimize.fmin(self.func, yd, args=(tck, xd, yd), full_output=True, disp=False)
    #         dist[idata] = output[1]#scipy.spatial.distance.euclidean([xdata[i], ydata[i]], [xp, yp])
    #     return dist



    def lnprior(self, theta):
        lnp = 0.
        for itheta in range(len(theta)):
            if not (self.para_priors[itheta][0] < theta[itheta] < self.para_priors[itheta][1]):
                return -np.inf
        return lnp

    def lnlikelihood(self, theta):
        # generate the distance distribution for models

        fdnu = 1.0
        for itheta in range(0, self.Nfdnu_thetas):
            if itheta==0:
                fdnu *= theta[itheta]
            else:
                fdnu *= self.fdnu_value[itheta-1]**theta[itheta]

        fnumax = 1.0
        for itheta in range(self.Nfdnu_thetas, self.Nfdnu_thetas+self.Nfnumax_thetas):
            if itheta==self.Nfdnu_thetas:
                fnumax *= theta[itheta]
            else:
                fnumax *= self.fnumax_value[itheta-1-self.Nfdnu_thetas]**theta[itheta]

        ### method 1: 1d distance
        # distp = self.dist(self.xmod*fnumax, self.ymod*(fnumax**0.75)/fdnu, self.tck)

        # # percentile
        # qpdv = np.nanpercentile(distp, self.qs)
        # # if (not np.isfinite(qpdv[0])): print(fnumax)

        # # calculate distances between two distributions
        # sum_dist = 0.
        # q = self.qs.shape[0]
        # sum_dist = np.sum((self.qobs - qpdv)**2.0)
        # metric = 1./sum_dist

        # # change of metric
        # for irange in range(q-1):
        #     sum_dist += 0.5*((self.qobs[irange+1] - qpdv[irange+1])**2.0 + (self.qobs[irange] - qpdv[irange])**2.0)
        # metric = (1.+ 1./(q-1.)*sum_dist)**-1.

        # print(qpdv,np.log(metric))

        ### method 3
        # bins = [self.bins[0].shape[0]-1, self.bins[1].shape[0]-1] 
        # ranges = [[self.bins[0].min(), self.bins[0].max()], [self.bins[1].min(), self.bins[1].max()]]
        # hmod = histogram2d(self.xmod, self.ymod*(fnumax**0.75)/fdnu, bins=bins, range=ranges)
        hmod, _, _ = np.histogram2d(self.xmod*fnumax, self.ymod*(fnumax**0.75)/fdnu, bins=self.bins)
        
        metric = np.sum(hmod*self.hobs)

        return metric # np.log

    def lnpost(self, theta):
        lnp = self.lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        else:
            lnlikelihood = self.lnlikelihood(theta)
            return lnp + lnlikelihood

    def run(self, para_priors, nwalkers=100, nsteps=2000, nburn=1000):
        self.para_priors = para_priors
        self.para_guess = np.array([np.mean(iprior) for iprior in self.para_priors])
        # self.ndim = len(self.fdnu_value)+len(self.fnumax_value)

        # configure
        self.para_priors
        self.disto = self.dist(self.xobs, self.yobs, self.tck)
        self.qs = np.array([5, 30, 50, 70, 90])
        self.qobs = np.nanpercentile(self.disto, self.qs)
        
		# run mcmc with ensemble sampler
        print("enabling Ensemble sampler.")
        pos0 = [self.para_guess + 1.0e-8*np.random.randn(self.ndim) for j in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnpost)

        # burn-in
        print("start burning in. nburn:", nburn)
        for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
            self._display_bar(j, nburn)
        sys.stdout.write("\n")
        pos, lnpost, rstate = result
        sampler.reset()

        # actual iteration
        print("start iterating. nsteps:", nsteps)
        for j, result in enumerate(sampler.sample(pos, iterations=nsteps, lnprob0=lnpost)):
            self._display_bar(j, nsteps)
        sys.stdout.write("\n")

        # modify samples
        self.samples = sampler.chain[:,:,:].reshape((-1,self.ndim))

        # save estimation result
        # 16, 50, 84 quantiles
        result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(self.samples, [16, 50, 84],axis=0)))))
        self.para_fit = result[:,0]

        # maximum
        para_fitmax = np.zeros(self.ndim)
        for ipara in range(self.ndim):
            n, bins, _ = plt.hist(self.samples[:,ipara], bins=80)
            idx = np.where(n == n.max())[0][0]
            para_fitmax[ipara] = bins[idx:idx+1].mean()
        self.para_fitmax = para_fitmax

        self.result = np.concatenate([result, para_fitmax.reshape(self.ndim,1)], axis=1)

        # save acceptance fraction
        self.acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
        print("Mean acceptance fraction: {:0.3f}".format(self.acceptance_fraction[0]))

        return

    def output(self, filepath, ifOutputSamples=False):
        self.filepath = filepath
        if not os.path.exists(filepath): os.mkdir(filepath)

        # save samples if the switch is toggled on
        if ifOutputSamples: np.save(self.filepath+"samples.npy", self.samples)

        # save guessed parameters
        np.savetxt(self.filepath+"guess.txt", self.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

        # plot triangle and save
        para_names = ["theta{:0.0f}".format(i) for i in range(self.ndim)]
        fig = corner.corner(self.samples, labels=para_names, quantiles=(0.16, 0.5, 0.84), truths=self.para_fitmax)
        fig.savefig(self.filepath+"triangle.png")
        plt.close()

        # save estimation result
        np.savetxt(self.filepath+"summary.txt", self.result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f", "%0.8f"), header="50th quantile, 16th quantile sigma, 84th quantile sigma, maximum")

        # save mean acceptance rate
        np.savetxt(self.filepath+"acceptance_fraction.txt", self.acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

        # plot traces and save
        fig = self._plot_mcmc_traces(self.ndim, self.samples, para_names)
        plt.savefig(self.filepath+'traces.png')
        plt.close()

        # plot fitting results and save
        # power_fit = self.LikelihoodsObj.model(self.para_fit, x=self.freq)
        # power_guess = self.LikelihoodsObj.model(self.para_guess, x=self.freq)
        # fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fit,
        #                     self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
        #                     self.PriorsObj.priorGuess)
        # plt.savefig(self.filepath+"fitmedian.png")
        # plt.close()

        fig = self._plot_fit_results(self.para_fit)
        plt.savefig(self.filepath+"fit.png")
        plt.close()

        fig = self._plot_fit_results(self.para_fitmax)
        plt.savefig(self.filepath+"fitmax.png")
        plt.close()

        return

filepath = "sample/test_2d_hist_dnunumax/"
xobs, yobs = np.asarray(xobs), np.asarray(yobs)
xpdv, ypdv = np.asarray(xpdv), np.asarray(ypdv)
mass = np.asarray(mass)

obj = fit_1d_dnu_numax(xobs, yobst, xpdv, ypdvt, tck,
     fix_fdnu=True, fix_fnumax=False, fdnu_value=[g, teff], fnumax_value=[g])

# para_priors = [[0., 10.], [-10., 10.], [-10., 10.]]
# obj.run(para_priors, nburn=3000, nsteps=2000)
# obj.output(filepath)


para_fit = np.array([0.988, 0.04])
fig = obj._plot_fit_results(para_fit)
plt.savefig(filepath+"fit.png")
plt.close()
