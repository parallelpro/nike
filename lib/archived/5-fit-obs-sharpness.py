import numpy as np 
from astropy.io import ascii


rootpath = ""#"/headnode2/yali4742/nike/"
import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.append(rootpath) 
from lib.histdist import model5, distance_to_edge, distfit, model5_prob, model6_prob, half_gauss_prob
from lib.wrapper import sharpness_fit
from scipy.optimize import minimize
import os
import emcee
import corner

def return_2dmap_axes(numberOfSquareBlocks):

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

def plot_mcmc_traces(ndim, samples, para_names):
        fig, axes = return_2dmap_axes(ndim)
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

def display_bar(j, nburn, width=30):
        n = int((width+1) * float(j) / nburn)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        return

def fit(filepath, distance, diagram, hist_model, edges, tck, xobs, yobs):
        if not os.path.exists(filepath): os.mkdir(filepath)

        # edges
        xedge, yedge = edges[:,0], edges[:,1]

        # calculate observational distance
        dist = distance_to_edge(xobs, yobs, xedge, yedge, tck, diagram=diagram, distance=distance)
        # dist = dist[dist>0.]
        # dist = dist[(dist>-0.2) ]#& (dist<4)]
        obj_obs = distfit(dist, hist_model, density=True)

        hist_model.set_priors(obj_obs.histx, obj_obs.histy, dist)
        prior_guess = hist_model.prior_guess
        para_guess = hist_model.para_guess
        def lnprior(theta):
                for ip in range(len(prior_guess)):
                        if not (prior_guess[ip][0] <= theta[ip] <= prior_guess[ip][1]):
                                return -np.inf
                return 0.

        def lnlikelihood(theta):
                return np.sum(np.log(hist_model.ymodel(theta, dist)))
                # chi2 = (hist_model.ymodel(theta, obj_obs.histx) - obj_obs.histy)**2.0/2.0
                # return -np.sum((chi2))
        
        def lnpost(theta):
                lp = lnprior(theta)
                if not np.isfinite(lp):
                        return -np.inf
                else:
                        return lnlikelihood(theta)

        def minus_lnpost(theta):
                lp = lnprior(theta)
                if not np.isfinite(lp):
                        return np.inf
                else:
                        return -lnlikelihood(theta)

        ndim, nwalkers, nburn, nsteps = 3, 500, 1000, 1000

        ifmle = True
        if ifmle:
            # mle
            res = minimize(minus_lnpost, para_guess)
            para_fit = res.x

            xfit = np.linspace(obj_obs.histx.min(), obj_obs.histx.max(), 500)
            yfit = hist_model.ymodel(para_fit, xfit)

        else:
            # run mcmc with ensemble sampler
            print("enabling Ensemble sampler.")
            pos0 = [para_guess + 1.0e-6*np.random.randn(ndim) for j in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)

            # # burn-in
            print("start burning in. nburn:", nburn)
            for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
                    display_bar(j, nburn)
            sys.stdout.write("\n")
            pos, lnpost, rstate = result
            sampler.reset()

            # actual iteration
            print("start iterating. nsteps:", nsteps)
            for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
                    display_bar(j, nsteps)
            sys.stdout.write("\n")

            # modify samples
            samples = sampler.chain[:,:,:].reshape((-1,ndim))

            # save estimation result
            # 16, 50, 84 quantiles
            result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                    zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
            para_fit = result[:,0]

            result = np.concatenate([result, para_fit.reshape(ndim,1)], axis=1)

            # save acceptance fraction
            # acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
            # np.savetxt(filepath+"acceptance_fraction.txt", acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

            # save samples if the switch is toggled on
            if False: np.save(filepath+"samples.npy", samples)

            # save guessed parameters
            # np.savetxt(filepath+"guess.txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

            # plot triangle and save
            para_names = hist_model.para_name
            fig = corner.corner(samples, labels=para_names, quantiles=(0.16, 0.5, 0.84))
            fig.savefig(filepath+"triangle.png")
            plt.close()

            # plot traces and save
            fig = plot_mcmc_traces(ndim, samples, para_names)
            plt.savefig(filepath+'traces.png')
            plt.close()

            # save estimation result
            np.savetxt(filepath+"summary.txt", result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f", "%0.8f"), header="50th quantile, 16th quantile sigma, 84th quantile sigma, maximum")

            xfit = np.linspace(obj_obs.histx.min(), obj_obs.histx.max(), 500)
            yfit = hist_model.ymodel(para_fit, xfit)

        # plot fitting results and save
        fig = plt.figure(figsize=(12,6))
        axes = fig.subplots(nrows=1, ncols=1)
        axes.hist(dist,density=True,bins=250)
        obj_obs.plot_hist(ax=axes, histkwargs={"color":"red", "label":"Kepler"})
        axes.plot(xfit, yfit, "k--", label="Kepler fit")
        # obj_obs.plot_fit(ax=axes, theta=para_fit, fitkwargs={"color":"black", "label":"Kepler fit"})

        axes.grid(True)
        axes.set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
        axes.set_ylim(0., obj_obs.histy.max()*1.5)
        axes.legend()
        plt.savefig(filepath+"fitmedian.png")
        plt.close()


if __name__ == "__main__":
        # read in unperturbed mass and radius, with edges
        samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy")
        mr_edges_obs = np.load(rootpath+"sample/obs/mr_edge_samples.npy")
        mr_tck_obs = np.load(rootpath+"sample/obs/mr_spline_tck.npy", allow_pickle=True)
        tnu_edges_obs = np.load(rootpath+"sample/obs/tnu_edge_samples.npy")
        tnu_tck_obs = np.load(rootpath+"sample/obs/nike_spline_tck.npy", allow_pickle=True)

        hist_model = half_gauss_prob()

        # obs dnu
        filepath = rootpath+"sample/obs/sharpness_tnu/dnu/"
        xobs, yobs = samples_obs[:,0], samples_obs[:,1] 
        idx = (xobs>= np.min(tnu_edges_obs[:,0])) #& (yobs<=(yedge_obs.max()))
        xobs, yobs = xobs[idx], yobs[idx]
        fit(filepath, "vertical", "tnu", hist_model, tnu_edges_obs, tnu_tck_obs, xobs, yobs)

        # obs numax
        filepath = rootpath+"sample/obs/sharpness_tnu/numax/"
        xobs, yobs = samples_obs[:,0], samples_obs[:,1] 
        idx = (yobs>= np.min(tnu_edges_obs[:,1])) #& (yobs<=(yedge_obs.max()))
        xobs, yobs = xobs[idx], yobs[idx]
        fit(filepath, "horizontal", "tnu", hist_model, tnu_edges_obs, tnu_tck_obs, xobs, yobs)

        # obs radius
        filepath = rootpath+"sample/obs/sharpness_mr/radius/"
        xobs, yobs = samples_obs[:,3], samples_obs[:,4] 
        idx = xobs<=2.2
        xobs, yobs = xobs[idx], yobs[idx]
        fit(filepath, "vertical", "mr", hist_model, mr_edges_obs, mr_tck_obs, xobs, yobs)

        # obs mass
        filepath = rootpath+"sample/obs/sharpness_mr/mass/"
        xobs, yobs = samples_obs[:,3], samples_obs[:,4] 
        idx = (yobs <= np.max(mr_edges_obs[:,1]))
        xobs, yobs = xobs[idx], yobs[idx]
        fit(filepath, "horizontal", "mr", hist_model, mr_edges_obs, mr_tck_obs, xobs, yobs)