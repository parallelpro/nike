import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping

__all__ = ["distfit", "model1", "model2"]

class model1:
    def __init__(self):
        # model1
        self.prior_guess = [[0., 1000.], [1e-6, 1.], [-1, 1.], [1e-6, 3.]]
        self.para_guess = [500., 0.5, 0.5, 1.0]
        self.para_name = ["H", "gamma", "x0", "tau"]
        return

    def ymodel(self, theta, x):
        H, gamma, x0, tau = theta
        # if (x is None): x = self.histx
        ymodel = np.zeros(x.shape[0])
        idx = x<x0
        ymodel[idx] = H/((x[idx]-x0)**2./gamma**2. + 1)
        idx = x>=x0
        ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau)
        return ymodel        

class model2:
    def __init__(self):
        # # model2
        self.prior_guess = [[0., 1000.], #H
                            [1e-6, 3.], # tau1
                            [-0.1, 0.3], # x0
                            [1e-6, 3.], #tau2
                            [0.3, 0.9], # x1
                            [-10, -1e-10]] #k
        self.para_guess = [500., 0.5, 0.1, 0.5, 0.7, -1]
        self.para_name = ["H", "tau1", "x0", "tau2", "x1", "k"]
        return

    def ymodel(self, theta, x):
        H, tau1, x0, tau2, x1, k = theta
        # if (x is None): x = self.histx
        ymodel = np.zeros(x.shape[0])
        idx = x<x0
        ymodel[idx] = H*np.exp((x[idx]-x0)/tau1)
        idx = (x>=x0) & (x<x1)
        ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau2) 
        idx = x>=x1
        ymodel[idx] = k*(x[idx]-x1) + H*np.exp(-(x1-x0)/tau2)       
        return ymodel    


# how to fit
class distfit:
    def __init__(self, dist, bins, model):
        hist, bin_edges = np.histogram(dist, bins=bins)
        self.dist = dist
        self.bins = bins
        self.histx = (bin_edges[:-1]+bin_edges[1:])/2.
        self.histy = hist
        self.model = model
        return


    def lnlikelihood(self, theta, x=None, y=None):
        if (x is None): x = self.histx
        if (y is None): y = self.histy
        ymodel = self.model.ymodel(theta, x=x)
        lnlikelihood = -np.sum((ymodel-y)**2.)/2.
        return lnlikelihood

    def minus_lnlikelihood(self, theta, **kwargs):
        return -self.lnlikelihood(theta, **kwargs)
    
    def fit(self, prior_guess=None, para_guess=None):
		# maximize likelihood function by scipy.optimize.minimize function
        
		# print(bounds)
		# print(self.para_guess)

        minimizer_kwargs={"bounds":self.model.prior_guess}
        result = basinhopping(self.minus_lnlikelihood, self.model.para_guess, minimizer_kwargs=minimizer_kwargs)
        self.para_fit = result.x
        return

    def output(self, filepath, ax=None, histkwargs={}, fitkwargs={}):
        st = "LS"
        # output
        # save guessed parameters
        np.savetxt(filepath+st+"guess.txt", self.model.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

        # save estimation result
        np.savetxt(filepath+st+"summary.txt", self.para_fit, delimiter=",", fmt=("%0.8f"), header="parameter")

        # plot fitting results and save
        if (ax is None):
            fig = plt.figure(figsize=(12,6))
            # axes = fig.subplots(nrows=2,ncols=1,squezzz=False).reshape(-1)
            ax = fig.add_subplot(111)
        self.plot_hist(ax=ax, histkwargs=histkwargs)
        self.plot_fit(self.para_fit, self.model.para_name, ax=ax, fitkwargs=fitkwargs)

        plt.savefig(filepath+st+"fit.png")
        plt.close()
        return

    def plot_hist(self, ax=None, histkwargs={}):
        ax.hist(self.dist, histtype="step", bins=self.bins, zorder=0, **histkwargs)
        return ax        
    
    def plot_fit(self, theta=None, para_name=None, ax=None, ifWritePara=True, fitkwargs={}):
        if (theta is None):
            theta = self.para_fit
        if (para_name is None):
            para_name = self.model.para_name
        if (ax is None):
            fig = plt.figure(figsize=(12,6))
            # axes = fig.subplots(nrows=2,ncols=1,squezzz=False).reshape(-1)
            ax = fig.add_subplot(111)

        yfit = self.model.ymodel(theta, self.histx)
        ax.plot(self.histx, yfit, **fitkwargs)

        if ifWritePara:
            for ipara in range(len(para_name)):
                ax.text(0.95, 0.9-ipara*0.04, "{:s}: {:0.3f}".format(para_name[ipara], theta[ipara]), ha="right", va="top", transform=ax.transAxes)
        return ax

if __name__ == "__main__":
    ### 1 observation
    dist = np.load("sample/obs/nike_dist.npy")
    filepath = "sample/sharpness/model2/obs/"
    bins = np.linspace(-0.5, 12, 1000)
    obj = distfit(dist, bins, model2())
    obj.fit()

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 4)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Ncount")
    ax.grid(which="both")

    obj.output(filepath, ax=ax, histkwargs={"color":"red"}, fitkwargs={"color":"black", "linestyle":"--"})



    ### 2 model
    dist = np.load("sample/padova/nike_dist.npy")
    filepath = "sample/sharpness/model2/padova/"
    bins = np.linspace(-0.5, 12, 1000)
    obj = distfit(dist, bins, model2())
    obj.fit()

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1, 4)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Ncount")
    ax.grid(which="both")

    obj.output(filepath, ax=ax, histkwargs={"color":"black"}, fitkwargs={"color":"red", "linestyle":"--"})

