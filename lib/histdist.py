import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping, minimize
from scipy.special import erf 
import scipy.spatial.distance

__all__ = ["distfit", "model1", "model2", "model3", "distance_to_edge"]

def distance_to_edge(xdata, ydata, xedge, yedge, tck_in_nike, diagram="nike", distance="shortest"):
    if not (diagram in ["nike", "tnu"]):
        raise ValueError("diagram should be in ['nike', 'tnu']")
    if not (distance in ["shortest", "vertical", "horizontal"]):
        raise ValueError("distance should be in ['shortest', 'vertical', 'horizontal']")
    # Ndata, Nedge = xdata.shape[0], xedge.shape[0]

    if distance=="vertical": #y
        Xa = np.array([xdata]).T
        Xb = np.array([xedge]).T
        Y = scipy.spatial.distance.cdist(Xa, Xb)
        argdist = np.argmin(Y, axis=1)
        dist = np.abs(ydata - yedge[argdist])
    if distance=="horizontal": #x
        Xa = np.array([ydata]).T
        Xb = np.array([yedge]).T
        Y = scipy.spatial.distance.cdist(Xa, Xb)
        argdist = np.argmin(Y, axis=1)
        dist = np.abs(xdata - xedge[argdist]) 
    if distance=="shortest":
        Xa = np.array([xdata, ydata]).T
        Xb = np.array([xedge, yedge]).T
        Y = scipy.spatial.distance.cdist(Xa, Xb)
        dist = np.min(Y, axis=1) 

    # signs: left or right?
    if diagram=="nike":
        idx_right = xdata>10.0**scipy.interpolate.splev(np.log10(ydata), tck_in_nike, der=0)
    if diagram=="tnu":
        idx_right = xdata>10.0**scipy.interpolate.splev(np.log10(xdata**0.75/ydata), tck_in_nike, der=0)
        
    dist[idx_right] = -dist[idx_right]

    return dist

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

    def sharpness(self, theta):
        # metric 1: H/gamma
        H, gamma = theta[0:2]
        metric = H/gamma
        return metric

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

class model3:
    def __init__(self):
        # # model3
        self.prior_guess = [[-1., 0.3], #xc
                            [1e-6, 100.], # s
                            [1e-6, 1000.], #H
                            [-0.1, 0.3], # x0
                            [1e-6, 3.], #tau
                            [0.3, 0.9], # x1
                            [-10, -1e-10]] #k
        self.para_guess = [0.1, 5., 10., 0.1, 0.5, 0.7, -1]
        self.para_name = ["xc", "s", "H", "x0", "tau", "x1", "k"]
        return

    def ymodel(self, theta, x):
        xc, s, H, x0, tau, x1, k = theta
        # if (x is None): x = self.histx
        ymodel = np.zeros(x.shape[0])
        idx = x<x0
        A = H/(erf(s*(x0-xc))+1)
        ymodel[idx] = A*(erf(s*(x[idx]-xc))+1)
        idx = (x>=x0) & (x<x1)
        ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau) 
        idx = x>=x1
        ymodel[idx] = k*(x[idx]-x1) + H*np.exp(-(x1-x0)/tau)       
        return ymodel

    def sharpness(self, theta):
        # use the derivative at xc as the sharpness metric
        xc, s, H, x0 = theta[0:4]
        A = H/(erf(s*(x0-xc))+1)
        metric = A * 2/np.sqrt(np.pi) * s
        return metric

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
        # result = minimize(self.minus_lnlikelihood, self.model.para_guess, **minimizer_kwargs)
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
        self.plot_fit(self.para_fit, self.model.para_name, ax=ax, ifWritePara=True, fitkwargs=fitkwargs)

        plt.savefig(filepath+st+"fit.png")
        plt.close()
        return

    def plot_hist(self, ax=None, histkwargs={}):
        # ax.hist(self.dist, histtype="step", bins=self.bins, zorder=0, **histkwargs)
        ax.step(self.histx, self.histy, **histkwargs)
        return ax        
    
    def plot_fit(self, theta=None, para_name=None, ax=None, 
                    oversampling=5, ifWritePara=False, fitkwargs={}):
        if (theta is None):
            theta = self.para_fit
        if (para_name is None):
            para_name = self.model.para_name
        if (ax is None):
            fig = plt.figure(figsize=(12,6))
            # axes = fig.subplots(nrows=2,ncols=1,squezzz=False).reshape(-1)
            ax = fig.add_subplot(111)

        xfit = np.linspace(np.min(self.histx), np.max(self.histx), len(self.histx)*oversampling)
        yfit = self.model.ymodel(theta, xfit)
        ax.plot(xfit, yfit, **fitkwargs)

        if ifWritePara:
            for ipara in range(len(para_name)):
                ax.text(0.95, 0.9-ipara*0.04, "{:s}: {:0.3f}".format(para_name[ipara], theta[ipara]), ha="right", va="top", transform=ax.transAxes)
        return ax

    # def perturb()

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

