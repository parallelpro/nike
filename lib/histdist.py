import numpy as np 
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping, minimize
from scipy.special import erf 
import scipy.spatial.distance
import scipy.interpolate

import sys
import emcee
import corner

__all__ = ["distfit", "model6", "model_heb", "model_rgb",
            "distance_to_edge", "distance_to_bump", "reduce_samples"]

def display_bar(j, nburn, width=30):
        n = int((width+1) * float(j) / nburn)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        return

def reduce_samples(Ndata, Ntarget):
    # factor = int(Ndata/Ntarget)
    # if factor <=0: factor = 1
    # idx = np.zeros(Ndata, dtype=bool)
    # idx[np.arange(0,int(Ndata/factor))*factor] = True
    idx = np.arange(0,Ntarget)*Ndata/Ntarget
    idx = np.array(idx, dtype=int)
    return idx

# def distance_to_edge(xdata, ydata, xedge, yedge, tck, diagram="nike", distance="shortest"):

#     if not (diagram in ["nike", "tnu", "mr"]):
#         raise ValueError("diagram should be in ['nike', 'tnu', 'mr']")
#     if not (distance in ["shortest", "vertical", "horizontal"]):
#         raise ValueError("distance should be in ['shortest', 'vertical', 'horizontal']")
#     # Ndata, Nedge = xdata.shape[0], xedge.shape[0]

#     if distance=="vertical": #y
#         if diagram=="tnu":
#             dist = np.zeros(xdata.shape[0])
#             # need to separate left or right according to numax
#             yp = yedge[xedge==xedge.min()][0]
#             idx_data = (ydata >= yp) & (xdata<=31)
#             idx_edge = (yedge >= yp) & (xedge<=31)
#             if xdata[idx_data].shape[0] != 0:
#                 Xa = np.array([xdata[idx_data]]).T
#                 Xb = np.array([xedge[idx_edge]]).T
#                 Y = scipy.spatial.distance.cdist(Xa, Xb)
#                 argdist = np.argmin(Y, axis=1)
#                 dist[idx_data] = np.abs(ydata[idx_data] - yedge[idx_edge][argdist]) 
#             Xa = np.array([xdata[~idx_data]]).T
#             Xb = np.array([xedge[~idx_edge]]).T
#             Y = scipy.spatial.distance.cdist(Xa, Xb)
#             argdist = np.argmin(Y, axis=1)
#             dist[~idx_data] = np.abs(ydata[~idx_data] - yedge[~idx_edge][argdist]) 
#         else:
#             Xa = np.array([xdata]).T
#             Xb = np.array([xedge]).T
#             Y = scipy.spatial.distance.cdist(Xa, Xb)
#             argdist = np.argmin(Y, axis=1)
#             dist = np.abs(ydata - yedge[argdist])
#     if distance=="horizontal": #x
#         if diagram=="tnu":
#             dist = np.zeros(xdata.shape[0])
#             # need to separate left or right according to numax
#             xp = xedge[yedge==yedge.min()][0]
#             idx_data = xdata >= xp
#             idx_edge = xedge >= xp
#             if xdata[idx_data].shape[0] != 0:
#                 Xa = np.array([ydata[idx_data]]).T
#                 Xb = np.array([yedge[idx_edge]]).T
#                 Y = scipy.spatial.distance.cdist(Xa, Xb)
#                 argdist = np.argmin(Y, axis=1)
#                 dist[idx_data] = np.abs(xdata[idx_data] - xedge[idx_edge][argdist]) 

#             Xa = np.array([ydata[~idx_data]]).T
#             Xb = np.array([yedge[~idx_edge]]).T
#             Y = scipy.spatial.distance.cdist(Xa, Xb)
#             argdist = np.argmin(Y, axis=1)
#             dist[~idx_data] = np.abs(xdata[~idx_data] - xedge[~idx_edge][argdist]) 
#         elif diagram=="mr":
#             dist = np.zeros(xdata.shape[0])
#             # need to separate left or right according to mass
#             xp = xedge[yedge==yedge.max()][0]
#             idx_data = xdata >= xp
#             idx_edge = xedge >= xp
#             if xdata[idx_data].shape[0] != 0:
#                 Xa = np.array([ydata[idx_data]]).T
#                 Xb = np.array([yedge[idx_edge]]).T
#                 Y = scipy.spatial.distance.cdist(Xa, Xb)
#                 argdist = np.argmin(Y, axis=1)
#                 dist[idx_data] = np.abs(xdata[idx_data] - xedge[idx_edge][argdist]) 

#             Xa = np.array([ydata[~idx_data]]).T
#             Xb = np.array([yedge[~idx_edge]]).T
#             Y = scipy.spatial.distance.cdist(Xa, Xb)
#             argdist = np.argmin(Y, axis=1)
#             dist[~idx_data] = np.abs(xdata[~idx_data] - xedge[~idx_edge][argdist])             
#         else:
#             Xa = np.array([ydata]).T
#             Xb = np.array([yedge]).T
#             Y = scipy.spatial.distance.cdist(Xa, Xb)
#             argdist = np.argmin(Y, axis=1)
#             dist = np.abs(xdata - xedge[argdist]) 
#     if distance=="shortest":
#         Xa = np.array([xdata, ydata]).T
#         Xb = np.array([xedge, yedge]).T
#         Y = scipy.spatial.distance.cdist(Xa, Xb)
#         dist = np.min(Y, axis=1) 

#     # signs: left or right?
#     if diagram=="nike":
#         idx_right = xdata>10.0**scipy.interpolate.splev(np.log10(ydata), tck, der=0)
#     if diagram=="tnu":
#         idx_right = xdata>10.0**scipy.interpolate.splev(np.log10(xdata**0.75/ydata), tck, der=0)
#     if diagram=="mr":
#         idx_right = ydata<scipy.interpolate.splev(xdata, tck, der=0)
#     dist[idx_right] = -dist[idx_right]

#     return dist


def distance_to_edge(xdata, ydata, xedge, yedge, tcks, tp, 
                    diagram="tnu", distance="vertical", return_idx=False):

    if not (diagram in ["tnu", "mr"]):
        raise ValueError("diagram should be in ['tnu', 'mr']")
    if not (distance in ["vertical", "horizontal"]):
        raise ValueError("distance should be in ['vertical', 'horizontal']")
    # Ndata, Nedge = xdata.shape[0], xedge.shape[0]

    if distance=="vertical": #y
        if diagram=="tnu":
            ridx = (xdata >= xedge.min()) & (xdata <= tp[0])
            xdata, ydata = xdata[ridx], ydata[ridx]
            dist = np.zeros(xdata.shape[0])

            # first consider clumps, upper part
            idx = xedge<tp[0]
            yp = yedge[idx][xedge[idx]==xedge[idx].min()][0]
            idx_data = (ydata >= yp) & (xdata<=31) 
            idx_edge = (yedge >= yp) & (xedge<=31) 
            if xdata[idx_data].shape[0] != 0:
                Xa = np.array([xdata[idx_data]]).T
                Xb = np.array([xedge[idx_edge]]).T
                Y = scipy.spatial.distance.cdist(Xa, Xb)
                argdist = np.argmin(Y, axis=1)
                dist[idx_data] = (ydata[idx_data] - yedge[idx_edge][argdist]) 
            
            # next, consider clumps (lower part) and secondary clumps
            idx_data = ~idx_data
            idx_edge = ~idx_edge
            if xdata[idx_data].shape[0] != 0: 
                Xa = np.array([xdata[idx_data]]).T
                Xb = np.array([xedge[idx_edge]]).T
                Y = scipy.spatial.distance.cdist(Xa, Xb)
                argdist = np.argmin(Y, axis=1)
                dist[idx_data] = -(ydata[idx_data] - yedge[idx_edge][argdist]) 

        else:# diagram=="mr":
            ridx = (xdata<=tp[0])
            xdata, ydata = xdata[ridx], ydata[ridx]
            dist = np.zeros(xdata.shape[0])

            Xa = np.array([xdata]).T
            Xb = np.array([xedge]).T
            Y = scipy.spatial.distance.cdist(Xa, Xb)
            argdist = np.argmin(Y, axis=1)
            dist = (ydata - yedge[argdist])
            
            
    if distance=="horizontal": #x
        if diagram=="tnu":
            ridx = (ydata >= yedge.min()) & (xdata <= tp[0])
            xdata, ydata = xdata[ridx], ydata[ridx]
            dist = np.zeros(xdata.shape[0])
            
            # first consider clumps, left part
            idx = xedge<tp[0]
            xp = xedge[idx][yedge[idx]==yedge[idx].min()][0]
            idx_data = xdata <= xp
            idx_edge = xedge <= xp
            if xdata[idx_data].shape[0] != 0:
                Xa = np.array([ydata[idx_data]]).T
                Xb = np.array([yedge[idx_edge]]).T
                Y = scipy.spatial.distance.cdist(Xa, Xb)
                argdist = np.argmin(Y, axis=1)
                dist[idx_data] = -(xdata[idx_data] - xedge[idx_edge][argdist]) 
                
            # next, consider clumps (right part)
            idx_data = (xdata > xp) & (xdata<= tp[0])
            idx_edge = (xedge > xp) & (xedge<= tp[0])
            if xdata[idx_data].shape[0] != 0:
                Xa = np.array([ydata[idx_data]]).T
                Xb = np.array([yedge[idx_edge]]).T
                Y = scipy.spatial.distance.cdist(Xa, Xb)
                argdist = np.argmin(Y, axis=1)
                dist[idx_data] = (xdata[idx_data] - xedge[idx_edge][argdist]) 

            # # finally, consider secondary clumps
            # idx_data = (xdata > tp[0])
            # idx_edge = (xedge > tp[0])
            # if xdata[idx_data].shape[0] != 0:
            #     Xa = np.array([ydata[idx_data]]).T
            #     Xb = np.array([yedge[idx_edge]]).T
            #     Y = scipy.spatial.distance.cdist(Xa, Xb)
            #     argdist = np.argmin(Y, axis=1)
            #     dist[idx_data] = -(xdata[idx_data] - xedge[idx_edge][argdist]) 
            
        else:# diagram=="mr":
            ridx = (ydata <= yedge[xedge<tp[0]].max()) & (xdata<=tp[0])
            xdata, ydata = xdata[ridx], ydata[ridx]
            dist = np.zeros(xdata.shape[0])

            # first, consider left part of clumps
            idx = xedge<tp[0]
            xp = xedge[idx][yedge[idx]==yedge[idx].max()][0]
            idx_data = xdata <= xp
            idx_edge = xedge <= xp
            if xdata[idx_data].shape[0] != 0:
                Xa = np.array([ydata[idx_data]]).T
                Xb = np.array([yedge[idx_edge]]).T
                Y = scipy.spatial.distance.cdist(Xa, Xb)
                argdist = np.argmin(Y, axis=1)
                dist[idx_data] = -(xdata[idx_data] - xedge[idx_edge][argdist]) 

            # next, consider right part of clumps
            idx_data = (xdata > xp) & (xdata<=tp[0])
            idx_edge = (xedge > xp) & (xedge<=tp[0])
            if xdata[idx_data].shape[0] != 0:
                Xa = np.array([ydata[idx_data]]).T
                Xb = np.array([yedge[idx_edge]]).T
                Y = scipy.spatial.distance.cdist(Xa, Xb)
                argdist = np.argmin(Y, axis=1)
                dist[idx_data] = (xdata[idx_data] - xedge[idx_edge][argdist])             

            # # finally, secondary clumps
            # idx_data = (xdata>tp[0])
            # idx_edge = (xedge>tp[0])
            # if xdata[idx_data].shape[0] != 0:
            #     Xa = np.array([ydata[idx_data]]).T
            #     Xb = np.array([yedge[idx_edge]]).T
            #     Y = scipy.spatial.distance.cdist(Xa, Xb)
            #     argdist = np.argmin(Y, axis=1)
            #     dist[idx_data] = -(xdata[idx_data] - xedge[idx_edge][argdist])      
            
#     # signs: left or right?
#     if diagram=="tnu":
#         # clumps
#         idx_right = xdata>10.0**scipy.interpolate.splev(np.log10(xdata**0.75/ydata), tcks[0], der=0)
#         idx = idx_right & (xdata<tp[0])
#         dist[idx] = -dist[idx]
        
#         # secondary clumps
#         idx_right = xdata>10.0**scipy.interpolate.splev(np.log10(xdata**0.75/ydata), tcks[1], der=0)
#         idx = idx_right & (xdata>=tp[0])
#         dist[idx] = -dist[idx]        
#     if diagram=="mr":
#         # clumps
#         idx_right = ydata<scipy.interpolate.splev(xdata, tcks[0], der=0)
#         idx = idx_right & (xdata<tp[0])
#         dist[idx] = -dist[idx]
        
#         # secondary clumps
#         idx_right = ydata<scipy.interpolate.splev(xdata, tcks[1], der=0)
#         idx = idx_right & (xdata>=tp[0])
#         dist[idx] = -dist[idx]

    if not return_idx:
        return dist, xdata, ydata
    else:
        return dist, xdata, ydata, ridx

def distance_to_bump(xdata, ydata, bump_points,
                    distance="vertical", return_idx=False):

    if not (distance in ["vertical", "horizontal"]):
        raise ValueError("distance should be in ['vertical', 'horizontal']")
    # Ndata, Nedge = xdata.shape[0], xedge.shape[0]

    k, b = bump_points

    if distance=="vertical": #y
        dist = ydata - (k*xdata + b)
    if distance=="horizontal": #x         
        dist = xdata - (ydata-b)/k

    return dist, xdata, ydata


# class model1:
#     def __init__(self):
#         # model1
#         self.prior_guess = [[0., 1000.], [1e-6, 1.], [-1, 1.], [1e-6, 3.]]
#         self.para_guess = [500., 0.5, 0.5, 1.0]
#         self.para_name = ["H", "gamma", "x0", "tau"]
#         return

#     def ymodel(self, theta, x):
#         H, gamma, x0, tau = theta
#         # if (x is None): x = self.histx
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         ymodel[idx] = H/((x[idx]-x0)**2./gamma**2. + 1)
#         idx = x>=x0
#         ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau)
#         return ymodel

#     def sharpness(self, theta):
#         # metric 1: H/gamma
#         H, gamma = theta[0:2]
#         metric = H/gamma
#         return metric

# class model2:
#     def __init__(self):
#         # # model2
#         self.prior_guess = [[0., 1000.], #H
#                             [1e-6, 3.], # tau1
#                             [-0.1, 0.3], # x0
#                             [1e-6, 3.], #tau2
#                             [0.3, 0.9], # x1
#                             [-10, -1e-10]] #k
#         self.para_guess = [500., 0.5, 0.1, 0.5, 0.7, -1]
#         self.para_name = ["H", "tau1", "x0", "tau2", "x1", "k"]
#         return

#     def ymodel(self, theta, x):
#         H, tau1, x0, tau2, x1, k = theta
#         # if (x is None): x = self.histx
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         ymodel[idx] = H*np.exp((x[idx]-x0)/tau1)
#         idx = (x>=x0) & (x<x1)
#         ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau2) 
#         idx = x>=x1
#         ymodel[idx] = k*(x[idx]-x1) + H*np.exp(-(x1-x0)/tau2)       
#         return ymodel    

# class model3:
#     def __init__(self):
#         # # model3
#         self.para_name = ["xc", "s", "H", "x0", "tau", "x1", "k"]
#         return
    
#     def set_priors(self, histx, histy, dist):
#         maximum_center = histx[histy==histy.max()][0]
#         self.prior_guess = [[(histx.min()+maximum_center)/2.,  0.8*maximum_center], #xc
#                             [0.2, 10.], # s
#                             [1e-6, histy.max()*2.0], #H
#                             [0.8*maximum_center, maximum_center*2.0], # x0
#                             [1e-6, 100.], #tau
#                             [maximum_center*2.0, histx.max()]] # x1
#                             # [-1e2, -1e-10]] #k]
#         self.para_guess = [np.mean(bound) for bound in self.prior_guess]
#         return

#     def ymodel(self, theta, x):
#         xc, s, H, x0, tau, x1 = theta
#         # if (x is None): x = self.histx
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         # if (erf(s*(x0-xc))+1)==0 : print(s, x0, xc)
#         A = H/(erf(s*(x0-xc))+1)
#         ymodel[idx] = A*(erf(s*(x[idx]-xc))+1)
#         idx = (x>=x0) & (x<x1)
#         ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau) 
#         idx = x>=x1
#         ymodel[idx] = H*np.exp(-(x1-x0)/tau) # +k*(x[idx]-x1)
#         return ymodel

#     def sharpness(self, theta):
#         # use the derivative at xc as the sharpness metric
#         xc, s, H, x0 = theta[0:4]
#         A = H/(erf(s*(x0-xc))+1)
#         metric = A * 2/np.sqrt(np.pi) * s
#         return metric


# class model4:
#     def __init__(self):
#         # # model3
#         self.para_name = ["sigma", "H", "x0", "tau" ] #"x1", "k"
#         return
    
#     def set_priors(self, histx, histy, dist):
#         x0 = histx[histy==histy.max()][0]
#         H = histy.max()
#         idx = histx<=x0
#         integration = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         sig = integration/(np.sqrt(np.pi/2.0)*H)
#         idx = histx>=x0
#         integration = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         tau = integration/H
#         self.prior_guess = [[1e-6, sig*5.0], #sigma
#                             [1e-6, histy.max()*2.], #H
#                             [histx.min(), histx.max()], # x0
#                             [tau*1e-3, tau*1e3]] #tau
#                             # [maximum_center+0.5*sig, histx.max()]] # x1
#                             # [-1e2, -1e-10]] #k]
#         self.para_guess = [sig, H, x0, tau]
#         return

#     def ymodel(self, theta, x):
#         sigma, H, x0, tau = theta #, x1
#         # if (x is None): x = self.histx
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         ymodel[idx] = H*np.exp(-((x[idx]-x0)**2.0)/(2*sigma**2.0))
#         idx = (x>=x0) #& (x<x1)
#         ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau) 
#         # idx = x>=x1
#         # ymodel[idx] = H*np.exp(-(x1-x0)/tau) # +k*(x[idx]-x1)
#         return ymodel

#     def sharpness(self, theta):
#         # use the derivative at xc as the sharpness metric
#         sigma, H = theta[0:2]
#         metric = H/sigma
#         return metric


# class model5:
#     def __init__(self):
#         # # model3
#         self.para_name = ["sigma", "H", "x0", "tau" ] #"x1", "k"
#         self.ndim = 4
#         return
    
#     def set_priors(self, histx, histy, dist):
#         x0 = histx[histy==histy.max()][0]
#         H = histy.max()
#         idx = histx<=x0
#         integration = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         sig = integration/(np.sqrt(np.pi/2.0)*H)
#         idx = histx>=x0
#         integration = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         tau = integration/H
#         self.prior_guess = [[1e-6, sig*10.0], #sigma
#                             [1e-6, histy.max()*2.], #H
#                             [histx.min(), histx.max()], # x0
#                             [tau*1e-3, tau*1e3]] #tau
#                             # [maximum_center+0.5*sig, histx.max()]] # x1
#                             # [-1e2, -1e-10]] #k]
#         self.para_guess = [sig, H, x0, tau]
#         return

#     def ymodel(self, theta, x):
#         sigma, H, x0, tau = theta #, x1
#         # if (x is None): x = self.histx
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         ymodel[idx] = H*np.exp(-((x[idx]-x0)**2.0)/(2*sigma**2.0))
#         idx = (x>=x0) #& (x<x1)
#         ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau) 
#         # idx = x>=x1
#         # ymodel[idx] = H*np.exp(-(x1-x0)/tau) # +k*(x[idx]-x1)
#         return ymodel
    

#     def sharpness(self, theta):
#         # use the derivative at xc as the sharpness metric
#         metric = theta[0] # sigma
#         return metric

# class model5_prob:
#     def __init__(self):
#         # # model3
#         self.para_name = ["sigma", "x0", "tau" ] #"x1", "k"
#         self.ndim = 3
#         return
    
#     def set_priors(self, histx, histy, dist):
#         x0 = histx[histy==histy.max()][0]
#         H = histy.max()
#         idx = histx<=x0
#         integration = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         sig = integration/(np.sqrt(np.pi/2.0)*H)
#         idx = histx>=x0
#         integration = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         tau = integration/H
#         self.prior_guess = [[1e-6, sig*10.0], #sigma
#                             [histx.min(), histx.max()], # x0
#                             [tau*1e-3, tau*1e3]] #tau
#         self.para_guess = [sig, x0, tau]
#         return

#     def ymodel(self, theta, x):
#         sigma, x0, tau = theta #, x1
#         # if (x is None): x = self.histx
#         x1, x2 = x.min(), x.max()
#         S1 = -(np.sqrt(np.pi/2.0)*sigma) * erf((x1-x0)/(np.sqrt(2)*sigma))
#         S2 = tau - tau*np.exp((x0-x2)/tau)
#         # H = 1./(np.sqrt(np.pi/2.)*sigma + tau)
#         H = 1.0/(S1+S2)
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         ymodel[idx] = H*np.exp(-((x[idx]-x0)**2.0)/(2*sigma**2.0))
#         idx = (x>=x0) #& (x<x1)
#         ymodel[idx] = H*np.exp(-(x[idx]-x0)/tau) 
#         return ymodel
    
#     def sharpness(self, theta):
#         # use the derivative at xc as the sharpness metric
#         metric = theta[0] # sigma
#         return metric


class model_heb:
    def __init__(self):
        # # model3
        self.para_name = ["sigma", "x0", "H", "gamma"] #"x1", "k"
        self.ndim = 4
        return
    
    def set_priors(self, histx, histy, dist):
        x0 = histx[histy==histy.max()][0]
        H = histy.max()
        idx = histx<=x0
        int1 = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
        idx = histx>=x0
        int2 = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
        sig = int1/(np.sqrt(np.pi/2.0)*H)
        gamma = int2*2/H/np.pi
        self.prior_guess = [[1e-3, sig*20.0], #sigma
                            [histx.min(), histx.max()], # x0
                            [0.1*H, 2*H], #H
                            [gamma*1e-3, gamma*1e3]] #gamma
        self.para_guess = [sig, x0, H, gamma]
        return

    def ymodel(self, theta, x):
        sigma, x0, H, gamma = theta #, x1
        # if (x is None): x = self.histx
        # S1 = -(np.sqrt(np.pi/2.0)*sigma) * erf((x1-x0)/(np.sqrt(2)*sigma))
        # S2 = gamma*np.arctan((x2-x0)/gamma)
        # H = 1.0/(S1+S2)
        ymodel = np.zeros(x.shape[0])
        idx = x<x0
        ymodel[idx] = H*np.exp(-((x[idx]-x0)**2.0)/(2*sigma**2.0))
        idx = (x>=x0) #& (x<x1)
        ymodel[idx] = H/(1+(x[idx]-x0)**2.0/gamma**2.0)
        return ymodel
    
    def sharpness(self, theta):
        # use the derivative at xc as the sharpness metric
        metric = [theta[0], theta[2]/theta[0]*np.exp(-0.5)] # sigma
        return metric
    
    def e_sharpness(self, theta, etheta):
        e_metric = [etheta[0],
            np.exp(-0.5)*((1./theta[0]*etheta[2])**2.0 + (-theta[2]/theta[0]**2.0*etheta[0])**2.0)**0.5]
        return e_metric
        
model6 = model_heb
class model_rgb:
    def __init__(self):
        # # model3
        self.para_name = ["sigma", "x0", "H", "c"] #"x1", "k"
        self.ndim = 4
        return
    
    def set_priors(self, histx, histy, dist):
        sig = np.mean(np.abs(histx))
        x0 = 0.
        H = np.max(histy)
        c = np.median(histy)

        self.prior_guess = [[1e-3, np.max(histx)], #sigma
                            [histx.min()/5., histx.max()/5.], # x0
                            [0.1*H, 2*H], #H
                            [c*1e-3, c*2.]] #gamma
        self.para_guess = [sig, x0, H, c]
        return

    def ymodel(self, theta, x):
        sigma, x0, H, c = theta #, x1
        ymodel = H*np.exp(-((x-x0)**2.0)/(2*sigma**2.0)) + c
        return ymodel
    
    def sharpness(self, theta):
        # use the derivative at xc as the sharpness metric
        metric = [theta[0], theta[2]/theta[0]*np.exp(-0.5)] # sigma
        return metric
    
    def e_sharpness(self, theta, etheta):
        e_metric = [etheta[0],
            np.exp(-0.5)*((1./theta[0]*etheta[2])**2.0 + (-theta[2]/theta[0]**2.0*etheta[0])**2.0)**0.5]
        return e_metric

class model7:
    def __init__(self):
        # # model3
        self.para_name = ["sigma", "x0", "H", "c", "k"] #"x1", "k"
        self.ndim = 5
        return
    
    def set_priors(self, histx, histy, dist):
        sig = np.mean(np.abs(histx))
        x0 = 0.
        H = np.max(histy)
        c = np.median(histy)
        k = 0.

        self.prior_guess = [[1e-3, np.max(histx)], #sigma
                            [histx.min()/5., histx.max()/5.], # x0
                            [0.1*H, 2*H], #H
                            [c*1e-3, c*2.], #c
                            [-1000., 1000.]] #k
        self.para_guess = [sig, x0, H, c, k]
        return

    def ymodel(self, theta, x):
        sigma, x0, H, c, k = theta #, x1
        ymodel = H*np.exp(-((x-x0)**2.0)/(2*sigma**2.0)) + c + k*x
        return ymodel
    
    def sharpness(self, theta):
        # use the derivative at xc as the sharpness metric
        metric = [theta[0], theta[2]/theta[0]*np.exp(-0.5)] # sigma
        return metric
    
    def e_sharpness(self, theta, etheta):
        e_metric = [etheta[0],
            np.exp(-0.5)*((1./theta[0]*etheta[2])**2.0 + (-theta[2]/theta[0]**2.0*etheta[0])**2.0)**0.5]
        return e_metric

# class model6_prob:
#     def __init__(self):
#         # # model3
#         self.para_name = ["sigma", "x0", "gamma" ] #"x1", "k"
#         self.ndim = 3
#         return
    
#     def set_priors(self, histx, histy, dist):
#         x0 = histx[histy==histy.max()][0]
#         H = histy.max()
#         idx = histx<=x0
#         int1 = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))
#         idx = histx>=x0
#         int2 = np.sum(histy[idx]) * np.median(np.diff(histx[idx]))

#         int = int1+int2
#         int1 = int1/int
#         int2 = int2/int
#         H = H/int

#         sig = int1/(np.sqrt(np.pi/2.0)*H)

#         gamma = int2*2/H/np.pi
#         self.prior_guess = [[1e-3, sig*20.0], #sigma
#                             [histx.min(), histx.max()], # x0
#                             [gamma*1e-3, gamma*1e3]] #gamma
#         self.para_guess = [sig, x0, gamma]
#         return

#     def ymodel(self, theta, x):
#         sigma, x0, gamma = theta #, x1
#         # if (x is None): x = self.histx
#         x1, x2 = x.min(), x.max()
#         S1 = -(np.sqrt(np.pi/2.0)*sigma) * erf((x1-x0)/(np.sqrt(2)*sigma))
#         S2 = gamma*np.arctan((x2-x0)/gamma)
#         # H = 1./(np.sqrt(np.pi/2.)*sigma + tau)
#         H = 1.0/(S1+S2)
#         ymodel = np.zeros(x.shape[0])
#         idx = x<x0
#         ymodel[idx] = H*np.exp(-((x[idx]-x0)**2.0)/(2*sigma**2.0))
#         idx = (x>=x0) #& (x<x1)
#         ymodel[idx] = H/(1+(x[idx]-x0)**2.0/gamma**2.0)
#         return ymodel
    
#     def sharpness(self, theta):
#         # use the derivative at xc as the sharpness metric
#         metric = theta[0] # sigma
#         return metric



# how to fit
class distfit:
    def __init__(self, dist, model, bins=None, density=False):
        if (bins is None):
            # p90 = np.percentile(dist, 85)
            hist, bin_edges = np.histogram(dist, bins="fd")
            # # idx_xmax = np.where(hist==hist.max())[0][0]
            # # xmax = np.mean(bin_edges[idx_xmax:idx_xmax+1])
            # # xmax_pencentile = np.sum(dist<=xmax)/len(dist)*100
            # pdown = np.percentile(dist, 0.)#xmax_pencentile/3.0)
            # pup = np.percentile(dist, 100.)#-4*pdown
            # dist = dist[((dist<=pup) & (dist>=pdown))]
            # idx = bin_edges <= bin_edges[np.where(hist==hist.max())[0][0]]
            # hist, bin_edges = np.histogram(dist, bins=bin_edges[idx], density=density)
            hist, bin_edges = np.histogram(dist, bins=len(bin_edges)*2, density=density) 
        else: 
            hist, bin_edges = np.histogram(dist, bins=bins)
        self.dist = dist
        self.bins = bin_edges
        self.histx = (bin_edges[:-1]+bin_edges[1:])/2.
        self.histy = hist
        self.model = model
        self.model.set_priors(self.histx, self.histy, self.dist)
        return


    def lnlikelihood(self, theta, x=None, y=None):
        if (x is None): x = self.histx
        if (y is None): y = self.histy
        ymodel = self.model.ymodel(theta, x=x)
        lnlikelihood = -np.sum((ymodel-y)**2.)/2.
        return lnlikelihood

    def minus_lnlikelihood(self, theta, **kwargs):
        return -self.lnlikelihood(theta, **kwargs)
    
    def lnprior(self, theta):
        for ip in range(len(self.model.prior_guess)):
            if not (self.model.prior_guess[ip][0] <= theta[ip] <= self.model.prior_guess[ip][1]):
                    return -np.inf
        return 0.

    def lnpost(self, theta, **kwargs):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
                return -np.inf
        else:
                return self.lnlikelihood(theta)
    
    def fit(self, ifmcmc=False):#prior_guess=None, para_guess=None, 
		# maximize likelihood function by scipy.optimize.minimize function
        
		# print(bounds)
		# print(self.para_guess)

        if ifmcmc: 
            # run mcmc with ensemble sampler
            ndim = self.model.ndim
            nwalkers, nburn, nsteps = 500, 1000, 1000

            print("enabling Ensemble sampler.")
            pos0 = [self.model.para_guess + 1.0e-1*np.random.randn(ndim) for j in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost)

            # # burn-in
            print("start burning in. nburn:", nburn)
            for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
                    # display_bar(j, nburn)
                    pass
            sys.stdout.write("\n")
            pos, _, _ = result
            sampler.reset()

            # actual iteration
            print("start iterating. nsteps:", nsteps)
            for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
                    # display_bar(j, nsteps)
                    pass
            sys.stdout.write("\n")

            # modify samples
            self.samples = sampler.chain[:,:,:].reshape((-1,ndim))

            # save estimation result
            # 16, 50, 84 quantiles
            result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                    zip(*np.percentile(self.samples, [16, 50, 84],axis=0)))))
            self.result = result
            self.para_fit = result[:,0]
            self.e_para_fit = (result[:,1]+result[:,2])/2.0
            # save guessed parameters
            # np.savetxt(filepath+"guess.txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

        else:
            minimizer_kwargs={"bounds":self.model.prior_guess}
            result = basinhopping(self.minus_lnlikelihood, self.model.para_guess, minimizer_kwargs=minimizer_kwargs)
            # result = minimize(self.minus_lnlikelihood, self.model.para_guess, **minimizer_kwargs)
            self.para_fit = result.x
            self.e_para_fit = np.zeros(len(self.para_fit))-999.
            # self.e_para_fit = np.diag(result.hess_inv)**0.5
        return

    def output(self, filepath, ax=None, ifmcmc=False, histkwargs={}, fitkwargs={}):
        st = ""
        # output
        # save guessed parameters
        np.savetxt(filepath+st+"guess.txt", self.model.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

        # plot fitting results and save
        if (ax is None):
            fig = plt.figure(figsize=(12,6))
            # axes = fig.subplots(nrows=2,ncols=1,squezzz=False).reshape(-1)
            ax = fig.add_subplot(111)
        self.plot_hist(ax=ax, histkwargs=histkwargs)
        self.plot_fit(theta=self.para_fit, para_name=self.model.para_name, ax=ax, ifWritePara=True, fitkwargs=fitkwargs)

        plt.savefig(filepath+st+"fit.png")
        plt.close()

        if ifmcmc:
            # plot triangle and save
            fig = corner.corner(self.samples, labels=self.model.para_name, show_titles=True, quantiles=(0.16, 0.5, 0.84))
            fig.savefig(filepath+st+"triangle.png")
            plt.close()

            # save estimation result
            np.savetxt(filepath+st+"summary.txt", self.result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f"), header="50th quantile, 16th quantile sigma, 84th quantile sigma")
        else:
            # save estimation result
            np.savetxt(filepath+st+"summary.txt", self.para_fit, delimiter=",", fmt=("%0.8f"), header="parameter")

        return

    def plot_hist(self, scale=1, ax=None, histkwargs={}):
        if (ax is None):
            fig = plt.figure(figsize=(12,6))
            # axes = fig.subplots(nrows=2,ncols=1,squezzz=False).reshape(-1)
            ax = fig.add_subplot(111)
        # ax.hist(self.dist, histtype="step", bins=self.bins, zorder=0, **histkwargs)
        ax.step(self.histx, self.histy/scale, **histkwargs)
        return ax        
    
    def plot_fit(self, scale=1, theta=None, para_name=None, ax=None, 
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
        ax.plot(xfit, yfit/scale, **fitkwargs)

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

