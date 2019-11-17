'''
This is to measure the horizontal/vertical scatter on dnu-numax diagram, 
without manipulating anything else (for example to see as a function of
mass/metallicity).

'''

# import numpy as np 
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import sys
# sys.path.append(rootpath) 
# from lib.histdist import model3, model4, distfit, distance_to_edge, reduce_samples
# import seaborn as sns
# from multiprocessing import Pool
# import random
# import os

# distance = "vertical"
# diagram = "mr"
# hist_model = model4()


# # read in unperturbed mass and radius, with edges
# tnu_samples_obs = np.load(rootpath+"sample/obs/tnu_samples.npy")
# mr_edges_obs = np.load(rootpath+"sample/obs/mr_edge_samples.npy")
# tck_obs = np.load(rootpath+"sample/obs/mr_spline_tck.npy", allow_pickle=True)
# # read in unperturbed mass and radius, with edges
# tnu_samples_pdv = np.load(rootpath+"sample/padova_oversampling/tnu_samples.npy")
# mr_edges_pdv = np.load(rootpath+"sample/padova_oversampling/mr_edge_samples.npy")
# tck_pdv = np.load(rootpath+"sample/padova_oversampling/mr_spline_tck.npy", allow_pickle=True)


# feh_limits = [-3.0, -0.20, 0.02, 1.0]

# for ifeh in range(len(feh_limits)-1):
#     feh = "{:0.0f}".format(ifeh)
        
#     # set up observations
#     # calculate observational distance
#     ycut = 0.1

#     xedge_obs, yedge_obs = mr_edges_obs[:,0], mr_edges_obs[:,1]
#     # idx = (xedge_obs**0.75/yedge_obs>=ycut) #& (yedge<=3.5)
#     # xedge_obs, yedge_obs = xedge_obs[idx], yedge_obs[idx]

#     xobs, yobs, feh_obs = tnu_samples_obs[:,3], tnu_samples_obs[:,4], tnu_samples_obs[:,2]
#     idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
#     idx = idx & (feh_obs >= feh_limits[ifeh]) & (feh_obs < feh_limits[ifeh+1])
#     xobs, yobs, feh_obs = xobs[idx], yobs[idx], feh_obs[idx]


#     hdist_obs = distance_to_edge(xobs, yobs, xedge_obs, yedge_obs, tck_obs, diagram=diagram, distance=distance)
#     obj_obs = distfit(hdist_obs, hist_model)
#     obj_obs = distfit(hdist_obs, hist_model, bins=obj_obs.bins)
#     obj_obs.fit()
#     sharpness_obs = hist_model.sharpness(obj_obs.para_fit)
#     Nobs = hdist_obs.shape[0]

#     # set up models
#     xedge_pdv, yedge_pdv = mr_edges_pdv[:,0], mr_edges_pdv[:,1]
#     # idx = (xedge_pdv**0.75/yedge_pdv>=ycut) #& (yedge<=3.5)
#     # xedge_pdv, yedge_pdv = xedge_pdv[idx], yedge_pdv[idx]

#     mass, radius, feh_pdv = tnu_samples_pdv[:,3], tnu_samples_pdv[:,4], tnu_samples_pdv[:,2]
#     idx = (mass<=1.9) #& (radius<=yedge_pdv.max())#
#     idx = idx & (feh_pdv >= feh_limits[ifeh]) & (feh_pdv < feh_limits[ifeh+1]) 
#     mass, radius, feh_pdv = mass[idx], radius[idx], feh_pdv[idx]
#     idx = reduce_samples(mass.shape[0], xobs.shape[0])
#     mass, radius, feh_pdv = mass[idx], radius[idx], feh_pdv[idx]



#     # # # customization
#     # step 1: deal with numax
#     filepath = rootpath+"sample/sharpness/perturb_gif_mr/radius_feh/radius_"+feh+"/"
#     if not os.path.exists(filepath): os.mkdir(filepath)
#     output = filepath+"data_radius_feh_"+feh
#     montecarlo = 120
#     mass_perturb = np.arange(0.00, 0.30, 1.0) # np.arange(0.00, 0.06, 0.11)#
#     radius_perturb = np.arange(0.00, 0.05, 0.002) # np.arange(0., 0.05, 0.1)
#     # # # end of customization


#     sharpness_med = np.zeros((mass_perturb.shape[0], radius_perturb.shape[0]), dtype=float)
#     sharpness_std = np.zeros((mass_perturb.shape[0], radius_perturb.shape[0]), dtype=float)

#     for imass in range(mass_perturb.shape[0]):
#         for iradius in range(radius_perturb.shape[0]):
#             print(imass, " mass: ", mass_perturb[imass], " , ", iradius, " radius:", radius_perturb[iradius])
            
#             # initiate a plot
#             fig = plt.figure(figsize=(12,12))
#             axes = fig.subplots(nrows=2, ncols=1)
#             obj_obs.plot_hist(ax=axes[0], histkwargs={"color":"red", "label":"Observations", "zorder":100})
#             obj_obs.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Observations fit", "linestyle":"--", "zorder":100})

#             # add perturbations
#             Ndata = mass.shape[0]
#             # sharpness_pdv_mcs = np.zeros(montecarlo)

#             # for imc in range(montecarlo):
#             def simulation(imc):
#                 # print(imc, "/", montecarlo)
#                 fmass = (np.array([random.gauss(0,1) for i in range(Ndata)]) * mass_perturb[imass] + 1)
#                 fradius = (np.array([random.gauss(0,1) for i in range(Ndata)]) * radius_perturb[iradius] + 1)
#                 xdata, ydata = (mass*fmass), (radius*fradius)
                
#                 dist = distance_to_edge(xdata, ydata, xedge_pdv, yedge_pdv, tck_pdv, diagram=diagram, distance=distance)
#                 obj = distfit(dist, hist_model, bins=obj_obs.bins)
#                 obj.fit()
#                 sharpness_pdv = hist_model.sharpness(obj.para_fit)
#                 return xdata, ydata, obj, sharpness_pdv

            
#             pool = Pool(processes=12)
#             result = pool.map(simulation, np.arange(0,montecarlo).tolist())
#             xdata, ydata, obj, _ = result[0]
#             sharpness_pdv_mcs = [result[i][-1] for i in range(len(result))]

#             sharpness_med[imass, iradius] = np.median(sharpness_pdv_mcs)
#             sharpness_std[imass, iradius] = np.std(sharpness_pdv_mcs)

#             # obj.histy = obj.histy/(Ndata/Nobs)  
#             obj.plot_hist(ax=axes[0], histkwargs={"color":"blue", "label":"Galaxia"})
#             obj.plot_fit(ax=axes[0], fitkwargs={"color":"black", "label":"Galaxia fit"})

            
#             axes[0].text(0.9, 0.6, "Galaxia slope: {:0.1f} $\pm$ {:0.1f}".format(sharpness_med[imass, iradius], sharpness_std[imass, iradius]), ha="right", va="top", transform=axes[0].transAxes)
#             axes[0].text(0.9, 0.55, "Observational slope: {:0.1f}".format(sharpness_obs), ha="right", va="top", transform=axes[0].transAxes)
#             axes[0].set_title("Scatter in radius: {:0.2f}%, Scatter in mass: {:0.2f}%".format(radius_perturb[iradius]*100,mass_perturb[imass]*100))   
#             axes[0].legend()
#             axes[0].grid(True)
#             axes[0].set_xlim(obj_obs.histx.min(), obj_obs.histx.max())
#             axes[0].set_ylim(0., obj_obs.histy.max()*1.5)


#             # diagram axes[1]
#             axes[1].plot(xobs, yobs, "r.", ms=1)
#             axes[1].plot(xdata, ydata, "b.", ms=1)
#             axes[1].plot(xedge_obs, yedge_obs, "k--")
#             axes[1].plot(xedge_pdv, yedge_pdv, "k-")

#             axes[1].grid(True)
#             axes[1].axis([0., 3., 5., 20.])
#             # axes[1].set_xscale("log")


#             plt.savefig(filepath+"{:0.0f}_mass_{:0.0f}_radius.png".format(imass, iradius))
#             plt.close()

#     # save data
#     data = {"mass_perturb":mass_perturb, "radius_perturb":radius_perturb, "montecarlo":montecarlo,
#             "sharpness_med":sharpness_med, "sharpness_std":sharpness_std, "sharpness_obs":sharpness_obs}
#     np.save(output, data)
