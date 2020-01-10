'''
This is to measure the horizontal/vertical scatter on dnu-numax diagram, 
without manipulating anything else (for example to see as a function of
mass/metallicity).

'''
if __name__ == "__main__":
        rootpath = "/mnt/c/Users/yali4742/Dropbox (Sydney Uni)/Work/nike/"#"/headnode2/yali4742/nike/"#
        import numpy as np 
        import matplotlib
        matplotlib.use("Agg")
        import sys
        sys.path.append(rootpath) 
        from lib.histdist import model6
        from lib.wrapper import sharpness_fit_perturb, sharpness_fit_rescale, sharpness_fit_perturb_mcmc
        import os
 
        # fdnu corrected sharma+2016
        obsdir = rootpath+"sample/yu/"
        moddir = rootpath+"sample/padova/"

        # read in unperturbed mass and radius, with edges
        obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
        apk = np.load(obsdir+"apk18.npy", allow_pickle=True).tolist()
        edges_obs = np.load(obsdir+"mr_edge_samples.npy")
        tck_obs, tp_obs = np.load(obsdir+"mr_spline_tck.npy", allow_pickle=True)

        # read in unperturbed mass and radius, with edges
        pdv = np.load(moddir+"padova.npy", allow_pickle=True).tolist()
        edges_pdv = np.load(moddir+"mr_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"mr_spline_tck.npy", allow_pickle=True)

        # # to exclude those points which lies above the edge (so no horizontal distance).
        # idx = obs["radius"] <= np.max(edges_obs[:,1])
        # for key in obs.keys():
        #         obs[key] = obs[key][idx]   

        # idx = pdv["radius"] <= np.max(edges_obs[:,1])
        # for key in pdv.keys():
        #         pdv[key] = pdv[key][idx]

        distance = "horizontal"
        diagram = "mr"
        hist_model = model6()

        montecarlo = 120

        # trial 1: obtain an lower limit of mass relations
        # trial 1.1: no binning
        xobs, yobs = obs["mass"], obs["radius"]
        e_xobs, e_yobs = obs["e_mass"], obs["e_radius"]
        e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        idx = (xobs<=1.9) 
        xobs, yobs, e_xobs, e_yobs = xobs[idx], yobs[idx], e_xobs[idx], e_yobs[idx]

        xpdv, ypdv = pdv["mass"], pdv["radius"]
        idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        xpdv, ypdv= xpdv[idx], ypdv[idx]


        # filepath = rootpath+"sample/sharpness/sharma16/mass/perturb/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # xperturb = np.arange(0.00, 0.15, 0.005) # np.arange(0.00, 0.06, 0.11)#
        # yperturb = np.arange(0.00, 0.05, 1.0) # np.arange(0., 0.05, 0.1)
        # sharpness_fit_perturb(xobs, yobs, xobs, None, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, xpdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         Ndata=xobs.shape[0], ifmcmc=True)


        # # trial 1.2: demonstrate observational errors, no binnings
        # filepath = rootpath+"sample/sharpness/sharma16/mass/rescale/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # scalar = np.arange(0., 1.5, 0.1)
        # sharpness_fit_rescale(xobs, yobs, None, e_xobs, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, None, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, scalar, montecarlo,
        #         Ndata=xobs.shape[0], ifmcmc=True)

        # testing...
        # filepath = rootpath+"sample/sharpness/sharma16/mass/perturb_test/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # # xobs, yobs = apk["mass"], apk["radius"]
        # # e_xobs = np.abs(1.5 * (apk["teff"]/5777.0)**0.5 * (apk["numax"]/3090.0)**3.0 * (apk["dnu"]/135.1)**-4.0 * (apk["e_teff"]/5777.0))/apk["mass"]
        # # idx = (xobs<=1.9)
        # # xobs, yobs, e_xobs = xobs[idx], yobs[idx], e_xobs[idx]
        # xperturb = np.arange(0.00, 0.20, 0.02) # np.arange(0.00, 0.06, 0.11)#
        # yperturb = np.array([0.00]) # np.arange(0., 0.05, 0.1)
        # sharpness_fit_perturb(xobs, yobs, None, e_xobs, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, None, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         Ndata=xobs.shape[0], ifmcmc=False)
 

        # # trial 1.3: mass effect
        # # all = np.percentile(samples_obs[:,3], np.linspace(0,100,6))
        # # zvalue_limits = [all[:-1].tolist(), all[1:].tolist()]
        # zvalue_limits = [[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
        #                 [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]]
        # zvalue_name = "mass"

        # xobs, yobs, zobs = obs["numax"], obs["dnu"], obs["mass"]
        # e_xobs, e_yobs = obs["e_numax"], obs["e_dnu"]
        # e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        # # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # # xobs, yobs, zobs = xobs[idx], yobs[idx], zobs[idx]

        # xpdv, ypdv, zpdv = pdv["numax"], pdv["dnu"], pdv["mass"]
        # # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # # xpdv, ypdv, zpdv = xpdv[idx], ypdv[idx], zpdv[idx]

        # filepath = rootpath+"sample/sharpness/sharma16/mass_mass/perturb/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # xperturb = np.arange(0.00, 0.15, 0.005) # np.arange(0.00, 0.06, 0.11)#
        # yperturb = np.arange(0.00, 0.05, 1.0) # np.arange(0., 0.05, 0.1)
        # sharpness_fit_perturb(xobs, yobs, zobs, None, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, zpdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         zvalue_limits=zvalue_limits, zvalue_name=zvalue_name,
        #         Ndata=xobs.shape[0], ifmcmc=True)

        # filepath = rootpath+"sample/sharpness/sharma16/mass_mass/rescale/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # scalar = np.arange(0., 1.5, 0.1)
        # sharpness_fit_rescale(xobs, yobs, zobs, e_xobs, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, zpdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, scalar, montecarlo,
        #         zvalue_limits=zvalue_limits, zvalue_name=zvalue_name,
        #         Ndata=xobs.shape[0], ifmcmc=True)


        # # trial 1.4: feh effect
        # fehs = np.arange(-0.3, 0.4, 0.1)
        # Nbin = fehs.shape[0]-1
        # # zvalue_limits = [[-3.0, -0.20, 0.02],
        # #                 [-0.20, 0.02, 1.0]]
        # zvalue_name = "feh"

        # for ibin in range(Nbin):
        #         # read in unperturbed mass and radius, with edges
        #         iobs = np.load(obsdir+"feh/{:0.0f}/apk18.npy".format(ibin), allow_pickle=True).tolist()
        #         iedges_obs = np.load(obsdir+"feh/{:0.0f}/mr_edge_samples.npy".format(ibin))
        #         itck_obs, itp_obs = np.load(obsdir+"feh/{:0.0f}/mr_spline_tck.npy".format(ibin), allow_pickle=True)

        #         # read in unperturbed mass and radius, with edges
        #         ipdv = np.load(moddir+"feh/{:0.0f}/padova.npy".format(ibin), allow_pickle=True).tolist()
        #         iedges_pdv = np.load(moddir+"feh/{:0.0f}/mr_edge_samples.npy".format(ibin))
        #         itck_pdv, itp_pdv = np.load(moddir+"feh/{:0.0f}/mr_spline_tck.npy".format(ibin), allow_pickle=True)       

        #         # # # nothing to exclude.
        #         # # idx = np.ones(obs["radius"].shape[0], dtype=bool)
        #         # # for key in obs.keys():
        #         # #         obs[key] = obs[key][idx]   

        #         # # idx = np.ones(obs["radius"].shape[0], dtype=bool)
        #         # # for key in pdv.keys():
        #         # #         pdv[key] = pdv[key][idx]  

        #         ixobs, iyobs = iobs["mass"], iobs["radius"]
        #         ie_xobs, ie_yobs = iobs["e_mass"], iobs["e_radius"]
        #         ie_xobs, ie_yobs = ie_xobs/ixobs, ie_yobs/iyobs

        #         ixpdv, iypdv = ipdv["mass"], ipdv["radius"]

        #         filepath = rootpath+"sample/sharpness/sharma16/mass_feh/perturb/"
        #         if not os.path.exists(filepath): os.mkdir(filepath)
        #         filepath = rootpath+"sample/sharpness/sharma16/mass_feh/perturb/{:0.0f}/".format(ibin)
        #         if not os.path.exists(filepath): os.mkdir(filepath)

        #         xperturb = np.arange(0.00, 0.15, 0.005) # np.arange(0.00, 0.06, 0.11)#
        #         yperturb = np.arange(0.00, 0.05, 1.0) # np.arange(0., 0.05, 0.1)
        #         sharpness_fit_perturb(ixobs, iyobs, ixobs, None, None, iedges_obs, itck_obs, itp_obs,
        #                 ixpdv, iypdv, ixpdv, iedges_pdv, itck_pdv, itp_pdv,
        #                 diagram, distance, hist_model,
        #                 filepath, xperturb, yperturb, montecarlo,
        #                 Ndata=ixobs.shape[0], ifmcmc=True)

        #         filepath = rootpath+"sample/sharpness/sharma16/mass_feh/rescale/"
        #         if not os.path.exists(filepath): os.mkdir(filepath)
        #         filepath = rootpath+"sample/sharpness/sharma16/mass_feh/rescale/{:0.0f}/".format(ibin)
        #         if not os.path.exists(filepath): os.mkdir(filepath)

        #         scalar = np.arange(0., 2.0, 0.1)
        #         sharpness_fit_rescale(ixobs, iyobs, ixobs, ie_xobs, None, iedges_obs, itck_obs, itp_obs,
        #                 ixpdv, iypdv, ixpdv, iedges_pdv, itck_pdv, itp_pdv,
        #                 diagram, distance, hist_model,
        #                 filepath, scalar, montecarlo,
        #                 Ndata=ixobs.shape[0], ifmcmc=True)


        # trial 2: mcmc fit
        xobs, yobs = obs["mass"], obs["radius"]
        e_xobs, e_yobs = obs["e_mass"], obs["e_radius"]
        e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # xobs, yobs = xobs[idx], yobs[idx]
        # e_xobs, e_yobs = e_xobs[idx], e_yobs[idx]

        xpdv, ypdv = pdv["mass"], pdv["radius"]
        # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # xpdv, ypdv= xpdv[idx], ypdv[idx]

        filepath = rootpath+"sample/sharpness/sharma16/mass/perturb_mcmc_test/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        sharpness_fit_perturb_mcmc(xobs, yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                diagram, distance, hist_model, filepath, ifmcmc=True)