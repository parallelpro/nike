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

        # # trial 1: obtain an lower limit of numax relations
        # # trial 1.1: no binning
        # xobs, yobs = obs["numax"], obs["dnu"]
        # e_xobs, e_yobs = obs["e_numax"], obs["e_dnu"]
        # e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        # # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # # xobs, yobs = xobs[idx], yobs[idx]
        # # e_xobs, e_yobs = e_xobs[idx], e_yobs[idx]

        # xpdv, ypdv = pdv["numax"], pdv["dnu"]
        # # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # # xpdv, ypdv= xpdv[idx], ypdv[idx]


        # filepath = rootpath+"sample/sharpness/sharma16/numax/perturb/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # xperturb = np.arange(0.00, 0.050, 0.002)#0.001) # np.arange(0.00, 0.06, 0.11)#
        # yperturb = np.array([0.])
        # sharpness_fit_perturb(xobs, yobs, xobs, None, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, xpdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         Ndata=xobs.shape[0], ifmcmc=True)


        # # trial 1.2: demonstrate observational errors, no binns
        # filepath = rootpath+"sample/sharpness/sharma16/numax/rescale/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # scalar = np.arange(0., 2.0, 0.1)
        # sharpness_fit_rescale(xobs, yobs, xobs, e_xobs, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, xpdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, scalar, montecarlo,
        #         Ndata=xobs.shape[0], ifmcmc=True)


        # # trial 1.3: mass effect
        # # all = np.percentile(obs[:,3], np.linspace(0,100,6))
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

        # filepath = rootpath+"sample/sharpness/sharma16/numax_mass/perturb/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # xperturb = np.arange(0.00, 0.035, 0.002)#0.001) # np.arange(0.00, 0.06, 0.11)#
        # yperturb = np.array([0.])
        # sharpness_fit_perturb(xobs, yobs, zobs, None, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, zpdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         zvalue_limits=zvalue_limits, zvalue_name=zvalue_name,
        #         Ndata=xobs.shape[0], ifmcmc=True)

        # scalar = np.arange(0., 2.0, 0.1)
        # sharpness_fit_rescale(xobs, yobs, zobs, e_xobs, None, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, xpdv, edges_pdv, tck_pdv, tp_pdv,
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
        #         iedges_obs = np.load(obsdir+"feh/{:0.0f}/tnu_edge_samples.npy".format(ibin))
        #         itck_obs, itp_obs = np.load(obsdir+"feh/{:0.0f}/nike_spline_tck.npy".format(ibin), allow_pickle=True)

        #         # read in unperturbed mass and radius, with edges
        #         ipdv = np.load(moddir+"feh/{:0.0f}/padova.npy".format(ibin), allow_pickle=True).tolist()
        #         iedges_pdv = np.load(moddir+"feh/{:0.0f}/tnu_edge_samples.npy".format(ibin))
        #         itck_pdv, itp_pdv = np.load(moddir+"feh/{:0.0f}/nike_spline_tck.npy".format(ibin), allow_pickle=True)       

        #         # to exclude those points which lies below the edge (so no horizontal distance).
        #         idx = iobs["dnu"]>np.min(iedges_obs[:,1])
        #         for key in iobs.keys():
        #                 iobs[key] = iobs[key][idx]

        #         idx = ipdv["dnu"]>np.min(iedges_pdv[:,1])
        #         for key in ipdv.keys():
        #                 ipdv[key] = ipdv[key][idx]

        #         ixobs, iyobs = iobs["numax"], iobs["dnu"]
        #         ie_xobs, ie_yobs = iobs["e_numax"], iobs["e_dnu"]
        #         ie_xobs, ie_yobs = ie_xobs/ixobs, ie_yobs/iyobs

        #         ixpdv, iypdv = ipdv["numax"], ipdv["dnu"]

        #         filepath = rootpath+"sample/sharpness/sharma16/numax_feh/perturb/"
        #         if not os.path.exists(filepath): os.mkdir(filepath)
        #         filepath = rootpath+"sample/sharpness/sharma16/numax_feh/perturb/{:0.0f}/".format(ibin)
        #         if not os.path.exists(filepath): os.mkdir(filepath)

        #         xperturb = np.arange(0.00, 0.035, 0.002)#0.001) # np.arange(0.00, 0.06, 0.11)#
        #         yperturb = np.array([0.])
        #         sharpness_fit_perturb(ixobs, iyobs, ixobs, None, None, iedges_obs, itck_obs, itp_obs,
        #                 ixpdv, iypdv, ixpdv, iedges_pdv, itck_pdv, itp_pdv,
        #                 diagram, distance, hist_model,
        #                 filepath, xperturb, yperturb, montecarlo,
        #                 Ndata=ixobs.shape[0], ifmcmc=True)

        #         filepath = rootpath+"sample/sharpness/sharma16/numax_feh/rescale/"
        #         if not os.path.exists(filepath): os.mkdir(filepath)
        #         filepath = rootpath+"sample/sharpness/sharma16/numax_feh/rescale/{:0.0f}/".format(ibin)
        #         if not os.path.exists(filepath): os.mkdir(filepath)

        #         scalar = np.arange(0., 2.0, 0.1)
        #         sharpness_fit_rescale(ixobs, iyobs, ixobs, ie_xobs, None, iedges_obs, itck_obs, itp_obs,
        #                 ixpdv, iypdv, ixpdv, iedges_pdv, itck_pdv, itp_pdv,
        #                 diagram, distance, hist_model,
        #                 filepath, scalar, montecarlo,
        #                 Ndata=ixobs.shape[0], ifmcmc=True)


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
        sharpness_fit_perturb_mcmc(xobs, yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                diagram, distance, hist_model, filepath, ifmcmc=True)