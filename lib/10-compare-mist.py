if __name__ == "__main__":
        rootpath = "/mnt/c/Users/yali4742/Dropbox (Sydney Uni)/Work/nike/"#"/headnode2/yali4742/nike/"#
        import numpy as np 
        import matplotlib
        matplotlib.use("Agg")
        import sys
        sys.path.append(rootpath) 
        from lib.histdist import model6
        from lib.wrapper import sharpness_fit_perturb_mcmc, sharpness_fit_rescale_mcmc
        import os
     
        ## trial 1: obtain an lower limit of numax relations
        # fdnu corrected sharma+2016
        obsdir = rootpath+"sample/yu/"
        moddir = rootpath+"sample/mist/"
        
        # read in unperturbed mass and radius, with edges
        obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
        edges_obs = np.load(obsdir+"tnu_edge_samples.npy")
        tck_obs, tp_obs = np.load(obsdir+"nike_spline_tck.npy", allow_pickle=True)

        # read in unperturbed mass and radius, with edges
        pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()
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

        # trial 1.1: no binning
        xobs, yobs = obs["numax"], obs["dnu"]
        e_xobs, e_yobs = obs["e_numax"], obs["e_dnu"]
        e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        idx = (obs["mass"]<=1.9) & (obs["mass"]>=0.8) #& (yobs<=(yedge_obs.max()))
        xobs, yobs, e_xobs, e_yobs = xobs[idx], yobs[idx], e_xobs[idx], e_yobs[idx]

        xpdv, ypdv = pdv["numax"], pdv["dnu"]
        idx = (pdv["mass"]<=1.9) & (pdv["mass"]>=0.8) #& (radius<=yedge_pdv.max())#
        xpdv, ypdv= xpdv[idx], ypdv[idx]


        filepath = rootpath+"sample/sharpness/mist/numax/perturb/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        sharpness_fit_perturb_mcmc(xobs, yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                diagram, distance, hist_model, filepath, ifmcmc=True)


        # trial 1.2: demonstrate observational errors, no binns
        filepath = rootpath+"sample/sharpness/mist/numax/rescale/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        sharpness_fit_rescale_mcmc(xobs, yobs, e_xobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                diagram, distance, hist_model, filepath, ifmcmc=True)


        ### trial 2: obtain an lower limit of dnu relations
        # fdnu corrected sharma+2016
        obsdir = rootpath+"sample/yu/"
        moddir = rootpath+"sample/mist/"

        # read in unperturbed mass and radius, with edges
        obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
        edges_obs = np.load(obsdir+"tnu_edge_samples.npy")
        tck_obs, tp_obs = np.load(obsdir+"nike_spline_tck.npy", allow_pickle=True)

        # read in unperturbed mass and radius, with edges
        pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()
        edges_pdv = np.load(moddir+"tnu_edge_samples.npy")
        tck_pdv, tp_pdv = np.load(moddir+"nike_spline_tck.npy", allow_pickle=True)

        # to exclude those points which lies left to the edge (so no vertical distance).
        idx = obs["numax"]>=np.min(edges_obs[:,0])
        for key in obs.keys():
                obs[key] = obs[key][idx]

        idx = pdv["numax"]>=np.min(edges_pdv[:,0])
        for key in pdv.keys():
                pdv[key] = pdv[key][idx]

        distance = "vertical"
        diagram = "tnu"
        hist_model = model6()

        montecarlo = 120


        # trial 2.1: no binning
        xobs, yobs = obs["numax"], obs["dnu"]
        e_xobs, e_yobs = obs["e_numax"], obs["e_dnu"]
        e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs        
        idx = (obs["mass"]<=1.9) & (obs["mass"]>=0.8) #& (yobs<=(yedge_obs.max()))
        xobs, yobs, e_xobs, e_yobs = xobs[idx], yobs[idx], e_xobs[idx], e_yobs[idx]
        
        xpdv, ypdv = pdv["numax"], pdv["dnu"]
        idx = (pdv["mass"]<=1.9) & (pdv["mass"]>=0.8) #& (radius<=yedge_pdv.max())#
        xpdv, ypdv= xpdv[idx], ypdv[idx]


        filepath = rootpath+"sample/sharpness/mist/dnu/perturb/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        sharpness_fit_perturb_mcmc(xobs, yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                diagram, distance, hist_model, filepath, ifmcmc=True)


        # trial 2.2: demonstrate observational errors, no binns
        filepath = rootpath+"sample/sharpness/mist/dnu/rescale/"
        if not os.path.exists(filepath): os.mkdir(filepath)
        sharpness_fit_rescale_mcmc(xobs, yobs, e_yobs, edges_obs, tck_obs, tp_obs,
                xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
                diagram, distance, hist_model, filepath, ifmcmc=True)


        # ### trial 3: obtain an lower limit of mass relations
        # # fdnu corrected sharma+2016
        # obsdir = rootpath+"sample/yu/"
        # moddir = rootpath+"sample/mist/"

        # # read in unperturbed mass and radius, with edges
        # obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
        # edges_obs = np.load(obsdir+"mr_edge_samples.npy")
        # tck_obs, tp_obs = np.load(obsdir+"mr_spline_tck.npy", allow_pickle=True)

        # # read in unperturbed mass and radius, with edges
        # pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()
        # edges_pdv = np.load(moddir+"mr_edge_samples.npy")
        # tck_pdv, tp_pdv = np.load(moddir+"mr_spline_tck.npy", allow_pickle=True)

        # # # to exclude those points which lies above the edge (so no horizontal distance).
        # # idx = obs["radius"] <= np.max(edges_obs[:,1])
        # # for key in obs.keys():
        # #         obs[key] = obs[key][idx]   

        # # idx = pdv["radius"] <= np.max(edges_obs[:,1])
        # # for key in pdv.keys():
        # #         pdv[key] = pdv[key][idx]

        # distance = "horizontal"
        # diagram = "mr"
        # hist_model = model6()

        # montecarlo = 120

        # # trial 3.1: no binning
        # xobs, yobs = obs["mass"], obs["radius"]
        # e_xobs, e_yobs = obs["e_mass"], obs["e_radius"]
        # e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        # idx = (xobs<=1.9) & (xobs>=0.8)
        # xobs, yobs, e_xobs, e_yobs = xobs[idx], yobs[idx], e_xobs[idx], e_yobs[idx]

        # xpdv, ypdv = pdv["mass"], pdv["radius"]
        # idx = (xpdv<=1.9) & (xpdv>=0.8)#& (radius<=yedge_pdv.max())#
        # xpdv, ypdv= xpdv[idx], ypdv[idx]


        # filepath = rootpath+"sample/sharpness/mist/mass/perturb/"
        # if not os.path.exists(filepath): os.mkdir(filepath)
        # sharpness_fit_perturb_mcmc(xobs, yobs, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model, filepath, ifmcmc=True)


        # # trial 3.2: demonstrate observational errors, no binnings
        # filepath = rootpath+"sample/sharpness/mist/mass/rescale/"
        # if not os.path.exists(filepath): os.mkdir(filepath)
        # sharpness_fit_rescale_mcmc(xobs, yobs, e_xobs, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model, filepath, ifmcmc=True)



        # ### trial 4: obtain an lower limit of radius relations
        # # fdnu corrected sharma+2016
        # obsdir = rootpath+"sample/yu/"
        # moddir = rootpath+"sample/mist/"

        # # read in unperturbed mass and radius, with edges
        # obs = np.load(obsdir+"yu18.npy", allow_pickle=True).tolist()
        # edges_obs = np.load(obsdir+"mr_edge_samples.npy")
        # tck_obs, tp_obs = np.load(obsdir+"mr_spline_tck.npy", allow_pickle=True)

        # # read in unperturbed mass and radius, with edges
        # pdv = np.load(moddir+"mist.npy", allow_pickle=True).tolist()
        # edges_pdv = np.load(moddir+"mr_edge_samples.npy")
        # tck_pdv, tp_pdv = np.load(moddir+"mr_spline_tck.npy", allow_pickle=True)

        # # # # nothing to exclude.
        # # # idx = np.ones(obs["radius"].shape[0], dtype=bool)
        # # # for key in obs.keys():
        # # #         obs[key] = obs[key][idx]   

        # # # idx = np.ones(obs["radius"].shape[0], dtype=bool)
        # # # for key in pdv.keys():
        # # #         pdv[key] = pdv[key][idx]  

        # distance = "vertical"
        # diagram = "mr"
        # hist_model = model6()

        # montecarlo = 120


        # # trial 4.1: no binning
        # xobs, yobs = obs["mass"], obs["radius"]
        # e_xobs, e_yobs = obs["e_mass"], obs["e_radius"]
        # e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs
        # idx = (xobs<=1.9) & (xobs>=0.8)
        # xobs, yobs, e_xobs, e_yobs = xobs[idx], yobs[idx], e_xobs[idx], e_yobs[idx]

        # xpdv, ypdv = pdv["mass"], pdv["radius"]
        # idx = (xpdv<=1.9) & (xpdv>=0.8)#& (radius<=yedge_pdv.max())#
        # xpdv, ypdv= xpdv[idx], ypdv[idx]


        # filepath = rootpath+"sample/sharpness/mist/radius/perturb/"
        # if not os.path.exists(filepath): os.mkdir(filepath)
        # sharpness_fit_perturb_mcmc(xobs, yobs, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model, filepath, ifmcmc=True)


        # # trial 4.2: demonstrate observational errors, no binnings
        # filepath = rootpath+"sample/sharpness/mist/radius/rescale/"
        # if not os.path.exists(filepath): os.mkdir(filepath)
        # sharpness_fit_rescale_mcmc(xobs, yobs, e_yobs, edges_obs, tck_obs, tp_obs,
        #         xpdv, ypdv, edges_pdv, tck_pdv, tp_pdv,
        #         diagram, distance, hist_model, filepath, ifmcmc=True)
