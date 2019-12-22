'''
This is to measure the horizontal/vertical scatter on dnu-numax diagram, 
without manipulating anything else (for example to see as a function of
mass/metallicity).

'''
if __name__ == "__main__":
        rootpath = "/Users/yali4742/Dropbox (Sydney Uni)/Work/nike/"#"/headnode2/yali4742/nike/"#
        import numpy as np 
        import matplotlib
        matplotlib.use("Agg")
        import sys
        sys.path.append(rootpath) 
        from lib.histdist import model6
        from lib.wrapper import sharpness_fit_perturb, sharpness_fit_rescale
        import os

        # fdnu corrected sharma+2016
        obsdir = rootpath+"sample/obs/"
        moddir = rootpath+"sample/padova/"

        # read in unperturbed mass and radius, with edges
        samples_obs = np.load(obsdir+"tnu_samples.npy")
        samples_obs_err = np.load(obsdir+"tnu_samples_err.npy")
        edges_obs = np.load(obsdir+"tnu_edge_samples.npy")
        tck_obs = np.load(obsdir+"nike_spline_tck.npy", allow_pickle=True)

        # read in unperturbed mass and radius, with edges
        samples_pdv = np.load(moddir+"tnu_samples.npy")
        edges_pdv = np.load(moddir+"tnu_edge_samples.npy")
        tck_pdv = np.load(moddir+"nike_spline_tck.npy", allow_pickle=True)

        # to exclude those points which lies left to the edge (so no vertical distance).
        idx = samples_obs[:,0]>=np.min(edges_obs[:,0])
        samples_obs = samples_obs[idx, :]
        samples_obs_err = samples_obs_err[idx, :]

        idx = samples_pdv[:,0]>=np.min(edges_pdv[:,0])
        samples_pdv = samples_pdv[idx, :]

        distance = "vertical"
        diagram = "tnu"
        hist_model = model6()

        montecarlo = 120

        # trial 1: obtain an lower limit of dnu relations
        # trial 1.1: no binning
        xobs, yobs = samples_obs[:,0], samples_obs[:,1]
        e_xobs, e_yobs = samples_obs_err[:,0], samples_obs_err[:,1]
        e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs        
        # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # xobs, yobs = xobs[idx], yobs[idx]
        # e_xobs, e_yobs = e_xobs[idx], e_yobs[idx]
        
        xpdv, ypdv = samples_pdv[:,0], samples_pdv[:,1]
        # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # xpdv, ypdv= xpdv[idx], ypdv[idx]


        filepath = rootpath+"sample/sharpness/sharma16/dnu/perturb/"
        if not os.path.exists(filepath): os.mkdir(filepath)

        xperturb = np.array([0.])
        yperturb = np.arange(0.00, 0.02, 0.001)#0.001) #np.arange(0., 0.05, 0.1)
        sharpness_fit_perturb(xobs, yobs, xobs, None, None, edges_obs, tck_obs,
                xpdv, ypdv, xpdv, edges_pdv, tck_pdv,
                diagram, distance, hist_model,
                filepath, xperturb, yperturb, montecarlo,
                Ndata=xobs.shape[0], ifmcmc=True, cores=10)

        # trial 1.2: demonstrate observational errors, no binns
        # filepath = rootpath+"sample/sharpness/sharma16/dnu/rescale/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # scalar = np.arange(0., 2.0, 0.1)
        # sharpness_fit_rescale(xobs, yobs, xobs, None, e_yobs, edges_obs, tck_obs,
        #         xpdv, ypdv, xpdv, edges_pdv, tck_pdv,
        #         diagram, distance, hist_model,
        #         filepath, scalar, montecarlo,
        #         Ndata=xobs.shape[0], ifmcmc=False)

        # # trial 1.3: mass effect
        # all = np.percentile(samples_obs[:,3], np.linspace(0,100,6))
        # zvalue_limits = [all[:-1].tolist(), all[1:].tolist()]
        # # zvalue_limits = [[0.3, 1.14, 1.47],
        # #                 [1.14, 1.47, 5.24]]
        # zvalue_name = "mass"

        # xobs, yobs, zobs = samples_obs[:,0], samples_obs[:,1], samples_obs[:,3]
        # e_xobs, e_yobs = samples_obs_err[:,0], samples_obs_err[:,1]
        # # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # # xobs, yobs, zobs = xobs[idx], yobs[idx], zobs[idx]

        # xpdv, ypdv, zpdv = samples_pdv[:,0], samples_pdv[:,1], samples_pdv[:,3]
        # # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # # xpdv, ypdv, zpdv = xpdv[idx], ypdv[idx], zpdv[idx]

        # filepath = rootpath+"sample/sharpness/perturb_gif_tnu/dnu_mass/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # sharpness_fit(xobs, yobs, zobs, edges_obs, tck_obs,
        #         xpdv, ypdv, zpdv, edges_pdv, tck_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         zvalue_limits=zvalue_limits, zvalue_name=zvalue_name,
        #         e_xobso=e_xobs, e_yobso=e_yobs)


        # # trial 1.4: feh effect
        # all = np.percentile(samples_obs[:,2], np.linspace(0,100,6))
        # zvalue_limits = [all[:-1].tolist(), all[1:].tolist()]
        # # zvalue_limits = [[-3.0, -0.20, 0.02],
        # #                 [-0.20, 0.02, 1.0]]
        # zvalue_name = "feh"

        # xobs, yobs, zobs = samples_obs[:,0], samples_obs[:,1], samples_obs[:,2]
        # e_xobs, e_yobs = samples_obs_err[:,0], samples_obs_err[:,1]
        # # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # # xobs, yobs, zobs = xobs[idx], yobs[idx], zobs[idx]

        # xpdv, ypdv, zpdv = samples_pdv[:,0], samples_pdv[:,1], samples_pdv[:,2]
        # # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # # xpdv, ypdv, zpdv = xpdv[idx], ypdv[idx], zpdv[idx]

        # filepath = rootpath+"sample/sharpness/perturb_gif_tnu/dnu_feh/"
        # if not os.path.exists(filepath): os.mkdir(filepath)

        # sharpness_fit(xobs, yobs, zobs, edges_obs, tck_obs,
        #         xpdv, ypdv, zpdv, edges_pdv, tck_pdv,
        #         diagram, distance, hist_model,
        #         filepath, xperturb, yperturb, montecarlo,
        #         zvalue_limits=zvalue_limits, zvalue_name=zvalue_name,
        #         e_xobso=e_xobs, e_yobso=e_yobs)


        # trial 1.5: using uncorrected dnu relations

        # fdnu corrected sharma+2016
        obsdir = rootpath+"sample/obs_nc/"
        moddir = rootpath+"sample/padova_nc/"

        # read in unperturbed mass and radius, with edges
        samples_obs = np.load(obsdir+"tnu_samples.npy")
        samples_obs_err = np.load(obsdir+"tnu_samples_err.npy")
        edges_obs = np.load(obsdir+"tnu_edge_samples.npy")
        tck_obs = np.load(obsdir+"nike_spline_tck.npy", allow_pickle=True)

        # read in unperturbed mass and radius, with edges
        samples_pdv = np.load(moddir+"tnu_samples.npy")
        edges_pdv = np.load(moddir+"tnu_edge_samples.npy")
        tck_pdv = np.load(moddir+"nike_spline_tck.npy", allow_pickle=True)

        # to exclude those points which lies left to the edge (so no vertical distance).
        idx = samples_obs[:,0]>=np.min(edges_obs[:,0])
        samples_obs = samples_obs[idx, :]
        samples_obs_err = samples_obs_err[idx, :]

        idx = samples_pdv[:,0]>=np.min(edges_pdv[:,0])
        samples_pdv = samples_pdv[idx, :]

        distance = "vertical"
        diagram = "tnu"
        hist_model = model6()

        montecarlo = 120

        xobs, yobs = samples_obs[:,0], samples_obs[:,1]
        e_xobs, e_yobs = samples_obs_err[:,0], samples_obs_err[:,1]
        e_xobs, e_yobs = e_xobs/xobs, e_yobs/yobs        
        # idx = (xobs<=2.2) #& (yobs<=(yedge_obs.max()))
        # xobs, yobs = xobs[idx], yobs[idx]
        # e_xobs, e_yobs = e_xobs[idx], e_yobs[idx]
        
        xpdv, ypdv = samples_pdv[:,0], samples_pdv[:,1]
        # idx = (xpdv<=1.9) #& (radius<=yedge_pdv.max())#
        # xpdv, ypdv= xpdv[idx], ypdv[idx]


        filepath = rootpath+"sample/sharpness/kb95/dnu/perturb/"
        if not os.path.exists(filepath): os.mkdir(filepath)

        xperturb = np.array([0.])
        yperturb = np.arange(0.00, 0.02, 0.001)#0.001) #np.arange(0., 0.05, 0.1)
        sharpness_fit_perturb(xobs, yobs, xobs, None, None, edges_obs, tck_obs,
                xpdv, ypdv, xpdv, edges_pdv, tck_pdv,
                diagram, distance, hist_model,
                filepath, xperturb, yperturb, montecarlo,
                Ndata=xobs.shape[0], ifmcmc=True, cores=10)

