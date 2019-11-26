import numpy as np 
from numpy.lib.recfunctions import append_fields
from fdnu import asfgrid
from isochrones import get_ichrone
import ebf
'''
I changed lines 34, 86, 138, 149 in mist/models.py to force vcrit=0.

'''

# method 1
tracks = get_ichrone('mist', tracks=True)
mass, age, feh = (1.03, 9.27, -0.11)
a=tracks.generate(mass, age, feh, return_dict=True, accurate=True) 

# # method 2
iso = get_ichrone('mist')

# read in Galaxia stars
rootpath = ''
synp=ebf.read(rootpath+"sample/kepler_galaxia_mrtd5.ebf")

masses, ages, fehs = synp["smass"], synp["log_age"], synp["feh"]
Nstar = masses.shape[0]
keys = ['nu_max','radius','logg','Teff','delta_nu','phase','Mbol','feh',
        'logTeff','initial_mass','density','mass','logL','age']#list(a.keys())
res = np.zeros(Nstar, dtype=np.dtype([(keys[i], np.float64) for i in range(len(keys))]))
for i in range(Nstar):
    print(i, "/", Nstar)
    tmass, tage, tfeh = (float(masses[i]), float(ages[i]), float(fehs[i]))#+np.log10(0.019/0.0142)
    try:
        # method 1:
        # t = tracks.generate(tmass, tage, tfeh, return_dict=True, accurate=True)
        # res[i] = tuple(t.values())

        # # method 2:
        eep = iso.get_eep(tmass, tage, tfeh, accurate=True)
        t = iso.interp_value([eep, tage, tfeh], keys)
        res[i] = tuple(t.tolist())
    except (RuntimeError, ValueError):
        pass

# res = np.load("sample/kepler_galaxia_mist.npy") # delete this line
idx = (res["age"]>0) & (res["phase"]>=2) & (res["phase"]<=3)
nres = res[idx]

# # # append dnu from scaling relations
# new_dt = np.dtype(nres.dtype.descr+[("dnu_scaling", np.float64)])
# numax_scaling1 =  3090.0 * nres["mass"] * nres["radius"]**-2. * (nres["Teff"]/5777.)**-0.5
# dnu_scaling1 = 135.1 * (nres["mass"] * nres["radius"]**-3.)**0.5
feh_galaxia = nres["feh"] #- np.log10(0.019/0.0142) # from feh_mist to feh_galaxia

# # # append corrected dnu using oscillation frequencies
s = asfgrid.Seism(datadir="fdnu/")
logz = feh_galaxia + np.log10(0.019)
teff = nres["Teff"]
mass = nres["mass"]
mini = nres["initial_mass"]
logg = nres["logg"]
dnu, numax, fdnu = s.get_dnu_numax(np.zeros(len(nres))+2, logz, teff, mini, mass, logg)

new_dt = np.dtype(nres.dtype.descr+[("dnu_scaling", np.float64), 
                                    ("feh_mist", np.float64), 
                                    ("dnu", np.float64), 
                                    ("numax", np.float64), 
                                    ("fdnu", np.float64)])
nnres = np.zeros(nres.shape[0], dtype=new_dt)
for name in nres.dtype.names:
    nnres[name] = nres[name]
nnres["dnu_scaling"] = dnu/fdnu
nnres["feh_mist"] = nnres["feh"]
nnres["feh"] = feh_galaxia
nnres["dnu"] = dnu
nnres["numax"] = numax
nnres["fdnu"] = fdnu

np.save("sample/kepler_galaxia_mist_uncofeh", nnres)

