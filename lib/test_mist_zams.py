import numpy as np 
import sys
sys.path.append('C:\\Users\\yali4742\\Dropbox (Sydney Uni)\\github')
from asteroseismology import EEP
import matplotlib.pyplot as plt 

filepath = "sample/grids/mist/"
files = ["00"+"{:0.0f}".format(i).zfill(2)+"000M.track.eep" for i in range(8,31,1)]
mass, radius = [np.zeros(len(files)) for i in range(2)]

fig = plt.figure(figsize=(6,8))
axes = fig.subplots(nrows=2,ncols=1,squeeze=False).reshape(-1)
for ifile in range(len(files)): #
    eep = EEP(filepath+files[ifile])
    idx=eep.eeps["phase"]>=0
    idx1=eep.eeps["phase"]==3
    idx_zams = eep.eeps["log_L"][idx1] == np.min(eep.eeps["log_L"][idx1])

    axes[0].plot(eep.eeps["log_Teff"][idx], eep.eeps["log_L"][idx], "k--")
    axes[0].plot(eep.eeps["log_Teff"][idx1][idx_zams], eep.eeps["log_L"][idx1][idx_zams], "r.")
    
    mass[ifile] = eep.eeps["star_mass"][idx1][idx_zams][0]
    radius[ifile] = 10.0**eep.eeps["log_R"][idx1][idx_zams][0]

axes[0].set_xlim(3.8, 3.4)
axes[0].set_ylim(1.00, 4.00)

axes[1].plot(mass,radius,"r-",zorder=100)
axes[1].set_xlabel("mass")
axes[1].set_ylabel("radius")

kepler=np.load("sample/obs/tnu_samples.npy")
m,r=kepler[:,3], kepler[:,4]
axes[1].plot(m,r,"k.")
plt.tight_layout()
plt.savefig("sample/test_mist_zams.png")
plt.close()