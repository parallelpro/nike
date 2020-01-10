import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import ascii
import matplotlib.colors as mcolors

inputdir = "sample/girardi/"

# kepler data
data = np.load("sample/obs/nike_samples.npy")
xobso, yobso, feho = data[:,0], data[:,1], data[:,2]
zs = feho#np.concatenate((feh))
min_, max_ = zs.min(), zs.max()
n = mcolors.PowerNorm(vmin=min_, vmax=max_,gamma=1.8)

feh_limits = np.percentile(feho, np.linspace(0., 100., 7))

for j in range(len(feh_limits)-1):
    idx = (feho>=feh_limits[j]) &(feho<=feh_limits[j+1])
    xobs, yobs, feh = xobso[idx], yobso[idx], feho[idx]

    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(12,6),squeeze=False)
    axes=axes.reshape(-1)

    inputfiles = ["feh_m149.csv", "feh_p007.csv"]
    colors=["purple", "red"]
    labels=["[M/H]=-1.49", "[M/H]=+0.07"]
    for i in range(len(inputfiles)):
        data = ascii.read(inputdir+inputfiles[i])
        m, L, T = data["mass"],10.**data["logL"], 10.**data["logT"]
        numax = m * L**-1. * (T/5777)**3.5 * 3090
        dnu = m**0.5 * L**-0.75 * (T/5777)**3.0 * 135.1
        axes[0].plot(numax, numax**0.75/dnu, ".", color=colors[i], label=labels[i], ms=10, zorder=10)
        axes[0].plot(numax, numax**0.75/dnu, "-", color=colors[i], zorder=10)

    # from matplotlib import gridspec

    # from matplotlib.colors import ListedColormap


    c=axes[0].scatter(xobs, yobs, marker=".", c=feh, cmap="jet", norm=n)
    axes[0].axis([15., 150., 2.5, 4.7])
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].set_xlabel("numax")
    axes[0].set_ylabel("numax^0.75/dnu")
    axes[0].set_title("{:0.3f} <= [Fe/H] <= {:0.3f}".format(feh_limits[j], feh_limits[j+1]))
    plt.colorbar(c, ax=axes).set_label("[Fe/H]")
    plt.savefig(inputdir+"zams_feh_{:0.0f}.png".format(j))
    plt.close()