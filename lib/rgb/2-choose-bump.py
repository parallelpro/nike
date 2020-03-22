'''
Develop a UI to handpick the most probable (k, b) combination for the bump location.

'''

import numpy as np

from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, output_file, show



rootpath = "/Volumes/Data/Onedrive/Work/nike/"

import sys
sys.path.append(rootpath) 
import os
from lib.histdist import distfit, model_rgb, distance_to_bump
import scipy.signal

# set up plots
ax1 = figure(plot_width=400, plot_height=400)

# initial configuration in bump data
obsdir = rootpath+"sample/rgb/yu/"
moddir = rootpath+"sample/rgb/mist/"
apk = np.load(obsdir+"apk18.npy", allow_pickle=True).tolist()
mist = np.load(moddir+"mist.npy", allow_pickle=True).tolist()

xobs, yobs = apk["teff"], apk["numax"]
xpdv, ypdv = mist['teff'], mist['numax']
eobs = apk["e_teff"]/apk["teff"]
bump_obs = np.load(obsdir+"numax_bump.npy")
bump_pdv = np.load(moddir+"numax_bump.npy")
distance = 'vertical'
hist_model = model_rgb()
bins = np.arange(-30, 30, 2.0)


# # set up distributions
# hdist_obs, xobs, yobs = distance_to_bump(xobs, yobs, bump_obs, distance=distance)
# obj_obs = distfit(hdist_obs, hist_model, bins=bins)
# obj_obs.fit(ifmcmc=False)
# hdist_pdv, xpdv, ypdv = distance_to_bump(xpdv, ypdv, bump_pdv, distance=distance)
# obj_pdv = distfit(hdist_pdv, hist_model, bins=bins)
# obj_pdv.fit(ifmcmc=False)


# # use the parametric model to set up a region to compare
# Ndata = xpdv.shape[0]
# weight = np.zeros(obj_obs.histx.shape, dtype=bool)
# sigma, x0 = obj_obs.para_fit[0], obj_obs.para_fit[1]
# idx = (obj_obs.histx >= x0-4*sigma) & (obj_obs.histx <= x0+4*sigma)
# weight[idx] = True
# eobs_cut = eobs[np.abs(hdist_obs) <= 3*sigma]
    
# if distance=="vertical":
#     fy1_base = np.random.normal(size=Ndata) * 10.0**scipy.signal.resample(np.log10(eobs_cut), Ndata)
#     fp1 = ypdv*fy1_base
#     # fp1 = fy1_base
#     fy2_base = np.random.normal(size=Ndata)
#     fp2 = ypdv*fy2_base


# def model(theta):#, obj_obs, xpdv, ypdv):

#     # theta[0]: offset in distance
#     # theta[1]: perturb

#     # disturb with artificial scatter
#     # xdata, ydata = (xpdv + xpdv*(fx2_base*theta[1])), (ypdv + ypdv*(fy2_base*theta[1]))

#     hdist = hdist_pdv + fp1 + fp2*theta
#     obj = distfit(hdist, hist_model, bins=obj_obs.bins)

#     # normalize the number of points in the weighted region
#     if np.sum(obj.histy[weight])!=0:
#         number_reduction_factor = 1. / np.sum(obj.histy[weight])*np.sum(obj_obs.histy[weight])
#     else:
#         number_reduction_factor = 0.
#     histy = obj.histy * number_reduction_factor
#     return histy, hdist, number_reduction_factor


# # Obs
# ax1.step(obj_obs.histx, obj_obs.histy, color='red')

# # Model
# for s in [0., 0.1, 0.2]:
#     hdist = hdist_pdv + fp1 + fp2*s
#     histy, _, _ = model(s)
#     ax2.step(obj_obs.histx, histy)


# Kepler stars on panel
ax1.scatter(xobs, yobs, radius=0.1, color='red')


xline = np.linspace(np.min(xobs), np.max(xobs), 10)
yline = np.zeros(xline.shape)

# dataColumn = ColumnDataSource(data=dict(xdata=xobs, ydata=yobs, dist=np.zeros(len(xobs))))
lineColumn = ColumnDataSource(data=dict(xline=xline, yline=yline))

# draw the bump line
ax1.line('xline', 'yline', source=lineColumn, color='black', line_width=3, line_alpha=0.6)

# sliders
# source = ColumnDataSource(data=data)
kguess = (np.max(yobs)-np.min(yobs))/(np.max(xobs)-np.min(xobs)) *5.
k_slider = Slider(start=-1, end=0, value=1, step=0.001, title="k")
b_slider = Slider(start=-1000, end=1000, value=1, step=0.001, title="b")


callback = CustomJS(args=dict(lineColumn=lineColumn, kval=k_slider, bval=b_slider),
                    code="""
    const data = lineColumn.data ;
    const k = kval.value ;
    const b = bval.value ;
    const x = data['xline']
    const y = data['yline']
    for (var i = 0; i < x.length; i++) {
        y[i] = b + k*x[i] ;
    }
    lineColumn.change.emit();
""")


k_slider.js_on_change('value', callback)
b_slider.js_on_change('value', callback)

layout = row(
    ax1, 
    column(k_slider, b_slider),
)

output_file("2.html", title="der")

show(layout)