#!/usr/bin/env python3
# coding=utf-8

################################################################################
#                                                                              #
# Variabilitectron - Searching for fast transients into XMM-Newton data        #
#                                                                              #
# Generating lightcurve plots                                                  #
#                                                                              #
# Inés Pastor Marazuela (2018) - ines.pastor.marazuela@gmail.com               #
#                                                                              #
################################################################################


# Built-in imports

from math import *
from os.path import sys

# Third-party imports

import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Pdf")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic
from astropy.io import fits
from astropy.time import Time

###
# Parsing arguments
###

parser = argparse.ArgumentParser()
parser.add_argument("-lc", help="Path to the source's lightcurve fits file", nargs='?', type=str)
args = parser.parse_args()

# Defining variables

input=(args.lc).split("/")
obs=input[-1][:10]
src=input[-1][11:25].replace("_", "+")
dt=(input[-1].split("_"))[-1].replace(".lc", "")
out=(input[-1][:25] + "_lc_{0}.pdf".format(dt))

###
# Extracting information from fits files
###

hdulist = fits.open(args.lc)
data    = hdulist[1].data
head    = hdulist[1].header
hdulist.close()

cts  = data[:]['RATE']
time = data[:]['TIME']/1000
std  = data[:]['ERROR']

t0 = time[0]
time = time - t0

cdt  = np.where(np.isfinite(cts) == True)
time = time[cdt]
cts  = cts[cdt]
std  = std[cdt]

for i in range(len(cts)) :
    if cts[i] < 0 :
        cts[i] = 0

# Max, min, etc

xmin = time[0]
xmax = time[-1]
ymin = 0
ymax = np.max(cts + std)

###
# Plotting
###

#seaborn-colorblind
color = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(7,5))

# Source
plt.errorbar(time, cts, yerr=std, fmt='-', color=color[0], alpha=0.5, zorder=5)
plt.plot(time, cts, 'o', color=color[0], zorder=6)
#plt.plot(time, cts, "o-", linewidth=0.7, markersize=2, color=color[0], label="Source",zorder=2)
#plt.fill_between(time, cts - std, cts + std, alpha=0.2, color=color[0])


#plt.plot(time[index], cts[index], 'x', color=color[1], zorder=5)
# Labels
#plt.legend(loc='upper right', fontsize=10)
plt.xlabel("Time (ks)", fontsize=16)
plt.ylabel("counts s$^{-1}$", fontsize=16)
# Text
plt.text(0.05, 0.90, "OBS " + obs, transform = ax.transAxes, fontsize=16)
plt.text(0.05, 0.80, src, transform = ax.transAxes, fontsize=16)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)


plt.minorticks_on()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')


plt.tick_params(axis='both', which='both', direction='in', labelsize=14)
plt.ticklabel_format(style='sci',axis='x')

plt.savefig(out, pad_inches=0, bbox_inches='tight')
plt.show()
