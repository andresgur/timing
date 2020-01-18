#!/usr/bin/env python3
# coding=utf-8

# Built-in imports
from os.path import sys
sys.path.insert(0, '/home/agurpide/scripts/pythonscripts')

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
import os

#my code
import plot_utils.plot_functions as pf
from timing_analysis.lightcurves import load




###
# Parsing arguments
###
parser = argparse.ArgumentParser()
parser.add_argument("-lc", help="Path to the source's lightcurve fits file", nargs='+', type=str)
parser.add_argument("-o", "--outputfile",nargs='?',dest="outputfile",help="Output file name if any",default="")
args = parser.parse_args()

# Defining variables
lightcurves=args.lc

outputfile=args.outputfile

###
# Extracting information from fits files
###
file_count=0
xmin=100
xmax=0
ymin=0
ymax=0

colors = pf.create_color_array(len(lightcurves))

for lightcurve in lightcurves:

    if os.path.isfile(lightcurve):
        time,cts,std,time_res = load.load_ligth_curve(lightcurve)
        # Max, min, etc
        xmin = min(time[0],xmin)
        xmax = max(time[-1],xmax)
        currentymax = np.max(cts + std)
        ymax = max(currentymax,ymax)
        ###
        # Plotting
        ###
        # Source
        plt.errorbar(time, cts, yerr=std, fmt='o', color=colors[file_count],ls="-",linewidth=0.5,elinewidth=0.5,markersize=1,errorevery=10,label="%s" %lightcurve)
        #plt.scatter(time, cts,color=colors[file_count],label="%s" %lightcurve,s=1)
        file_count += 1
    else:
        print("Lightcurve %s not found" %lightcurve)


#plt.scatter(time, cts, color=color[0],s=1)
#plt.plot(time, cts, "o-", linewidth=0.7, markersize=2, color=color[0], label="Source",zorder=2)
#plt.fill_between(time, cts - std, cts + std, alpha=0.2, color=color[0])


#plt.plot(time[index], cts[index], 'x', color=color[1], zorder=5)
# Labels
#plt.legend(loc='upper right', fontsize=10)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("counts s$^{-1}$", fontsize=16)

plt.xlim(xmin, xmax)
#plt.ylim(ymin, ymax)

pf.set_plot_format(14)
plt.ticklabel_format(style='sci',axis='x',scilimits=(1,1000))

if outputfile!="":
    plt.savefig(outputfile, pad_inches=0, bbox_inches='tight')
else:
    if len(lightcurves)==1:
        outputfile=lightcurve.replace("lc", "pdf")
        plt.savefig(outputfile,  bbox_inches='tight')
    elif len(lightcurves)>1:
        outputfile="lightcurve.pdf"
        plt.savefig(outputfile,  bbox_inches='tight')
print("Written output to %s" %outputfile)

plt.show()
