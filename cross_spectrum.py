#!/usr/bin/env python3
# coding=utf-8
from os.path import sys
sys.path.insert(0, '/home/agurpide/scripts/pythonscripts')
import matplotlib.pyplot as plt
from timing_analysis.lightcurves import rebin as rebin_lc
from plot_utils import plot_functions as pf
import numpy as np
import argparse
from stingray import Lightcurve,Crossspectrum, AveragedCrossspectrum


parser = argparse.ArgumentParser()
parser.add_argument("-lc", help="Path to the source's lightcurve fits file", nargs=2, type=str)
parser.add_argument("-s",  help="Lightcurve segments in seconds to perform the power spectrum", nargs='?', type=float,default=5.0)
parser.add_argument("-r",  help="Rebinning of the lightcurve in seconds", nargs='?', type=float,default=2.0)
args = parser.parse_args()

reference_lc=args.lc[0]
interest_lc=args.lc[1]

rebin = args.r

segments_seconds=args.s

fig, axes = plt.subplots(2,sharex=True,figsize=(16,14))
fig.subplots_adjust(hspace=0)

axes[1].ticklabel_format(style='sci',axis='x')
axes[1].set_xlabel('Frequency (Hz)',fontsize=20)
axes[0].set_ylabel('Cross Spectral Amplitude',fontsize=20)
axes[1].set_ylabel('Time Lag (s)',fontsize=20)
#rebin lcs
ref_rebinned = rebin_lc.rebin_lc(reference_lc,rebin)
interest_rebinned = rebin_lc.rebin_lc(interest_lc,rebin)

avg_cross_spec = AveragedCrossspectrum(interest_rebinned, ref_rebinned, segments_seconds,norm='leahy')
rebinned_cross_spec = Crossspectrum.rebin_log(avg_cross_spec, f=0.01)
freq_lags, freq_lags_err = rebinned_cross_spec.time_lag()

axes[0].plot(rebinned_cross_spec.freq, rebinned_cross_spec.power, lw=1, color='green')
axes[1].errorbar(rebinned_cross_spec.freq, freq_lags, yerr=freq_lags_err,fmt="o", lw=1, color='green',markersize=2)
#major ticks
pf.format_axis(axes[0])
pf.format_axis(axes[1])

plt.show()
