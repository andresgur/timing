# !/usr/bin/env python3
# coding=utf-8

# Lightcurve utils
# Built-in imports


# imports
import numpy as np
from astropy.io import fits


def load_ligth_curve(lightcurve):
    """Load lightcurve from fits file. Returns the time counts and errors."""
    print('Reading fits file %s' % lightcurve)
    hdulist = fits.open(lightcurve)
    data = hdulist[1].data
    head = hdulist[1].header
    hdulist.close()

    telescope = head['TELESCOP']
    time_res = head['TIMEDEL']
    times = data[:]['TIME']
    error = data[:]['ERROR']
    cts = data[:]['RATE']

    # set Nan values filtered by the GTI to 0
    valid_counts = np.where(np.isfinite(cts) != True)
    cts[valid_counts] = 0
    error[valid_counts] = 0

    # Nustar lightcurves are set at T = 0
    if telescope == 'NuSTAR':
        print("Reading Nustar lightcurve")
        times += head['TSTART']

    print("Lightcurve start time %.3f s" % times[0])
    print("Lightcurve time resolution %.3f s" % time_res)
    duration = (times[-1] - times[0])
    print("Lightcurve duration %.3f s" % duration)
    return times, cts, error, time_res
