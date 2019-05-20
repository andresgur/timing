#!/usr/bin/env python3
# coding=utf-8
# Creates power spectrum out of the input lightcurve and averaged power spectrum out of the rebinned lightcurve, according to input parameters.


import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import multiprocessing as mp
from operator import attrgetter
from stingray import Lightcurve, Powerspectrum, AveragedPowerspectrum
import logging
import os
from astropy.modeling.fitting import SherpaFitter
from astropy.modeling import models
sys.path.insert(0, '/home/agurpide/scripts/pythonscripts')
from timing_analysis.lightcurves.load import load_ligth_curve
from plot_utils import plot_functions as pf


# methods
def compute_powers(lc, segments_seconds, rebinning_f):
    """Compute power and averaged power spectrum."""
    pow_spec = Powerspectrum(lc, norm='leahy', gti=None)

    print("Number of powers %i" % (len(pow_spec.power)))

    try:
        # if segment is larger than lightcurve, average and power spectrum are the same at this stage
        if segments_seconds > lc.tseg:
            avg_pow_spec = pow_spec
        else:
            # compute average power spectrum using chunks of the input lightcurve
            avg_pow_spec = AveragedPowerspectrum(lc, segments_seconds, norm='leahy')

        print('Number of light segments used %i' % avg_pow_spec.m)

        # rebin average power spectrum in log in frequency
        rebinned_pow_spec = Powerspectrum.rebin_log(avg_pow_spec, f=rebinning_f)

        print("Frequency resolution of the averaged power of the first bin after rebinning is %.2f mHz"
              % (1000 * rebinned_pow_spec.df))

    except AssertionError as ae:
        print("Impossible to compute average power spectrum -> some lightcurve chunks have counts = 0: %s \n \
        Try taking the entire lightcurve and rebinned it later in frequency or a different lightcurve chunk size" % (ae))
        rebinned_pow_spec = None
    except Exception as ex:
        print("Unknown error %s" % ex)

    print(" The last %.2f s of lightcurve where discarded when computing the averaged power spectrum" % (lc.tseg % segments_seconds))

    return pow_spec, rebinned_pow_spec


def parallelPowspec(lc, norm='leahy', gti=None):
    """Compute power spectra in parallel. Wrapper function for PowerSpectrum."""
    return Powerspectrum(lc, norm=norm, gti=gti)


def add_zero_padding(stingra_lcs):
    """Add zero padding to lightcurves given the longest lightcurve in the input list.

    Finds the lightcurve with the longest length and appends zeros to the rest of the lightcurves provided that they have the same
    binning.
    Parameters:
    -----------
    stingra_lcs : The list of Stingray lightcurves to be zero padded

    """
    # find lc with largest duration
    longest_lc = max(stingr_lcs, key=attrgetter('tseg'))

    logger.info("Lightcurve with longest duration: has %.3f s" % (longest_lc.tseg))

    for index, stingr_lc in enumerate(stingra_lcs):

        if longest_lc.tseg == stingr_lc.tseg and longest_lc.n == stingr_lc.n:
            # to avoid stingray from crashing when creating an empty lc
            print("Skipping reference lightcurve")
            continue
        # check that lightcurves have same temporal resolution
        assert stingr_lc.dt - longest_lc.dt < 1e-6, "Only lightcurves with same temporal resolution can be added together. \n \
         (%.15f != %.15f s)" % (longest_lc.dt, stingr_lc.dt)

        time_diff = longest_lc.tseg - stingr_lc.tseg

        binstoadd = int(round(time_diff / stingr_lc.dt))

        logger.debug("Adding %.3f s of zeros sampled at %.3f s to lightcurve with duration %.3f resulting in %i bins to be added" % (time_diff, stingr_lc.dt,
                     stingr_lc.tseg, binstoadd))

        times_fill = np.linspace(stingr_lc.time[-1] + stingr_lc.dt, stingr_lc.time[-1] + time_diff, binstoadd, endpoint=True)
        counts = np.zeros(len(times_fill))
        # stingray cannot create LC with one bin. To solve this add the last time of the original lightcurve
        # and add the counts of the last bin so when taking the average it will remain the same
        if times_fill.size == 1:
            logger.warning("Only one bin to be added and stingray cannot create a single bin lc. Adding the last bin of the\
            lightcurve copying the number of counts")
            times_fill = np.linspace(stingr_lc.time[-1], stingr_lc.time[-1] + time_diff, binstoadd + 1, endpoint=True)
            counts = np.zeros(len(times_fill))
            counts[0] = stingr_lc.counts[-1]

        zero_padding = Lightcurve(times_fill, counts, gti=None)

        # add zero padding to make all lc of same length
        stingra_lcs[index] = stingr_lc.join(zero_padding)

        new_time_diff = longest_lc.tseg - stingra_lcs[index].tseg

        assert new_time_diff == 0, "Padding failed. Lightcurves have not the same duration."

    print("Padded %i lightcurves" % len(lightcurves))

    return stingra_lcs


def plot_power(pow_spec, ax, color, nyq_freq, plot_noise_level=False):
    """Plot a power spectrum with error bars and histogram like form.

    Parameters
    ----------
    pow_spec : The power spectrum to be plot
    ax : The matplotlib axis
    color : color for the data plot
    nyq_freq : The Nyquist frequency to set the maximum value in the x axis*
    plot_noise_level : Whether to add a line to indicate the poissonian noise level

    """
    # power_spec.m gives the number of power averaged together in each bin (either by frequency binning or averaging powspec)
    ax.errorbar(pow_spec.freq, pow_spec.power, yerr=2 / np.sqrt(pow_spec.m), color=color, ls="-",
                linewidth=0.5, elinewidth=1, markersize=4, errorevery=2, drawstyle='steps-mid')

    ax.set_xlim(pow_spec.df, nyq_freq)
    if plot_poisson_level:
        ax.axhline(y=2, color='gray', ls='--')
    '''
    twin_y = ax.twiny()
    xticks = ax.get_xticks()
    xlabels = ['{:1.1f}'.format(time) for time in 1/xticks]
    twin_y.set_xticks(xticks)
    twin_y.set_xticklabels(xlabels)
    twin_y.set_xlim(ax.get_xlim())
    twin_y.set_xlabel("Time (s)", fontsize=15)
    twin_y.set_xscale("log")
    #twin_y.ticklabel_format(style='plain', axis='x')
    pf.format_axis(twin_y)
    '''


def plot_lc(lc, lc_ax, color='green', segments_seconds=None):
    """Plot lightcurve."""
    lc_ax.errorbar(lc.time - lc.tstart, lc.countrate, yerr=lc.countrate_err, color=color, linewidth=0.5, marker='o', ms=2, errorevery=10)
    # print vertical lines indicating where the lightcurve was chopped
    if segments_seconds is not None:
        [lc_ax.axvline(x=segment, lw=1, color='gray', ls='--', alpha=0.5) for segment in np.arange(0, lc.tseg, segments_seconds)]
        lc_ax.text(min(segments_seconds + 0.5, lc.tseg - 10), max(lc.countrate), "%.3f s" % segments_seconds)


    # lc_ax.set_xlim(0, lc.tseg)
    # lc_ax.set_ylim(0, max(lc.countrate + lc.countrate * 0.1))
def setmodel_hints(model):
    for index, paramname in enumerate(model.param_names):

        if 'x_break' in paramname:
            max = 10**-1
            min = 10**-5
            value = 10**-2
        elif 'amplitude' in paramname:
            max = 30.0
            min = 0.0
            value = 5.0
        elif 'alpha_1' in paramname:
            max = 0.0
            min = -4.0
            value = -1.0
        elif 'alpha_2' in paramname:
            max = 0.0
            min = -4.0
            value = -2.0

        model.bounds[paramname] = (min, max)
        model.parameters[index] = value


# ---------------------------------------------------------------------------------
# main
parser = argparse.ArgumentParser()
parser.add_argument("input_lc", help="Path to the source's lightcurve fits file", nargs='+', type=str)
parser.add_argument("-s", help="Lightcurve segments in number of time bins to perform the power spectrum", nargs='?', type=int, default=500)
parser.add_argument("-r", help="Rebinning of the lightcurve in number of bins", nargs='?', type=int, default=2)
parser.add_argument("-f", help="Rebinning of the averaged power spectrum in Hz", nargs='?', type=float, default=0.01)
parser.add_argument("--add", dest='add', help="Add input lightcurves to create one single lightcurve", action='store_true')
parser.add_argument("-t", "--threads", help="Number of threads for parallel processing", nargs='?', type=int, default=3)
parser.add_argument("-o", "--outdir", help="Output dir", nargs='?', type=str, default='powers')
parser.add_argument("--fit", dest='fit', help='Flag to fit the average power spectrum', action='store_true')
args = parser.parse_args()

lightcurves = args.input_lc
time_bins = args.r
segments = args.s
rebinning_f = args.f
threads = args.threads
outdir = args.outdir

# constants
power_law_prefix = 'pow_'
white_noise_prefix = 'white_noise_'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

scriptname = os.path.basename(__file__)
# logger

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(scriptname)

fig = plt.figure(figsize=(16, 16))

plotlightcurve = 1
ylogscale = 1
color_whenadding = 'green'
plot_poisson_level = True

if plotlightcurve:
    grid = GridSpec(2, 2)
    lc_ax = plt.subplot(grid[0, 0:])
    pow_ax = plt.subplot(grid[1, 0])
    avg_pow_ax = plt.subplot(grid[1, 1])
else:
    grid = GridSpec(2, 1)
    pow_ax = plt.subplot(grid[0, 0])
    avg_pow_ax = plt.subplot(grid[1, 0])

plt.minorticks_on()
# lightcurves
if plotlightcurve:
    lc_ax.set_xlabel('Time (s)', fontsize=20)
    lc_ax.set_ylabel('Cts/s', fontsize=20)
    lc_ax.ticklabel_format(style='sci', axis='x', scilimits=(1, 1000))
    lc_ax.tick_params(length=10, width=2, direction='in', labelsize='17')
    # minor ticks
    lc_ax.tick_params(which='minor', length=5, width=1, direction='in')

avg_pow_ax.ticklabel_format(style='plain', axis='x')
pow_ax.set_xlabel('Frequency (Hz)', fontsize=20)
pow_ax.set_ylabel('Power', fontsize=20)
pow_ax.set_xscale("log")
pow_ax.tick_params(length=10, width=2, direction='in', labelsize='17')
# minor ticks
pow_ax.tick_params(which='minor', length=5, width=1, direction='in')

avg_pow_ax.ticklabel_format(style='plain')
avg_pow_ax.set_xlabel('Frequency (Hz)', fontsize=20)
avg_pow_ax.set_ylabel('Averaged Powers', fontsize=20)
avg_pow_ax.set_xscale("log")
avg_pow_ax.tick_params(length=10, width=2, direction='in', labelsize='17')
# minor ticks
avg_pow_ax.tick_params(which='minor', length=5, width=1, direction='in')


if ylogscale:
    pow_ax.set_yscale("log")
    avg_pow_ax.set_yscale("log")

colors = pf.create_color_array(len(lightcurves))

if args.add:

    logger.debug("Adding %i powerspectra..." % len(lightcurves))

    # create lightcurves
    lc_arrays = [load_ligth_curve(lightcurve) for lightcurve in lightcurves if os.path.isfile(lightcurve)]
    stingr_lcs = [Lightcurve(times, cts, input_counts=False, err=stds, gti=None) for times, cts, stds, time_res in lc_arrays]

    # [logger.debug("Number of bins in lightcurve: %i" % lc.n) for lc in stingr_lcs]

    # add zero padding to all lightcurves so they have same time resolution
    padded_lcs = add_zero_padding(stingr_lcs)

    # only rebin if the input rebinning is biggger than the original resolution of the lightcurves
    rebin_secs = time_bins * padded_lcs[0].dt

    if time_bins >= 2:

        # rebin lightcurves
        padded_lcs = [lc.rebin(rebin_secs) for lc in padded_lcs]

    # at this point all will have same dt and T
    nyq_freq = padded_lcs[0].n / (2 * padded_lcs[0].tseg)

    logger.info("Power spectrum frequency resolution: %.3f Hz " % (nyq_freq * 1000))

    # [logger.info("Duration: %.4f s \n Binning: %.4f \n Nbins: %i" % (reb_lc.tseg, reb_lc.dt, reb_lc.n)) for reb_lc in padded_lcs]

    [plot_lc(lc, lc_ax, color=color) for lc, color in zip(padded_lcs, colors)]

    # sort lc by temporal resolution
    # sorted(padded_lcs, key=attrgetter('dt'), reverse=True)

    # Setup a list of processes that we want to run
    pool = mp.Pool(processes=threads)
    lc_powers = pool.map(parallelPowspec, padded_lcs)
    '''
    for lc in padded_lcs:
        print(lc)
        Powerspectrum(lc, 'leahy', gti=None)
    '''

    added_powers = lc_powers[0].power
    frequencies = lc_powers[0].freq
    nphotons = lc_powers[0].nphots1
    time_bins = lc_powers[0].n

    # add variables for all power spectra
    for pow in lc_powers[1:]:
        added_powers += pow.power
        nphotons += pow.nphots1
        time_bins += pow.n
    # create combined power spectrum
    pow_spec = Powerspectrum()

    pow_spec.power = added_powers
    pow_spec.freq = frequencies
    # the number of powers combined
    pow_spec.m = len(lc_powers)
    pow_spec.nphots = nphotons
    pow_spec.n = time_bins
    pow_spec.norm = 'leahy'
    pow_spec.df = 1 / padded_lcs[0].tseg

    plot_power(pow_spec, pow_ax, color_whenadding, nyq_freq, plot_poisson_level)

    # rebin average power spectrum in log in frequency
    avg_pow_spec = Powerspectrum.rebin_log(pow_spec, f=rebinning_f)

    plot_power(avg_pow_spec, avg_pow_ax, color_whenadding, nyq_freq, plot_poisson_level)

    rms, rms_error = pow_spec.compute_rms(pow_spec.df, nyq_freq)
    ybottom, ytop = pow_ax.get_ylim()
    xbottom, xtop = pow_ax.get_xlim()
    pow_ax.text(xtop * 0.45, ytop * 0.55, "%.1f $\pm$ %.1f" % (rms, rms_error), fontsize=14)

    if args.fit:

        powmodel = models.BrokenPowerLaw1D()

        setmodel_hints(powmodel)

        sfit = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')
        sfit.est_config['numcores'] = threads

        output = sfit(powmodel, avg_pow_spec.freq, avg_pow_spec.power, weights=2 / np.sqrt(avg_pow_spec.m))
        print(sfit.fit_info)
        param_errors = sfit.est_errors(sigma=1)

        logger.info(sfit._fitmodel.sherpa_model)
        avg_pow_ax.plot(avg_pow_spec.freq, output(avg_pow_spec.freq), color='blue')
    avg_pow_ax.legend(loc='best', fontsize=14)

else:

    for lightcurve, color in zip(lightcurves, colors):

        time, cts, std, time_res = load_ligth_curve(lightcurve)

        lc = Lightcurve(time, cts, input_counts=False, err=std)

        print("Lightcurve total length is %.3f s which gives a f resolution of %.2f mHz" % (lc.tseg, 1000 / lc.tseg))

        print("Number of bins in the lightcurve: %i" % lc.n)

        rebin_secs = time_bins * lc.dt

        rebinned_lc = lc.rebin(rebin_secs)

        nyq_freq = rebinned_lc.n / (2 * rebinned_lc.tseg)

        print('Rebinned lightcurve to %.3f s giving a maximum frequency of %.5f Hz' % (rebin_secs, nyq_freq))

        segments_seconds = segments * rebinned_lc.dt

        nsegments = lc.tseg // segments_seconds

        # case when segments_seconds>lenght of the lc
        if nsegments == 0:
            nsegments = 1

        print('Taking lightcurve segments of %.2f s giving %i lightcurves' % (segments_seconds, nsegments))

        if segments_seconds > lc.tseg:
            print("Warning: \n Light curve segment is higher than lightcurve duration \n Only one segment of total duration %.2f s will be used" % lc.tseg)
            segments_seconds = rebinned_lc.tseg

        if plotlightcurve:
            plot_lc(rebinned_lc, lc_ax, color, segments_seconds)

        pow_spec, avg_pow_spec = compute_powers(rebinned_lc, segments_seconds, rebinning_f)

        plot_power(pow_spec, pow_ax, color, nyq_freq, plot_poisson_level)

        if avg_pow_spec is not None:
            # logger.debug("Value of MW in each frequency bin of the avg power spectrum:")
            # print(avg_pow_spec.m)
            plot_power(avg_pow_spec, avg_pow_ax, color, nyq_freq)

            if args.fit:
                powmodel = models.BrokenPowerLaw1D()

                setmodel_hints(powmodel)

                sfit = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='confidence')

                sfit.est_config['numcores'] = threads

                output = sfit(powmodel, avg_pow_spec.freq, avg_pow_spec.power, err=2 / np.sqrt(avg_pow_spec.m))
                print(sfit.fit_info)
                #param_errors = sfit.est_errors(sigma=1)

                logger.info(sfit._fitmodel.sherpa_model)
                #avg_pow_ax.plot(avg_pow_spec.freq, output(avg_pow_spec.freq), color=color)

plt.show()

if len(lightcurves) == 1:
    outputfilename = lightcurves[0].replace(".lc", "pow")
    pf.save_plot(fig, outdir + "/" + outputfilename)
else:
    pf.save_plot(fig, outdir + "/powerspec")
