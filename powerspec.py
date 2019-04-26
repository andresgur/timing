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


def add_zero_padding(stingra_lcs):
    '''Add zero padding to lightcurves given the longest lightcurve in the input list.

    Finds the lightcurve with the longest length and appends zeros to the rest of the lightcurves provided that they have the same
    binning, otherwise skips them.
    Parameters:
    -----------
    stingra_lcs : The list of Stingray lightcurves to be zero padded
    '''
    # find lc with largest duration
    longest_lc = max(stingr_lcs, key=attrgetter('tseg'))

    print("Lightcurve with longest duration: has %.3f s" % (longest_lc.tseg))

    # create zero padding to make all lightcurves last the same so they have the same frequency resolution
    print("Filling lightcurves with zero padding...")

    for index, stingr_lc in enumerate(stingra_lcs):

        if longest_lc.tseg == stingr_lc.tseg and longest_lc.n == stingr_lc.n:
            # to avoid stingray from crashing when creating an empty lc
            print("Skipping reference lightcurve")
            continue

        time_diff = longest_lc.tseg - stingr_lc.tseg

        if (stingr_lc.dt - longest_lc.dt >= 2e-4):
            print("Warning: Lightcurve %s does not have same rebinning (%.15f s!= %.15f s) as the original lightcurve. \
            Skipping it..." % (stingr_lc, longest_lc.dt, stingr_lc.dt))
            continue

        print("Adding %.3f s of zeros sampled at %.3f s to lightcurve with duration %.3f" % (time_diff, longest_lc.dt, stingr_lc.tseg))

        binstoadd = int(time_diff // longest_lc.dt)

        times_fill = np.linspace(stingr_lc.time[-1] + longest_lc.dt, stingr_lc.time[-1] + time_diff, binstoadd, endpoint=True)
        zero_padding = Lightcurve(times_fill, np.zeros(len(times_fill)))
        # add zero padding to make all lc of same length
        stingra_lcs[index] = stingr_lc.join(zero_padding)

    print("Padded %i lightcurves" % len(lightcurves))

    return stingra_lcs


def plot_power(pow_spec, ax, color, nyq_freq):
    """Plot a power spectrum with error bars and histogram like form.
    Parameters
    ----------
    pow_spec : The power spectrum to be plot
    ax : The matplotlib axis
    color : color for the data plot
    nyq_freq : The Nyquist frequency to set the maximum value in the x axis"""
    # power_spec.m gives the number of power averaged together in each bin (either by frequency binning or averaging powspec)
    ax.errorbar(pow_spec.freq, pow_spec.power, yerr=2 / np.sqrt(pow_spec.m), color=color, ls="-",
                linewidth=0.5, elinewidth=1, markersize=4, errorevery=1, drawstyle='steps-mid')

    ax.set_xlim(pow_spec.df, nyq_freq)
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


# ---------------------------------------------------------------------------------
# main
parser = argparse.ArgumentParser()
parser.add_argument("-lc", help="Path to the source's lightcurve fits file", nargs='+', type=str)
parser.add_argument("-s", help="Lightcurve segments in number of time bins to perform the power spectrum", nargs='?', type=int, default=500)
parser.add_argument("-r", help="Rebinning of the lightcurve in number of bins", nargs='?', type=int, default=2)
parser.add_argument("-f", help="Rebinning of the averaged power spectrum in Hz", nargs='?', type=float, default=0.01)
parser.add_argument("-a", help="Add input lightcurves to create one single lightcurve", nargs='?', type=bool, default=False)
args = parser.parse_args()

lightcurves = args.lc
time_bins = args.r
segments = args.s
rebinning_f = args.f

addlc = args.a

fig = plt.figure(figsize=(16, 16))

plotlightcurve = 1
ylogscale = 1
color_whenadding = 'green'

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

avg_pow_ax.ticklabel_format(style='plain', axis='x')
pow_ax.set_xlabel('Frequency (Hz)', fontsize=20)
pow_ax.set_ylabel('Power', fontsize=20)
pow_ax.set_xscale("log")

avg_pow_ax.ticklabel_format(style='plain')
avg_pow_ax.set_xlabel('Frequency (Hz)', fontsize=20)
avg_pow_ax.set_ylabel('Averaged Powers', fontsize=20)
avg_pow_ax.set_xscale("log")

if ylogscale:
    pow_ax.set_yscale("log")
    avg_pow_ax.set_yscale("log")

print("Processing %i lightcurves" % len(lightcurves))

colors = pf.create_color_array(len(lightcurves))

if addlc:

    total_times = np.array([])
    total_counts = np.array([])
    total_err = np.array([])

    concatenated = None

    # create lightcurves
    lc_arrays = [load_ligth_curve(lightcurve) for lightcurve in lightcurves]
    stingr_lcs = [Lightcurve(times, cts, input_counts=False, err=stds) for times, cts, stds, time_res in lc_arrays]
    # [plot_lc(lc, lc_ax, 'blue') for lc in stingr_lcs]

    [print("Number of bins in lightcurve: %i" % lc.n) for lc in stingr_lcs]

    # add zero padding to all lightcurves so they have same time resolution
    padded_lcs = add_zero_padding(stingr_lcs)

    [print("Duration: %.3f s \n Binning: %.3f" % (padded_lc.tseg, padded_lc.dt)) for padded_lc in padded_lcs]

    # only rebin if the input rebinning is biggger than the original resolution of the lightcurves
    rebin_secs = time_bins * padded_lcs[0].dt

    if time_bins >= 2:

        # rebin lightcurves, skipping those with different dt
        padded_lcs = [lc.rebin(rebin_secs) for lc in padded_lcs]

    # at this point all will have same dt and T
    nyq_freq = padded_lcs[0].n / (2 * padded_lcs[0].tseg)

    print('Rebinned lightcurve to %.3f s giving a maximum frequency of %.5f Hz' % (rebin_secs, nyq_freq))

    print("After rebinning:")
    [print("Duration: %.3f s \n Binning: %.3f" % (reb_lc.tseg, reb_lc.dt)) for reb_lc in padded_lcs]

    [plot_lc(lc, lc_ax, color=color) for lc, color in zip(padded_lcs, colors)]

    # pool = mp.Pool(processes=4)
    # lc_powers = [pool.apply(Powerspectrum, args=(lc, 'leahy', None, )) for lc in padded_lcs]
    lc_powers = [Powerspectrum(lc, 'leahy', gti=None) for lc in padded_lcs]

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
    new_powspec = Powerspectrum()

    new_powspec.power = added_powers
    new_powspec.freq = frequencies
    # the number of powers combined
    new_powspec.m = len(lc_powers)
    new_powspec.nphots1 = nphotons
    new_powspec.n = time_bins

    plot_power(new_powspec, pow_ax, color_whenadding, nyq_freq)

    # rebin average power spectrum in log in frequency
    rebinned_pow_spec = Powerspectrum.rebin_log(new_powspec, f=rebinning_f)

    plot_power(rebinned_pow_spec, avg_pow_ax, color_whenadding, nyq_freq)

    plt.show()


    '''
    times, cts, stds, time_res = load_ligth_curve(lightcurves[0])
    print("Lightcurve start time (ms): %.4f" % concatenated.tstart)
    binsize = time_res
    print("Time resolution for the concatenated lightcurve: %.3f" % binsize)

    print("Reading all time stamps and count rates")

    Lightcurve(times, cts, input_counts=False, err=stds)

    # process remaining lightcurves and concatenate them
    for lightcurve in lightcurves[1:]:
        times, cts, stds, time_res = load_ligth_curve(lightcurve)

        # create lightcurve with zeros between obs with same binsize
        time_between = times[0] - (concatenated.tstart + concatenated.tseg)

        print('Time in between lightcurves: %.4f' % time_between)
        if time_between < 0:
            print("Error: please input ligthcurves to append in chronological order.")
            sys.exit()
        lc_fill = Lightcurve(times_fill, np.zeros(len(times_fill)))
        print("Lightcurve duration: %.2f s" % (times[-1] - times[0]))
        current_lc = Lightcurve(times, cts, input_counts=False, err=stds)
        concatenated = concatenated.join(current_lc)

    # there seems to be a bug when creating the power spectrum in the library so recreate the lightcurve
    lc = Lightcurve(concatenated.time, concatenated.counts, err=concatenated.counts_err)

    print("Lightcurve total length is %.3f s which gives a f resolution of %.2f mHz" % (lc.tseg, 1000 / lc.tseg))
    '''

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

        plot_power(pow_spec, pow_ax, color, nyq_freq)

        if avg_pow_spec is not None:
            print("Value of MW in each frequency bin of the avg power spectrum:")
            print(avg_pow_spec.m)
            plot_power(avg_pow_spec, avg_pow_ax, color, nyq_freq)

# major ticks
if plotlightcurve:
    pf.format_axis(lc_ax)

pf.format_axis(pow_ax)

pf.format_axis(avg_pow_ax)

if len(lightcurves) == 1:
    outputfilename = lightcurves[0].replace(".lc", "pow")
    pf.save_plot(fig, outputfilename)
else:
    pf.save_plot(fig, "powerspec")
