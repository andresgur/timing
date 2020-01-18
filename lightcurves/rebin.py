import sys
sys.path.insert(0, '/home/agurpide/scripts/pythonscripts')
import numpy as np
from timing_analysis.lightcurves.load import load_ligth_curve
from stingray import Lightcurve

def rebin_lc(lc_file,rebin_sec):
    nyq_freq = 1/2*rebin_sec
    print('Rebinning lightcurve to %.2f giving a maximum frequency of %.4f Hz' %(rebin_sec,nyq_freq))
    time,cts,std,time_res = load_ligth_curve(lc_file)
    lc = Lightcurve(time,cts)
    rebinned_lc = lc.rebin(rebin_sec)
    return rebinned_lc
