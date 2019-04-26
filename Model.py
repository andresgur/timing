from os.path import sys
sys.path.insert(0, '/home/agurpide/scripts/pythonscripts')
from stingray import Lightcurve,Powerspectrum,AveragedPowerspectrum


class Model:
    def __init__(self):
        self.lc = None
        self.std = None
        self.rebinnedlc = None
