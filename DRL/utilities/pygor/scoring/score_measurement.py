import numpy as np
import scipy.signal
from Pygor import Experiment
from Pygor import Meas


class Scoring():
    def __init__(self,pygor=None,**kwargs):
        if pygor is None:
            self.pygor = Experiment()
        else:
            self.pygor = pygor
            
            
