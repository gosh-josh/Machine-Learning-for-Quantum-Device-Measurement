import numpy as np
import h5py
from .hyper_corner_solvers import Hyper_corner
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
LEN_DAC=16

class Current_Tuner():
    def __init__(self,device,tuningbias,bias=6e-10):
        self.device=device
        self.device.set_bias_params(np.zeros_like(self.device.get_biases(),dtype=np.float))
        self.device.set_gate_params(self.device.get_gatebounds()[0])
        self.pinchbias=self.device.do0d()
        self.device.set_gate_params(self.device.get_gatebounds()[1])
        self.shortbias=self.set_bias_to_target(tuningbias,self.pinchbias+bias)[0]
    def set_bias_to_target(self,var,target):
        STEP=0.5
        offsets=[]
        if(var=="all"):
            variables=self.device.get_biases()
        else:
            variables=[var]
        for bias in variables:
            val = self.device.getval(bias)
            current = self.device.do0d()
            while abs(current-target)>1e-11:
                if(current>current-target):
                    val-=STEP
                    self.device.setval(bias,val)
                    current=self.device.do0d()
                else:
                    val+=STEP
                    self.device.setval(bias,val)
                    current=self.device.do0d()
            offsets.append(val)
        return offsets
    def get_maxc_minc(self):
        return None
    def solve_hyper_corner(self,plungers=None,method='halfstep'):
        if plungers==None:
            gates = self.device.get_gates()
        else:
            gates = [gate for gate in self.device.gates if gate not in self.device.plungers]
            gates+= plungers
            pass
        
            
    
        
    
        
            