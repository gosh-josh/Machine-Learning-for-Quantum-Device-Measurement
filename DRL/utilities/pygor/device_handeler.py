import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import sys
from .Pygor import Experiment
from .measurement_functions import measurement_funcs as meas
try:
    import cPickle as pickle
except ImportError:
    import pickle
LEN_DAC=16

class Device():
    def __init__(self,pygor=None,sdevice=None):
    
        if pygor is None:
            pygor = pygor=Experiment()
    
    
        if sdevice==None:
            self.pygor=pygor
            self.server=pygor.server
            self.savedir = pygor.savedir
            self.savedata = pygor.savedata
            self.gates=["c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16"]
            self.biases=["c1","c2"]
            self.plungers=["c4","c10"]
            self.gatemin=[-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000,-2000]
            self.gatemax=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.biasmin=[-200,-200]
            self.biasmax=[200,200]
            self.dacs=np.array(["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16"],dtype='<U20')
            self.labels=np.array(["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16"],dtype='<U20')
        else:
            self.pygor=pygor
            self.gates=sdevice.gates
            self.biases=sdevice.biases
            self.plungers=sdevice.plungers
            self.gatemin=sdevice.gatemin
            self.gatemax=sdevice.gatemax
            self.biasmin=sdevice.biasmin
            self.biasmax=sdevice.biasmax
            #gatebounds
            self.biasbounds=sdevice.biasbounds
            self.dacs=sdevice.dacs
            self.labels=sdevice.labels
            
        
        
    def set_gates(self,gates,minval,maxval):
        if(len(gates)==len(minval)==len(maxval)):
            if(all(gate in self.dacs for gate in gates))and(len(gates) == len(set(gates))):
                self.gates=gates
                self.gatemax=maxval
                self.gatemin=minval
                return self.gates
    def get_gates(self):
        return self.gates
        
        
    def set_biases(self,biases,minval,maxval):
        if(len(biases)==len(minval)==len(maxval)):
            if(all(bias in self.dacs for bias in biases))and(len(biases) == len(set(biases))):
                self.biases=biases
                self.biasmax=maxval
                self.biasmin=minval
                return self.biases
    def get_biases(self):
        return self.biases
        
    def set_plungers(self,plungers):
        if(all(plunger in self.gates for plunger in plungers))and(len(plungers) == len(set(plungers))):
            self.plungers=plungers
            return self.plungers
    def get_plungers(self):
        return self.plungers        
    
    def set_gatebounds(self,minval,maxval):
        if(len(minval)==len(maxval))and(len(minval)==len(self.gates)):
            self.gatemin=minval
            self.gatemax=maxval
            return self.gatemin,self.gatemax
    def get_gatebounds(self):
        return self.gatemin,self.gatemax
            
    def set_biasbounds(self,minval,maxval):
        if(len(minval)==len(maxval))and(len(minval)==len(self.gates)):
            self.biasmin=minval
            self.biasmax=maxval
            return self.biasmin,self.biasmax
    def get_biasbounds(self,minval,maxval):
        return self.biasmin,self.biasmax
    
            
    def relabel_dac(self,newvar):
        self.labels=newvar
    def name_dacs(self):
        for gate in self.gates:
                if sys.version_info[0] < 3:
                    newname = raw_input("Rename %s: "%(str(gate)))
                    self.labels[self.dacs==gate]=newname
                else:
                    newname = input("Rename %s: "%(str(gate)))
                    self.labels[self.dacs==gate]=newname
            
        for bias in self.biases:
                if sys.version_info[0] < 3:
                    newname = raw_input("Rename %s: "%(str(bias)))
                    self.labels[self.dacs==bias]=newname
                else:
                    newname = input("Rename %s: "%(str(bias)))
                    self.labels[self.dacs==bias]=newname
                    
                    
                    
                    
    def do0d(self):
        return meas.do0d(self)
    def do1d(self,var1,min1,max1,res1):
        
        
        if(var1 in self.labels):
            dacvar = self.dacs[self.labels==var1]
            return meas.do1d(self,dacvar[0],min1,max1,res1)
        else:
            return meas.do1d(self,var1,min1,max1,res1)
            
            
            
            
            
    def do2d(self,var1,min1,max1,res1,var2,min2,max2,res2):
        if(var1 in self.labels)and(var2 in self.labels):
            dacvar1 = self.dacs[self.labels==var1]
            dacvar2 = self.dacs[self.labels==var2]
            return meas.do2d(self,dacvar1[0],min1,max1,res1,dacvar2[0],min2,max2,res2)
        else:
            return meas.do2d(self,var1,min1,max1,res1,var2,min2,max2,res2)
            
            
            
    def getval(self,var):
        if(var in self.labels):
            dacvar = self.dacs[self.labels==var]
            return meas.getval(self,dacvar[0])
        else:
            return meas.getval(self,var)
            
            

    def setval(self,var,val):
        if(var=="all"):
            gate_vals=np.full_like(self.gates,val)
            bias_vals=np.full_like(self.bias,val)
            self.set_gate_params(gate_vals)
            self.set_bias_params(bias_vals)
        if(var in self.labels):
            dacvar = self.dacs[self.labels==var]
            return meas.setval(self,dacvar[0],val)
        else:
            return meas.setval(self,var,val)
            
            
    def set_gate_params(self,gate_params):
        if(len(gate_params)==len(self.gates)):
            currentparams = np.array(self.pygor.get_params())
            currentparams[np.in1d(self.dacs,self.gates)]=np.maximum(np.minimum(gate_params,self.gatemax),self.gatemin)
            return meas.set_params(self,currentparams)
            
            
            
            
    def set_bias_params(self,bias_params):
        if(len(bias_params)==len(self.biases)):
            currentparams = np.array(self.pygor.get_params())
            currentparams[np.in1d(self.dacs,self.biases)]=np.maximum(np.minimum(bias_params,self.biasmax),self.biasmin)
            return meas.set_params(self,currentparams)
            
            
    def save(self,fname):
        sdevice=saved_device(self)
        with open('%s.pkl'%(self.pygor.savedir+"\\"+fname), 'wb') as output:
            pickle.dump(sdevice, output, pickle.HIGHEST_PROTOCOL)
    @classmethod
    def load(cls,fname,pygor):
        with open(fname, 'rb') as f:
            sdevice = pickle.load(f)
        return cls(pygor,sdevice)
        
        
            
            
class saved_device():
    def __init__(self,device):
        self.gates=device.gates
        self.biases=device.biases
        self.plungers=device.plungers
        self.gatemin=device.gatemin
        self.gatemax=device.gatemax
        self.biasmin=device.biasmin
        self.biasmax=device.biasmax
        self.biasbounds=device.biasbounds
        self.dacs=device.dacs
        self.labels=device.labels
        
    
        
            