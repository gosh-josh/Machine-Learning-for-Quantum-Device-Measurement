import numpy as np
import h5py
import os
import sys
sys.path.append('../')
sys.path.append('../')
sys.path.append('../Measurement')
sys.path.append('../utilities/pygor')
import matplotlib.pyplot as plt
try: # Python 3
    import xmlrpc.client as rc
except ImportError: # Python 2
    import xmlrpclib as rc
import time
from measurement_functions import measurement_funcs as meas
from bokeh.plotting import output_notebook

LEN_DAC=16

class Experiment():
    def __init__(self,savedir="C:\\Pygor\\",mode='none',debug=False,verbose=False,xmlip="http://129.67.86.107:8000/RPC2",**kwargs):
        if(mode != 'none'):
            while True:
                if isinstance(kwargs.get('savedata',None), str):
                    savename=kwargs.get('savedata',None)
                    savedata=True
                else:
                    if sys.version_info[0] < 3:
                        savename = raw_input("Name the experiment: ")
                    else:
                        savename = input("Name the experiment: ")
                    print(os.path.isfile(savedir+savename))
                if(savename != ""):
                    try:
                        os.makedirs(savedir+savename)
                        os.makedirs(savedir+savename+"//images")
                        os.makedirs(savedir+savename+"//numpydata")
                        os.makedirs(savedir+savename+"//h5")
                        break
                    except FileExistsError:
                        print("File already exists")
                else:
                    print("file is null")
            savedir+=savename
        
        self.savedata=kwargs.get('savedata',None)
        self.figcounter=0
        if xmlip is None:
            self.server = mock_server()
            print("Creating mock server")
        else:
            self.server = rc.ServerProxy(xmlip,verbose=verbose)
            
        self.savedir=savedir
        
        self.mode = mode
        
        
        
        self.allcontrol_r = invert_dict(self.server.get_control_labels())
        
        
        if(mode == 'jupyter'):
            output_notebook()
    
        
        
        
        
        
        
    def get_params(self,key=None):
        """Bulk get all 16 voltages
          
        Returns:
            array length 16 of current voltages
        """
        return meas.get_params(self,key)
    def set_params(self,params,key=None):
        """Bulk sets all 16 voltages
        
        Args:
          params: list of length containing the voltages to set DAC channels (c1 is params[0] ect)
          settletime: sets the settle time after all voltages are changed
          
        Returns:
            array length 16 of current voltages
        """
        return meas.set_params(self,params,key)
    def getval(self,var):
        """Get a variables current voltage
        
        Args:
          var:  variable "c1"-"c16" to be set
          
        Returns:
            value of var
        """
        return meas.getval(self,str(var))
    def setval(self,var,val):
        """Set a variable to a specific voltage
        
        Args:
          var:  variable "c1"-"c16" to be set ("call" can also be used to change all dacs to var)
          val: target value of var
          settletime: sets the settle time when var is changed
          
        Returns:
            final value of var
        """
        if(var=="call"):
            return self.set_params(np.full([LEN_DAC],val))
        else:
            return meas.setval(self,str(var),val)
            
    
    
    
    def setvals(self,varz,vals):
        """Set a variable to a specific voltage
        
        Args:
          var:  variable "c1"-"c16" to be set ("call" can also be used to change all dacs to var)
          val: target value of var
          settletime: sets the settle time when var is changed
          
        Returns:
            final value of var
        """

        return meas.setvals(self,varz,vals)
        
    def getvals(self,varz):
        """Set a variable to a specific voltage
        
        Args:
          var:  variable "c1"-"c16" to be set ("call" can also be used to change all dacs to var)
          val: target value of var
          settletime: sets the settle time when var is changed
          
        Returns:
            final value of var
        """

        return meas.getvals(self,varz)
    
    def do0d(self):
        """Take a single current measurement
          
        Returns:
            single current value
        """
        return meas.do0d(self)
    def do1d(self,var1,min1,max1,res1,**kwargs):
        """Do a 1d scan with defult settings
        
        Args:
          var1: scan variable "c1"-"c16"
          min1: inital value of var1
          max1: final value of var1
          res1: resolution in var1
          settletime: sets the settle time between measurements
          
        Returns:
            1d array of current values
        """
        self.figcounter+=1
        return meas.do1d(self,str(var1),min1,max1,res1,**kwargs)
        
    def do2d(self,var1,min1,max1,res1,var2,min2,max2,res2,**kwargs):
        """Do a 2d grid scan in a raster pattern with defult settings
        
        Args:
          var1: first scan variable "c1"-"c16" (slow in raster)
          min1: inital value of var1
          max1: final value of var1
          res1: resolution in var1
          var2: second scan variable "c1"-"c16" (fast in raster)
          min2: inital value of var2
          max2: final value of var2
          res2: resolution in var2
          
        Returns:
            2d array of current values
        """
        self.figcounter+=1
        return meas.do2d(self,str(var1),min1,max1,res1,str(var2),min2,max2,res2,**kwargs)
    def do3d(self,var1,min1,max1,res1,var2,min2,max2,res2,var3,min3,max3,res3):
        """Do a multiple 2d grid scans in a raster pattern with defult settings and return a 3d scan 
        
        Args:
          var1: first scan variable "c1"-"c16" (slow in raster)
          min1: inital value of var1
          max1: final value of var1
          res1: resolution in var1
          var2: second scan variable "c1"-"c16" (fast in raster)
          min2: inital value of var2
          max2: final value of var2
          res2: resolution in var2
          var3: second scan variable "c1"-"c16" (fast in raster)
          min3: inital value of var3
          max3: final value of var3
          res3: resolution in var3
          
        Returns:
            3d array of current values
        """
        data = meas.do3d(self,str(var1),min1,max1,res1,str(var2),min2,max2,res2,str(var3),min3,max3,res3)
        return data
    def set_settletime(self,settletime):
        """Set settling time
        
        Args:
          settletime: settle time (default 0.02)
          
        Returns:
            current settletime
        """
        return self.server.set_settletime(settletime)
    def set_shuttle(self,shuttle):
        """Set shuttling gates
        
        Args:
          shuttle: set shuttle (default False)
          
        Returns:
            current shuttle
        """
        return self.server.set_shuttle(shuttle)
    def save(self,fname):
        pass
    @classmethod
    def load(cls,fname,pygor):
        pass
        
        
 
        
class mock_server():
    def __init__(self,chans=1):
        self.params = {}
        self.param_labels = {}
        
        self.params["dac"] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.param_labels["dac"] = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16']
        
        self.chans = chans
        
    def pushparams(self,key,params):
        self.params[key] = params
        return params
    def pullparams(self,key=None):
        if key is None:
            return self.params
        else:   
            return self.params[key]
    def get_control_labels(self):
        return self.param_labels
    def do0d(self):
        return [[0]]*self.chans
    def do1d(self,var,minv,maxv,res):
        data = [None]*res
        key1="dac"
        var1_index = self.param_labels["dac"].index(var)
        vals1=np.linspace(minv,maxv,res)
        
        for i,val1 in enumerate(vals1):
            var1_params = self.pullparams(key1)
            var1_params[var1_index]=val1
            self.pushparams(key1,var1_params)
            aqui = self.do0d()
            data[i]=aqui
        return data
    
    
    
        return [np.zeros(res).tolist()]*self.chans
    def get_measure_shape(self):
        return [[1]]*self.chans, [{}]*self.chans
    
def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = [key] 
            else: 
                inverse[item].append(key) 
    return inverse
        
    
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("--------------------------------------------------")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nWelcome to Pygor!")
print("--------------------------------------------------")
print("To start the server:  pygor = Experimentfrontend()")
print("New experiment created in: pygor.savedir")

print("\nTo take measurements use functions \npygor.do2d, pygor.do1d and pygor.do0d.")
print("To control use pygor.setval and pygor.getval.")
print("Function syntax is identical to Igor!")
print("--------------------------------------------------")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")