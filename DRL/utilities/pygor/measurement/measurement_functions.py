import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from data import Data
from Data_handeler.metadata import create_data_arguments
from Measurement.do2d import *
from Measurement.do1d import *
from Measurement.calculate import *


Data3d = None

class measurement_funcs():
    @staticmethod
    def do2d(pygor,var1,min1,max1,res1,var2,min2,max2,res2,**kwargs):
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
            type = kwargs.get('type',None)
            
            do2d_func = {'do2d':standard_do2d}
            
            
            if type is None:
                data_h = standard_do2d(pygor,var1,min1,max1,res1,var2,min2,max2,res2)
            else:
                data_h = do2d_func[type](pygor,var1,min1,max1,res1,var2,min2,max2,res2)
            
            
            return data_h
            
           
            
    @staticmethod
    def do1d(pygor,var1,min1,max1,res1,**kwargs):
            """Do a 1d scan keyword arument 'type' determines how.
            
            
            Args:
            var1: scan variable(s) eg "c1"-"c16"
            min1: inital value(s)
            max1: final value or direction
            res1: resolution in var1
            
            Returns:
                data handler object. (call data_handler.data to obtain data)
            """
            
            
            
        
            type = kwargs.get('type',None)
            
            do1d_func = {'do1d':standard_do1d,
                         'combo':do1d_combo,
                         'server':pygor.server.do1d,
                         'do1d_queue':standard_do1d,
                         'combo_queue':do1d_combo,
                         'server_queue':pygor.server.do1d}
           
            
            if type is None:
                if (pygor.mode=='none') or (pygor.mode=='fast'):
                    data_t = pygor.server.do1d(var1,min1,max1,res1)
                    shapes,metadata_list = pygor.server.get_measure_shape()
                        
                    all_params = pygor.get_params("all")
                    for metadata in metadata_list:
                        metadata['params']=all_params
                    x_vals = np.linspace(min1,max1,res1)
                    variables, values, data_list = create_data_arguments(shapes,metadata_list,[var1],[res1],[x_vals])
                    
                    for iter,aqu in enumerate(data_t):
                        for chan_num in range(len(data_list)):
                            data_list[chan_num][iter] = aqu[chan_num][0]
                
                    data_h = Data(variables,values,data_list,mode=pygor.mode,metadata=metadata_list,label="measurement_%s"%(pygor.figcounter),savedir=pygor.savedir)
                else:
                    data_h = standard_do1d(pygor,var1,min1,max1,res1)
            elif 'queue' in type:
                args = kwargs.get('args',[True]*4)
                args_list = [var1,min1,max1,res1]
                for i,arg in args:
                    if arg:
                        iterator = args_list[i]
                
                cur_args = [var1[:],min1[:],max1[:],res1[:]]
                data_h = []
                for i in range(len(iterator)):
                    for j,arg in enumerate(args):
                        if arg:
                            cur_args[j] = args_list[j][i]
                    data_h += [do1d_func[type](pygor,cur_args[0],cur_args[1],cur_args[2],cur_args[3])]
            
            else:
                data_h = do1d_func[type](pygor,var1,min1,max1,res1)
            
            
            return data_h
    @staticmethod
    def do0d(pygor):
            """Take a single current measurement
            
            Returns:
                single current value
            """
            return pygor.server.do0d()
            
    @staticmethod  
    def setval(pygor,var,val):
            """Set a variable to a specific voltage
            
            Args:
            var:  variable "c1"-"c16" to be set
            val: target value of var
            settletime: sets the settle time when var is changed
            shuttle: defines if value should be shuttled at top speed (WARNING USE FOR ONLY BIAS OR ONLY GATE JUMPS OF <50mV)
            
            Returns:
                final value of var
            """
            
            
            check = pygor.allcontrol_r
            if(var in check):
                v=int(var[1:])
                params = pygor.get_params(check[var][0])
                params[v-1]=float(val)
                newp = pygor.set_params(params,key=check[var][0])
                return newp[v-1]
            return None
    
    @staticmethod  
    def setvals(pygor,varz,vals):
        """Perform multiple setvals
        
        Args:
        varz: list of variables
        vals: list of target values equal in length to varz
        
        Returns:
            list of changes
        """
        
        allcontrol = pygor.server.get_control_labels()
        varkeys = {}
        varvals = {}
        for i,var in enumerate(varz):
            try:
                for controlkey,control in allcontrol.iteritems():
                        if(var in control):
                            if(controlkey in list(varkeys.keys())):
                                varkeys[controlkey]+=[var]
                                varvals[controlkey]+=[vals[i]]
                            else:
                                varkeys[controlkey]=[var]
                                varvals[controlkey]=[vals[i]]
            except AttributeError:
                for controlkey,control in allcontrol.items():
                        if(var in control):
                            if(controlkey in list(varkeys.keys())):
                                varkeys[controlkey]+=[var]
                                varvals[controlkey]+=[vals[i]]
                            else:
                                varkeys[controlkey]=[var]
                                varvals[controlkey]=[vals[i]]
                                
                                
                                
        for varkey in list(varkeys.keys()):
            params = pygor.get_params(varkey)
            indexes = [allcontrol[varkey].index(i) for i in varkeys[varkey]]
            params = np.array(params)
            params[indexes]=varvals[varkey]
            pygor.set_params(params,key=varkey)
            
        
        return vals
    
    @staticmethod           
    def getval(pygor,var):
            """Get a variables current voltage
            
            Args:
            var:  variable "c1"-"c16" to be set
            
            Returns:
                value of var
            """
            
            
            
            
            check = pygor.allcontrol_r
            if(var in check):
                v=int(var[1:])
                params = pygor.get_params(check[var][0])
                return params[v-1]
            return None
    
    
    @staticmethod  
    def getvals(pygor,varz):
        """Perform multiple getvals
        
        Args:
        varz: list of variables
        
        Returns:
            list of vals
        """
        
        
        
        
        vals = []
        for var in varz:
            vals += [pygor.getval(var)]
        return vals
            
    @staticmethod
    def set_params(pygor,params,key=None):
            """Bulk sets all 16 voltages
            
            Args:
            params: list of length containing the voltages to set DAC channels (c1 is params[0] ect)
            settletime: sets the settle time after all voltages are changed
            
            
            Returns:
                array length 16 of current voltages
            """
            
            if key is None:
                key="dac"
            
            
            if(len(params)==16):
                if(type(params) is list):
                    return pygor.server.pushparams(key,params)
                else:
                    return pygor.server.pushparams(key,params.tolist())
            return None
            
    @staticmethod
    def get_params(pygor,key=None):
            """Bulk get all 16 voltages
            
            Returns:
                array length 16 of current voltages
            """
            
            if key is None:
                key="dac"
            elif key == "all":
                return pygor.server.pullparams() 
            
            return pygor.server.pullparams(key) 
            
    @staticmethod
    def do3d(pygor,var1,min1,max1,res1,var2,min2,max2,res2,var3,min3,max3,res3):
        data = np.zeros([len(pygor.do0d()),res1,res2,res3])
        zvals = np.linspace(min3,max3,res3)
        for i in range(res3):
            pygor.setval(var3,zvals[i])
            data[:,:,:,i]=pygor.do2d(var1,min1,max1,res1,var2,min2,max2,res2).data
        data_h = Data3d([var1,var2,var3],np.array([np.linspace(min1,max1,res1),np.linspace(min2,max2,res2),zvals]),data,label="measurement_%s_isosurface"%(pygor.figcounter),savedir=pygor.savedir)
        data_h.plot()
        data_h.save()
        return data_h
        
    @staticmethod
    def do1d_combo(pygor,vararr,minarr,maxarr,step):
    
        def get_data_at_i(i,x_vals,values,data):
            for v in values:
                v[0]=x_vals[:i]
            return  values,[d[:i] for d in data]
    
    
    
    
        if(len(vararr)==len(minarr)==len(minarr)):
            all_vals=[]
            for i in range(len(vararr)):
                all_vals += [np.linspace(minarr[i],maxarr[i],step)]
            all_vals = np.array(all_vals)
            
            x_vals = np.zeros([len(vararr),step])
            for i in range(len(vararr)):
                x_vals[i,:] = np.linspace(0,maxarr[i]-minarr[i],step)
            x_vals = np.sqrt(np.sum(np.power(x_vals,2),axis=0))
            
            
            
            shapes,metadata_list = pygor.server.get_measure_shape()
                        
            all_params = pygor.get_params("all")
            for metadata in metadata_list:
                metadata['params']=all_params
                        
                        
            variables, values, data_list = create_data_arguments(shapes,metadata_list,[str(vararr)],[step],[x_vals])
            
            
            pygor.figcounter+=1
            
            
            data_h=None
            
            for i in range(step):
                
                pygor.setvals(vararr,all_vals[:,i].tolist())
                point=pygor.do0d()
                
                for chan_num, chan in enumerate(point):
                    data_list[chan_num][i]=chan[0]
                                
                if(pygor.mode=='jupyter') or (pygor.mode=='realtime'):
                    if data_h is None:
                        vl_tmp,d_tmp=get_data_at_i(i,x_vals,values,data_list)
                        data_h = Data(variables,vl_tmp,d_tmp,mode=pygor.mode,metadata=metadata_list,label="measurement_%s"%(pygor.figcounter),savedir=pygor.savedir)
                        data_h.plot()
                    else:
                        vl_tmp,d_tmp=get_data_at_i(i,x_vals,values,data_list)
                        data_h.update_all(vl_tmp,d_tmp)

            if data_h is None:
                data_h = Data(variables,values,data_list,mode=pygor.mode,metadata=metadata_list,label="measurement_%s"%(pygor.figcounter),savedir=pygor.savedir)
                            
            if(pygor.mode!='none'):
                data_h.save()
            else:
                return data_h
                            
            if(pygor.mode!='save_only') and (data_h.plotted == False):
                data_h.plot()
                            
            return data_h
           
            
    @staticmethod
    def get_numerical_gradient(pygor,vararr,step=-2.5):
        basedata = pygor.do0d()
        baseval = pygor.getvals(vararr)
        shiftval = np.zeros([len(basedata),len(vararr)])
        for i, var in enumerate(vararr):
            stepvar = list(baseval)
            stepvar[i] += step
            pygor.setvals(vararr,stepvar)
            shiftval[:,i] = pygor.do0d()
        pygor.setvals(vararr,baseval)
        return (shiftval.transpose() - np.array(basedata)).transpose()/(step*1e-3)
        
    @staticmethod
    def calculate(pygor,vararr,vals=None,**kwargs):
            """Calculate paramaters for a measurement
            Args:
                - vararr: 1D array list of variables
    
            Returns:
                -ret: value or list of values that depend on type
        
            Recomended use:
            kwarg 'type' is used to select specific implementation of 
            calculate function. The default is 'parallel_1d' used 
            to calculate starting and stopping points of 1d traces. 
            All calculate is implemented in 'calculate.py'.
            """
    
            type = kwargs.get('type',None)
            
            calculate_func = {'parallel_1d':parallel_1d}
            
            if vals is None:
                params = pygor.getvals(vararr)
            else:
                params = pygor.setvals(vararr,vals)
            
            if type is None:
                ret = parallel_1d(params,**kwargs)
            else:
                ret = calculate_func(params,**kwargs)
            
            
            return ret
    
        

        
        
        
            