import sys
import os
sys.path.append('../')
sys.path.append('../../')

sys.path.append('E:/Dropbox/03.Research/09.QuantumProjects/Pygor')
import numpy as np

from runpy import run_path
#settings = run_path("E:/Dropbox/03.Research/09.QuantumProjects/Pygor/data.py")
#sys.path.append(os.path.dirname(os.path.expanduser("E:/Dropbox/03.Research/09.QuantumProjects/Pygor/data.p")))

from data import Data
from Data_handeler.metadata import create_data_arguments

#from Pygor.data import Data
#from Pygor.Data_handeler.metadata import create_data_arguments



def standard_do2d(pygor,var1,min1,max1,res1,var2,min2,max2,res2):
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
            
            min1=int(min1)
            max1=int(max1)
            min2=int(min2)
            max2=int(max2)
            
            def get_data_at_i(i,x_vals,values,data):
                for v in values:
                    v[1]=x_vals[:i]
                return  values,[d[:,:i].T for d in data]
   
            check = pygor.allcontrol_r
            if(var1 in check)&(var2 in check):
            
                if(res1!=0)&(res2!=0):
                        currentparams = pygor.getval([var1,var2])

                        x_vals = np.linspace(min1,max1,res1)
                        y_vals = np.linspace(min2,max2,res2)
                        
                        
                        
                        shapes,metadata_list = pygor.server.get_measure_shape()
                        
                        all_params = pygor.get_params("all")
                        for metadata in metadata_list:
                            metadata['params']=all_params
                        
                        
                        variables, values, data_list = create_data_arguments(shapes,metadata_list,[var2,var1],[res2,res1],[y_vals,x_vals])
    
                        
                        

                        
                        data_h=None
    
                        for i,xval in enumerate(x_vals):
                            
                            

                            currentparams = pygor.setval(var1,float(xval))
                            
                            line=pygor.server.do1d(var2,min2,max2,res2)
                            
                            
                            for chan_num in range(len(shapes)):
                                #print(data_list[chan_num][i],[ln[chan_num][0] for ln in line])

                                
                                data_list[chan_num][:,i] = [ln[chan_num][0] for ln in line]
                                #data_list[chan_num][i]=np.array(chan)[0]
                                
                                
                            if(pygor.mode=='jupyter') or (pygor.mode=='realtime'):
                                if data_h is None:
                                    vl_tmp,d_tmp=get_data_at_i(i+1,x_vals,values,data_list)
                                    if(i!=0):
                                        data_h = Data(variables,vl_tmp,d_tmp,mode=pygor.mode,metadata=metadata_list,label="measurement_%s"%(pygor.figcounter),savedir=pygor.savedir)
                                        data_h.plot()
                                else:
                                    vl_tmp,d_tmp=get_data_at_i(i+1,x_vals,values,data_list)
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
            return None
  
        
        
            