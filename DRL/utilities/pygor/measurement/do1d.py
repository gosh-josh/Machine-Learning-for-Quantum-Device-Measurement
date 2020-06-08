import numpy as np

import sys
sys.path.insert(0, '')
sys.path.append('E:/Dropbox/03.Research/09.QuantumProjects/Pygor')

from data import Data
from Data_handeler.metadata import create_data_arguments

def standard_do1d(pygor,var1,min1,max1,res1):
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
            
            """
            if(pygor.fast):
                xvals = np.linspace(min1,max1,res1)
                data=pygor.server.do1d(var1,min1,max1,res1)
                data_h=Data([var1],xvals[np.newaxis,...],np.array(data),label="measurement_%s"%(pygor.figcounter),savedir=pygor.savedir)
                data_h.plot()
                if(pygor.savedata):
                        data_h.save()
                return data_h
                        
            """
            
            def get_data_at_i(i,x_vals,values,data):
                for v in values:
                    v[0]=x_vals[:i]
                return  values,[d[:i] for d in data]
   
            check = pygor.allcontrol_r
            if(var1 in check):
            
                if(res1!=0):
                        currentparams = pygor.getval(var1)
                        x_vals = np.linspace(min1,max1,res1)
                        
                        shapes,metadata_list = pygor.server.get_measure_shape()
                        
                        all_params = pygor.get_params("all")
                        for metadata in metadata_list:
                            metadata['params']=all_params
                        
                        
                        variables, values, data_list = create_data_arguments(shapes,metadata_list,[var1],[res1],[x_vals])
                        
                        
                        
                        data_h=None
                        
                        for i,xval in enumerate(x_vals):
                            
                            currentparams = pygor.setval(var1,float(xval))
                            
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
            return None
            
            
            
            
            
            
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
           