import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import gc
from Data_handeler.data_handeler import Data3d, Data2d, Data1d
from bokeh.plotting import figure, output_file, show
from bokeh.io import push_notebook
from bokeh.models.widgets import Panel, Tabs
from bokeh.embed import file_html
from bokeh.resources import CDN
import uuid 
import pickle

           
            
class Data():
    def __init__(self,variables,values,data,**kwargs):
        """
        Class Data()
        
        Constructor accepts:
        
            - variables: list containing lists of strings of the labels of each variable for each data array
            - values: list of list of numpy arrays containing axes values the variables have been varied by
            - data: list of numpy arrays contaiaing data
        
        ####constraints####
        len(variables)==len(values)==len(data) - must be true
        
        len(variables[i])==len(values[i])==len(data[i].shape) - must be true for any i withing array bounds
        """
    
        
        self.full_data = {}
        self.handlers = {}
        self.data = data
        self.kwargs = kwargs
        self.plotted = False
        
        if 'mode' not in kwargs:
            self.mode = 'jupyter'
        else:
            self.mode = kwargs['mode']
            
        metadata =  kwargs.get('metadata',None)
        self.savedir = kwargs.get('savedir',None)
        
        #Sanity checks
        assert len(variables)==len(values)==len(data)
        
        if metadata is not None:
            assert len(metadata)==len(data)
        
        
        
        #Config channel labels 
        if 'channels' in kwargs:
            assert len(kwargs['channels']) == len(data)
            channels = kwargs['channels']
        else:
            channels = []
            for i in range(len(data)):
                channels+=['chan%i'%(i)]
               
        #Check if data object has been given a name if not create one
        if 'label' in kwargs:
            self.label = kwargs['label']
        else:
            self.label = uuid.uuid4().hex[:6].upper()
        
       
        
        #Loop through channels to check consistancy and create plotting handlers
        for i,data_chan in enumerate(data):
            varz = variables[i]
            vals = values[i]
            if len(varz)!= len(vals) != len(data_chan.shape):
                warnings.warn("Variables, values and data shape are not equal in size for channel %s aborting plotting"%(channels[i]))
                self.handlers[channels[i]] = None
                continue
            else:
                self.full_data[channels[i]] = {}
                self.full_data[channels[i]]['data'] = data_chan
                self.full_data[channels[i]]['varz'] = varz
                self.full_data[channels[i]]['vals'] = vals
                if metadata is not None:
                    self.full_data[channels[i]]['metadata'] = metadata[i]
                if len(data_chan.shape)==1:
                    self.handlers[channels[i]] = Data1d(self.full_data[channels[i]],channels[i],**kwargs)
                elif len(data_chan.shape)==2:
                    self.handlers[channels[i]] = Data2d(self.full_data[channels[i]],channels[i],**kwargs)
                elif len(data_chan.shape)==3:
                    self.handlers[channels[i]] = Data3d(self.full_data[channels[i]],channels[i],**kwargs)
                else:
                    warnings.warn("Data for channel %s is too highly dimensional aborting plotting"%(channels[i]))
                    self.handlers[channels[i]] = None

    def plot(self):
        """
        plot
       
        Plots all given data and stores handle
        """
        
        if(self.mode == 'none') or (self.mode == 'save_only'):
            warnings.warn("Attempting to plot data using incorrect mode. Please change modes to plot data.")
        else:
            tab_list =[]
            for channel, handler in self.handlers.items():
                fig = handler.plot()
                tab_list += [Panel(child=fig, title=channel)]
            
            
            self.tabs = Tabs(tabs=tab_list)
            if(self.mode == 'jupyter'):
                self.target = show(self.tabs, notebook_handle=True) 
                self.plotted = True
            else:
                self.target = None
                show(tabs)
                self.plotted = True
            
    def update_all(self,new_values,new_data):
        """
        update_all
       
        updates all current plots with newly aquired data
        
        """
        assert len(new_values)==len(new_data)== len(self.handlers)
        self.data = new_data
        for i,(channel, handler) in enumerate(self.handlers.items()):
            self.full_data[channel]['data'] = new_data[i]
            self.full_data[channel]['vals'] = new_values[i]
            handler.update(new_data[i],new_values[i])
            
        if(self.mode == 'jupyter') and (self.target is not None):
            push_notebook(handle=self.target)
            
    def change_cmap(self,cmaps):
        for i,(channel, handler) in enumerate(self.handlers.items()):
            if isinstance(handler,Data2d):
                handler.change_cmap(cmaps[i])
        if(self.mode == 'jupyter') and (self.target is not None):
            push_notebook(handle=self.target)
                
    
    def save(self,path=None):
    
        #Check if method has been handed a path
        if path is not None:
            data_path = path+"%s.pickle"%(self.label)
            graph_path = path+"%s.html"%(self.label)
        else:
            data_path = self.savedir+"//numpydata//%s.pkl"%(self.label)
            graph_path = self.savedir+"//images//%s.html"%(self.label)
        
        #If mode is data aquisition only warn user
        if(self.mode == 'none'):
            warnings.warn("Attempting to save data using incorrect mode. Please change modes to save data.")
            
        #If mode is save only only save data
        elif(self.mode == 'save_only'):
            
            #Saving data
            with open(data_path, 'wb') as file_handle:
                pickle.dump(self.full_data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #Else save plot and data
        else:
        
            #Saving data
            with open(data_path, 'wb') as file_handle:
                pickle.dump(self.full_data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            #Saving plot
            html = file_html(self.tabs, CDN, self.label)
            with open(graph_path,"w") as f:
                f.write(html)
        
    @classmethod
    def load(cls,pygor,label=None):
        if isinstance(pygor,str):
            path = pygor
        else:
            path = pygor.savedir+"//numpydata//%s.npz"%(label)
        npzdata = np.load(path)
        
        data = npzdata['data']
        vals = npzdata['vals']
        variables = npzdata['variables']
        if label is not None:
            return cls(variables,vals,data,label=label)
        else:
            return cls(variables,vals,data,label)
        
        
        
"""       
class Data3d():
    def __init__(self,variables,vals,data,datalabels=None,label="in_progress",savedir=None):
        if(len(variables)==len(vals)==3):
            if datalabels is None:
                datalabels=[]
                for i in range(data.shape[0]):
                    datalabels+=["chan%i"%(i+1)]
            self.datalabels=datalabels
            self.variables=variables
            self.vals=vals
            self.data=data
            self.ax_array=None
            self.fig=None
            self.cmap='RdBu_r'
            self.padding = 0.5
            self.cbars=[]
            self.savedir = savedir
            self.label = label
            try:
                from skimage import measure
                self.isosurface=[]
                for i in range(data.shape[0]):
                    self.isosurface += [measure.marching_cubes_lewiner(data[i],spacing=(  (np.max(vals[0])-np.min(vals[0]))/(vals[0].size)  ,  (np.max(vals[1])-np.min(vals[1]))/(vals[1].size)  ,  (np.max(vals[2])-np.min(vals[2]))/(vals[2].size)  ))]
            except:
                print("couldn't extract isosurface")
                self.isosurface = None
            

    def plot(self,clear=False):
        if self.isosurface is not None:
            columns = np.minimum(self.data.shape[0],3)
            rows = int(np.ceil(self.data.shape[0]/3))
        
            if self.fig is None:
                fig, ax_array = plt.subplots(rows, columns,subplot_kw={"projection":'3d'},squeeze=False)
                fig.suptitle(self.label)
            else:
                fig = self.fig
                ax_array = self.ax_array
                fig.suptitle(self.label)
                
            xmult = (self.vals[0][-1]-self.vals[0][0])/np.abs(self.vals[0][-1]-self.vals[0][0])
            ymult = (self.vals[1][-1]-self.vals[1][0])/np.abs(self.vals[1][-1]-self.vals[1][0])
            zmult = (self.vals[2][-1]-self.vals[2][0])/np.abs(self.vals[2][-1]-self.vals[2][0])
            
            print(xmult,ymult,zmult)
            
            if(len(self.variables)==3):
                
                for i,ax_row in enumerate(ax_array):
                    for j,axes in enumerate(ax_row):
                        if(i*columns+j==self.data.shape[0]):
                            break
                        if(clear):
                            axes.cla()
                        axes.plot_trisurf(xmult*(self.isosurface[i*columns+j][0][:, 0]-self.vals[0][0]), ymult*(self.isosurface[i*columns+j][0][:,1]-self.vals[1][0]), self.isosurface[i*columns+j][1], zmult*(self.isosurface[i*columns+j][0][:, 2]-self.vals[2][0]),cmap=self.cmap, lw=1)
                        axes.set_xlabel(self.variables[0])
                        axes.set_ylabel(self.variables[1])
                        axes.set_zlabel(self.variables[2])
                        axes.set_xlim([np.min(self.vals[0]),np.max(self.vals[0])])
                        axes.set_ylim([np.min(self.vals[1]),np.max(self.vals[1])])
                        axes.set_zlim([np.min(self.vals[2]),np.max(self.vals[2])])
                        print(i,j,i*(columns)+j)
                        
                fig.subplots_adjust(wspace=self.padding, hspace=self.padding)
                fig.savefig("plot.pdf",bbox_inches='tight')
                plt.show()
                self.ax_array=ax_array
                self.fig=fig
    def update(self,data):
        if self.fig is not None:
            self.data=data
            self.plot(clear=True)
            plt.show()
    def change_cmap(self,cmap):
        if(len(self.variables)==3):
            self.cmap=cmap
            self.update(self.data)
    def change_padding(self,padding):
        self.padding=padding
        self.update(self.data)
    
    def save(self,path=None):
        if path is None and self.savedir is not None:
            path = self.savedir+"//numpydata//%s.npz"%(self.label)
            pathfig1 = self.savedir+"//images//%s.png"%(self.label)
            pathfig2 = self.savedir+"//images//%s.pdf"%(self.label)
            self.fig.savefig(pathfig1,bbox_inches='tight')
            self.fig.savefig(pathfig2,bbox_inches='tight')
        if path is not None:
            try:
                np.savez(path,variables=self.variables,vals=self.vals,data=self.data)
            except:
                print('file path does not exist')

                
        
    @classmethod
    def load(cls,pygor,label=None):
        if isinstance(pygor,str):
            path = pygor
        else:
            path = pygor.savedir+"//numpydata//%s.npz"%(label)
        npzdata = np.load(path)
        
        data = npzdata['data']
        vals = npzdata['vals']
        variables = npzdata['variables']
        
        return cls(variables,vals,data)

            
            
"""
        
            
        

        
            