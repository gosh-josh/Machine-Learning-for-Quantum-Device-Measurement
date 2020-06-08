import numpy as np
import os
import numpy as np
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, ColumnDataSource
from bokeh.plotting import figure 
from bokeh.resources import CDN
from bokeh.embed import file_html
import matplotlib as plt
import matplotlib.cm as cm

           
class Data1d():
    def __init__(self,full_data,chan_label,**kwargs):
        self.full_data = full_data
        self.chan_label = chan_label
            
        self.kwargs = kwargs
        self.fig = None
        

    def plot(self):
    
    
        source = ColumnDataSource(data=dict(x=[], y=[]))
        
        source.data = dict(x=self.full_data['vals'][0],y=self.full_data['data'])
        
        
        if 'tools' in self.kwargs:
            TOOLS = self.kwargs['tools']
        else:
            TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset"
       
            
        x_label = self.full_data['varz'][0]
        y_label = self.chan_label
        
        TOOLTIPS = [(x_label, "$x"),(y_label, "$y")]
        
        self.fig = figure(title="%s data"%(y_label), x_axis_label=x_label, y_axis_label=y_label,tooltips=TOOLTIPS)
        
        self.d_s = self.fig.line(x="x", y="y", source=source)
        
        
        return self.fig
        
        
        
    def update(self,data,vals):
        if self.fig is not None:
            self.full_data['data']=data
            self.full_data['vals']=vals[0]
            self.d_s.data_source.remove('x')
            self.d_s.data_source.remove('y')
            self.d_s.data_source.add(data,name = 'y')
            self.d_s.data_source.add(vals[0],name = 'x')
            self.d_s.data_source.trigger("data", self.d_s.data_source.data, self.d_s.data_source.data)
            
    def stream(self,data,rollmax=1000):
        pass






     
class Data2d():
    
    def __init__(self,full_data,chan_label,**kwargs):
        self.full_data = full_data
        self.chan_label = chan_label
   
        self.kwargs = kwargs
        self.fig = None

    def plot(self,clear=False):
        if 'cmap' in self.kwargs:
            cmap = self.kwargs['cmap']
            #use matplotlib to produce a nice cmap
            
        else:
            cmap = "bwr"
            
            
        if (cmap == "yutian"):
            bokehpalette = [plt.colors.rgb2hex(m) for m in np.append(np.loadtxt('yutian_cmap.txt',dtype=np.uint16)/(2**16),np.ones([256,1]),axis=1)]
        else:
            colormap =cm.get_cmap(cmap) 
            bokehpalette = [plt.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
        
        if 'tools' in self.kwargs:
            TOOLS = self.kwargs['tools']
        else:
            TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset"
            
        z_label = self.chan_label
        x_label = self.full_data['varz'][0]
        y_label = self.full_data['varz'][1]
            
            
        TOOLTIPS = [(z_label, "@image"),("(%s,%s)"%(x_label,y_label), "($x, $y)")]
        
        x_vals = self.full_data['vals'][0]
        y_vals = self.full_data['vals'][1]
        
        x_ran = [x_vals[0], x_vals[-1]]
        y_ran = [y_vals[0], y_vals[-1]]
        x_width = abs(np.diff(x_ran)[0])
        y_width = abs(np.diff(y_ran)[0])
        
        
        
        self.fig = figure(x_range=x_ran, y_range=y_ran, x_axis_label=x_label, y_axis_label=y_label, toolbar_location = 'right',tools=TOOLS,tooltips=TOOLTIPS)
        
        data = self.full_data['data']
        
        mapper = LinearColorMapper(palette=bokehpalette,low=data.min(),high=data.max())
        
        self.d_s = self.fig.image(image=[data], x=x_ran[0], y=y_ran[0], dw=x_width, dh=y_width, color_mapper=mapper)
        
        color_bar = ColorBar(color_mapper=mapper, label_standoff=5, orientation='horizontal', location=(0,0))
        
        self.fig.add_layout(color_bar, 'above')
        
        return self.fig
       
       
    def update(self,data,vals):
        if self.fig is not None:
        
            self.full_data['data']=data
            
            
            
            self.d_s.data_source.remove('image')
            self.d_s.data_source.add([data],name = 'image')
            
            self.d_s.data_source.trigger("data", self.d_s.data_source.data, self.d_s.data_source.data)
            self.fig.select_one(LinearColorMapper).update(low=data.min(), high=data.max())
            
            if np.array_equal(vals[0],self.full_data['vals'][0]):
                self.full_data['vals'][0]=vals[0]
                
                #Update x axis and x width
                x_vals = self.full_data['vals'][0]
                x_ran = [x_vals[0], x_vals[-1]]
                x_width = abs(np.diff(x_ran)[0])
                
                
                self.fig.x_range.start = x_ran[0]
                self.fig.x_range.end = x_ran[-1]
                self.d_s.glyph.dw = x_width
                self.d_s.glyph.x = x_ran[0]
                
            if np.array_equal(vals[1],self.full_data['vals'][1]):
                self.full_data['vals'][1]=vals[1]
                
                #Update y axis and y width
                y_vals = self.full_data['vals'][1]
                y_ran = [y_vals[0], y_vals[-1]]
                y_width = abs(np.diff(y_ran)[0])
                
                
                self.fig.y_range.start = y_ran[0]
                self.fig.y_range.end = y_ran[-1]
                self.d_s.glyph.dh = y_width
                self.d_s.glyph.y = y_ran[0]
                
                
        
            
            
            
            
    def change_cmap(self,cmap):
    
    
    
    
    
        if (cmap == "yutian"):
            bokehpalette = [plt.colors.rgb2hex(m) for m in np.append(np.loadtxt('yutian_cmap.txt',dtype=np.uint16)/(2**16),np.ones([256,1]),axis=1)]
        else:
            colormap =cm.get_cmap(cmap) 
            bokehpalette = [plt.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
            
            
        #mapper = LinearColorMapper(palette=bokehpalette,low=self.full_data['data'].min(),high=self.full_data['data'].max())
        
        self.d_s.data_source.remove('image')
        self.d_s.data_source.add([self.full_data['data']],name = 'image')
        
        self.d_s.data_source.trigger("data", self.d_s.data_source.data, self.d_s.data_source.data)
        self.fig.select_one(LinearColorMapper).update(palette=bokehpalette)
        



         
           
           
           
class Data3d():
    def __init__(self,full_data):
        self.full_data = full_data
            

    def plot(self,clear=False):
        pass

    def update(self,data):
        if self.fig is not None:
            pass
   
            
        

        
            