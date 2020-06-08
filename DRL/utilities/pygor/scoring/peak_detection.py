import numpy as np
import scipy.signal


def find_peaks_1d(trace,prominence,height,offset=0,maxval=6e-10,prenorm=False,**kwargs):
    
    if not prenorm:
        trace_norm=trace.copy()-offset
        trace_norm[trace_norm<0]=0
        trace_norm = (trace_norm)/((maxval-offset)) #normalize the current amplitude
    else:
        trace_norm = trace
      
    peaks, data = scipy.signal.find_peaks(trace_norm,prominence=prominence,height=height)
    return peaks, data
    
    
def nearest_neighbor_rejection(values,additional=None,**kwargs):

    values = [np.array(values[0]),np.array(values[1])]


    values_comp = np.abs(values[0][...,np.newaxis] - values[1][np.newaxis,...])
    
    while values_comp.shape[0]!=values_comp.shape[1]:
        
        ind,vl = min(enumerate(values_comp.shape), key=lambda x: x[1])
        mx,vl = max(enumerate(values_comp.shape), key=lambda x: x[1])
        closest_values = np.amin(values_comp,axis=ind)
        furthest_value = closest_values.argmax()
        values_comp=np.delete(values_comp,furthest_value,axis=mx)
        values[mx]=np.delete(values[mx],[furthest_value])
        
        if additional is not None:
            additional[mx]=np.delete(additional[mx],[furthest_value])
            
    return values,additional
        

