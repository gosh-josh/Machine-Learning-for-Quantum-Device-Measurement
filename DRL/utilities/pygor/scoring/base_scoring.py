import numpy as np
from .peak_detection import find_peaks_1d, nearest_neighbor_rejection

#=========================================================================================================================
#Gradient Scoring
#=========================================================================================================================

def invariant_gradient_score(trace,x_vals, height=0.0178, prominence=0.107,offset=-2e-10,maxval=5e-10,pad=(2,2),conv=[-1,-1,0,1,1],**kwargs):
    """Score a current trace.
    
      peakscore - gives score for sum of average observed gradient immediately left and right of peaks

    Args:
        - trace: 1d numpy array of current values
        - x_vals: 1d numpy array of voltage values
    kwargs:
        - pinch: define the point between maxval and offset at which current is considered pinched.
        - height: minimum height of detected peaks normalised between 0 and 1.
        - prominence: the prominence normalised between 0 and 1.
        - offset: current value when no current is flowing.
        - maxval: current value when no current is blocked.
    """
    spacing = abs(np.diff(x_vals)[0])
    
    #Normalise trace between 0 and 1 and remove noise level
    example_trace_norm=trace-offset
    example_trace_norm[example_trace_norm<0]=0
    example_trace_norm = (example_trace_norm)/((maxval-offset))

    trace_grad = np.gradient(example_trace_norm,spacing)
    
    def interpolate_trace(data,x_vals,target_spacing=1):
        newx_vals = np.arange(x_vals[0],x_vals[-1],target_spacing)
        new_data = np.interp(newx_vals,x_vals,data)
        return new_data
    
    if(spacing!=1):
        example_trace_norm = interpolate_trace(example_trace_norm,x_vals)
        trace_grad = interpolate_trace(trace_grad,x_vals)

        
    peaks, data = find_peaks_1d(example_trace_norm,prominence,height,prenorm=True)
    
    
    grad_edges = np.convolve(np.lib.pad(trace_grad,pad,'edge'),conv,mode='valid')
    peakgradscore=[]
    for peak in peaks:
        try:
            peakgradscore+=[np.maximum(grad_edges[peak],0)]
        except IndexError:
            pass
    if(len(peakgradscore)>0):
        ret_type = kwargs.get('value',None)
        if ret_type is None:
            return np.average(peakgradscore)*100
        elif ret_type == 'full':
            return np.array(peakgradscore)*100
        elif ret_type == 'last':
            return peakgradscore[0]*100
        else:
            raise ValueError("Couldn't interpret string given for 'value' keyword argument")
    return 0
    
    
    
def original_gradient_score(example_trace,x_vals, pinch=0.2,height=0.0178, prominence=0.107,offset=0,maxval=6e-10,debug=True):
    """old version of the grasdient score.

    The score function includes two components:
    pinchscore - gives a small score for observing a pinch off plus extra reward for gradient 
    peakscore - gives score for sum of average observed gradient immediately left and right of peaks
  
    Args:
    example_trace: the current trace.
    x_vals: 1d numpy array of voltage values
    pinch: define the point between maxval and offset at which current is considered pinched.
    height: minimum height of detected peaks normalised between 0 and 1.
    prominence: the prominence normalised between 0 and 1.
    offset: current value when no current is flowing.
    maxval: current value when no current is blocked.
    """
    #Normalise trace between 0 and 1 and remove noise level
    vrange = abs(x_vals[0]-x_vals[-1])
  
  
    example_trace=example_trace
    example_trace_norm=example_trace.copy()-offset
    example_trace_norm[example_trace_norm<0]=0
    example_trace_norm = (example_trace_norm)/((maxval-offset))
  
    smoothing = np.convolve(example_trace_norm, np.ones((3,))/3, mode='valid')
    smoothgrad = np.gradient(smoothing,(vrange/example_trace.size))
    trace_grad = np.gradient(example_trace_norm,(vrange/example_trace.size))
    if(len(example_trace_norm[(example_trace_norm)>pinch])>0) and (len(example_trace_norm[(example_trace_norm)<pinch])>0):
        pinchscore=0.1
        gradmax=np.max(smoothgrad)
        pinchscore+=gradmax
    else:
        pinchscore=0
        
    peaks, data = find_peaks_1d(example_trace_norm,prominence,height,prenorm=True)
    
    
    peakgradscore=[]
    for peak in peaks:
        try:
            peakgradscore+=[(trace_grad[peak-1]+trace_grad[peak-2]-trace_grad[peak+1]-trace_grad[peak+2])/4]
        except IndexError:
            pass
    if(len(peakgradscore)>0):
        if(debug):
            return np.minimum(1,(4.8*pinchscore))+peakgradscore[0]*100
        else:
            return np.minimum(1,(4.8*pinchscore))+peakgradscore[0]*100
    else:
        if(debug):
            return np.minimum(1,4.8*pinchscore)
        else:
            return np.minimum(1,4.8*pinchscore)
        
        
        
        
def max_gradient_score(example_trace,x_vals, thresh=0.2,offset=0,maxval=6e-10,smoothing = 3):
    """old version of the pinch score which simply score the max observed gradient.
  
    Args:
    example_trace: the current trace.
    x_vals: 1d numpy array of voltage values
    pinch: define the point between maxval and offset at which current is considered pinched.
    height: minimum height of detected peaks normalised between 0 and 1.
    prominence: the prominence normalised between 0 and 1.
    offset: current value when no current is flowing.
    maxval: current value when no current is blocked.
    """

    #Normalise trace between 0 and 1 and remove noise level
    vrange = abs(x_vals[0]-x_vals[-1])
  
    example_trace=example_trace
    example_trace_norm=example_trace.copy()-offset
    example_trace_norm[example_trace_norm<0]=0
    example_trace_norm = (example_trace_norm)/((maxval-offset))
  
    smoothing = np.convolve(example_trace_norm, np.ones((smoothing,))/smoothing, mode='valid')
    smoothgrad = np.gradient(smoothing,(vrange/example_trace.size))
    
    if(len(example_trace_norm[(example_trace_norm)>thresh])>0) and (len(example_trace_norm[(example_trace_norm)<thresh])>0):
        gradmax=np.max(smoothgrad)
        pinchscore=gradmax
    else:
        pinchscore=0
        
    return pinchscore
    
    

#=========================================================================================================================
#Count Scoring
#=========================================================================================================================

def peak_count_score(trace,x_vals,height=0.0178, prominence=0.107,offset=-2e-10,maxval=5e-10,**kwargs):
    """Score a trace by the amount of observed peaks.
  
    Args:
    example_trace: the current trace.
    x_vals: 1d numpy array of voltage values
    pinch: define the point between maxval and offset at which current is considered pinched.
    height: minimum height of detected peaks normalised between 0 and 1.
    prominence: the prominence normalised between 0 and 1.
    offset: current value when no current is flowing.
    maxval: current value when no current is blocked.
    """

    example_trace_norm=trace-offset
    example_trace_norm[example_trace_norm<0]=0
    example_trace_norm = (example_trace_norm)/((maxval-offset))
    
    peaks, data = find_peaks_1d(example_trace_norm,prominence,height,prenorm=True)
    
    return len(peaks)
    
def threshold_peak_count_score(trace,x_vals,height=0.0178, prominence=0.107,offset=-2e-10,maxval=5e-10,score_threshold=0.5,**kwargs):
    
    kwargs['value'] = 'full'
    all_peak_scores = invariant_gradient_score(trace,x_vals,height=height,prominence=prominence,offset=offset,maxval=maxval,**kwargs)
    if isinstance(all_peak_scores,np.ndarray):
        valid_peaks = all_peak_scores[all_peak_scores>score_threshold]
    else:
        valid_peaks = []
    
    return len(valid_peaks)
    
    
    
    
#=========================================================================================================================
#Space Scoring
#=========================================================================================================================
    
    
def dot_periodicity_score(traces,x_vals,height=0.0178, prominence=0.107,weight=0.1,default=0.2,func2=np.average,**kwargs):
#
#STILL NEEDS WORk
#
#

    #if len(traces) != 2:
    #    raise ValueError("Periodicity score requires list of traces that is length 2, instead got length %i."%(len(traces)))

    peaks = []
    for trace in traces:
        peaks += [find_peaks_1d(trace,prominence,height,**kwargs)[0]]

    peaks_xvals = []
    peaks_xav = []
    for i in range(len(traces)):
        peaks_xvals += [x_vals[peaks[i]]] 
        if len(peaks_xvals[i])>=2:
            peaks_xav += [np.convolve(peaks_xvals[i],[0.5, 0.5],'valid')]
        else:
            peaks_xav += [None]
       
       
    
    
    i_iter = np.where(np.triu(np.ones([len(peaks_xav),len(peaks_xav)],dtype=np.bool),1))
    
    
    scores = []
    for i in range(i_iter[0].size):
        comp_xav = [peaks_xav[i_iter[0][i]]  , peaks_xav[i_iter[1][i]]]
        if (comp_xav[0] is None) or (comp_xav[1] is None):
            scores += [default]
        else:
            comp_xav , blnk = nearest_neighbor_rejection(comp_xav)
            dist = np.linalg.norm(comp_xav[0]-comp_xav[1])/np.sqrt(comp_xav[0].size)
            scores += [dist]
            
            
    return 1-np.tanh(func2(scores)*weight)
    
    
    
    
    
def dot_spacing_score(traces,x_vals,height=0.0178, prominence=0.107,func1=np.average,**kwargs):
    
    peaks = []
    peaks_xvals = []
    for trace in traces:
        pks = find_peaks_1d(trace,prominence,height,**kwargs)[0]
        peaks += [pks]
        peaks_xvals += [x_vals[pks]] 
        
    i_iter = np.where(np.triu(np.ones([len(peaks_xvals),len(peaks_xvals)],dtype=np.bool),1))
        
    diff = []
    scores = []
    for i in range(i_iter[0].size):
        
        comp = [peaks_xvals[i_iter[0][i]]  , peaks_xvals[i_iter[1][i]]]
        
        if len(comp[0])<2 or len(comp[1])<2:
            scores += [0]
        else:
            new_comp,blnk = nearest_neighbor_rejection(comp)
            
            diffs = [np.diff(new_comp[0]),np.diff(new_comp[1])]
            
            dist = np.linalg.norm((diffs[0]-diffs[1])/np.average(np.minimum(diffs[0],diffs[1])))/np.sqrt(diffs[0].size)
            
            scores += [dist]

    return func1(scores)
    
    
def stable_dot_score(traces,x_vals,height=0.0178, prominence=0.107,**kwargs):
    
    

    mult = dot_periodicity_score(traces,x_vals,height=height,prominence=prominence,**kwargs)
    
    val = dot_spacing_score(traces,x_vals,height=height,prominence=prominence,**kwargs)
    
    #print(mult,val)
  
    score = mult*val
    
    return score


    
    

