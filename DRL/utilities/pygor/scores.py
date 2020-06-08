
from .Scoring.base_scoring import *

def gradient_scoring(trace,x_vals,**kwargs):
    """ Score 1D trace using gradient based methods
    
    Args:
        - trace: 1D array of measurments
        - x_vals: 1D array of x values
    
    Returns:
        -score: single value that represents how good a trace is
        
    Recomended use:
    kwarg 'type' is used to select specific implementation of 
    score function. The default is 'invariant_gradient_score'
    which is the recomended implementation. All gradient scoring
    is implemented in 'base_scoring.py'.
    
    recomended trace resolution is 1mV
    """

    type = kwargs.get('type',None)
    
    scoring_function = {'old':original_gradient_score,
                        'max':max_gradient_score,
                        'inv':invariant_gradient_score}
    
    if type is None:
        score = invariant_gradient_score(trace,x_vals,**kwargs)
    else:
        score = scoring_function[type](trace,x_vals,**kwargs)
       
    return score
    
    
def count_scoring(trace,x_vals,**kwargs):
    """ Score 1D trace using peak count based methods
    
    Args:
        - trace: 1D array of measurments
        - x_vals: 1D array of x values
    
    Returns:
        -score: single value that represents how good a trace is
        
    Recomended use:
    kwarg 'type' is used to select specific implementation of 
    score function. The default is 'threshold_peak_count_score'
    which is the recomended implementation. All count scoring
    is implemented in 'base_scoring.py'.
    
    recomended trace resolution is 1mV
    """
    type = kwargs.get('type',None)
    
    scoring_function = {'basic':peak_count_score,
                        'threshold':threshold_peak_count_score,
                        'basic_binary':peak_count_score,
                        'threshold_binary':threshold_peak_count_score}
                        
    if type is None:
        score = threshold_peak_count_score(trace,x_vals,**kwargs)
    else:
        score = scoring_function[type](trace,x_vals,**kwargs)
        if 'binary' in type:
            score = int(score>0)
    
    return score
    
    
def space_scoring(traces,x_vals,**kwargs):
    """ Score list of 1D traces using peak space methods
    
    Args:
        - traces: list of 1D arrays of measurments
        - x_vals: 1D array of x values
    
    Returns:
        -score: single value that represents how good traces are
        
    Recomended use:
    kwarg 'type' is used to select specific implementation of 
    score function. The default is 'stable_dot_score'
    which is the recomended implementation. All space scoring
    is implemented in 'base_scoring.py'.
    
    recomended trace resolution is 1mV
    """
    type = kwargs.get('type',None)
    
    scoring_function = {'dot_score':stable_dot_score,
                        'scpacing_score':dot_spacing_score,
                        'periodicity_score':dot_periodicity_score}
                        
    if type is None:
        score = stable_dot_score(traces,x_vals,**kwargs)
    else:
        score = scoring_function[type](traces,x_vals,**kwargs)
    
    return score

    
    
    
