import numpy as np


def parallel_1d(point,l=100,s=60,n=4,side='minus'):
           
            
            if len(point) != 2:
                raise ValueError("Expected list of size 2 as input, instead got list of size %i."%(len(point)))
                
            s = float(s/(n-1))
            
            
            l_step = (l)/np.sqrt(2)
            s_step = (s)/np.sqrt(2)
            
            if side == 'minus':
                point_c = point - (np.array([l_step,l_step])/2)
            elif side == 'plus':
                point_c = point + (np.array([l_step,l_step])/2)
            else:
                point_c = point
    
            anc_point = point_c+(np.array([s_step,-s_step])*(((n-2)/2)+0.5))+(np.array([l_step,l_step])/2)
            trace_start = [anc_point[...]]
            trace_end = [anc_point[...]-(np.array([l_step,l_step]))]
    
            for i in range(n-1):
                anc_point = anc_point+np.array([-s_step,s_step])
                trace_start += [anc_point[...]]
                trace_end += [anc_point[...]-(np.array([l_step,l_step]))]
                
               
            return trace_start,trace_end
        
            