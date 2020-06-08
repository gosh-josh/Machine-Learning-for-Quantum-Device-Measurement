import numpy as np


def variable_metadata(shapes,metadata,variables):
    assert len(shapes)==len(metadata)
    variables_list = []
    for i in range(len(shapes)):
        temp_arr = [variables+metadata[i].get('variables',[])]
        if(len(temp_arr)<len(shapes[i])):
            temp_arr += [["UNK"]*(-len(temp_arr)+len(shapes[i]))]
        variables_list+=temp_arr
    return variables_list
    
def data_shapes(shapes,res):
    data_list = []
    for i in range(len(shapes)):
        shape = res + np.array(shapes[i])[np.array(shapes[i])!=1].tolist()
        data_list += [np.zeros(shape,dtype=np.float)]
    return data_list
    
def ranges_metadata(shapes,metadata,ranges):
    assert len(shapes)==len(metadata)
    vals_list = []
    for i in range(len(shapes)):
        vals_temp = ranges+metadata[i].get('values',[])
        if(len(vals_temp)<len(shapes[i])):
            vals_temp += [None]*(-len(vals_temp)+len(shapes[i]))
        vals_list+=[vals_temp]
    return vals_list
    
    
def create_data_arguments(shapes,metadata,variables,res,ranges):
    varz = variable_metadata(shapes,metadata,variables)
    vals = ranges_metadata(shapes,metadata,ranges)
    data = data_shapes(shapes,res)
    return varz,vals,data