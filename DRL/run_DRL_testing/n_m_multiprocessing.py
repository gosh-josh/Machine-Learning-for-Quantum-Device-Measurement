
import sys

import math

sys.path.append('../')

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')

from scipy.optimize import minimize

from datetime import datetime
import numpy as np
import timeit
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tqdm import tqdm
import warnings
import pickle
from tqdm import tqdm
from matplotlib import cm
from mpl_toolkits import mplot3d

from nelder_mead_offline import Nelder_Mead_tuning,read_full_scan
import multiprocessing
from multiprocessing import Pool



def dfprocessing_func(coordinates):

    print('Multi-processing')
    window_size = 32
    MaxStep = 30


    zwolak = Nelder_Mead_tuning(image, coordinates, window_size, MaxStep)
    res = zwolak.main()

    success = zwolak.bias_triangle_found

    step_count = zwolak.evaluations

    if np.int(step_count) == MaxStep:
        step_count = 0

    steps = step_count

    return steps #,success


name = 'regime_2'
scan= read_full_scan('regime_2_full_scan_data_time','../run_on_device/benchmark')
#scan= read_full_scan('full_scan_data_time','../run_on_device/benchmark')
image = scan['Scan data.data'].data[0]


success = np.zeros_like(image)
steps = np.zeros_like(image)

n = 1

def load_dict(file_name):
    strFile = "test_data/{}.pickle".format(file_name)
    dict = pickle.load(open(strFile, "rb"))
    return dict

dict_name = 'bench_mark_nelder_mead_regime_2'
dict = load_dict(dict_name)
original_success = dict['success']

count = 0
window_size = 32

if __name__ == '__main__':
    for index_row in tqdm(range(np.int((np.shape(image)[0])))):
        print("Index row",index_row)
        with Pool(5) as p:
            a = [0]*np.shape(image)[1]
            for index, i in enumerate(a):
                a[index] = [index_row,index]
            print(np.shape(a))
            steps[index_row,:] = p.map(dfprocessing_func, a)

    '''for index_col in range(np.int(((np.shape(image)[1])-window_size)/n)):

        if original_success[index_row,index_col] == 1:
            count += 1
            print("Counter", count)
            i = n*index_row+window_size
            j = n*index_col+window_size
            a = [i,j]

            p = multiprocessing.Pool(2)
            success, steps = p.map(dfprocessing_func,a,image)
            p.terminate()
            p.join()

            success[i - np.int(n / 2):i + np.int(n / 2),
            j - np.int(n / 2):j + np.int(
                n / 2)], steps[i - np.int(n / 2):i + np.int(n / 2),
                         j - np.int(n / 2):j + np.int(n / 2)] = success, steps

            print("Steps", steps)'''




plot_3d(steps)
plot_superimposed(steps, image,name)

plot_superimposed(success, image,name)

plot_3d(success)

plot_3d(-steps)



benchmark_nelder_mead = {
    'steps':steps,
    'success':success,
    'triangle_probability':triangle_probability,
    'image_function':image_function,
    'name':name,
    'image':image,
    'n':n,
    'window_size':window_size
}

'''savefilename = 'bench_mark_nelder_mead_high_def_'+name
pickle_out = open('test_data/'+savefilename+".pickle","wb")
pickle.dump(benchmark_nelder_mead, pickle_out)
pickle_out.close()'''