import sys

import math

sys.path.append('../')

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')
sys.path.append('../utilities/pygor')

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

class Nelder_Mead_tuning:
    def __init__(self,image,start_pixel,window_size, MaxStep, probalility_matrix = None):
        self. image = image
        self.model_binary_classifier = self.load_cnn()
        self.params_list = []
        self.objective_function_list = []
        self.starting_params = np.copy(start_pixel)
        self.bounds1 = np.array([0 ,np.shape(self.image)[0]])
        self.bounds2 = np.array([0 ,np.shape(self.image)[1]])
        self.window_size = window_size
        self.probability_matrix = probalility_matrix

        self.simplex_1  = self.starting_params
        if (self.starting_params[0] + 75) < self.bounds1[1]:
            self.simplex_2 = self.starting_params + np.array([75,0])
        else:
            self.simplex_2 = self.starting_params + np.array([-75, 0])

        if (self.starting_params[1] + 75) < self.bounds2[1]:
            self.simplex_3 = self.starting_params + np.array([0,75])
        else:
            self.simplex_3 = self.starting_params + np.array([0,-75])

        self.initial_simplex = [self.simplex_1,self.simplex_2,self.simplex_3]

        self.bias_triangle_found = False
        self.MaxStep = MaxStep
        self.data_store = []
        self.current_simplex = np.copy(self.initial_simplex)
        self.evaluations = 0

    def main(self):
        while self.bias_triangle_found == False and self.evaluations <= self.MaxStep:
            res = self.optimisation()
            self.current_simplex = res.final_simplex[0]
        return res

    def optimisation(self):
        res = minimize(self.objective_function,self.current_simplex[0],method='Nelder-Mead', options = {'maxiter':1, 'initial_simplex':self.current_simplex})
        return res

    def get_data(self,param1,param2):
        data = self.image[np.int(param1 - self.window_size / 2):np.int(param1 + self.window_size / 2),
               np.int(param2 - self.window_size / 2):np.int(param2 + self.window_size / 2)]
        return data

    def evaluate_vec_norm(self,data):
        prob = self.predict_cnn(data)
        dif = np.array([1, 0]) - np.array([prob, 1 - prob])
        vec_norm = np.linalg.norm(dif)
        return vec_norm, prob

    def objective_function(self,params):
        self.evaluations += 1
        param1, param2 = params[0],params[1]

        if param1-self.window_size/2 < self.bounds1[0] or param1 + self.window_size/2 > self.bounds1[1] or param2 - self.window_size/2 < self.bounds2[0] or param2 + self.window_size/2 > self.bounds2[1]:
            #print("Out of bounds (return 2), ",param1,param2)
            return 2

        if self.probability_matrix == None: #.all()
            data = self.get_data(param1,param2)
            vec_norm, prob = self.evaluate_vec_norm(data)
        else:
            data = self.get_data(param1,param2)
            prob = self.probability_matrix[np.int(param1),np.int(param2)]
            dif = np.array([1, 0]) - np.array([prob, 1 - prob])
            vec_norm = np.linalg.norm(dif)

        self.params_list.append([param1,param2])
        self.objective_function_list.append(vec_norm)

        #print('iteration ', len(self.params_list))

        if prob > 0.5:
            #plt.imshow(data)
            #plt.title('Bias triangle')
            #plt.colorbar()
            #plt.show()
            #print("Bias triangle found")
            self.bias_triangle_found = True
            self.data_store.append(data)

            return vec_norm
        self.bias_triangle_found = False
        return vec_norm

    def evaluate_function(self):
        n = np.int(self.window_size/2)
        self.image_function = np.ones_like(image) * 2
        self.triangle_probability = np.zeros_like(image)
        for index_row in tqdm(range(np.int((np.shape(self.image)[0] - self.window_size * 2)/n))):
            for index_col in range(np.int((np.shape(self.image)[1] - self.window_size * 2)/n)):
                data = self.get_data(n*index_row+self.window_size, n*index_col+self.window_size)
                vec_norm, prob = self.evaluate_vec_norm(data)
                self.image_function[n*index_row + self.window_size - np.int(n/2):n*index_row + self.window_size + np.int(n/2), n*index_col + self.window_size- np.int(n/2):n*index_col + self.window_size + np.int(n/2)] = vec_norm
                self.triangle_probability[n*index_row + self.window_size - np.int(n/2):n*index_row + self.window_size + np.int(n/2), n*index_col + self.window_size- np.int(n/2):n*index_col + self.window_size + np.int(n/2)] = prob

        return self.image_function, self.triangle_probability

    def normalise(self, x):
        x_max = np.amax(x)
        x_min = np.amin(x)
        y = (x - x_min) / (x_max - x_min)
        return y

    def load_cnn(self):
        model_binary_classifier = models.load_model(
            '../../classifier/bias_triangle_binary_classifier.h5')
        return model_binary_classifier

    def predict_cnn(self, measurement):
        x, y = np.shape(measurement)
        test_image = tf.image.resize(self.normalise(np.array(measurement)).reshape(-1, x, y, 1), (32, 32))
        cnn_prediction = self.model_binary_classifier.predict(test_image, steps=1)
        return cnn_prediction[0][0]

def load_image(file_name):
    try:
        strFile = "../data/{}.p".format(file_name)

        image = pickle.load(open(strFile, "rb"))

    except:
        strFile = "../data/{}.npy".format(file_name)
        image = np.load(strFile)

    return image


def plot_3d(data):
    x1_ori = np.arange(np.shape(data)[0])
    x2_ori = np.arange(np.shape(data)[1])

    x1g_ori, x2g_ori = np.meshgrid(x1_ori, x2_ori)

    X_ori = np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection="3d")
    # plot_trisurf(x, y, z,cmap=’viridis’, edgecolor=’none’)
    # Plot the surface.
    ax.plot_wireframe(x1g_ori, x2g_ori, data.reshape(x1g_ori.shape), color='red', alpha=0.7)

    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_zlabel('f(x)', fontsize=18)

    plt.show()

def plot_superimposed(data,image,name):
    #plt.imshow(image)
    #plt.imshow(data, alpha=0.4)
    #plt.colorbar()
    #plt.show()

    my_cmap = cm.viridis.reversed()
    my_cmap.set_under('k', alpha=0)

    extent = [0,1,0,1]

    # Overlay the two images
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='viridis', extent=extent)
    clim = im.properties()['clim']
    im2 = ax.imshow(data, cmap=my_cmap,
                    interpolation='none',
                    clim=0.1, extent=extent)

    clb = fig.colorbar(im, ax=ax)
    clb2 = fig.colorbar(im2, ax=ax)
    clb.set_label('(A)', labelpad=-62, y=1.06, rotation=0)
    clb2.set_label(name, labelpad=52, y=1.06, rotation=0)

    plt.xlabel('V5 (mV)')
    plt.ylabel('V9 (mV)')

    ax.set_yticks([0,1])
    ax.set_xticks([0,1])
    ax.yaxis.set_label_coords(-0.025, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.025)
    plt.show()

def read_full_scan(file,current_dir):
    infile = open(current_dir+'/'+str(file)+'.pickle','rb')
    scan = pickle.load(infile)
    infile.close()

    '''image = scan['Scan data.data']

    scan_time = scan['Scan time (s)']'''

    return scan

name = 'regime_2'
scan= read_full_scan('regime_2_full_scan_data_time','../run_on_device/benchmark')
#scan= read_full_scan('full_scan_data_time','../run_on_device/benchmark')
image = scan['Scan data.data'].data[0]

window_size = 32
MaxStep = 30

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

for index_row in tqdm(range(np.int(((np.shape(image)[0])-window_size)/n))):
    for index_col in range(np.int(((np.shape(image)[1])-window_size)/n)):

        if original_success[index_row,index_col] == 1:
            count += 1
            print("Counting", count)
            '''zwolak = Nelder_Mead_tuning(image,[n*index_row+window_size, n*index_col+window_size],window_size, MaxStep)
            res = zwolak.main()

            success[n * index_row + window_size - np.int(n / 2):n * index_row + window_size + np.int(n / 2),
                    n * index_col + window_size - np.int(n / 2):n * index_col + window_size + np.int(n / 2)] = zwolak.bias_triangle_found

            step_count = zwolak.evaluations
            if np.int(step_count) == MaxStep:
                step_count = 0
            steps[n * index_row + window_size - np.int(n / 2):n * index_row + window_size + np.int(n / 2),
                    n * index_col + window_size - np.int(n / 2):n * index_col + window_size + np.int(n / 2)] = step_count
'''
print("Count", count)

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