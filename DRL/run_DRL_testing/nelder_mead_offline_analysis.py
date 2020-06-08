import sys

import math

sys.path.append('../')

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import minimize

from datetime import datetime
import numpy as np
import timeit
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
from tensorflow.keras import models
from tqdm import tqdm
import warnings
import pickle
from tqdm import tqdm
import matplotlib
from matplotlib import cm
from mpl_toolkits import mplot3d

def load_dict(file_name):
    strFile = "test_data/{}.pickle".format(file_name)
    dict = pickle.load(open(strFile, "rb"))
    return dict

def plot_3d(data):
    x1_ori = np.arange(np.shape(data)[0])
    x2_ori = np.arange(np.shape(data)[1])

    x1g_ori, x2g_ori = np.meshgrid(x1_ori, x2_ori)

    X_ori = np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection="3d")
    # Plot the surface.

    surf = ax.plot_surface(x1g_ori, x2g_ori, data.reshape(x1g_ori.shape),cmap = 'hsv',alpha = 0.9)
    #surf = ax.plot_wireframe(x1g_ori, x2g_ori, data.reshape(x1g_ori.shape), cmap='bwr', alpha=0.9)
    #ax.contour(x1g_ori, x2g_ori, data.reshape(x1g_ori.shape), extend3d=True, cmap=cm.coolwarm)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_zlabel('f(x)', fontsize=18)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_superimposed(data,image,name,zmin,zmax):

    matplotlib.rcParams.update({'font.size': 18})

    my_cmap = cm.bwr#.reversed()
    my_cmap.set_under('k', alpha=0)
    my_cmap.set_over('k', alpha=0)

    extent = [0,1,0,1]

    # Overlay the two images
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='viridis', extent=extent)
    clim = im.properties()['clim']
    im2 = ax.imshow(data, cmap=my_cmap,
                    interpolation='none',
                    clim=[zmin,zmax], extent=extent,alpha = 0.9)


    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    cax2 = divider.append_axes("right", size="5%", pad=0.4)

    clb = fig.colorbar(im, ax=ax, cax=cax1)
    #clb.set_ticks([])
    clb2 = fig.colorbar(im2, ax=ax, cax=cax2)
    clb.set_label('(A)  N', labelpad=8, y=1.08, rotation=0)
    ax.set_xlabel('v5 (mV)', labelpad=-17)
    ax.set_ylabel('v9 (mV)', labelpad=-75)
    xticks = ['v5_max', 'v5_min']
    yticks = ['v9_min', 'v9_max']
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_xticklabels(xticks, rotation='horizontal')
    ax.set_yticklabels(yticks, rotation='horizontal')    # ax.set_yticklabels([np.int(block_centre_voltages[-1][-1][1]),np.int(block_centre_voltages[0][0][1])],rotation='vertical')
    plt.tight_layout()
    plt.savefig('classification_figures/' + name +'.pdf', transparent=True)
    plt.show()

dict_name = 'bench_mark_nelder_mead_regime_2'
dict = load_dict(dict_name)

steps = dict['steps']
success = dict['success']
triangle_probability = dict['triangle_probability']
image_function = dict['image_function']
name = dict['name']
image = dict['image']
n = dict['n']
window_size = dict['window_size']

#plot_superimposed(triangle_probability,image,'Triangle probability',0.1,1.0)
#plot_superimposed(image_function,image, 'Optimisation function',0,2.35)
plot_superimposed(steps, image, 'regime_2_steps',0.5,30)
#plot_superimposed(success, image, 'success',0.05,1.05)

plot_3d(image_function)
plot_3d(triangle_probability)

count = 0

for index_row in tqdm(range(np.shape(success)[0])):
    for index_col in range(np.shape(success)[1]):

        if success[index_row,index_col] == 1:
            count += 1


print(count)