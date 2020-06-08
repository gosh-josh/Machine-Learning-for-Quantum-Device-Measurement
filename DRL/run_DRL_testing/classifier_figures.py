import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../utilities')
sys.path.append('../environments')
sys.path.append('../data')
sys.path.append('../utilities/pygor')


import numpy as np
import pickle
import matplotlib
import tensorflow as tf

import matplotlib.pyplot as plt
from quan_T4_2d_new import Quantum_T4_2D
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 16})

def plot_image(env,File_name,image):

    actions_x = np.zeros_like(env.image)
    actions_y = np.zeros_like(env.image)

    X = np.zeros_like(env.image)
    Y = np.zeros_like(env.image)

    X_2 = []
    Y_2 = []
    actions_x_2 = []
    actions_y_2 = []



    new_image = np.ones_like(env.image)*-1
    n = 2
    z = 1.0e-11
    z2 = 2.5e-10

    q = 1.0
    m = 10

    new_image[m * env.bw - n: (m + 1) * env.bw + n, m * env.bh:(m + 1) * env.bh] = z

    X_2.append(0.5)
    Y_2.append(0.5)
    actions_x_2.append(-q)
    actions_y_2.append(q)

    X_2.append(0.5)
    Y_2.append(0.5)
    actions_x_2.append(q)
    actions_y_2.append(-q)

    X_2.append(0.5)
    Y_2.append(0.5)
    actions_x_2.append(-q)
    actions_y_2.append(0.0)

    X_2.append(0.5)
    Y_2.append(0.5)
    actions_x_2.append(q)
    actions_y_2.append(0.0)

    X_2.append(0.5)
    Y_2.append(0.5)
    actions_x_2.append(0.0)
    actions_y_2.append(-q)

    X_2.append(0.5)
    Y_2.append(0.5)
    actions_x_2.append(0.0)
    actions_y_2.append(q)

    new_image[(m + 1) * env.bw - n: (m + 2) * env.bw + n, m * env.bh:(m + 1) * env.bh] = z2
    new_image[(m - 1) * env.bw - n: (m) * env.bw + n, m * env.bh:(m + 1) * env.bh] = z2

    new_image[m * env.bw - n: (m + 1) * env.bw + n, (m + 1) * env.bh:(m + 2) * env.bh] = z2
    new_image[m * env.bw - n: (m + 1) * env.bw + n, (m - 1) * env.bh:(m) * env.bh] = z2

    new_image[(m - 1) * env.bw - n: (m) * env.bw + n, (m - 1) * env.bh:(m) * env.bh] = z2
    new_image[(m + 1) * env.bw - n: (m + 2) * env.bw + n, (m + 1) * env.bh:(m + 2) * env.bh] = z2

    for index_row, row in enumerate(env.threshold_test):
        for index_col, item in enumerate(row):

            new_image[index_row*env.bw-n: index_row*env.bw+n,index_col*env.bh:(index_col+1)*env.bh] = z
            new_image[index_row*env.bw: (index_row+1)*env.bw,index_col*env.bh-n:index_col*env.bh+n] = z

    my_cmap = cm.Oranges
    my_cmap.set_under('k', alpha=0)

    extent = [0, 1, 0, 1]

    # Overlay the two images
    fig, ax = plt.subplots()

    im = ax.imshow(image, cmap='viridis', extent=extent)
    clim = im.properties()['clim']

    im2 = ax.imshow(new_image, cmap=my_cmap,
                    interpolation='none',
                    clim=clim, extent=extent,alpha = 0.7)
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)

    #clb = fig.colorbar(im2, ax=ax, cax=cax1)
    clb = fig.colorbar(im, ax=ax, cax=cax1)
    clb.set_label('(A)', labelpad=-8, y=1.06, rotation=0)
    ax.set_xlabel('v5 (mV)', labelpad=-10)
    ax.set_ylabel('v9 (mV)', labelpad=-40)

    ax.set_yticklabels(['v9 min',' ',' ',' ',' ', 'v9 max'])
    ax.set_xticklabels(['v5 max',' ',' ',' ',' ', 'v5 min'])
    # ax.set_yticklabels([np.int(block_centre_voltages[-1][-1][1]),np.int(block_centre_voltages[0][0][1])],rotation='vertical')
    quiver = ax.quiver(Y_2, X_2, actions_x_2, actions_y_2,color = 'white')
    plt.tight_layout()
    #plt.savefig('../run_on_device/benchmark/figures/' + File_name + '_segmentation.pdf', transparent=True)

    plt.show()

def plot_fig(env,File_name,image):

    threshold = np.ones_like(env.image)*-1
    triangle = np.ones_like(env.image)*-1
    n = 2
    z = 1.0e-9

    for index_row, row in enumerate(env.threshold_test):
        for index_col, item in enumerate(row):
            if item > 0:

                threshold[index_row*env.bw:index_row*env.bw+env.bw,index_col*env.bh:index_col*env.bh+env.bh] = item  #image[index_row*env.bw:index_row*env.bw+env.bw,index_col*env.bh:index_col*env.bh+env.bh]

                if (index_row == 11 and index_col == 9):#env.isquantum[index_row, index_col] == 1: # or (index_row == 11 and index_col == 9): #

                    triangle[index_row * env.bw - n: index_row * env.bw + n,
                    index_col * env.bh-n:(index_col + 1) * env.bh+n] = z
                    triangle[(index_row+1) * env.bw - n: (index_row+1) * env.bw + n,
                    index_col * env.bh-n:(index_col + 1) * env.bh+n] = z
                    triangle[index_row * env.bw-n: (index_row + 1) * env.bw+n,
                    index_col * env.bh - n:index_col * env.bh + n] = z
                    triangle[index_row * env.bw-n: (index_row + 1) * env.bw+n,
                    (index_col+1) * env.bh - n:(index_col+1) * env.bh + n] = z

                    plt.imshow(image[index_row*env.bw:index_row*env.bw+env.bw,index_col*env.bh:index_col*env.bh+env.bh])
                    #plt.title(str(index_row) +','+ str(index_col))
                    plt.axis('off')
                    plt.show()
                    #np.save('regime_2_9_11_triangle',image[index_row*env.bw:index_row*env.bw+env.bw,index_col*env.bh:index_col*env.bh+env.bh])

    my_cmap = cm.Reds #.reversed()
    my_cmap.set_under('k', alpha=0)

    my_cmap2 = cm.binary.reversed()
    my_cmap2.set_under('k', alpha=0)

    extent = [0,1,0,1]

    # Overlay the two images
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap = 'viridis', extent=extent,alpha = 0.7)
    clim = im.properties()['clim']
    im2 = ax.imshow(threshold, cmap=my_cmap,
                    interpolation='none',clim = 0.5,
                     extent=extent,alpha = 0.9)
    '''im3 = ax.imshow(triangle, cmap=my_cmap2,
                    interpolation='none',clim = clim,
                    extent=extent)'''
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)

    clb = fig.colorbar(im2,ax = ax,cax=cax1)

    #clb.set_label('(A)', labelpad=-8, y=1.06, rotation=0)
    ax.set_xlabel('v5 (mV)',labelpad=-10)
    ax.set_ylabel('v9 (mV)',labelpad=-40)

    ax.set_yticklabels(['v9 min',' ',' ',' ',' ','v9 max'])
    ax.set_xticklabels(['v5 max', ' ',' ',' ',' ','v5 min'])
    # ax.set_yticklabels([np.int(block_centre_voltages[-1][-1][1]),np.int(block_centre_voltages[0][0][1])],rotation='vertical')
    plt.tight_layout()
    #plt.savefig('../run_on_device/benchmark/figures/'+File_name+'_pre_classifier.pdf',transparent = True)
    plt.show()


File_Name_List = ["rotated_T4_scan_data_res_320_win_320", "rotated_T4_scan_data_res_350_win_350",
                  "rotated_T4_scan_data_res_400_win_400_sep", "rotated_T4_scan_data_res_480_win_480"]

#File_Name_List = ["florians_rotated"]

'''n_env = len(File_Name_List)
env_list=[0]*n_env

#env_list[0] = Quantum_T4_2D(File_Name_List[0],isRepeat=True,offset=2.0e-10)
#plot_fig(env_list[0],File_Name_List[0])

for n in range(n_env):
    env_list[n] = Quantum_T4_2D(File_Name_List[n],isRepeat=True,offset=2.0e-10)
    plot_fig(env_list[n],File_Name_List[n])
'''

def read_full_scan(file,current_dir):
    infile = open(current_dir+'/'+str(file)+'.pickle','rb')
    scan = pickle.load(infile)
    infile.close()

    image = scan['Scan data.data']

    scan_time = scan['Scan time (s)']

    return scan# image, scan_time

name = 'regime_'
MaxStep = 200
scan= read_full_scan('regime_2_full_scan_data_time','../run_on_device/benchmark')
#scan= read_full_scan('full_scan_data_time','../run_on_device/benchmark')
image = scan['Scan data.data'].data[0]

#image, scan_time = read_full_scan('full_scan_data_time','../run_on_device/benchmark')
env = Quantum_T4_2D(" ",file = False, image = image, bh=32, bw=32)

plot_fig(env,name,image)

plot_image(env,name,image)