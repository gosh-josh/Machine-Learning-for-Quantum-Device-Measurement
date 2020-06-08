import sys
sys.path.append('../')
sys.path.append('../utilities')
sys.path.append('../environments')

import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from quan_T4_2d import Quantum_T4_2D

from play_episodes import play_train_episode, play_test_episode_from_location,burn_in_experience

from drl_models import Dueling_DQN_PER_2D

#File_Name_List=["rotated_T4_scan_data_res_320_win_320","rotated_T4_scan_data_res_350_win_350","rotated_T4_scan_data_res_400_win_400_sep","rotated_T4_scan_data_res_480_win_480","rotated_seb_pkl","rotated_B2_data_res_360_win_360"]
File_Name_List=['rotated_B2_data_res_360_win_360']

for file_name in File_Name_List:
    newenv=Quantum_T4_2D(file_name)
    where_is_bias_triangle = newenv.isquantum
    f, axarr = plt.subplots(newenv.dim[0], newenv.dim[1], figsize=(10, 10))
    for ii in range(newenv.dim[0]):
        for jj in range(newenv.dim[1]):
            if where_is_bias_triangle[ii, jj] == 1:
                ones = np.ones_like(newenv.image_smallpatch_data[ii][jj][4])
                ones[0,0] = 0
                #triangles.append((ii, jj))
                axarr[ii, jj].imshow(ones)
                axarr[ii, jj].axis('off')
            else:
                axarr[ii, jj].imshow(newenv.image_smallpatch_data[ii][jj][4])
                axarr[ii, jj].axis('off')
    f.show()
    plt.show()

    plt.imshow(newenv.prediction)
    plt.colorbar()
    plt.title("CNN Prediction")
    plt.show()
    plt.imshow(newenv.threshold_test)
    plt.colorbar()
    plt.title("Pre-classifier prediction")
    plt.show()


'''image = newenv.image

mid_point_x = math.floor(len(image[:,0])/2.0)
mid_point_y = math.floor(len(image[0,:])/2.0)

trace_x = image[mid_point_x,:]
trace_y = image[:,mid_point_y]

plt.plot(trace_x)
plt.plot(trace_y)

print(max(trace_y))
print(min(trace_y))

print(max(trace_x))
print(min(trace_x))

threshold_1_list = np.zeros_like(trace_x)
threshold_2_list = np.zeros_like(trace_x)

trace_range = max(trace_x) - min(trace_x)

threshold_1 = trace_range*0.3
threshold_2 = trace_range*0.0

threshold_1_list[:] = threshold_1
threshold_2_list[:] = threshold_2

plt.plot(trace_x)
plt.plot(threshold_1_list)
plt.plot(threshold_2_list)
plt.show()

heat_map = np.zeros_like(image)
for i in range(len(trace_y)):
    for j in range(len(trace_x)):
        #heat_map[i,j] = data7[i,j]
        if (image[i,j] > threshold_2 )and (image[i,j] < threshold_1 ):
            heat_map[i,j] = 1

plt.imshow(heat_map)
plt.show()

threshold_test = np.zeros_like(where_is_bias_triangle)

for i in range(len(where_is_bias_triangle[:,0])):
    for j in range(len(where_is_bias_triangle[0,:])):
        statistics = newenv.data[i][j]
        means = statistics[:9]
        for mean in means:
            if (mean > threshold_2) and (mean < threshold_1):
                threshold_test[i, j] = 1

plt.imshow(threshold_test)
plt.show()

f, axarr = plt.subplots(newenv.dim[0], newenv.dim[1], figsize=(10, 10))

for ii in range(newenv.dim[0]):
    for jj in range(newenv.dim[1]):
        if where_is_bias_triangle[ii, jj] == 1 and threshold_test[ii,jj] == 1:
            ones = np.ones_like(newenv.image_smallpatch_data[ii][jj][4])
            ones[0,0] = 0
            #triangles.append((ii, jj))
            axarr[ii, jj].imshow(ones)
            axarr[ii, jj].axis('off')
        else:
            axarr[ii, jj].imshow(newenv.image_smallpatch_data[ii][jj][4])
            axarr[ii, jj].axis('off')
f.show()
plt.show()

f, axarr = plt.subplots(newenv.dim[0], newenv.dim[1], figsize=(10, 10))

for ii in range(newenv.dim[0]):
    for jj in range(newenv.dim[1]):
        if threshold_test[ii,jj] == 1:
            ones = np.ones_like(newenv.image_smallpatch_data[ii][jj][4])
            ones[0,0] = 0
            #triangles.append((ii, jj))
            axarr[ii, jj].imshow(ones)
            axarr[ii, jj].axis('off')
        else:
            axarr[ii, jj].imshow(newenv.image_smallpatch_data[ii][jj][4])
            axarr[ii, jj].axis('off')
f.show()
plt.show()'''