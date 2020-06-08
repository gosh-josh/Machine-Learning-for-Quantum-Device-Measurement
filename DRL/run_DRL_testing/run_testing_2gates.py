import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')
from utility_plot_arrow import plot_arrow_to_file

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
from matplotlib import cm
from tensorflow.keras import models

import random
from tqdm import tqdm
from prioritized_experience_replay import Memory
from quan_T4_2d_new import Quantum_T4_2D

import pickle
import os
from play_episodes import play_train_episode, play_test_episode_from_location, burn_in_experience

from drl_models import Dueling_DQN_PER_2D

import warnings

IM_SIZE = 2  # 80
N_CHANEL = 9  # this is the representation of a block by 9 blocks
K = 6  # env.action_space.n
D = IM_SIZE * N_CHANEL
hidden_layer_sizes = [128, 64, 32]
gamma = 0.5
starting_pixel_loc_list = [[20, 340], [320, 15]]

# number of random test
batch_sz = 32
count = 0

model = Dueling_DQN_PER_2D(D=D, K=K, batch_sz=batch_sz, hidden_layer_sizes=hidden_layer_sizes,
                           gamma=gamma, lr=2.3e-6, N_CHANEL=N_CHANEL, IM_SIZE=IM_SIZE, scope="DDQN")

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()

MODEL_PATH="../logs/2d/save_models/2d_mean_std"

saver.restore(sess, MODEL_PATH)
model.set_session(sess)

#newenv = Quantum_T4_2D('rotated_T4_scan_data_res_350_win_350')
#newenv = Quantum_T4_2D('rotated_B2_data_res_360_win_360')
#newenv = Quantum_T4_2D('florians_rotated')

def read_full_scan(file,current_dir):
    infile = open(current_dir+'/'+str(file)+'.pickle','rb')
    scan = pickle.load(infile)
    infile.close()

    image = scan['Scan data.data']

    scan_time = scan['Scan time (s)']

    return scan# image, scan_time

name = 'regime_2'
MaxStep = 200
scan= read_full_scan('regime_2_full_scan_data_time','../run_on_device/benchmark')
image = scan['Scan data.data'].data[0]

#image, scan_time = read_full_scan('full_scan_data_time','../run_on_device/benchmark')
newenv = Quantum_T4_2D(" ",file = False, image = image, bh=32, bw=32)
# -------------------------------------------------------------------------------------------------------------------------

steps = np.zeros_like(newenv.image)
n = newenv.bw
for index_row in tqdm(range(np.int(newenv.dim[0]))):
    for index_col in range(np.int(newenv.dim[1])):
        newenv.reset_at_loc([index_row,index_col])
        episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(
            newenv, model, 0.0, MaxStep=MaxStep)
        steps[n * index_row - np.int(n / 2):n * index_row + np.int(n / 2),
                n * index_col - np.int(n / 2):n * index_col + np.int(n / 2)] = num_steps_in_episode

rand_steps = np.zeros_like(newenv.image)
n = newenv.bw
for index_row in tqdm(range(np.int(newenv.dim[0]))):
    for index_col in range(np.int(newenv.dim[1])):
        newenv.reset_at_loc([index_row,index_col])
        episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(
            newenv, model, 1.0, MaxStep=MaxStep)
        rand_steps[n * index_row - np.int(n / 2):n * index_row + np.int(n / 2),
                n * index_col - np.int(n / 2):n * index_col + np.int(n / 2)] = num_steps_in_episode


step = np.zeros_like(newenv.image)
for index_row in tqdm(range(np.shape(step)[0])):
    for index_col in range(np.shape(step)[1]):
        i = np.int(index_row/newenv.bw)
        j = np.int(index_col / newenv.bh)
        newenv.reset_at_loc([i,j])
        episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(
            newenv, model, 0.0, MaxStep=MaxStep)
        step[index_row,index_col] =  num_steps_in_episode


rand_step = np.zeros_like(newenv.image)
for index_row in tqdm(range(np.shape(step)[0])):
    for index_col in range(np.shape(step)[1]):
        i = np.int(index_row/newenv.bw)
        j = np.int(index_col / newenv.bh)
        newenv.reset_at_loc([i,j])
        episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(
            newenv, model, 1.0, MaxStep=MaxStep)
        rand_step[index_row,index_col] =  num_steps_in_episode


m = 16
steps2 = steps[m:-m,m:-m]
step = step.flatten()
rand_step = rand_step.flatten()

step_mean = np.mean(step)
step_std = np.std(step)

rand_step_mean = np.mean(rand_step)
rand_step_std = np.std(rand_step)

plt.clf()
plt.hist([step, rand_step], np.linspace(0, 150), color=['r', 'b'], label=['Algorithm', 'Random'], alpha=0.6)
plt.title("Steps Regime 1")
plt.xlabel('Number of steps')
plt.ylabel('Frequency')
plt.legend()
# plt.savefig('fig/offline/basel2_hist_29_starting position' + str(starting_pixel_loc[0]) + '_' + str(
#    starting_pixel_loc[0]),transparent=True)
plt.show()

plt.clf()
height = [newenv.dim[0] * newenv.dim[1], rand_step_mean, step_mean]
error = [[0, rand_step_mean - np.percentile(rand_step, 10), step_mean - np.percentile(step, 10)],
         [0, np.percentile(rand_step, 90) - rand_step_mean, np.percentile(step, 90) - step_mean]]
plt.bar(['Grid Scan', 'Random', 'DRL Algorithm'], height, yerr=error, color=['coral', 'goldenrod', 'lightseagreen'],
        alpha=0.7, ecolor='black', capsize=10)
# plt.title('Benchmarking')
plt.title('Benchmark_random_step')
plt.ylabel('Number of steps')
# plt.savefig('fig/offline/basel2_29_starting position' + str(starting_pixel_loc[0]) + '_' + str(
#    starting_pixel_loc[0]),transparent=True)
plt.show()

step = step.reshape([672,672])
rand_step = rand_step.reshape([672,672])
my_cmap = cm.bwr  # .reversed()
n = 25
extent = [0, 1, 0, 1]
# Overlay the two images
fig, ax = plt.subplots()
im = ax.imshow(rand_step, cmap='bwr', extent=extent,norm=DivergingNorm(n))
clim = im.properties()['clim']
im2 = ax.imshow(rand_step, cmap=my_cmap,
                interpolation='none',norm=DivergingNorm(n),
                clim=clim, extent=extent)
clb = fig.colorbar(im, ax=ax)
clb.set_label('Steps', labelpad=-42, y=1.04, rotation=0)
plt.xlabel('V5 (mV)')
plt.ylabel('V9 (mV)')
xticks = ['v5_max', 'v5_min']
yticks = ['v9_min', 'v9_max']
ax.set_xticks(np.arange(len(xticks)))
ax.set_yticks(np.arange(len(yticks)))
ax.set_xticklabels(xticks, rotation='horizontal')
ax.set_yticklabels(yticks, rotation='horizontal')
ax.yaxis.set_label_coords(-0.025, 0.5)
ax.xaxis.set_label_coords(0.5, -0.025)
plt.show()

data = {
    'steps':step,
    'rand_steps':rand_step,
    'image':image,
    'name':name
}

savefilename = 'benchmark_drl_'+name
pickle_out = open(savefilename+".pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()

model.session.close()