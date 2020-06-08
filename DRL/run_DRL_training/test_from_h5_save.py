#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:38:24 2020

@author: sebastian
"""
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from pympler import asizeof
from tensorflow.keras import models

from tqdm import tqdm
from prioritized_experience_replay import Memory
from quan_T4_2d_new import Quantum_T4_2D

import pickle
import os
from play_episodes import play_train_episode, play_test_episode_from_location,burn_in_experience, plot_policy

from drl_models import Dueling_DQN_PER_2D


import warnings

IM_SIZE = 2 #80
N_CHANEL=9 # this is the representation of a block by 9 blocks
K = 6 #env.action_space.n
D = IM_SIZE*N_CHANEL
hidden_layer_sizes = [128,64,32]
gamma = 0.5
starting_pixel_loc_list = [[20, 340], [320, 15]]

# number of random test
batch_sz=32
count=0

#tf.reset_default_graph()

model = Dueling_DQN_PER_2D(D=D, K=K, batch_sz=batch_sz, hidden_layer_sizes=hidden_layer_sizes,
                           gamma=gamma, lr=2.3e-6, N_CHANEL=N_CHANEL, IM_SIZE=IM_SIZE, scope="DDQN")

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()

MODEL_PATH="../logs/2d/save_models/2d_mean_std"

saver.restore(sess, MODEL_PATH)
model.set_session(sess)

#filename = '/basel2/basel2_29'
def read_full_scan(file,current_dir):
    infile = open(current_dir+'/'+str(file)+'.pickle','rb')
    scan = pickle.load(infile)
    infile.close()

    image = scan['Scan data.data']

    scan_time = scan['Scan time (s)']

    return image, scan_time

filename = 'regime_1'
MaxStep = 200
image, scan_time = read_full_scan('full_scan_data_time','../run_on_device/benchmark')
#env = Quantum_T4_2D(" ",file = False, image = image, bh=32, bw=32)

#filelist = ['/test_orientation/B2','/test_orientation/B2_90','/test_orientation/B2_180','/test_orientation/B2_270']
#filelist = ['/test_orientation/B2_270']

def show_performance(filename, starting_pixel_loc):
    n = 1000
    
    env = Quantum_T4_2D(" ",starting_pixel_loc = starting_pixel_loc,file = False, image = image, bh=32, bw=32)
    newenv = Quantum_T4_2D(" ",starting_pixel_loc = starting_pixel_loc,file = False, image = image, bh=32, bw=32)

    
    dimension = env.dim[0]
    points = np.random.randint(0,dimension,[n,2])
    
    MaxStep = 300

    reward = []
    step = []
    rand_reward = []
    rand_step = []

    for i in range(n):
        
        env.reset_at_loc(points[i])
    
        epsilon = 0.0

        episode_reward, num_steps_in_episode, env.visit_map, loc_state_list, null, position_list_x, position_list_y = play_test_episode_from_location(env, model, epsilon,MaxStep=MaxStep)
        reward.append(episode_reward)
        step.append(num_steps_in_episode)

    for i in range(n):
    
        newenv.reset_at_loc(points[i])
                                
        epsilon = 1.1

        episode_reward_rand, num_steps_in_episode_rand, newenv.visit_map, loc_state_list, null, position_list_x, position_list_y = play_test_episode_from_location(newenv, model, epsilon,MaxStep=MaxStep)
        rand_reward.append(episode_reward_rand)
        rand_step.append(num_steps_in_episode_rand)

    step_mean = np.mean(step)
    step_std = np.std(step)

    rand_step_mean = np.mean(rand_step)
    rand_step_std = np.std(rand_step)
    
    plt.clf()
    plt.hist([step,rand_step],np.linspace(0,MaxStep),color=['r','b'],label=['Algorithm','Random'],alpha=0.6)
    plt.title("Steps "+filename)
    plt.xlabel('Number of steps')
    plt.ylabel('Frequency')
    plt.legend()
    #plt.savefig('fig/offline/basel2_hist_29_starting position' + str(starting_pixel_loc[0]) + '_' + str(
    #    starting_pixel_loc[0]),transparent=True)
    plt.show()
    
    plt.clf()
    height = [env.dim[0]*env.dim[1],rand_step_mean,step_mean]
    error = [[0,rand_step_mean - np.percentile(rand_step,10),step_mean - np.percentile(step,10)],[0, np.percentile(rand_step,90)- rand_step_mean,np.percentile(step,90)- step_mean]]
    plt.bar(['Grid Scan','Random','DRL Algorithm'],height, yerr=error,color = ['coral','goldenrod','lightseagreen'],alpha=0.7, ecolor='black', capsize=10)
    #plt.title('Benchmarking')
    plt.title('Benchmark_random_step')
    plt.ylabel('Number of steps')
    #plt.savefig('fig/offline/basel2_29_starting position' + str(starting_pixel_loc[0]) + '_' + str(
    #    starting_pixel_loc[0]),transparent=True)
    plt.show()

    '''plt.clf()
    plt.imshow(env.image)
    plt.title(filename)
    plt.show()'''

    return step, rand_step, step_mean,step_std, rand_step_mean,rand_step_std

step, rand_step, step_mean,step_std, rand_step_mean,rand_step_std = show_performance(filename,[0,0])

#np.save('/fig/offline/Benchmarks_offline_10000samples_drl.npy',drl)
#np.save('/fig/offline/Benchmarks_offline_10000samples_random_agent.npy',random_agent)

#drl_zero,drl_zero_std,random_zero,random_zero_std = show_performance(filename,[0,0])
#drl_half,drl_half_std,random_half,random_half_std = show_performance(filename,[175,175])
#drl_end,drl_end_std,random_end,random_end_std = show_performance(filename,[310,310])

#height = [drl_zero,random_zero,drl_half,random_half,drl_end,random_end]
#error = [drl_zero_std,random_zero_std,drl_half_std,random_half_std,drl_end_std,random_end_std]
#plt.bar(['drl_zero','random_zero','drl_half','random_half','drl_end','random_end'],height,yerr=error,color = ['coral','lightseagreen','coral','lightseagreen','coral','lightseagreen'],alpha=0.7, ecolor='black', capsize=10)
#plt.title('Benchmarking')#
#plt.title('Steps ' + filename)
#plt.ylabel('Number of steps')
#plt.savefig('Benchmark_random_step_basel2_29',transparent=True)
#plt.show()

model.session.close()


'''n_iter = 100

epsilon_list = np.linspace(0,1,3)
reward_mean = np.zeros_like(epsilon_list)
step_mean = np.zeros_like(epsilon_list)
for index,epsilon in enumerate(epsilon_list):
    rewards = [] 
    steps = []
    memory_of_model = np.zeros(n_iter)
    memory_of_env = np.zeros(n_iter)
    for i in range(n_iter):
        newenv.reset_at_rand_loc()
        episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(newenv, model, epsilon,MaxStep=100)
        rewards.append(episode_reward)
        steps.append(num_steps_in_episode)
        memory_of_model[i] = asizeof.asizeof(model)
        memory_of_env[i] = asizeof.asizeof(newenv)
    reward_mean[index] = np.mean(rewards)
    step_mean[index] = np.mean(steps)
    print('Epsilon',epsilon)
    print('Mean reward',reward_mean[index])
    print('Mean step count',step_mean[index])

    plt.plot(memory_of_model)
    plt.title("Model memory")
    plt.show()
    plt.plot(memory_of_env)
    plt.title("Model env")
    plt.show()

    
plt.clf()
plt.plot(epsilon_list,reward_mean)
plt.xlabel('Epsilon')
plt.ylabel('Mean Reward (1000 randomly initiated)')
plt.title('Performance on same device')
plt.show()

plt.clf()
plt.plot(epsilon_list,step_mean)
plt.xlabel('Epsilon')
plt.ylabel('Mean number of steps (1000 randomly initiated)')
plt.title('Performance on same device')
plt.show()

newenv = Quantum_T4_2D('rotated_B2_data_res_360_win_360')
n_iter = 1000

epsilon_list = np.linspace(0, 1, 15)
reward_mean = np.zeros_like(epsilon_list)
step_mean = np.zeros_like(epsilon_list)
for index, epsilon in enumerate(epsilon_list):
    rewards = []
    steps = []
    for i in range(n_iter):
        newenv.reset_at_rand_loc()
        episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(
            newenv, model, epsilon, MaxStep=100)
        rewards.append(episode_reward)
        steps.append(num_steps_in_episode)
        memory_of_env[i] = asizeof.asizeof(env)
        memory_of_model[i] = asizeof.asizeof(model)
        print("Memory size of env' = " + str(memory_of_env[i]) + " bytes")
        print("Memory size of model' = " + str(memory_of_model[i]) + " bytes")
    reward_mean[index] = np.mean(rewards)
    step_mean[index] = np.mean(steps)
    print('Epsilon', epsilon)
    print('Mean reward', reward_mean[index])
    print('Mean step count', step_mean[index])

plt.clf()
plt.plot(epsilon_list, reward_mean)
plt.xlabel('Epsilon')
plt.ylabel('Mean Reward (1000 randomly initiated)')
plt.title('Performance on new device')
plt.savefig("performance_reward.pdf",box_inches="tight")
plt.show()

plt.clf()
plt.plot(epsilon_list, step_mean)
plt.xlabel('Epsilon')
plt.ylabel('Mean number of steps (1000 randomly initiated)')
plt.title('Performance on new device')
plt.savefig("performance_steps.pdf",box_inches="tight")
plt.show()'''

#-------------------------------------------------------------------------------------------------------------------------

#data = np.load('/home/sebastian/PycharmProjects/Vu/Vu_algorithm/DPhil/release_drl_quantum_env/data/seb_pkl.p',allow_pickle=True)
#plt.imshow(data)
'''
f, axarr  = plt.subplots(newenv.dim[0],newenv.dim[1],figsize=(10,10))
for ii in range(newenv.dim[0]):
    for jj in range(newenv.dim[1]):
        axarr[ii,jj].imshow(newenv.image_smallpatch_data[ii][jj][4])
        axarr[ii,jj].axis('off')
f.show()
plt.show()
plt.clf()
'''

'''epsilon = 0.0
episode_reward, num_steps_in_episode, newenv.visit_map, loc_state_list, newenv, position_list_x, position_list_y = play_test_episode_from_location(newenv, model, epsilon,MaxStep=200)
print('Reward', episode_reward)
print('Number of steps', num_steps_in_episode)
plt.imshow(newenv.visit_map)
plt.show()'''

#print(newenv.current_pos)
#print(episode_reward)
#print(num_steps_in_episode)
'''map = newenv.visit_map
x,y  = np.shape(map)
for i in range(x):
    print(map[i,:])'''


'''f, axarr = plt.subplots(newenv.dim[0], newenv.dim[1], figsize=(10, 10))
# -----------------------------------------------------------------------------------
# Testing environment
for ii in range(newenv.dim[0]):
    for jj in range(newenv.dim[1]):
        ones = np.ones_like(newenv.image_smallpatch_data[ii][jj][4])
        ones[0,0] = 0
        axarr[ii, jj].imshow(ones,cmap = 'gist_gray')
        axarr[ii, jj].axis('off')

for ii in range(len(position_list_x)):
    for jj in range(len(position_list_y)):
        if (ii == jj):
            axarr[position_list_x[ii], position_list_y[jj]].imshow(newenv.image_smallpatch_data[position_list_x[ii]][position_list_y[jj]][4])
            axarr[position_list_x[ii], position_list_y[jj]].axis('off')

for ii in range(newenv.dim[0]):
    for jj in range(newenv.dim[1]):
        if where_is_bias_triangle[ii, jj] == 1:
            ones = np.ones_like(newenv.image_smallpatch_data[ii][jj][4])
            ones[0,0] = 0
            triangles.append((ii, jj))
            axarr[ii, jj].imshow(ones)
            axarr[ii, jj].axis('off')

f.show()
plt.show()
plt.clf()

print("Done ALL ---------------------------------------------------")

print('Reward', episode_reward)
print('Number of steps', num_steps_in_episode)

f, axarr = plt.subplots(newenv.dim[0], newenv.dim[1], figsize=(10, 10))
# -----------------------------------------------------------------------------------
# Testing environment
for ii in range(newenv.dim[0]):
    for jj in range(newenv.dim[1]):
        axarr[ii, jj].imshow(newenv.image_smallpatch_data[ii][jj][4])
        axarr[ii, jj].axis('off')

f.show()
plt.show()
plt.clf()

# -----------------------------------------------------------------------------------
# Testing environment

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


print(triangles)'''

'''
for i in triangles:
    myrow,mycol = i
    fig=plt.figure()
    plt.imshow(newenv.image_smallpatch_data[myrow][mycol][4])
    plt.show()
    plt.clf()
'''
