#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:22:51 2020

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
import matplotlib.cm as cm
from tensorflow.keras import models

from tqdm import tqdm
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

def plot_arrow_action(myactions, image,bw,bh):
    
    actions_x = np.zeros_like(myactions)
    actions_y = np.zeros_like(myactions)

    X = np.zeros_like(myactions)
    Y= np.zeros_like(myactions)
    
    C = [0] * np.shape(X)[0]
    
    for i in range(len(C)):
        C[i] = [0]* np.shape(X)[1]

    colours = [['c1'],['c2'],['c3'],['c4'],['c5'],['c6']]
    
    for i in range(np.shape(myactions)[0]):
        for j in range(np.shape(myactions)[1]):
            
            X[i,j] = i*bh
            Y[i,j] = j*bw
            
            if myactions[i,j] == 4: # down and right
                actions_x[i,j] = 1
                actions_y[i,j] = -1
                C[i][j] = colours[0]
            
            if myactions[i,j] == 5: # up and left
                actions_x[i,j] = -1
                actions_y[i,j] = 1
                C[i][j] = colours[3]
            
            if myactions[i,j] == 2: # right 
                actions_x[i,j] = 1
                actions_y[i,j] = 0
                C[i][j] = colours[1]
                
            if myactions[i,j] == 3: # left 
                actions_x[i,j] = -1
                actions_y[i,j] = 0
                C[i][j] = colours[5]
            
            if myactions[i,j] == 0: # down
                actions_x[i,j] = 0
                actions_y[i,j] = -1
                C[i][j] = colours[5]
                
            if myactions[i,j] == 1: # up
                actions_x[i,j] = 0
                actions_y[i,j] = 1
                C[i][j] = colours[2]
            
    return actions_x, actions_y, X, Y, C

filename = ''

file = 'full_scan_data_time'

infile = open('../run_on_device/benchmark/'+str(file)+'.pickle','rb')
scan = pickle.load(infile)
infile.close()

image = scan['Scan data.data']

plt.imshow(image)
plt.show()

env = Quantum_T4_2D(filename,image = image, file = False, starting_pixel_loc = [600,600],bh = 32, bw = 32)
#filename = 'florians_rotated'
#env = Quantum_T4_2D(filename, starting_pixel_loc = [600,600])

action,state = plot_policy(env,model)
#print(action)

plt.imshow(env.image, alpha = 0.7)
plt.colorbar()
plt.title('State')
#plt.show()

#plt.imshow(state[1:,:], alpha = 0.4)
#plt.colorbar()
#plt.title('State')
#plt.show()

#plt.imshow(action, alpha = 0.6,cmap = 'bwr')
#plt.colorbar()
#plt.title('Action')
#plt.show()
bw = env.bw
bh = env.bh
actions_x, actions_y, X, Y,C = plot_arrow_action(action,env.image,bw,bh)


#plt.quiver(Y[1:,:],X[1:,:],actions_x[1:,:], actions_y[1:,:])
plt.quiver(Y,X,actions_x, actions_y)

plt.show()
        

epsilon =0.0
MaxStep = 100   
episode_reward, num_steps_in_episode, env.visit_map, loc_state_list, null, position_list_x, position_list_y = play_test_episode_from_location(env, model, epsilon,MaxStep=MaxStep)

image = np.ones_like(env.image) * -2.0e-10

X_2 = np.zeros_like(X)
Y_2 = np.zeros_like(Y)
actions_x_2 = np.zeros_like(actions_x)
actions_y_2 = np.zeros_like(actions_y)

X_2 = []
Y_2 = []
actions_x_2 = []
actions_y_2 = []


def plot_path_arrows(env,image):
    for index_row,row in enumerate(env.visit_map):
        for index,item in enumerate(row):
            if item == 1:
                image[index_row*env.bw:index_row*env.bw+env.bw,index*env.bw:index*env.bw+env.bh] = env.image_smallpatch_data[index_row][index][4][:env.bw,:env.bh]
                X_2.append( X[index_row,index]+env.bw)
                Y_2.append(Y[index_row,index]+env.bw)
                actions_x_2.append( actions_x[index_row,index])
                actions_y_2.append(actions_y[index_row,index])

    my_cmap = cm.viridis.reversed()
    my_cmap.set_under('k', alpha=0)

    # Overlay the two images
    fig, ax = plt.subplots()

    im = ax.imshow(env.image, cmap='viridis')
    clim = im.properties()['clim']
    im2 = ax.imshow(image, cmap=my_cmap,
                    interpolation='none',
                    clim=clim)
    fig.colorbar(im, ax=ax)
    fig.colorbar(im2, ax=ax)
    # plt.title(run_number)
    #plt.savefig('figures/' + str(title) + '_' + str(run_number) + '.pdf', transparent=True)
    #plt.show()

   # plt.imshow(image,cmap = 'binary',alpha = 0.9)
    #plt.colorbar()
    
    plt.quiver(Y_2,X_2,actions_x_2, actions_y_2,color='black')
    
    #plt.imshow(env.image, alpha = 0.4)
    
    plt.show()


plot_path_arrows(env,image)