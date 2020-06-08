#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:25:20 2020

@author: sebastian
"""
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname
current_dir = dirname(__file__)

from data_analysis import read_in_dictionarys_drl, read_full_scan, plot_hist, plot_triangles, plot_trajectory, plot_trajectory_test

files_eps_0 = ['regime4_epsilon_0_starting_position_12_17',

 'regime4_epsilon_0_starting_position_18_14',

 'regime4_epsilon_0_starting_position_3_0',

 'regime4_epsilon_0_starting_position_11_16',

 'regime4_epsilon_0_starting_position_17_20',

 'regime4_epsilon_0_starting_position_16_18',

 'regime4_epsilon_0_starting_position_14_8',

 'regime4_epsilon_0_starting_position_4_14',

 'regime4_epsilon_0_starting_position_14_9',

 'regime4_epsilon_0_starting_position_2_8']

steps, times, visitmap, bias_triangles,windows, block_centre_voltages = read_in_dictionarys_drl(files_eps_0,current_dir)

files_eps_1 = ['regime4_epsilon_1_starting_position_12_17',

 'regime4_epsilon_1_starting_position_18_14',

 'regime4_epsilon_1_starting_position_3_0',

 'regime4_epsilon_1_starting_position_11_16',

 'regime4_epsilon_1_starting_position_17_20',

 'regime4_epsilon_1_starting_position_16_18',

 'regime4_epsilon_1_starting_position_14_8',

 'regime4_epsilon_1_starting_position_4_14',

 'regime4_epsilon_1_starting_position_14_9',

 'regime4_epsilon_1_starting_position_2_8']


rand_steps, rand_times, rand_visitmap, rand_bias_triangles,rand_windows, rand_block_centre_voltages = read_in_dictionarys_drl(files_eps_1,current_dir)

image, scan_time = read_full_scan('regime_original_3_full_scan_data_time',current_dir)

plot_hist(steps, rand_steps, times, rand_times,scan_time, log = True)

plt.imshow(image.data[0],cmap = 'bwr')
plt.colorbar()
plt.show()

#for i in range(10):
#    plot_trajectory_test(visitmap,image.data[0],i,'DRL',block_centre_voltages[0])
plot_trajectory_test(visitmap,image.data[0],6,'regime_3_trajectory',block_centre_voltages[0],bias_triangles)
        
plot_triangles(bias_triangles, windows)