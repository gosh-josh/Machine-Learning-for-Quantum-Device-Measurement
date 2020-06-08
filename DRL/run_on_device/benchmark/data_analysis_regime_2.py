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

from data_analysis import read_in_dictionarys_drl, read_full_scan, plot_hist, plot_triangles, plot_trajectory_test

files_eps_1 = ['regime2_epsilon_1_starting_position_17_9',

'regime2_epsilon_1_starting_position_12_11',

'regime2_epsilon_1_starting_position_20_19',

'regime2_epsilon_1_starting_position_12_4',

'regime2_epsilon_1_starting_position_3_20',

'regime2_epsilon_1_starting_position_4_12',

'regime2_epsilon_1_starting_position_19_17',

'regime2_epsilon_1_starting_position_2_18',

'regime2_epsilon_1_starting_position_2_12',

'regime2_epsilon_1_starting_position_5_14']

rand_steps, rand_times, rand_visitmap, rand_bias_triangles,rand_windows, rand_block_centre_voltages = read_in_dictionarys_drl(files_eps_1,current_dir)


files_eps_0 = ['regime2_epsilon_0_starting_position_17_9',

'regime2_epsilon_0_starting_position_12_11',

'regime2_epsilon_0_starting_position_20_19',

'regime2_epsilon_0_starting_position_12_4',

'regime2_epsilon_0_starting_position_3_20',

'regime2_epsilon_0_starting_position_4_12',

'regime2_epsilon_0_starting_position_19_17',

'regime2_epsilon_0_starting_position_2_18',

'regime2_epsilon_0_starting_position_2_12',

'regime2_epsilon_0_starting_position_5_14']

steps, times, visitmap, bias_triangles,windows, block_centre_voltages = read_in_dictionarys_drl(files_eps_0,current_dir)

image, scan_time = read_full_scan('regime_2_full_scan_data_time',current_dir)

plot_hist(steps, rand_steps, times, rand_times,scan_time, log = False, regime = 'regime_2')

plt.imshow(image.data[0])
plt.colorbar()
plt.show()

#plot_trajectory_test(visitmap,image.data[0],0,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)
#plot_trajectory_test(visitmap,image.data[0],1,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)
#plot_trajectory_test(visitmap,image.data[0],2,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)
#plot_trajectory_test(visitmap,image.data[0],3,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)
#plot_trajectory_test(visitmap,image.data[0],5,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)
#plot_trajectory_test(visitmap,image.data[0],6,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)

    
'''for i in range(10):
    plot_trajectory_test(visitmap,image.data[0],i,'regime_2_trajectory',block_centre_voltages[0],bias_triangles)'''
    
#plot_triangles(bias_triangles, windows)