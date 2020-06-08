#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:25:20 2020

@author: sebastian
"""


import sys


import numpy as np
import matplotlib.pyplot as plt
import pickle

from data_analysis import read_in_dictionarys_drl, read_full_scan, plot_hist, plot_triangles, plot_trajectory_test

from os.path import dirname
current_dir = dirname(__file__)

files_eps_1 = ['regime3_epsilon_1_starting_position_19_12',

               'regime3_epsilon_1_starting_position_18_11',

               'regime3_epsilon_1_starting_position_11_4',

               'regime3_epsilon_1_starting_position_5_10',

               'regime3_epsilon_1_starting_position_15_18',

               'regime3_epsilon_1_starting_position_20_16',

               'regime3_epsilon_1_starting_position_7_3',

               'regime3_epsilon_1_starting_position_1_6',

               'regime3_epsilon_1_starting_position_18_1',

               'regime3_epsilon_1_starting_position_19_19']

rand_steps, rand_times, rand_visitmap, rand_bias_triangles, rand_windows, rand_block_centre_voltages = read_in_dictionarys_drl(
    files_eps_1,current_dir)

files_eps_0 = ['regime3_epsilon_0_starting_position_19_12',

               'regime3_epsilon_0_starting_position_18_11',

               'regime3_epsilon_0_starting_position_11_4',

               'regime3_epsilon_0_starting_position_5_10',

               'regime3_epsilon_0_starting_position_15_18',

               'regime3_epsilon_0_starting_position_20_16',

               'regime3_epsilon_0_starting_position_7_3',

               'regime3_epsilon_0_starting_position_1_6',

               'regime3_epsilon_0_starting_position_18_1',

               'regime3_epsilon_0_starting_position_19_19']

steps, times, visitmap, bias_triangles, windows, block_centre_voltages = read_in_dictionarys_drl(files_eps_0,current_dir)

image, scan_time = read_full_scan('regime_original_3_full_scan_data_time',current_dir)

infile = open(str(current_dir)+'/'+'full_scan_regime_4' + '.pkl', 'rb')
scan = pickle.load(infile)
infile.close()


image = scan['chan0']['data']


plot_hist(steps, rand_steps, times, rand_times, scan_time, log=False)

plt.imshow(image, cmap='bwr')
plt.colorbar()
plt.show()
for i in range(10):
    plot_trajectory_test(visitmap,image,i,'regime_4_trajectory',block_centre_voltages[0],bias_triangles)

#plot_triangles(bias_triangles, windows)
