#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:26:10 2020

@author: sebastian
"""
import sys

sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle 
from data_analysis import read_in_dictionarys_drl, read_full_scan, plot_hist, plot_triangles, plot_trajectory_test, random_pixel_method, plot_normal
import matplotlib.cm as cm
from os.path import dirname
current_dir = dirname(__file__)

infile = open('epsilon_0_starting_position_9_12.pickle','rb')
data_1_eps_1 = pickle.load(infile)
infile.close()

for key in data_1_eps_1:
    print(key)
    #print(data_1_eps_1[key])
    print()
t_list = data_1_eps_1['Timed step list (s)']
    

files_eps_1 = ['epsilon_1_starting_position_9_12', 

'epsilon_1_starting_position_1_11', 

'epsilon_1_starting_position_9_9', 

'epsilon_1_starting_position_19_13', 

'epsilon_1_starting_position_0_2', 

'epsilon_1_starting_position_20_19', 

'epsilon_1_starting_position_5_6', 

'epsilon_1_starting_position_16_18', 

'epsilon_1_starting_position_17_19', 

'epsilon_1_starting_position_16_17']

rand_steps, rand_times, rand_visitmap, rand_bias_triangles,rand_windows, rand_block_centre_voltages = read_in_dictionarys_drl(files_eps_1,current_dir)


files_eps_0 = ['epsilon_0_starting_position_9_12',

'epsilon_0_starting_position_1_11', 

'epsilon_0_starting_position_9_9', 

'epsilon_0_starting_position_19_13', 

'epsilon_0_starting_position_0_2', 

'epsilon_0_starting_position_20_19', 

'epsilon_0_starting_position_5_6', 

'epsilon_0_starting_position_16_18', 

'epsilon_0_starting_position_17_19', 

'epsilon_0_starting_position_16_17']

steps, times, visitmap, bias_triangles,windows, block_centre_voltages = read_in_dictionarys_drl(files_eps_0,current_dir)

image, scan_time = read_full_scan('full_scan_data_time',current_dir)

plot_hist(steps, rand_steps, times, rand_times,scan_time, log = False, regime = 'regime_1')

extent = np.array([block_centre_voltages[0][0][0][0],block_centre_voltages[0][-1][-1][0],block_centre_voltages[0][-1][-1][1],block_centre_voltages[0][0][0][1]])

#extent = (-10,20,-30,40)

plt.imshow(image,extent=extent)
plt.title('Regime 1')
plt.colorbar()
plt.show()

plot_trajectory_test(visitmap,image,0,'regime_1_trajectory',block_centre_voltages[0],bias_triangles)
plot_trajectory_test(visitmap,image,1,'regime_1_trajectory',block_centre_voltages[0],bias_triangles)
plot_trajectory_test(visitmap,image,5,'regime_1_trajectory',block_centre_voltages[0],bias_triangles)
plot_trajectory_test(visitmap,image,6,'regime_1_trajectory',block_centre_voltages[0],bias_triangles)
plot_trajectory_test(visitmap,image,8,'regime_1_trajectory',block_centre_voltages[0],bias_triangles)

#print(block_centre_voltages)
    
#for i in range(10):
#    plot_trajectory_test(visitmap,image,i,'regime_1_trajectory',block_centre_voltages[0],bias_triangles)

#plot_triangles(bias_triangles, windows)

cvg = 0.01#100
#1 low
#2 high
#3 double dot
#4 single dot
pixels1,stats1, log_stats_mean_1, log_stats_std_1, real_stats_1 = random_pixel_method(np.array(image),np.array(image[0:32,-32:]) ,cvg,normalise=True)
pixels2,stats2, log_stats_mean_2, log_stats_std_2, real_stats_2 = random_pixel_method(np.array(image),np.array(image[68:100,68:100]) ,cvg,normalise=True)
pixels4,stats4, log_stats_mean_4, log_stats_std_4, real_stats_4 = random_pixel_method(np.array(image),np.array(windows[7][3][13]) ,cvg,normalise=False)
pixels3,stats3, log_stats_mean_3, log_stats_std_3, real_stats_3 = random_pixel_method(np.array(image),np.array(windows[8][10][11]) ,cvg,normalise=False)



plt.imshow(np.array(windows[7][3][13]))
plt.show()

plt.imshow(pixels4)

my_cmap = cm.viridis
my_cmap.set_under('k', alpha=0)

extent = [block_centre_voltages[0][0][0][0], block_centre_voltages[0][-1][-1][0], block_centre_voltages[0][-1][-1][1],
          block_centre_voltages[0][0][0][1]]

# Overlay the two images
fig, ax = plt.subplots()
im = ax.imshow(np.ones_like(pixels4)*-0.1, cmap='binary', extent=extent)

im2 = ax.imshow(pixels4, cmap=my_cmap,
                interpolation='none',
                clim=1.0e-11, extent=extent)
ax.set_yticks([])
ax.set_xticks([])

plt.tight_layout()
#plt.axis('off')
plt.savefig('figures/pixel_sampling_regime_1_single_dot.pdf', transparent=True)
plt.show()

'''mean_list = stats4[:9]
std_list = stats4[9:]
name_list = ['1','2','3','4','5','6','7','8','9']#['0.5','0.1','0.01','0.001']
plot_normal(mean_list,std_list,name_list)

mean_list = stats3[:9]
std_list = stats3[9:]
name_list = ['1','2','3','4','5','6','7','8','9']#['0.5','0.1','0.01','0.001']
plot_normal(mean_list,std_list,name_list)
'''

plt.plot(np.ones(50)*0.05,'black')
plt.plot(np.ones(50)*0.3,'black')
plt.fill_between(np.arange(0,50),np.ones(50)*0.05,np.ones(50)*0.3,color='black',alpha=0.2, label = 'Passes pre-classification')

plt.plot(np.ones_like(log_stats_mean_1)*real_stats_1[0],'b--')
plt.plot(np.ones_like(log_stats_mean_2)*real_stats_2[0],'y--')
plt.plot(np.ones_like(log_stats_mean_3)*real_stats_3[0],'g--')
plt.plot(np.ones_like(log_stats_mean_4)*real_stats_4[0],'r--')

plt.plot(log_stats_mean_1,'b',label='Low current')
plt.plot(log_stats_mean_2,'y',label = 'High current')
plt.plot(log_stats_mean_3,'g',label='Double dot')
plt.plot(log_stats_mean_4,'r',label = 'Single dot')
plt.ylim(-0.05,1.0)
plt.grid()
plt.legend()
plt.xlabel('Pixels measured')
plt.ylabel('Normalised current')
plt.show()