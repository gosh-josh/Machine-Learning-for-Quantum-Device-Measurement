#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:26:10 2020

@author: sebastian
"""

import sys

sys.path.append('../../')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

#matplotlib.rcParams['text.usetex'] = True
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import random
import math
import scipy.stats
import os
from os.path import dirname, join
current_dir = dirname(__file__)

arr = os.listdir()


def read_in_dictionarys_drl(files,current_dir):
    steps = []
    times = []
    visitmap = []
    bias_triangles = []
    windows = []
    block_centre_voltages = []
    
    
    for file in files:
        infile = open(str(current_dir)+'/'+str(file)+'.pickle','rb')
        data= pickle.load(infile)
        infile.close()
        
        steps.append(data['Number of steps'])
        times.append(data['Total tuning time (seconds)'])
        visitmap.append(data['Environment visit map'])
        bias_triangles.append(data['Bias triangle location'])
        windows.append(data['Small window measurements'])
        block_centre_voltages.append(data['Block centre voltages'])

    return steps, times, visitmap, bias_triangles,windows, block_centre_voltages

def read_full_scan(file,current_dir):
    infile = open(str(current_dir)+'/'+str(file)+'.pickle','rb')
    scan = pickle.load(infile)
    infile.close()

    image = scan['Scan data.data']

    scan_time = scan['Scan time (s)']

    return image, scan_time


def plot_hist(steps, rand_steps,times, rand_time,scan_time, log, regime):
    matplotlib.rcParams.update({'font.size': 16})
    steps_mean = np.mean(steps)
    rand_steps_mean = np.mean(rand_steps)
    time_mean = np.mean(times)
    rand_time_mean = np.mean(rand_time)
    #scan_time
    height_time = [scan_time,rand_time_mean,time_mean]
    error_time = [[0,rand_time_mean - np.percentile(rand_time,10),time_mean - np.percentile(times,10)],[0, np.percentile(rand_time,90)- rand_time_mean,np.percentile(times,90)- time_mean]]
    
    height_step= [21*21,rand_steps_mean,steps_mean]
    error_step = [[0,rand_steps_mean - np.percentile(rand_steps,10),steps_mean - np.percentile(steps,10)],[0, np.percentile(rand_steps,90)- rand_steps_mean,np.percentile(steps,90)- steps_mean]]
       
    colors = ['coral', 'goldenrod', 'lightseagreen']
    labels = ['(i)', '(ii)']
    x = np.arange(len(labels))
    width = 0.25  # the width of the bars
    
    
    fig, ax1 = plt.subplots()
    ax1.bar([0 - width,0,0+width ], height_time, width, yerr=error_time, color=colors,
            alpha=0.99, ecolor='black', capsize=10, label = ['Grid Scan', 'Random','DRL Algorithm'],log=log)
    
    ax2 = ax1.twinx()
    
    ax2.bar([1 - width,1,1+width ], height_step,width, yerr=error_step, color=colors,
            alpha=0.99, ecolor='black', capsize=10, label = ['Grid Scan', 'Random','DRL Algorithm'],log=log)
    ax2.set_ylabel('(ii) Number of measurements')
    
    ax1.set_ylabel('(i) Tuning time (s)')
    #ax1.set_title('Performance Benchmarking')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid()

    #ax2.legend(loc=1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('figures/' + regime + '_hist.pdf', transparent=True)
    plt.show()

    return

def plot_trajectory_test(visitmap, image, run_number,title,block_centre_voltages,bias_triangles):
    matplotlib.rcParams.update({'font.size': 16})
    image_trajectory = np.ones_like(image)*-2.0e-10#np.ones_like(image)*1.5e-10
    triangle = np.ones_like(image)*-2.0e-10
    
    bias_triangles = bias_triangles[run_number]
    n = 10
    z = 1.0e-9
    
    for index_row,row in enumerate(visitmap[run_number]): # 5 or 6
        for index,item in enumerate(row):
            if item == 1:
                image_trajectory[index_row*32:index_row*32+32,index*32:index*32+32] = image[index_row*32:index_row*32+32,index*32:index*32+32]
                if bias_triangles[index_row,index] == 1:
                    triangle[index_row*32-n:index_row*32,index*32-n:index*32+32+n] = z # n, 32 +2n
                    triangle[index_row*32+32:index_row*32+32+n,index*32-n:index*32+32+n] = z # n, 32 + 2n
                    triangle[index_row*32-n:index_row*32+32+n,index*32-n:index*32] = z # 32+2n, n
                    triangle[index_row*32-n:index_row*32+32+n,index*32+32:index*32+32+n] = z # 32 +2n, n  
                    
                
    my_cmap = cm.viridis.reversed()
    my_cmap.set_under('k', alpha=0)

    my_cmap2 = cm.bwr
    my_cmap2.set_under('k', alpha=0)
    
    extent = [block_centre_voltages[0][0][0],block_centre_voltages[-1][-1][0],block_centre_voltages[-1][-1][1],block_centre_voltages[0][0][1]]
    
    # Overlay the two images
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap ='viridis',extent = extent)
    clim=im.properties()['clim']
    im2 = ax.imshow(image_trajectory, cmap=my_cmap, 
              interpolation='none', 
              clim= clim,extent = extent)
    im3 = ax.imshow(triangle,cmap=my_cmap2, 
              interpolation='none', 
              clim= clim,extent = extent)

    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    cax2 = divider.append_axes("right", size="5%", pad=0.1)

    clb = fig.colorbar(im,ax = ax,cax = cax1)
    clb.set_ticks([])
    clb2 = fig.colorbar(im2,ax = ax,cax = cax2)
    clb.set_label('(A)', labelpad=40, y=1.05, rotation=0)

    ax.set_xlabel('v5 (mV)',labelpad=-17)
    ax.set_ylabel('v9 (mV)',labelpad=-55)
    
    ax.set_yticks([np.int(block_centre_voltages[-1][-1][1]),np.int(block_centre_voltages[0][0][1])])
    ax.set_xticks([block_centre_voltages[0][0][0],block_centre_voltages[-1][-1][0]])
    #ax.set_yticklabels([np.int(block_centre_voltages[-1][-1][1]),np.int(block_centre_voltages[0][0][1])],rotation='vertical')
    plt.tight_layout()
    plt.savefig('figures/'+str(title)+'_'+str(run_number)+'.pdf',transparent = True)
    plt.show()
    
        
    return

def plot_triangles(bias_triangles, windows):
    for index_t,t in enumerate(bias_triangles):
        for index1, item1 in enumerate(t):
            for index2, item2 in enumerate(item1):
                if item2 == 1:
                    plt.imshow(windows[index_t][index1][index2])
                    plt.title(str(index_t)+','+str(index1)+','+str(index2))
                    #plt.colorbar()
                    #plt.axis('off')
                    #plt.tight_layout()
                    #if index2 == 11:
                        #plt.savefig('figures/regime_1_triangle.pdf',transparent = True)
                    plt.show()
    return

def random_pixel_method(full_map,image,cvg,normalise=True): #(windows,index_t,index1,index2,cvg):
    #image = np.array(windows[index_t][index1][index2])

    # Need to add normalisation

    block_size = np.shape(image)[0]

    small_window_measurements = np.zeros((block_size,block_size))

    log_stats_mean = []
    log_stats_std = []

    stats = np.zeros(18)

    i = 0
    while any(stats == 0):
        small_window_measurements = sample_random_pixels(small_window_measurements,image)
        stats = block_splitting_statistics(full_map,small_window_measurements,normalise)
        log_stats_mean.append(stats[0])
        log_stats_std.append(stats[8])
        i+=1
    small_window_measurements = sample_random_pixels(small_window_measurements,image)
    new_stats = block_splitting_statistics(full_map,small_window_measurements,normalise)

    while sum(abs((stats - new_stats)/stats)) > cvg: #i<cvg: #
        i+=1
        small_window_measurements = sample_random_pixels(small_window_measurements,image)
        stats = new_stats
        new_stats = block_splitting_statistics(full_map,small_window_measurements,normalise)

        log_stats_mean.append(new_stats[0])
        log_stats_std.append(new_stats[9])

    small_window_statistics = block_splitting_statistics(full_map,small_window_measurements,normalise)

    real_stats = block_splitting_statistics(full_map,image,normalise)

    return small_window_measurements, small_window_statistics, log_stats_mean, log_stats_std, real_stats

def sample_random_pixels(small_window_measurements,image):
    block_size = np.shape(image)[0]
    x, y = random.randint(0, block_size - 1), random.randint(0, block_size - 1)

    point_measurement = image[x, y]

    small_window_measurements[x, y] = point_measurement

    return small_window_measurements

def block_splitting_statistics(full_map,measurement,normalise):

    measurement_size = np.shape(measurement)[0]
    n_over_2 = math.floor(measurement_size / 2.0)
    n_over_4 = math.floor(measurement_size / 4.0)
    n_3_over_4 = math.floor(3 * measurement_size / 4.0)

    # Split into blocks based:
    block_1 = measurement[0:n_over_2, 0:n_over_2]
    block_2 = measurement[0:n_over_2, n_over_2:measurement_size]
    block_3 = measurement[n_over_2:measurement_size, 0:n_over_2]
    block_4 = measurement[n_over_2:measurement_size, n_over_2:measurement_size]
    block_5 = measurement[n_over_4:n_3_over_4, n_over_4:n_3_over_4]
    block_6 = measurement[n_over_4:n_3_over_4, 0:n_over_2]
    block_7 = measurement[n_over_4:n_3_over_4, n_over_2:measurement_size]
    block_8 = measurement[0:n_over_2, n_over_4:n_3_over_4]
    block_9 = measurement[n_over_2:measurement_size, n_over_4:n_3_over_4]

    blocks = [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9]
    mean_current = []
    stds_current = []
    for block in blocks:
        data_set = []
        for row in block:
            for element in row:
                if element != 0.0:
                    data_set.append(element)
                    # print("Element",element)

        if data_set == []:
            data_set = 0

        mean_current.append(np.mean(data_set))
        stds_current.append(np.std(data_set))

    if normalise == True:
        normalised_mean, normalised_stds = np.zeros_like(mean_current), np.zeros_like(stds_current)
        for i in range(len(mean_current)):
            normalised_mean[i], normalised_stds[i] = normalise_function(full_map, mean_current[i], stds_current[i])
    else:
        normalised_mean, normalised_stds = mean_current, stds_current

    current_statistics = np.concatenate((normalised_mean, normalised_stds))

    return current_statistics  # mean_current, stds_current

def plot_normal(mean_list, std_list,name_list):

    x_min = -1.0
    x_max = 1.0

    x = np.linspace(x_min, x_max, 100)

    for index, mean in enumerate(mean_list):
        y = scipy.stats.norm.pdf(x, mean, std_list[index])

        plt.plot(x, y, label=name_list[index])

    plt.grid()

    #plt.xlim(x_min, x_max)
    #plt.ylim(0, 2.0)

    plt.title('How to plot a normal distribution in python with matplotlib', fontsize=10)

    plt.xlabel('x')
    plt.ylabel('Normal Distribution')
    plt.legend()
    plt.show()
    return

def normalise_function(full_map, mean, std):

    trace_0 = full_map[0,:]#full_map[:,0]
    trace_1 = full_map[0,:]

    '''plt.plot(trace_0)
    plt.title("Trace 0")
    plt.show()

    plt.plot(trace_1)
    plt.title("Trace 1")
    plt.show()'''

    environment_max_current = max([max(trace_0),max(trace_1)])
    environment_min_current = min([min(trace_0),min(trace_1)])

    standard_deviation_trace_0 = np.std(trace_0)
    standard_deviation_trace_1 = np.std(trace_1)

    standard_deviation_for_normalisation = max([min([standard_deviation_trace_0,standard_deviation_trace_1]),abs(standard_deviation_trace_0-standard_deviation_trace_1)])

    normalised_mean = (mean - environment_min_current)/(environment_max_current - environment_min_current)
    normalised_std = (std) / (standard_deviation_for_normalisation)

    return normalised_mean,normalised_std
