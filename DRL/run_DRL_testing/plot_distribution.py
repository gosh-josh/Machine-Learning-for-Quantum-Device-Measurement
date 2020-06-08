import sys

sys.path.append('../')
sys.path.append('../../')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib.colors import DivergingNorm
from matplotlib import cm
import numpy as np
import pickle
import Pygor


infile = open('benchmark_drl_regime_1.pickle', 'rb')
distribution = pickle.load(infile)
infile.close()

step = distribution['steps']
rand_step = distribution['rand_steps']

step = step.flatten()
rand_step = rand_step.flatten()

step_mean = np.mean(step)
step_std = np.std(step)

rand_step_mean = np.mean(rand_step)
rand_step_std = np.std(rand_step)

matplotlib.rcParams.update({'font.size': 18})

plt.clf()
plt.hist([step, rand_step], np.linspace(0, 210), color=['r', 'b'], label=['DRL decision agent', 'Random decision agent'], alpha=0.6)
#plt.title("Steps Regime 1")
plt.xlabel('Number of measurements')
plt.ylabel('Frequency')
#plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('../run_on_device/benchmark/figures/regime_2_performance_distribution.pdf')
plt.show()

step = step.reshape([672,672],order='A')
rand_step = rand_step.reshape([672,672],order='A')

matplotlib.rcParams.update({'font.size': 18})

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
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
clb = fig.colorbar(im, ax=ax,cax=cax)
clb.set_label('N', labelpad=-25, y=1.09, rotation=0)
ax.set_xlabel('v5 (mV)', labelpad=-17)
ax.set_ylabel('v9 (mV)', labelpad=-75)
xticks = ['v5_max', 'v5_min']
yticks = ['v9_min', 'v9_max']
ax.set_xticks(np.arange(len(xticks)))
ax.set_yticks(np.arange(len(yticks)))
ax.set_xticklabels(xticks, rotation='horizontal')
ax.set_yticklabels(yticks, rotation='horizontal')
plt.tight_layout()
plt.savefig('classification_figures/' + 'regime_1_random_agent' +'.pdf', transparent=True)
plt.show()

my_cmap = cm.bwr  # .reversed()
n = 25
extent = [0, 1, 0, 1]
# Overlay the two images
fig, ax = plt.subplots()
im = ax.imshow(step, cmap='bwr', extent=extent,norm=DivergingNorm(n))
clim = im.properties()['clim']
im2 = ax.imshow(step, cmap=my_cmap,
                interpolation='none',norm=DivergingNorm(n),
                clim=clim, extent=extent)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
clb = fig.colorbar(im, ax=ax,cax=cax)
clb.set_label('N', labelpad=-22, y=1.07, rotation=0)
ax.set_xlabel('v5 (mV)', labelpad=-17)
ax.set_ylabel('v9 (mV)', labelpad=-75)
xticks = ['v5_max', 'v5_min']
yticks = ['v9_min', 'v9_max']
ax.set_xticks(np.arange(len(xticks)))
ax.set_yticks(np.arange(len(yticks)))
ax.set_xticklabels(xticks, rotation='horizontal')
ax.set_yticklabels(yticks, rotation='horizontal')
plt.tight_layout()
plt.savefig('classification_figures/' + 'regime_1_drl_agent' +'.pdf', transparent=True)
plt.show()