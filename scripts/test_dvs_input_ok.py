#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:31:13 2019

@author: alberto
"""
import pyNN.nest as sim
import numpy as np
import scipy.io
from matplotlib import pyplot as plt

dt = 1.0
win = 20
duration = int(win*dt)

sim.setup(timestep=dt)

dataset_path = '../datasets/nmnist/NMNIST_small_test_data.mat'
#weights_path = '../nest/nmnist_2/'
weights_path = '../nest/nmnist_400/'

data = scipy.io.loadmat(dataset_path)

image, label = data['image'], data['label']

image = image.swapaxes(3,1)
image = image.swapaxes(2,3)

w0 = np.load(weights_path+'fc0.weight.npz')['arr_0']
b0 = np.load(weights_path+'fc0.bias.npz')['arr_0']

w1 = np.load(weights_path+'fc1.weight.npz')['arr_0']
b1 = np.load(weights_path+'fc1.bias.npz')['arr_0']

w2 = np.load(weights_path+'fc2.weight.npz')['arr_0']
b2 = np.load(weights_path+'fc2.bias.npz')['arr_0']


delay = 1.0
conns_0 = []
for i in range(w0.shape[1]):
    for j in range(w0.shape[0]):
        conns_0.append([i, j, w0[j, i], delay])

conns_1 = []
for i in range(w1.shape[1]):
    for j in range(w1.shape[0]):
        conns_1.append([i, j, w1[j, i], delay])
        
conns_2 = []
for i in range(w2.shape[1]):
    for j in range(w2.shape[0]):
        conns_2.append([i, j, w2[j, i], delay])

sample_idx = 999
true = label[sample_idx].argmax()

sample_image = image[sample_idx,:,:,:,:]

flat_image = np.empty((win, 34*34*2))

for t in range(win): 
    flat_image[t,:] = sample_image[:,:,:,t].reshape((1,-1))

fig=plt.figure('first layer potentials')
plt.plot(flat_image[1,:])
plt.show()

fl_i = flat_image.swapaxes(1,0).tolist()

spike_times = []

for neuron_id in range(34*34*2):
    spike_times.append([dt*(i+2.0) for i,x in enumerate(fl_i[neuron_id]) if x == 1])

v_th = 0.3

vth0 = v_th-b0
vth0[vth0<0.0] = 0.001

vth1 = v_th-b1
vth1[vth1<0.0] = 0.001

vth2 = v_th-b2
vth2[vth2<0.0] = 0.001

tau_syn = 1.0

cellparams = {'v_thresh': v_th, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 10.0, 'e_rev_I': -10.0, 'i_offset': 0.0,
              'cm': 0.1, 'tau_m': 0.8325, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}

cellvalues = {'gsyn_exc': 0.0, 'v': 0.0, 'gsyn_inh': 0.0}

input_celltype = sim.SpikeSourceArray(spike_times=spike_times)
fc_celltype = sim.IF_cond_exp(**cellparams)
input_pop = sim.Population(34*34*2, input_celltype)


fc0 = sim.Population(w1.shape[0], fc_celltype)
fc0.initialize(**cellvalues)
fc0.set(v_thresh=vth0)

fc1 = sim.Population(w1.shape[0], fc_celltype)
fc1.initialize(**cellvalues)
fc1.set(v_thresh=vth1)
#fc1.set(i_offset=b1)

fc2 = sim.Population(10, fc_celltype)
fc2.initialize(**cellvalues)
fc2.set(v_thresh=vth2)
#fc2.set(i_offset=b2)

proj0 = sim.Projection(input_pop,fc0, connector=sim.FromListConnector(conns_0))
proj1 = sim.Projection(fc0, fc1, connector=sim.FromListConnector(conns_1))
proj2 = sim.Projection(fc1, fc2, connector=sim.FromListConnector(conns_2))

#input_pop.record('spikes')
fc0.record(['spikes','v'])
#fc1.record(['spikes','v'])
#fc2.record(['spikes','v'])
fc2.record(['spikes'])

sim.run(duration-dt)

#spiketrains_i = input_pop.get_data().segments[-1].spiketrains
#spiketrains_0 = fc0.get_data().segments[-1].spiketrains
#spiketrains_1 = fc1.get_data().segments[-1].spiketrains
spiketrains_2 = fc2.get_data().segments[-1].spiketrains

st_count = []
for k, spiketrain in enumerate(spiketrains_2):
    spiketrain = [item for item in spiketrain]
    count=0
    for t in spiketrain:
        count += 1
    st_count.append(count)


predicted = st_count.index(max(st_count))
membranes_0 = fc0.get_data().segments[-1].analogsignals[0]
#membranes_1 = fc1.get_data().segments[-1].analogsignals[0]
#membranes_2 = fc2.get_data().segments[-1].analogsignals[0]

from pyNN.utility.plotting import Figure, Panel

Figure(
    Panel(membranes_0[:,15:25], ylabel="Membrane potential (mV)", xticks=True,
          xlabel="Time (ms)", yticks=True))

#Figure(
#    Panel(membranes_1[:,15:25], ylabel="Membrane potential (mV)", xticks=True,
#          xlabel="Time (ms)", yticks=True))
#
#Figure(
#    Panel(membranes_2[:,:10], ylabel="Membrane potential out (mV)", xticks=True,
#          xlabel="Time (ms)", yticks=True))


## animate sample
#fig=plt.figure('nmnist sample')
#for t in range(sample_image.shape[0]):
#    data = np.array([sample_image[t,0,:,:],
#                 np.zeros((34,34)), 
#                 sample_image[t,1,:,:]]).swapaxes(0,2)
#    plt.imshow(data)
#    print (t)
#    plt.pause(0.01)
    
plt.show()

print('predicted: '+ str(predicted) + ' true:'+ str(true))