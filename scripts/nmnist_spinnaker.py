#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:31:13 2019

@author: alberto
"""
import spynnaker8 as sim
import numpy as np
import scipy.io

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

weights = []
biases =[]

weights.append(np.load(weights_path+'fc0.weight.npz')['arr_0'])
biases.append(np.load(weights_path+'fc0.bias.npz')['arr_0'])

weights.append(np.load(weights_path+'fc1.weight.npz')['arr_0'])
biases.append(np.load(weights_path+'fc1.bias.npz')['arr_0'])

weights.append(np.load(weights_path+'fc2.weight.npz')['arr_0'])
biases.append(np.load(weights_path+'fc2.bias.npz')['arr_0'])

delay = 1.0
v_th = 0.3

cm = 0.1
tau_syn = 0.01

cellparams = {'v_thresh': v_th, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 10.0, 'e_rev_I': -10.0, 'i_offset': 0.0,
              'cm': cm, 'tau_m': 0.8325, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}

cellvalues = {'v': 0.0}

fc_celltype = sim.IF_cond_exp(**cellparams)
input_celltype = sim.SpikeSourceArray()
input_pop = sim.Population(34*34*2, input_celltype)

pops = [input_pop]
proj_exc = []
proj_inh = []

for layer in range(len(weights)):
    inh_synapses = []
    exc_synapses = []
    for i in range(weights[layer].shape[1]):
        for j in range(weights[layer].shape[0]):        
            if float(weights[layer][j, i])<0.0:
                inh_synapses.append([i, j, -1.0*weights[layer][j, i], delay])
            else:
                exc_synapses.append([i, j, weights[layer][j, i], delay])
    
    pops.append(sim.Population(weights[layer].shape[0], fc_celltype))
    
    vth = v_th-biases[layer]
    vth[vth<0.0] = 0.001
    
    pops[layer+1].set(v_thresh=vth)
    
    conn_exc = sim.FromListConnector(exc_synapses)
    conn_inh = sim.FromListConnector(inh_synapses)   
    proj_exc.append(sim.Projection(pops[layer],pops[layer+1], connector=conn_exc, receptor_type='excitatory'))   
    proj_inh.append(sim.Projection(pops[layer],pops[layer+1], connector=conn_inh, receptor_type='inhibitory'))

    
pops[-1].record(['spikes'])
#pops[0]. record(['spikes'])
#pops[1]. record(['spikes','v'])

num_to_test = 1000
acc = 0.0

num_timesteps = duration/dt

for sample_idx in range(num_to_test):

    true = label[sample_idx].argmax()
    sample_image = image[sample_idx,:,:,:,:]
    flat_image = np.empty((win, 34*34*2))
    
    for t in range(win): 
        flat_image[t,:] = sample_image[:,:,:,t].reshape((1,-1))

    fl_i = flat_image.swapaxes(1,0).tolist()
    
    spike_times = []
    segment_start = 2 + (duration/dt)*sample_idx
    
    for neuron_id in range(34*34*2):
        spike_times.append([segment_start+float(i) for i,x in enumerate(fl_i[neuron_id]) if x == 1])
    
    input_pop.set(spike_times=spike_times)
    
    for layer in range(len(weights)):
        pops[layer+1].initialize(**cellvalues)
    
    sim.run(duration)
    spiketrains_last = pops[-1].get_data().segments[-1].spiketrains
    #spiketrains_0 = pops[0].get_data().segments[-1].spiketrains
    #spiketrains_1 = pops[1].get_data().segments[-1].spiketrains
    #membranes_1 = pops[1].get_data().segments[-1].filter(name='v')[0]
    
    st_count = []
    for k, spiketrain in enumerate(spiketrains_last):
        spiketrain = [int(item)-segment_start for item in spiketrain if item>segment_start]
        count=0
        for t in spiketrain:
            count += 1
        st_count.append(count)
      
    predicted = st_count.index(max(st_count))
    print('predicted: '+ str(predicted) + ' true:'+ str(true))
    
    #sim.reset()
    
    if predicted == true:
        acc +=1.0

print('-----------------------------------------')
print('accuracy: ' +str(100*(acc/num_to_test)) + '%')


#from pyNN.utility.plotting import Figure, Panel
#
#Figure(
#    Panel(membranes_1[:,15:25], ylabel="Membrane potential (mV)", xticks=True,
#          xlabel="Time (ms)", yticks=True))
