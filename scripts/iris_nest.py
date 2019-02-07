#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:14:07 2018

@author: alberto
"""

from six.moves import cPickle
import numpy as np
import os
np.set_printoptions(threshold='nan')
import sys
import pandas as pd
import seaborn as sns

# tell pycharm where to find nest (Mac Alberto)
sys.path.insert(0, "/Applications/nest-2.12.0/lib/python2.7/site-packages")

def read_from_file2list(filename):
    inh_synapses = []
    exc_synapses = []
    data = open(filename)

    if data is None:
        print 'not found'
        return
    for line in data.readlines():
        if line[0] == '#':
            continue
        # print line.strip().split()
        value = line.strip().split()
        if float(value[3])<0:
            inh_synapses.append([int(float(value[0])), int(float(value[1])), -1*float(value[3]), float(value[2])])
        else:
            exc_synapses.append([int(float(value[0])), int(float(value[1])), float(value[3]), float(value[2])])
    return [exc_synapses, inh_synapses]
    
def save_confusion_matrix(results, path):
    actual = pd.Series(list(results[:,0]),name='Actual',dtype='uint')
    predicted = pd.Series(list(results[:,1]),name='Predicted',dtype='uint')
    df_confusion = pd.crosstab(actual, predicted)
    plot_confusion = sns.heatmap(df_confusion,cmap='Blues',linewidths=0.5,annot=True)
    figure = plot_confusion.get_figure()
    figure.savefig(path+'/confusion_matrix.png')
 
dataset_path = '../datasets/iris'

# max: 83.3%
layers_path = '../nest/iris'
snn_filepath = layers_path+'/Iris_nest'

# simulation variables
num_to_test = 30
#case_to_test = 2
num_classes = 3
duration = 100
dt = 1
rescale_fac = 500

num_timesteps=duration/dt
results = np.empty((num_to_test,3))

# simulation setup
import pyNN.nest as sim
sim.setup(timestep=dt)

# cell parameters
cellparams = {'v_thresh': 1.0, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 10.0, 'e_rev_I': -10.0, 'i_offset': 0.0,
              'cm': 0.09, 'tau_m': 1000.0, 'tau_syn_E': 0.01, 'tau_syn_I': 0.01, 'tau_refrac': 0.0}

cellvalues = {'gsyn_exc': 0.0, 'v': 0.0, 'gsyn_inh': 0.0}

# load test set
print("Loading test set from '.npz' files in {}.\n".format(dataset_path))    
x_test = np.load(os.path.join(dataset_path, 'x_test.npz'))['arr_0'][:num_to_test]
y_test = np.load(os.path.join(dataset_path, 'y_test.npz'))['arr_0'][:num_to_test]

#x_test = np.load(os.path.join(dataset_path, 'x_test.npz'))['arr_0'][case_to_test:case_to_test+1]
#y_test = np.load(os.path.join(dataset_path, 'y_test.npz'))['arr_0'][case_to_test:case_to_test+1]

# ------ Load network structure --------
s = cPickle.load(open(snn_filepath, 'rb'))
# Iterate over populations in assembly
layers = []
for label in s['labels']:
    celltype = getattr(sim, s[label]['celltype'])
    population = sim.Population(s[label]['size'], celltype,
                                     celltype.default_parameters,
                                     structure=s[label]['structure'],
                                     label=label)
    # Set the rest of the specified variables, if any.
    for variable in s['variables']:
        if getattr(population, variable, None) is None:
            setattr(population, variable, s[label][variable])
    if label != 'InputLayer':
        population.set(i_offset=s[label]['i_offset'])
    layers.append(population)
print('assembly loaded')

# ------ Load weights and delays --------
for i in range(len(layers) - 1):
    print ('Loading connections for layer ' + str(layers[i].label))
    filepath = os.path.join(layers_path, layers[i + 1].label)
    conn_list=read_from_file2list(filepath)
    exc_connector = sim.FromListConnector(conn_list[0])
    pro_exc = sim.Projection(layers[i], layers[i + 1], connector=exc_connector, receptor_type='excitatory')
    if conn_list[1]!=[]:
        inh_connector = sim.FromListConnector(read_from_file2list(filepath)[1])
        pro_inh = sim.Projection(layers[i], layers[i + 1], connector=inh_connector, receptor_type='inhibitory')
print('connections loaded')

# -------- Cell initialization -----------
#vars_to_record = ['spikes','v']
#if 'spikes' in vars_to_record:
    #layers[0].record(['spikes'])  # Input layer has no 'v'
for k in range(1,len(layers)):
    layers[k].set(**cellparams)
    layers[k].initialize(**cellvalues)
    #layers[k].record(vars_to_record)
layers[-1].record(['spikes'])
# -------- Run Simulation -----------------

for batch_idx in range(num_to_test):

    batch_idxs = range(batch_idx, (batch_idx + 1))
    x = x_test[batch_idxs, :]
    y = y_test[batch_idxs, :]

    truth = np.argmax(y, axis=1)

    # --- Main step ---
    print("\nStarting new simulation...")
    rates = x.flatten()

    layers[0].set(rate=list(rates*rescale_fac))

    sim.run(duration - dt)
#        mem_pot = layers[1].get_data().segments[-1].analogsignals

    #print(pro_exc.get('weight',format='list'))
    #print(pro_inh.get('weight',format='list'))

    # -----get spiketrains-------
    shape = [num_classes, int(num_timesteps)]
    output = np.zeros(shape, 'int32')
        
    spiketrains = layers[-1].get_data().segments[-1].spiketrains
    print(str(spiketrains))

    #segment_start = num_timesteps*batch_idx
    
    spiketrains_flat = np.zeros((np.prod(shape[:-1]), shape[-1]))
    for k, spiketrain in enumerate(spiketrains):
        #spiketrain = [int(item)-segment_start for item in spiketrain if item>segment_start]
        for t in spiketrain:
            spiketrains_flat[k, int(t / dt)] = t

    spiketrains = np.reshape(spiketrains_flat, shape)
     
    for l in range(shape[0]):
            for t in range(shape[1]):
                output[l, t] = np.count_nonzero(spiketrains[l, :t+1])
    # print (str(output))

    # ---- decide class -------
    guesses = np.argmax(output, 0)
    undecided = np.nonzero(np.sum(output, 0) == 0)
    # Assign negative value such that undecided samples count as
    # wrongly classified.
    guesses[undecided] = -1
    print ('guesses ' + str(guesses))
    # Get classification error of current batch, for each time step.
    top1err = guesses != np.broadcast_to(truth, guesses.shape)              

    top1 = not top1err[-1]

    # ----- print and save results ------
    results[batch_idx] = np.array([np.argmax(y_test[batch_idx]), guesses[-1], int(top1)])
    print ('...done: ' + str(batch_idx+1) + '/' + str(num_to_test))
    print ('ClassID: ' + str(results[batch_idx][0]) +
                      ', top-1: ' + str(top1)+ '\n')

    print ('v_thresh: ' + str(layers[1].get('v_thresh')))
    
    sim.reset()

acc1 = results.sum(axis=0)[2] / num_to_test
print ('top-1 acc: ' + str(acc1 * 100) + '%')

np.savez(layers_path+'/results_mnist_nest.npz',results)
save_confusion_matrix(results,layers_path)
sim.end()