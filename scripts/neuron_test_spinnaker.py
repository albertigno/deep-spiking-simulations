#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:59:17 2019

@author: alberto

trying to sumilate spiking neuron as implement in STBP paper
"""

import spynnaker8 as sim
#import numpy as np
#import scipy.io
#from matplotlib import pyplot as plt

dt = 1
duration = int(16*dt)

sim.setup(timestep=dt)

delay = 1.0

w2 = [[0, 0, 0.2, delay], [0, 1, 0.1, delay]]

w1 = [[0, 0, 0.2, delay]]

b = -0.02

# decay without presynaptic input or bias current = f(tau_m)

cellparams = {'v_thresh': 0.3, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 0.0, 'e_rev_I': 10.0, 'i_offset': 0.0,
              'cm': 0.0001, 'tau_m': 0.8325, 'tau_syn_E': 0.001, 'tau_syn_I': 0.001, 'tau_refrac': 0.0}

cellvalues = {'v': 0.25}

#cellvalues = {'v': -0.25} # debe haber simetria

t2 = [[2,3,4,8],[2,4,6,8]]
t1 = [2,12]

input_celltype = sim.SpikeSourceArray(spike_times=t1)
fc_celltype = sim.IF_cond_exp(**cellparams)

pop0 = sim.Population(1,input_celltype)
pop1 = sim.Population(1, fc_celltype)
pop1.initialize(**cellvalues)
#pop1.set(i_offset=b)

pop1.record(['spikes','v'])
pop0.record(['spikes'])

# create synapsis

conn = sim.FromListConnector(w1)
pro = sim.Projection(pop0, pop1, connector=conn)

sim.run(duration)

mem = pop1.get_data().segments[-1].filter(name='v')[0]

from pyNN.utility.plotting import Figure, Panel

Figure(
    Panel(mem, ylabel="Membrane potential (mV)", xticks=True,
          xlabel="Time (ms)", yticks=True))