import argparse
from pprint import pprint
import numpy as np
import pathlib
# import matplotlib.pyplot as plot

from brian2 import *
# from brian2tools import *
from brian2.devices import Device
defaultclock.dt = 0.05*ms

parser = argparse.ArgumentParser(description="Izhikevich and STDP model for BSA encoded data")
parser.add_argument("-v", dest="verbose", help="verbose", action='store_true')
parser.add_argument("-file", dest="filepath", help="input npy file", type=pathlib.Path, default='./EEGparticipant 015.npy')
parser.add_argument("-izhi_a", dest="izhi_a", help="Izhi model parameter a", type=float, default=0.02)
parser.add_argument("-izhi_b", dest="izhi_b", help="Izhi model parameter b", type=float, default=0.2)
parser.add_argument("-izhi_c", dest="izhi_c", help="Izhi model parameter c", type=float, default=-50)
parser.add_argument("-izhi_d", dest="izhi_d", help="Izhi model parameter d", type=float, default=-70)
parser.add_argument("-izhi_num", dest="izhi_num", help="Number of Izhi neurons", type=int, default=1)
parser.add_argument("-brian_cpp", dest="brian_cpp", help="Brian CPP standalone mode", action='store_true')



args = parser.parse_args()
verbose = args.verbose

brian_cpp = args.brian_cpp
if (brian_cpp):
    if verbose: print('Using Brian CPP standalone mode')
    standalone_path = './cpp_standalone'
    set_device('cpp_standalone', directory=standalone_path, build_on_run=True, clean=True)
    # prefs.codegen.target = 'cython'
    # prefs.codegen.cpp.compiler = 'unix'


# External input data from .npy file
# Take first ten seconds and train
if verbose: print('Loading spike train')
eeg_spike_data = np.load(args.filepath)
eeg_spike_channel = np.concatenate(eeg_spike_data[0:10, :, 0])

izhi_num = args.izhi_num
if verbose: print('Number of neurons: {:d}'.format(izhi_num))

izhi_a = args.izhi_a
izhi_b = args.izhi_b
izhi_c = args.izhi_c
izhi_d = args.izhi_d
# Izhi params
a = izhi_a/ms
b = izhi_b/ms
c = izhi_c*mV
d = izhi_d*mV


izhi_eqs = '''dvm/dt = (0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w : volt
         dw/dt = a*(b*vm-w) : volt/second
         '''

# Data collected at 100Hz, take first channel of first epoch
spike_times = np.nonzero(eeg_spike_channel)[0]*ms
num_input_neurons = 1
input_indices = np.zeros(len(spike_times))
spike_input_group = SpikeGeneratorGroup(num_input_neurons, input_indices, spike_times)

# Izhikevich neuronal model
izhi_hidden_group = NeuronGroup(izhi_num, izhi_eqs, threshold='vm > c', reset='vm = d', method='euler')

# STDP params
taupre = 20*ms
taupost = taupre
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_synapse = '''
    w_syn : 1
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)'''
on_pre = '''
    Apre += dApre
    w_syn = clip(w_syn + Apost, 0, gmax)'''
on_post = '''
    Apost += dApost
    w_syn = clip(w_syn + Apre, 0, gmax)'''

input_neurons = spike_input_group
output_neurons = izhi_hidden_group
S = Synapses(input_neurons, output_neurons, eqs_synapse, on_pre=on_pre, on_post=on_post)
S.connect()

S.w_syn = 'rand() * gmax'
st_mon = StateMonitor(S, 'w', record=[0, 1])
sp_mon = SpikeMonitor(input_neurons)

network = Network(S, input_neurons, output_neurons)
network.run(10*second, report='text', profile=True)
profiling_summary(net=network, show=10)
