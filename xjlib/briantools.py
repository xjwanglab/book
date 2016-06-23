from brian.globalprefs import *

import numpy as np

def accelerate():
    set_global_preferences(
        useweave=True,
        usecodegen=True,
        usecodegenweave=True,
        usecodegenstateupdate=True,
        usenewpropagate=True,
        usecodegenthreshold=True,
        gcc_options=['-ffast-math', '-march=native']
        )

def savespikes(spikemonitor, filename):
    print("Saving spike times to " + filename)
    np.savetxt(filename, spikemonitor.spikes, fmt='%-9d %25.18e',
               header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

def loadspikes(filename):
    print("Loading spike times from " + filename)
    return np.loadtxt(filename)
