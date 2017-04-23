"""
Leaky-Integrate-and-Fire model
@author: Guangyu Robert Yang @ 2017/4

"""
from __future__ import division
from collections import OrderedDict
import random as pyrand # Import before Brian floods the namespace

# Once your code is working, turn units off for speed
# import brian_no_units

from brian import *

# Make Brian faster
set_global_preferences(
    useweave=True,
    usecodegen=True,
    usecodegenweave=True,
    usecodegenstateupdate=True,
    usenewpropagate=True,
    usecodegenthreshold=True,
    gcc_options=['-ffast-math', '-march=native']
    )

#=========================================================================================
# Equations
#=========================================================================================

equation = '''
dV/dt  = (-(V - V_L) + I/g) / tau_m : mV
I      : amp
'''

#=========================================================================================
# Model Parameters
#=========================================================================================

modelparamsLIF = dict(
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    g         = 25*nS,
    tau_m     = 20*ms,
    tau_ref   = 2*ms,
    )

#=========================================================================================
# Model
#=========================================================================================

class Model(NetworkOperation):
    def __init__(self, modelparams='LIF', dt=0.02*ms, n_neuron=1):
        #---------------------------------------------------------------------------------
        # Initialize
        #---------------------------------------------------------------------------------

        # Create clocks
        clocks         = OrderedDict()
        clocks['main'] = Clock(dt)
        clocks['mons'] = Clock(0.1*ms)

        super(Model, self).__init__(clock=clocks['main'])

        #---------------------------------------------------------------------------------
        # Complete the model specification
        #---------------------------------------------------------------------------------

        # Model parameters
        if isinstance(modelparams, str):
            if modelparams == 'LIF':
                params = modelparamsLIF.copy()
            else:
                raise ValueError('Unknown model params')
        elif isinstance(modelparams, dict):
            params = modelparams.copy()
        else:
            raise ValueError('Unknown modelparams type')

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = OrderedDict() # Network objects

        net['neuron'] = NeuronGroup(n_neuron,
                             Equations(equation, **params),
                             threshold=params['Vth'],
                             reset=params['Vreset'],
                             refractory=params['tau_ref'],
                             clock=clocks['main'],
                             order=2, freeze=True)

        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        mons = OrderedDict()
        var_list = ['V']
        mons['spike'] = SpikeMonitor(net['neuron'], record=True)
        mons['pop']   = PopulationRateMonitor(net['neuron'], bin=0.1)
        for var in var_list:
            mons[var] = StateMonitor(net['neuron'], var, record=True, clock=clocks['mons'])

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params     = params
        self.I          = 0.
        self.net        = net
        self.mons       = mons
        self.clocks     = clocks
        self.n_neuron   = n_neuron

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects += self.net.values() + self.mons.values()

    def reinit(self, seed=123):
        # Re-initialize random number generators
        pyrand.seed(seed)
        np.random.seed(seed)

        # Reset network components, monitors, and clocks
        for n in self.net.values() + self.mons.values() + self.clocks.values():
            n.reinit()

        # Randomly initialize membrane potentials
        self.net['neuron'].V = self.params['V_L']

        # Set external current
        self.net['neuron'].I = self.I

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    dt = 0.02*ms
    T  = 3*second
    n_neuron = 100
    modelparams = 'LIF'

    # Setup the network
    model   = Model(modelparams, dt, n_neuron)
    network = Network(model)

    # Setup the stimulus
    model.I = np.arange(n_neuron)/n_neuron*1.0*nA
    model.reinit(seed=1234)

    # Run the network
    network.run(T, report='text')

    # Compute firing rate
    rates = [np.sum((model.mons['spike'][i]>100*ms))/(T-100*ms) for i in range(n_neuron)]

    # Plot the results
    plt.plot(model.I/nA, rates)
    plt.xlabel('I (nA)')
    plt.ylabel('Firing rate (sp/s)')
    plt.savefig('LIFmodel_fIcurve.pdf')
    plt.show()

