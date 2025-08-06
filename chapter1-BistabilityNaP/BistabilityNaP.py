"""
Bistability with NaP

Reference:
Wang X-J (2008)
Attractor network models
In Encyclopedia of Neuroscience, volume 1, pp. 667-679 Edited by Squire LR. Oxford: Academic Press.

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
dV/dt  = (-g_L*(V-V_L) -g_NaP*m_NaP*(V-V_Na) + I) / C_m : mV
m_NaP  = 1./(1+exp(-(V+45*mV)/(5*mV))) : 1
I      : amp
'''

#=========================================================================================
# Model Parameters
#=========================================================================================

modelparamsLIF = dict(
    V_L    = -70*mV,
    Vth    = 100*mV, # disabling spiking
    Vreset = -55*mV,

    g_L    = 25*nS,
    tau_m  = 20*ms,
    C_m    = 0.5*nF,
    tau_ref= 2*ms,

    V_Na   = 55*mV,
    g_NaP  = 15*nS
    )

#=========================================================================================
# Model
#=========================================================================================

class Model(NetworkOperation):
    def __init__(self, modelparams='LIF', dt=0.02*ms, n_neuron=1, stim=None):
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
        # External input
        #---------------------------------------------------------------------------------

        if stim is not None:
            net['neuron'].I = stim


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

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    dt = 0.02*ms
    T  = 0.5*second
    n_neuron = 1
    modelparams = 'LIF'

    # Set up the stimulus
    dt_stim = 1*ms
    i_stim  = int(T/dt_stim)+1
    t_stim  = np.arange(i_stim)/i_stim*T
    stim    = np.zeros(len(t_stim))
    stim[(50*ms<t_stim)*(t_stim<100*ms)]  = 1.0*nA
    stim[(350*ms<t_stim)*(t_stim<400*ms)] =-1.0*nA
    stim_    = TimedArray(stim, dt=dt_stim)

    # Setup the network
    model   = Model(modelparams, dt, n_neuron, stim_)
    network = Network(model)
    model.reinit(seed=1234)

    # Run the network
    network.run(T, report='text')

    # Plot the results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model.mons['V'].times/ms, model.mons['V'][0]/mV)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.subplot(2, 1, 2)
    plt.plot(t_stim/ms, stim/nA)
    plt.xlabel('Time (ms)')
    plt.ylabel('Input (nA)')
    plt.savefig('Bistability_NaP_trace.pdf')
    plt.show()