"""
Hodgkin-Huxley model

@author: Guangyu Robert Yang @ 2017/4

"""
from __future__ import division
from collections import OrderedDict
from scipy.signal import fftconvolve
import scipy.stats
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
dv/dt  = (gl*(El-v)-g_na*(m*m*m)*h*(v-ENa)-g_kd*(n*n*n*n)*(v-EK)+I)/Cm : volt
dm/dt  = alpham*(1-m)-betam*m : 1
dn/dt  = alphan*(1-n)-betan*n : 1
dh/dt  = alphah*(1-h)-betah*h : 1
alpham = 0.32*(mV**-1)*(13.*mV-v+VT)/(exp((13.*mV-v+VT)/(4.*mV))-1.)/ms : Hz
betam  = 0.28*(mV**-1)*(v-VT-40.*mV)/(exp((v-VT-40.*mV)/(5.*mV))-1.)/ms : Hz
alphah = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms : Hz
betah  = 4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms : Hz
alphan = 0.032*(mV**-1)*(15.*mV-v+VT)/(exp((15.*mV-v+VT)/(5.*mV))-1.)/ms : Hz
betan  = .5*exp((10.*mV-v+VT)/(40.*mV))/ms : Hz
I      : amp
'''

#=========================================================================================
# Model Parameters
#=========================================================================================

area = 20000*umetre**2
modelparams_type1 = dict(
    area    = area,
    Cm      = 1*ufarad*cm**-2 * area,
    gl      = 5e-5*siemens*cm**-2 * area,
    El      = -65*mV,
    EK      = -90*mV,
    ENa     = 50*mV,
    g_na    = 100*msiemens*cm**-2 * area,
    g_kd    = 30*msiemens*cm**-2 * area,
    VT      = -63*mV,
    tau_ref = 3*ms,
    Vth     = -20*mV, # Empirical threshold
    )

#=========================================================================================
# Model
#=========================================================================================

class Model(NetworkOperation):
    def __init__(self, modelparams, dt=0.02*ms, n_neuron=1):
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
            if modelparams == 'type1':
                params = modelparams_type1
            elif modelparams == 'type2':
                raise NotImplementedError('Type 2 not implemented yet')
                # params = modelparams_type2
        elif isinstance(modelparams, dict):
            params = modelparams.copy()
        else:
            raise ValueError('Unknown modelparams type')

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = OrderedDict() # Network objects

        threshold = EmpiricalThreshold(threshold=params['Vth'],
                                       refractory=params['tau_ref'],
                                       clock=clocks['main'])

        net['neuron'] = NeuronGroup(n_neuron,
                             Equations(equation, **params),
                             threshold=threshold,
                             clock=clocks['main'],
                             implicit=True, freeze=True)

        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        # net['pg'] = PoissonGroup(n_neuron, params['nu_ext'], clock=clocks['main'])
        # net['ic'] = IdentityConnection(net['pg'], net['neuron'], 'ge', weight=6*nS)

        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        mons = OrderedDict()
        var_list = ['v', 'm', 'n', 'h']
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
        self.net['neuron'].V = self.params['El']

        # Set other variables to zero
        for var in ['m', 'n', 'h']:
            setattr(self.net['neuron'], var, 0)

        # Set external current
        self.net['neuron'].I = self.I

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    dt = 0.02*ms
    T  = 5*second
    n_neuron = 100
    modelparams = 'type1'

    # Setup the network
    model   = Model(modelparams, dt, n_neuron)
    network = Network(model)

    # Setup the stimulus
    model.I = np.arange(n_neuron)/n_neuron*0.7*nA
    model.reinit(seed=1234)

    # Run the network
    network.run(T, report='text')

    # Compute firing rate
    rates = [np.sum((model.mons['spike'][i]>100*ms))/(T-100*ms) for i in range(n_neuron)]

    # Plot the results
    plt.plot(model.I/nA, rates)
    plt.xlabel('I (nA)')
    plt.ylabel('Firing rate (sp/s)')
    plt.savefig('hhmodel_fIcurve.pdf')
    plt.show()

