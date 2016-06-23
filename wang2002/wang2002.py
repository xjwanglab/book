"""
Probabilistic decision making by slow reverberation in cortical circuits.
X.-J. Wang, Neuron 2002.

http://dx.doi.org/10.1016/S0896-6273(02)01092-9

"""
import random as pyrand # Import before Brian floods the namespace

import brian as B
from brian.network import network_operation
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

#=========================================================================================
# Equations
#=========================================================================================

equations = dict(
    E = '''
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : pA
    I_AMPA        = gAMPA_E*S_AMPA*(V - V_E) : pA
    I_NMDA        = gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dx/dt         = -x/tau_x : 1
    dsNMDA/dt     = -sNMDA/tauNMDA + alpha*x*(1 - sNMDA) : 1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    ''',

    I = '''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : pA
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : pA
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA: 1
    S_NMDA: 1
    S_GABA: 1
    '''
    )

#=========================================================================================
# Parameters
#=========================================================================================

modelparams = dict(
    # Common LIF
    V_L    = -70*B.mV,
    Vth    = -50*B.mV,
    Vreset = -55*B.mV,

    # Excitatory LIF
    gE        = 25*B.nS,
    tau_m_E   = 20*B.ms,
    tau_ref_E = 2*B.ms,

    # Inhibitory LIF
    gI        = 20*B.nS,
    tau_m_I   = 10*B.ms,
    tau_ref_I = 1*B.ms,

    # Reversal potentials
    V_E = 0*B.mV,
    V_I = -70*B.mV,

    # NMDA nonlinearity
    a = 0.062*B.mV**-1,
    b = 3.57,

    # Synaptic time constants
    tauAMPA = 2*B.ms,
    tau_x   = 2*B.ms,
    tauNMDA = 100*B.ms,
    alpha   = 0.5*B.kHz,
    tauGABA = 5*B.ms,
    delay   = 0.5*B.ms,

    # External synaptic conductances
    gAMPA_ext_E = 2.1*B.nS,
    gAMPA_ext_I = 1.62*B.nS,

    # Unscaled recurrent synaptic conductances (excitatory)
    gAMPA_E = 80*B.nS,
    gNMDA_E = 264*B.nS,
    gGABA_E = 520*B.nS,

    # Unscaled recurrent synaptic conductances (inhibitory)
    gAMPA_I = 64*B.nS,
    gNMDA_I = 208*B.nS,
    gGABA_I = 400*B.nS,

    # Background noise
    nu_ext = 2.4*B.kHz,

    # Number of neurons
    N_E  = 1600,
    N_I  = 400,
    fsel = 0.15,

    # Hebb-strengthened weight
    wp = 1.7
    )

#=========================================================================================
# Model
#=========================================================================================

class Stimulus(object):
    def __init__(self, Ton, Toff, mu0, coh):
        self.Ton  = Ton
        self.Toff = Toff
        self.mu0  = mu0
        self.coh  = coh

    def s1(self, t):
        if self.Ton <= t < self.Toff:
            return self.mu0*(1 + self.coh/100)
        return 0

    def s2(self, t):
        if self.Ton <= t < self.Toff:
            return self.mu0*(1 - self.coh/100)
        return 0

class Model(B.NetworkOperation):
    def __init__(self, modelparams, stimparams, dt):
        #---------------------------------------------------------------------------------
        # Complete the model specification
        #---------------------------------------------------------------------------------

        # Model parameters
        params = modelparams.copy()

        # Rescale conductances by number of neurons
        for x in ['gAMPA_E', 'gAMPA_I', 'gNMDA_E', 'gNMDA_I']:
            params[x] /= params['N_E']
        for x in ['gGABA_E', 'gGABA_I']:
            params[x] /= params['N_I']

        # Make local variables for convenience
        N_E   = params['N_E']
        fsel  = params['fsel']
        wp    = params['wp']
        delay = params['delay']

        # Subpopulation size
        N1 = int(fsel*N_E)
        N2 = N1
        N0 = N_E - (N1 + N2)
        params['N0'] = N0
        params['N1'] = N1
        params['N2'] = N2

        # Hebb-weakened weight
        wm = (1 - wp*fsel)/(1 - fsel)
        params['wm'] = wm

        # Synaptic weights
        self.W = np.asarray([
            [1,  1,  1],
            [wm, wp, wm],
            [wm, wm, wp]
            ])

        #---------------------------------------------------------------------------------
        # Clock
        #---------------------------------------------------------------------------------

        clock = B.Clock(dt)
        super(Model, self).__init__(clock=clock)

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = dict()

        # E/I populations
        for x in ['E', 'I']:
            net[x] = B.NeuronGroup(params['N_'+x],
                                 B.Equations(equations[x], **params),
                                 threshold=params['Vth'],
                                 reset=params['Vreset'],
                                 refractory=params['tau_ref_'+x],
                                 clock=clock,
                                 order=2, freeze=True)

        # Excitatory subpopulations
        for x in xrange(3):
            net[x] = net['E'].subgroup(params['N'+str(x)])

        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        for x in ['E', 'I']:
            net['pg'+x] = B.PoissonGroup(params['N_'+x], params['nu_ext'], clock=clock)
            net['ic'+x] = B.IdentityConnection(net['pg'+x], net[x], 'sAMPA_ext',
                                             delay=delay)

        #---------------------------------------------------------------------------------
        # Recurrent input (pre-synaptic)
        #---------------------------------------------------------------------------------

        net['icAMPA'] = B.IdentityConnection(net['E'], net['E'], 'sAMPA', delay=delay)
        net['icNMDA'] = B.IdentityConnection(net['E'], net['E'], 'x',     delay=delay)
        net['icGABA'] = B.IdentityConnection(net['I'], net['I'], 'sGABA', delay=delay)

        #---------------------------------------------------------------------------------
        # External input (post-synaptic)
        #---------------------------------------------------------------------------------

        self.stimulus = Stimulus(stimparams['Ton'], stimparams['Toff'],
                                 stimparams['mu0'], stimparams['coh'])
        for i, stimulus in zip([1, 2], [self.stimulus.s1, self.stimulus.s2]):
            net['pg'+str(i)] = B.PoissonGroup(params['N'+str(i)], stimulus, clock=clock)
            net['ic'+str(i)] = B.IdentityConnection(net['pg'+str(i)], net[i],
                                                  'sAMPA_ext', delay=delay)

        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        for x in ['E', 'I']:
            net['sm'+x] = B.SpikeMonitor(net[x], record=True)

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params = params
        self.net    = net
        self.contained_objects += [v for k, v in self.net.items() if k not in xrange(3)]
        self.contained_objects += [self.get_recurrent_input()]

    def get_recurrent_input(self):
        @network_operation(when='start', clock=self.clock)
        def recurrent_input():
            # AMPA
            S = self.W.dot([self.net[i].sAMPA.sum() for i in xrange(3)])
            for i in xrange(3):
                self.net[i].S_AMPA = S[i]
            self.net['I'].S_AMPA = S[0]

            # NMDA
            S = self.W.dot([self.net[i].sNMDA.sum() for i in xrange(3)])
            for i in xrange(3):
                self.net[i].S_NMDA = S[i]
            self.net['I'].S_NMDA = S[0]

            # GABA
            S = self.net['I'].sGABA.sum()
            self.net['E'].S_GABA = S
            self.net['I'].S_GABA = S

        return recurrent_input

    def reinit(self):
        # Reset network components
        for n in self.net.values():
            n.reinit()

        # Reset membrane potential
        for x in ['E', 'I']:
            self.net[x].V = self.params['V_L']

        # Set synaptic variables to zero
        for x in ['sAMPA_ext', 'sAMPA', 'x', 'sNMDA']:
            setattr(self.net['E'], x, 0)
        for x in ['sAMPA_ext', 'sGABA']:
            setattr(self.net['I'], x, 0)

        # Reset clock
        self.clock.reinit()

#=========================================================================================
# Simulation
#=========================================================================================

class Simulation(object):
    def __init__(self, modelparams, stimparams, dt):
        self.model   = Model(modelparams, stimparams, dt)
        self.network = B.Network(self.model)

    def run(self, T, seed=1):
        # Initialize random number generators
        pyrand.seed(seed)
        np.random.seed(seed)

        # Make Brian faster
        accelerate()

        # Initialize the network and run
        self.model.reinit()
        self.network.run(T, report='text')

    def savespikes(self, filename_exc='spikesE.txt', filename_inh='spikesI.txt'):
        savespikes(self.model.net['smE'], filename_exc)
        savespikes(self.model.net['smI'], filename_inh)

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    stimparams = dict(
        Ton  = 1*B.second,
        Toff = 2*B.second,
        mu0  = 40*B.Hz,
        coh  = 6.4
        )

    dt = 0.02*B.ms
    T  = 5*B.second

    sim = Simulation(modelparams, stimparams, dt)
    sim.run(T)
    sim.savespikes()

    #-------------------------------------------------------------------------------------
    # Spike raster plot
    #-------------------------------------------------------------------------------------

    # Load spikes
    spikes = loadspikes('spikesE.txt')

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(spikes[:,1], spikes[:,0], 'o', ms=2, mfc='k', mew=0)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index')

    print("Saving raster plot to wang2002.pdf")
    plt.savefig('wang2002.pdf')
