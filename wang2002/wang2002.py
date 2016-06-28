"""
Probabilistic decision making by slow reverberation in cortical circuits.
X-J Wang, Neuron 2002.

http://dx.doi.org/10.1016/S0896-6273(02)01092-9

"""
from collections import OrderedDict

import random as pyrand # Import before Brian floods the namespace

# Once code is working, turn off unit-checking for speed
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

# sAMPA, x, sNMDA, sGABA are synaptic conductance stored pre-synatically
# S_AMPA, S_NMDA, S_GABA are synaptic conductance stored post-synaptically
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
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Excitatory LIF
    gE        = 25*nS,
    tau_m_E   = 20*ms,
    tau_ref_E = 2*ms,

    # Inhibitory LIF
    gI        = 20*nS,
    tau_m_I   = 10*ms,
    tau_ref_I = 1*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,

    # Synaptic time constants
    tauAMPA = 2*ms,
    tau_x   = 2*ms,
    tauNMDA = 100*ms,
    alpha   = 0.5*kHz,
    tauGABA = 5*ms,
    delay   = 0.5*ms,

    # External synaptic conductances
    gAMPA_ext_E = 2.1*nS,
    gAMPA_ext_I = 1.62*nS,

    # Unscaled recurrent synaptic conductances (onto excitatory)
    gAMPA_E = 80*nS,
    gNMDA_E = 264*nS,
    gGABA_E = 520*nS,

    # Unscaled recurrent synaptic conductances (onto inhibitory)
    gAMPA_I = 64*nS,
    gNMDA_I = 208*nS,
    gGABA_I = 400*nS,

    # Background noise
    nu_ext = 2.4*kHz,

    # Number of neurons
    N_E = 1600,
    N_I = 400,

    # Fraction of selective neurons
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

        self.set_coh(coh)

    def s1(self, t):
        return self.pos*(self.Ton <= t < self.Toff)

    def s2(self, t):
        return self.neg*(self.Ton <= t < self.Toff)

    def set_coh(self, coh):
        self.pos = self.mu0*(1 + coh/100)
        self.neg = self.mu0*(1 - coh/100)

class Model(NetworkOperation):
    def __init__(self, modelparams, clock, stimulus):
        #---------------------------------------------------------------------------------
        # Initialize
        #---------------------------------------------------------------------------------

        super(Model, self).__init__(clock=clock)

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

        # Synaptic weights between populations
        self.W = np.asarray([
            [1,  1,  1],
            [wm, wp, wm],
            [wm, wm, wp]
            ])

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = OrderedDict() # Network objects
        exc = OrderedDict() # Excitatory subpopulations

        # E/I populations
        for x in ['E', 'I']:
            net[x] = NeuronGroup(params['N_'+x],
                                 Equations(equations[x], **params),
                                 threshold=params['Vth'],
                                 reset=params['Vreset'],
                                 refractory=params['tau_ref_'+x],
                                 clock=clock,
                                 order=2, freeze=True)

        # Excitatory subpopulations
        for x in xrange(3):
            exc[x] = net['E'].subgroup(params['N'+str(x)])

        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        for x in ['E', 'I']:
            net['pg'+x] = PoissonGroup(params['N_'+x], params['nu_ext'], clock=clock)
            net['ic'+x] = IdentityConnection(net['pg'+x], net[x], 'sAMPA_ext',
                                             delay=delay)

        #---------------------------------------------------------------------------------
        # Recurrent input
        #---------------------------------------------------------------------------------

        # Change pre-synaptic variables
        net['icAMPA'] = IdentityConnection(net['E'], net['E'], 'sAMPA', delay=delay)
        net['icNMDA'] = IdentityConnection(net['E'], net['E'], 'x',     delay=delay)
        net['icGABA'] = IdentityConnection(net['I'], net['I'], 'sGABA', delay=delay)

        # Link pre-synaptic variables to post-synaptic variables
        @network_operation(when='start', clock=clock)
        def recurrent_input():
            # AMPA
            S = self.W.dot([self.exc[i].sAMPA.sum() for i in xrange(3)])
            for i in xrange(3):
                self.exc[i].S_AMPA = S[i]
            self.net['I'].S_AMPA = S[0]

            # NMDA
            S = self.W.dot([self.exc[i].sNMDA.sum() for i in xrange(3)])
            for i in xrange(3):
                self.exc[i].S_NMDA = S[i]
            self.net['I'].S_NMDA = S[0]

            # GABA
            S = self.net['I'].sGABA.sum()
            self.net['E'].S_GABA = S
            self.net['I'].S_GABA = S

        #---------------------------------------------------------------------------------
        # External input (post-synaptic)
        #---------------------------------------------------------------------------------

        for i, stimulus in zip([1, 2], [stimulus.s1, stimulus.s2]):
            net['pg'+str(i)] = PoissonGroup(params['N'+str(i)], stimulus, clock=clock)
            net['ic'+str(i)] = IdentityConnection(net['pg'+str(i)], exc[i],
                                                  'sAMPA_ext', delay=delay)

        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        mons = OrderedDict()
        for x in ['E', 'I']:
            mons['spikes'+x] = SpikeMonitor(net[x], record=True)

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params = params
        self.net    = net
        self.exc    = exc
        self.mons   = mons

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects += self.net.values() + self.mons.values()
        self.contained_objects += [recurrent_input]

    def reinit(self):
        # Reset network components
        for n in self.net.values() + self.mons.values():
            n.reinit()

        # Randomly initialize membrane potentials
        for x in ['E', 'I']:
            self.net[x].V = np.random.uniform(self.params['Vreset'], self.params['Vth'],
                                              size=self.params['N_'+x])

        # Set synaptic variables to zero
        for x in ['sAMPA_ext', 'sAMPA', 'x', 'sNMDA']:
            setattr(self.net['E'], x, 0)
        for x in ['sAMPA_ext', 'sGABA']:
            setattr(self.net['I'], x, 0)

#=========================================================================================
# Simulation
#=========================================================================================

class Simulation(object):
    def __init__(self, modelparams, stimparams, dt):
        self.clock    = Clock(dt)
        self.stimulus = Stimulus(stimparams['Ton'], stimparams['Toff'],
                                 stimparams['mu0'], stimparams['coh'])
        self.model    = Model(modelparams, self.clock, self.stimulus)
        self.network  = Network(self.model)

    def run(self, T, seed=1):
        # Initialize random number generators
        pyrand.seed(seed)
        np.random.seed(seed)

        # Initialize and run
        self.clock.reinit()
        self.model.reinit()
        self.network.run(T, report='text')

    def savespikes(self, filename_exc, filename_inh):
        print("Saving excitatory spike times to " + filename_exc)
        np.savetxt(filename_exc, self.model.mons['spikesE'].spikes, fmt='%-9d %25.18e',
                   header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

        print("Saving inhibitory spike times to " + filename_inh)
        np.savetxt(filename_inh, self.model.mons['spikesI'].spikes, fmt='%-9d %25.18e',
                   header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

    def loadspikes(self, *args):
        return [np.loadtxt(filename) for filename in args]

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    stimparams = dict(
        Ton  = 0.5*second, # Stimulus onset
        Toff = 1.5*second, # Stimulus offset
        mu0  = 40*Hz,      # Input rate
        coh  = 0           # Percent coherence
        )

    dt = 0.02*ms
    T  = 2*second

    sim = Simulation(modelparams, stimparams, dt)
    sim.stimulus.set_coh(1.6) # Shows how coherence can be changed
    sim.run(T, seed=1234)
    sim.savespikes('spikesE.txt', 'spikesI.txt')

    #-------------------------------------------------------------------------------------
    # Spike raster plot
    #-------------------------------------------------------------------------------------

    # Load excitatory spikes
    spikes, = sim.loadspikes('spikesE.txt')

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(spikes[:,1], spikes[:,0], 'o', ms=2, mfc='k', mew=0)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index')

    print("Saving raster plot to wang2002.pdf")
    plt.savefig('wang2002.pdf')
