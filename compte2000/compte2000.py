"""
Spatial working memory spiking neural circuit model
A. Compte, N. Brunel, P. Goldman-Rakic, X.-J. Wang 2000
doi: 10.1093/cercor/10.9.910

"""
from __future__ import division
from collections import OrderedDict
from scipy.signal import fftconvolve

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

equations = dict(
    E = '''
    dV/dt         = (-(V - V_L) + Isyn/gE) / tau_m_E : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA + I_stim : pA
    I_AMPA_ext    = -gAMPA_ext_E*sAMPA_ext*(V - V_E) : pA
    I_AMPA        = -gAMPA_E*S_AMPA*(V - V_E) : pA
    I_NMDA        = -gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = -gGABA_E*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dx/dt         = -x/tau_x : 1
    dsNMDA/dt     = -sNMDA/tauNMDA + alpha*x*(1 - sNMDA) : 1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    I_stim : pA
    ''',

    I = '''
    dV/dt         = (-(V - V_L) + Isyn/gI) / tau_m_I : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : pA
    I_AMPA_ext    = -gAMPA_ext_I*sAMPA_ext*(V - V_E) : pA
    I_AMPA        = -gAMPA_I*S_AMPA*(V - V_E) : pA
    I_NMDA        = -gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = -gGABA_I*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA: 1
    S_NMDA: 1
    S_GABA: 1
    '''
    )

#=========================================================================================
# Model Parameters
#=========================================================================================

modelparams_common = dict(
    # Number of neurons
    N_E = 2048,
    N_I = 512,

    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -60*mV,

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
    tauGABA = 10*ms)

modelparams_compte2000 = dict(
    # Unscaled recurrent synaptic conductances (excitatory)
    gAMPA_E = 0*nS,
    gNMDA_E = 0.381*2048*nS,
    gGABA_E = 1.336*512*nS,

    # Unscaled recurrent synaptic conductances (inhibitory)
    gAMPA_I = 0*nS,
    gNMDA_I = 0.292*2048*nS,
    gGABA_I = 1.024*512*nS,

    # External synaptic conductances
    gAMPA_ext_E = 3.1*nS,
    gAMPA_ext_I = 2.38*nS,

    # Background noise
    nu_ext = 1.8*kHz,

    # Connectivity footprint
    sigma_EE = 14.4, # Notice there is a typo in the original reported value
    JEE_plus = 1.62
    )

modelparams_murray2012 = dict(
    # Unscaled recurrent synaptic conductances (excitatory)
    gAMPA_E = 0*nS,
    gNMDA_E = 1001.9*nS,
    gGABA_E = 807.2*nS,

    # Unscaled recurrent synaptic conductances (inhibitory)
    gAMPA_I = 0*nS,
    gNMDA_I = 717.6*nS,
    gGABA_I = 566.2*nS,

    # External synaptic conductances
    gAMPA_ext_E = 9.3*nS,
    gAMPA_ext_I = 7.14*nS,

    # Background noise
    nu_ext = 0.6*kHz,

    # Connectivity footprint
    sigma_EE = 9.0,
    JEE_plus = 3.0
    )


#=========================================================================================
# Model
#=========================================================================================

class Model(NetworkOperation):
    def __init__(self, modelparams, clock):
        #---------------------------------------------------------------------------------
        # Initialize
        #---------------------------------------------------------------------------------

        super(Model, self).__init__(clock=clock)

        #---------------------------------------------------------------------------------
        # Complete the model specification
        #---------------------------------------------------------------------------------

        # Model parameters
        if isinstance(modelparams,str):
            params = modelparams_common.copy()
            if modelparams == 'compte2000':
                params.update(modelparams_compte2000)
            elif modelparams == 'murray2012':
                params.update(modelparams_murray2012)
        elif isinstance(modelparams,dict):
            params = modelparams.copy()
        else:
            IOError('Unknown modelparams type')

        # Rescale conductances by number of neurons
        for conductance in ['gAMPA_E', 'gAMPA_I', 'gNMDA_E', 'gNMDA_I']:
            params[conductance] /= params['N_E']
        for conductance in ['gGABA_E', 'gGABA_I']:
            params[conductance] /= params['N_I']

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = OrderedDict() # Network objects

        # E/I populations
        for pop in ['E', 'I']:
            net[pop] = NeuronGroup(params['N_'+pop],
                                 Equations(equations[pop], **params),
                                 threshold=params['Vth'],
                                 reset=params['Vreset'],
                                 refractory=params['tau_ref_'+pop],
                                 clock=clock,
                                 order=1, freeze=True)

        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        for pop in ['E', 'I']:
            net['pg'+pop] = PoissonGroup(params['N_'+pop], params['nu_ext'], clock=clock)
            net['ic'+pop] = IdentityConnection(net['pg'+pop], net[pop], 'sAMPA_ext')

        #---------------------------------------------------------------------------------
        # Recurrent input (pre-synaptic)
        #---------------------------------------------------------------------------------

        net['icAMPA'] = IdentityConnection(net['E'], net['E'], 'sAMPA')
        net['icNMDA'] = IdentityConnection(net['E'], net['E'], 'x')
        net['icGABA'] = IdentityConnection(net['I'], net['I'], 'sGABA')

        JEE_plus = params['JEE_plus']
        sigma_EE = params['sigma_EE']/360.*2*np.pi
        N_E      = params['N_E']

        from scipy.stats import norm

        temp = (2*norm.cdf(np.pi/sigma_EE)-1)/np.sqrt(2*np.pi)*sigma_EE
        JEE_minus = (1-JEE_plus*temp)/(1-temp)

        dtheta = 2*np.pi*((np.arange(N_E)+1)/N_E-0.5)
        self.w = JEE_minus+((JEE_plus-JEE_minus)*
                    np.exp(-dtheta**2/2/sigma_EE**2))

        @network_operation(when='start', clock=clock)
        def recurrent_input():
            # NMDA. Do convolution (substantially speed up)
            sNMDA = self.net['E'].sNMDA
            sNMDA_pad = np.concatenate((sNMDA[int(N_E/2):],sNMDA,sNMDA[:int(N_E/2)]))
            self.net['E'].S_NMDA = fftconvolve(self.w, sNMDA_pad,'same')
            self.net['I'].S_NMDA = self.net['E'].sNMDA.sum()

            # GABA
            S = self.net['I'].sGABA.sum()
            self.net['E'].S_GABA = S
            self.net['I'].S_GABA = S


        I_0 = 375*pA
        sigma_stim = 6./360.*2*np.pi
        theta_stim = 180./360.*2*np.pi
        dtheta = theta_stim - np.arange(N_E)/N_E*2*np.pi
        self.Istim0 = I_0*np.exp(-dtheta**2/2/sigma_stim**2)
        self.Ton = 500*ms
        self.Toff = 750*ms
        clock_stim = Clock(50*ms)
        @network_operation(when='start', clock=clock_stim)
        def stim_input(clock_alpha):
            t = clock_alpha.t
            # Stimulus
            if (t>=self.Ton) and (t<self.Toff):
                self.net['E'].I_stim = self.Istim0
            else:
                self.net['E'].I_stim = 0
        self.contained_objects += [stim_input]
        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        self.clock_mon = Clock(0.4*ms)
        mons = OrderedDict()
        for pop in ['E', 'I']:
            mons['spike'+pop] = SpikeMonitor(net[pop], record=True)
            mons['pop'+pop] = PopulationRateMonitor(net[pop], bin=0.1)
            for var in ['S_AMPA','S_NMDA','S_GABA', 'V', 'Isyn', 'I_AMPA', 'I_NMDA', 'I_GABA']:
                mons[var+pop] = StateMonitor(net[pop], var, record=True, clock=self.clock_mon)
                pass
        pop = 'E'
        for var in ['x','sNMDA','I_stim']:
            mons[var+pop] = StateMonitor(net[pop], var, record=True, clock=self.clock_mon)
        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params = params
        self.net    = net
        self.mons   = mons

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects += self.net.values() + self.mons.values()
        self.contained_objects += [recurrent_input]

    def reinit(self):
        # Reset network components
        for n in self.net.values() + self.mons.values():
            n.reinit()


        # Randomly initialize membrane potentials
        for pop in ['E', 'I']:
            self.net[pop].V = np.random.uniform(self.params['Vreset'], self.params['Vth'],
                                                size=self.params['N_'+pop])

        # Set synaptic variables to zero
        for var in ['sAMPA_ext', 'sAMPA', 'x', 'sNMDA']:
            setattr(self.net['E'], var, 0)
        for var in ['sAMPA_ext', 'sGABA']:
            setattr(self.net['I'], var, 0)

#=========================================================================================
# Simulation
#=========================================================================================

class Simulation(object):
    def __init__(self, modelparams, dt):
        self.clock    = Clock(dt)
        self.model    = Model(modelparams, self.clock)
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
        np.savetxt(filename_exc, self.model.mons['spikeE'].spikes, fmt='%-9d %25.18e',
                   header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

        print("Saving inhibitory spike times to " + filename_inh)
        np.savetxt(filename_inh, self.model.mons['spikeI'].spikes, fmt='%-9d %25.18e',
                   header='{:<8} {:<25}'.format('Neuron', 'Time (s)'))

    def loadspikes(self, *args):
        return [np.loadtxt(filename) for filename in args]

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    dt = 0.02*ms
    T  = 2.0*second

    sim = Simulation('murray2012', dt)
    sim.run(T, seed=1234)
#==============================================================================
#     sim.savespikes('spikesE.txt', 'spikesI.txt')
# 
#     #-------------------------------------------------------------------------------------
#     # Spike raster plot
#     #-------------------------------------------------------------------------------------
# 
#     # Load excitatory spikes
#     spikes, = sim.loadspikes('spikesE.txt')
# 
#     import matplotlib.pyplot as plt
# 
#     plt.figure()
# 
#     plt.plot(spikes[:,1], spikes[:,0], 'o', ms=2, mfc='k', mew=0)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Neuron index')
# 
#     print("Saving raster plot to wang2002.pdf")
#     plt.savefig('wang2002.pdf')
#==============================================================================


    model = sim.model
    mons = model.mons
    raster_plot(mons['spikeE'])
    plt.figure()
    plt.plot(mons['popE'].times,mons['popE'].rate)
    plt.xlabel('Time (second)')
    plt.ylabel('E population rate (sp/s)')
    #plt.savefig('EPopulationRate_Murray2012.pdf')
    
params = model.params
JEE_plus = params['JEE_plus']
sigma_EE = params['sigma_EE']/360.*2*np.pi
N_E      = params['N_E']
from scipy.stats import norm

temp = (norm.cdf(np.pi/sigma_EE)-norm.cdf(-np.pi/sigma_EE))/np.sqrt(2*np.pi)*sigma_EE
JEE_minus = (1-JEE_plus*temp)/(1-temp)
Theta_to, Theta_from = np.mgrid[0:N_E,0:N_E]
Theta_to = 2*np.pi*((Theta_to+0.5)/N_E-0.5)
Theta_from = 2*np.pi*((Theta_from+0.5)/N_E-0.5)

# Get the distance in periodic boundary conditions
dtheta = Theta_to-Theta_from
dtheta = np.minimum(abs(dtheta),2*np.pi-abs(dtheta))

W = JEE_minus+((JEE_plus-JEE_minus)*
                    np.exp(-dtheta**2/2/sigma_EE**2))
                    
