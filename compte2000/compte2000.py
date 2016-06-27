"""
Spatial working memory spiking neural circuit model

References:
J.D. Murray et al. Cerebral Cortex 2012
doi: 10.1093/cercor/bhs370

A. Compte, N. Brunel, P. Goldman-Rakic, X.-J. Wang Cerebral Cortex 2000
doi: 10.1093/cercor/10.9.910

Notice that with the Compte parameters, the spontaneous state takes ~10s
to reach its actual steady-state around 8Hz (not 3.5Hz).
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

equations = dict(
    E = '''
    dV/dt         = (-(V - V_L) + Isyn/gE) / tau_m_E : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA + Istim : pA
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
    Istim  : pA
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

modelparams_compte = dict(
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

modelparams_murray = dict(
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
# Stimulus Parameters
#=========================================================================================

stimparams = dict(
    # Peak input strength
    Ipeak         = 375*pA,

    # Input width and location
    sigma_stim    = 6.,
    theta_stim    = 180.,

    # Input onset and offset
    Ton           = 500*ms,
    Toff          = 750*ms
    )

#=========================================================================================
# Model
#=========================================================================================

class Model(NetworkOperation):
    def __init__(self, modelparams, stimparams, dt=0.02*ms):
        #---------------------------------------------------------------------------------
        # Initialize
        #---------------------------------------------------------------------------------

        # Create clocks
        clocks         = OrderedDict()
        clocks['main'] = Clock(dt)
        clocks['nmda'] = Clock(dt*10)  # NMDA update is less frequent
        clocks['mons'] = Clock(1.0*ms)

        super(Model, self).__init__(clock=clocks['main'])

        #---------------------------------------------------------------------------------
        # Complete the model specification
        #---------------------------------------------------------------------------------

        # Model parameters
        if isinstance(modelparams, str):
            params = modelparams_common.copy()
            if modelparams == 'compte':
                params.update(modelparams_compte)
            elif modelparams == 'murray':
                params.update(modelparams_murray)
        elif isinstance(modelparams, dict):
            params = modelparams.copy()
        else:
            raise ValueError('Unknown modelparams type')

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
                                 clock=clocks['main'],
                                 order=1, freeze=True)

        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        for pop in ['E', 'I']:
            net['pg'+pop] = PoissonGroup(params['N_'+pop], params['nu_ext'], clock=clocks['main'])
            net['ic'+pop] = IdentityConnection(net['pg'+pop], net[pop], 'sAMPA_ext')

        #---------------------------------------------------------------------------------
        # Recurrent connections
        #---------------------------------------------------------------------------------

        # Presynaptic variables
        net['icAMPA'] = IdentityConnection(net['E'], net['E'], 'sAMPA')
        net['icNMDA'] = IdentityConnection(net['E'], net['E'], 'x')
        net['icGABA'] = IdentityConnection(net['I'], net['I'], 'sGABA')

        # Recurrent NMDA connections
        N_E      = params['N_E']
        JEE_plus = params['JEE_plus']
        sigma_EE = deg2rad(params['sigma_EE'])

        tmp = (2*scipy.stats.norm.cdf(np.pi/sigma_EE)-1)/np.sqrt(2*np.pi)*sigma_EE
        JEE_minus = (1-JEE_plus*tmp)/(1-tmp)

        dtheta = 2*np.pi*((np.arange(N_E)+1)/N_E-0.5)
        self.w = JEE_minus+((JEE_plus-JEE_minus)*np.exp(-dtheta**2/2./sigma_EE**2))

        @network_operation(when='start', clock=clocks['nmda'])
        def recurrent_NMDA():
            sNMDA = self.net['E'].sNMDA
            sNMDA_pad = np.concatenate((sNMDA[int(N_E/2):],sNMDA,sNMDA[:int(N_E/2)]))
            # Convolution speeds up 2X
            self.net['E'].S_NMDA = fftconvolve(self.w, sNMDA_pad,'same')
            self.net['I'].S_NMDA = self.net['E'].sNMDA.sum()

        # Recurrent GABA connections
        @network_operation(when='start', clock=clocks['main'])
        def recurrent_GABA():
            S = self.net['I'].sGABA.sum()
            self.net['E'].S_GABA = S
            self.net['I'].S_GABA = S

        #---------------------------------------------------------------------------------
        # Stimulus
        #---------------------------------------------------------------------------------

        clocks['stim'] = Clock(10*ms)
        @network_operation(when='start', clock=clocks['stim'])
        def stimulus(clock):
            t = clock.t
            if self.stimparams['Ton'] <= t < self.stimparams['Toff']:
                self.net['E'].Istim = self.Istim
            else:
                self.net['E'].Istim = 0

        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        mons = OrderedDict()
        var_list = ['S_AMPA', 'S_NMDA', 'S_GABA', 'V',
                    'I_AMPA', 'I_NMDA', 'I_GABA', 'Isyn']
        for pop in ['E', 'I']:
            mons['spike'+pop] = SpikeMonitor(net[pop], record=True)
            mons['pop'+pop]   = PopulationRateMonitor(net[pop], bin=0.1)
            for var in var_list:
                mons[var+pop] = StateMonitor(net[pop], var, record=True, clock=clocks['mons'])

        pop = 'E'
        for var in ['x','sNMDA','Istim']:
            mons[var+pop] = StateMonitor(net[pop], var, record=True, clock=clocks['mons'])
        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params     = params
        self.stimparams = stimparams
        self.net        = net
        self.mons       = mons
        self.clocks     = clocks

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects += self.net.values() + self.mons.values()
        self.contained_objects += [recurrent_GABA,recurrent_NMDA]
        self.contained_objects += [stimulus]

    def reinit(self, seed=123):
        # Re-initialize random number generators
        pyrand.seed(seed)
        np.random.seed(seed)

        # Reset network components, monitors, and clocks
        for n in self.net.values() + self.mons.values() + self.clocks.values():
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

        # Set stimulus
        N_E = self.params['N_E']
        dtheta = stimparams['theta_stim'] - np.arange(N_E)/N_E*360.
        self.Istim  = stimparams['Ipeak']*np.exp(-dtheta**2/2/stimparams['sigma_stim']**2)


#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    dt = 0.2*ms
    T  = 3.0*second
    modelparams = 'murray'
    
    # Setup the network
    model   = Model(modelparams, stimparams, dt)
    network = Network(model)
    
    # Setup the stimulus parameters for this trial (optional)
    model.stimparams['theta_stim']  = 180
    model.reinit(seed=1234)
    network.run(T, report='text')
    
    # Plot results
    plt.figure()
    spike_id, spike_time = zip(*model.mons['spikeE'].spikes)
    plt.plot(spike_time, spike_id, 'o', ms=2, mfc='k', mew=0)
    plt.ylim([min(spike_id),max(spike_id)])
    plt.xlabel('Time (second)')
    plt.ylabel('Neuron index')
    plt.savefig('workingmemory_ringmodel.pdf')