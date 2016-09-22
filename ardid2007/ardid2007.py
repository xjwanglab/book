"""
Double ring model

References:
Ardid, Wang, Compte 2007 Journal of Neuroscience
doi: 10.1523/JNEUROSCI.1145-07.2007

How is the long-range connection modeled?

"""
from __future__ import division
from collections import OrderedDict
from scipy.signal import fftconvolve
import scipy.stats
import random as pyrand # Import before Brian floods the namespace

# Once your code is working, turn units off for speed
# import brian_no_units

from brian import *

# Notice scipy.fftpack.rfft behaves differently from numpy.fft.rfft
from numpy.fft import rfft, irfft

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
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dx/dt         = -x/tau_x : 1
    dsNMDA/dt     = -sNMDA/tauNMDA + alpha*x*(1 - sNMDA) : 1
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA + Istim : pA
    I_AMPA_ext    = -G_AMPA_ext*(V - V_E) : pA
    I_AMPA        = -G_AMPA*(V - V_E) : pA
    I_NMDA        = -G_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = -G_GABA*(V - V_I) : pA
    dG_AMPA_ext/dt= -G_AMPA_ext/tauAMPA : nS
    G_AMPA : nS
    G_NMDA : nS
    G_GABA : nS
    Istim  : pA
    ''',

    I = '''
    dV/dt         = (-(V - V_L) + Isyn/gI) / tau_m_I : mV
    dsGABA/dt     = -sGABA/tauGABA : 1
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : pA
    I_AMPA_ext    = -G_AMPA_ext*(V - V_E) : pA
    I_AMPA        = -G_AMPA*(V - V_E) : pA
    I_NMDA        = -G_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = -G_GABA*(V - V_I) : pA
    dG_AMPA_ext/dt= -G_AMPA_ext/tauAMPA : nS
    G_AMPA: nS
    G_NMDA: nS
    G_GABA: nS
    '''
    )


#=========================================================================================
# Model Parameters
#=========================================================================================

modelparams = dict()

modelparams['neuron'] = dict(
    # Number of neurons
    N_E = 1024,
    N_I = 256,

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
    tauGABA = 10*ms,

    # Background noise
    nu_ext = 1.8*kHz
    )

modelparams['mt'] = dict(
    # recurrent connectivity footprint
    sigma_EE = 14.4,
    JEE_plus = 1.62,

    # Unscaled recurrent synaptic conductances (onto excitatory)
    gAMPA_E = 0.005*1024*nS,
    gNMDA_E = 0.093*1024*nS,
    gGABA_E = 1.47*256*nS,

    # Unscaled recurrent synaptic conductances (onto inhibitory)
    gAMPA_I = 0.005*1024*nS,
    gNMDA_I = 0.195*1024*nS,
    gGABA_I = 0.391*256*nS,

    # External synaptic conductances
    gAMPA_ext_E = 15.0*nS,
    gAMPA_ext_I = 4.5*nS,
    )

modelparams['pfc'] = dict(
    # recurrent connectivity footprint
    sigma_EE = 14.4,
    JEE_plus = 1.62,

    # Unscaled recurrent synaptic conductances (onto excitatory)
    gAMPA_E = 0.391*1024*nS,
    gNMDA_E = 0.732*1024*nS,
    gGABA_E = 3.74*256*nS,

    # Unscaled recurrent synaptic conductances (onto inhibitory)
    gAMPA_I = 0.293*1024*nS,
    gNMDA_I = 0.566*1024*nS,
    gGABA_I = 2.87*256*nS,

    # External synaptic conductances
    gAMPA_ext_E = 3.1*nS,
    gAMPA_ext_I = 2.38*nS
    )

modelparams['conn'] = dict(
    # Connectivity footprint
    sigma_mt2pfc = 36.,
    sigma_pfc2mt = 72.,

    # Unscaled long range synaptic conductances (onto excitatory)
    gAMPA_E_mt2pfc = 0.005*1024*nS,
    gAMPA_E_pfc2mt = 0.146*1024*nS,

    # Unscaled long range synaptic conductances (onto inhibitory)
    gAMPA_I_mt2pfc = 0*nS,
    gAMPA_I_pfc2mt = 0.039*1024*nS
    )

#=========================================================================================
# Stimulus Parameters
#=========================================================================================

stimparams = dict(
    Ton  = 500 * ms,
    Toff = 750 * ms,
    I0_E = 1. * nA,
    I1_E = 0.9 * nA,
    I0_I = 0.2 * nA,
    I1_I = 0.18 * nA,
    mu   = 2.53,
    theta_stim = 180,
    Igate = 0.025 * nA
    )

#=========================================================================================
# Model
#=========================================================================================

class Model(NetworkOperation):
    def __init__(self, modelparams,
                 stimparams, dt=0.02*ms):
        #---------------------------------------------------------------------------------
        # Initialize
        #---------------------------------------------------------------------------------

        # Create clocks
        clocks         = OrderedDict()
        clocks['main'] = Clock(dt)
        clocks['nmda'] = Clock(dt*10)  # NMDA update is less frequent
        clocks['mons'] = Clock(1.0*ms)
        #clocks['mons'] = Clock(dt)

        super(Model, self).__init__(clock=clocks['main'])

        #---------------------------------------------------------------------------------
        # Complete the model specification
        #---------------------------------------------------------------------------------

        # Model parameters
        p = modelparams.copy()
        p_neuron = p['neuron']

        areas = ['mt','pfc']        # areas
        conns = ['pfc2mt','mt2pfc'] # connections

        # Rescale conductances by number of neurons
        for area in areas:
            for conductance in ['gAMPA_E', 'gAMPA_I', 'gNMDA_E', 'gNMDA_I']:
                p[area][conductance] /= p_neuron['N_E']
            for conductance in ['gGABA_E', 'gGABA_I']:
                p[area][conductance] /= p_neuron['N_I']

        for conn in conns:
            p['conn']['gAMPA_E_'+conn] /= p_neuron['N_E']
            p['conn']['gAMPA_I_'+conn] /= p_neuron['N_E']

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net = OrderedDict() # Network objects
        exc = OrderedDict()
        inh = OrderedDict()
        
        # E/I populations
        for pop in ['E', 'I']:
            net[pop] = NeuronGroup(len(areas)*p_neuron['N_'+pop],
                                 Equations(equations[pop], **p_neuron),
                                 threshold=p_neuron['Vth'],
                                 reset=p_neuron['Vreset'],
                                 refractory=p_neuron['tau_ref_'+pop],
                                 clock=clocks['main'],
                                 order=1, freeze=True)

        # Excitatory subpopulations
        for area in areas:
            exc[area] = net['E'].subgroup(p_neuron['N_E'])
            inh[area] = net['I'].subgroup(p_neuron['N_I'])

        #---------------------------------------------------------------------------------
        # Background AMPA input (post-synaptic)
        #---------------------------------------------------------------------------------

        for area in areas:
            for pop, target in zip(['E','I'], [exc,inh]):
                net['pg'+area+pop] = PoissonGroup(p_neuron['N_'+pop], p_neuron['nu_ext'],
                                             clock=clocks['main'])
                net['ic'+area+pop] = IdentityConnection(net['pg'+area+pop], target[area], 'G_AMPA_ext',
                                                   weight=p[area]['gAMPA_ext_'+pop])

        #---------------------------------------------------------------------------------
        # Recurrent connections
        #---------------------------------------------------------------------------------

        # Presynaptic variables
        net['icAMPA'] = IdentityConnection(net['E'], net['E'], 'sAMPA')
        net['icNMDA'] = IdentityConnection(net['E'], net['E'], 'x')
        net['icGABA'] = IdentityConnection(net['I'], net['I'], 'sGABA')

        def get_fw(N, sigma, J_plus=None):
            dtheta = 2*np.pi*np.minimum(np.arange(N),N-np.arange(N))/N
            if J_plus is not None:
                sigma = deg2rad(sigma)
                tmp = (2*scipy.stats.norm.cdf(np.pi/sigma)-1)/np.sqrt(2*np.pi)*sigma
                J_minus = (1-J_plus*tmp)/(1-tmp)
                w = J_minus+((J_plus-J_minus)*np.exp(-dtheta**2/2./sigma**2))
            else:
                w = np.exp(-dtheta**2/2./sigma**2)/sigma/np.sqrt(2*np.pi)
            return rfft(w)

        # Recurrent NMDA connections
        N_E      = p_neuron['N_E']
        self.fw = dict()
        for area in areas:
            self.fw[area] = get_fw(N_E, p[area]['sigma_EE'], p[area]['JEE_plus'])

        p_conn = p['conn']
        for conn in ['pfc2mt','mt2pfc']:
            self.fw[conn] = get_fw(N_E, p_conn['sigma_'+conn])

        # PFC, MT only have within area NMDA connections
        @network_operation(when='start', clock=clocks['nmda'])
        def recurrent_NMDA():
            for area in areas:
                fsNMDA = rfft(self.exc[area].sNMDA)
                self.exc[area].G_NMDA = irfft(self.fw[area] * fsNMDA, N_E) * p[area]['gNMDA_E']
                self.inh[area].G_NMDA = fsNMDA[0]  * p[area]['gNMDA_I']

        # Recurrent GABA connections
        @network_operation(when='start', clock=clocks['main'])
        def recurrent_GABA():
            for area in areas:
                S = self.inh[area].sGABA.sum()
                self.exc[area].G_GABA = S * p[area]['gGABA_E']
                self.inh[area].G_GABA = S * p[area]['gGABA_I']

        # AMPA
        @network_operation(when='start', clock=clocks['main'])
        def recurrent_AMPA():
            fsAMPA = dict()
            for area in areas:
                fsAMPA[area] = rfft(self.exc[area].sAMPA)
                self.exc[area].G_AMPA = irfft(self.fw[area] * fsAMPA[area], N_E) * p[area]['gAMPA_E']
                self.inh[area].G_AMPA = fsAMPA[area][0]  * p[area]['gAMPA_I']

            for conn in conns:
                area_from, area_to = conn.split('2')
                tmp = irfft(self.fw[conn] * fsAMPA[area_from], N_E)
                self.exc[area_to].G_AMPA += p_conn['gAMPA_E_'+conn] * tmp
                # slicing [3::4] start from index 3 and take one every 4 number
                self.inh[area_to].G_AMPA += p_conn['gAMPA_I_'+conn] * tmp[3::4]
        #---------------------------------------------------------------------------------
        # Stimulus
        #---------------------------------------------------------------------------------

        clocks['stim'] = Clock(10*ms)
        @network_operation(when='start', clock=clocks['stim'])
        def stimulus(clock):
            t = clock.t
            if self.stimparams['Ton'] <= t < self.stimparams['Toff']:
                self.exc['mt'].Istim = self.Istim['E']
                self.inh['mt'].Istim = self.Istim['I']
                self.exc['pfc'].Istim = self.stimparams['Igate']
                self.inh['pfc'].Istim = self.stimparams['Igate']
            else:
                self.exc['mt'].Istim = 0
                self.inh['mt'].Istim = 0
                self.exc['pfc'].Istim = 0
                self.inh['pfc'].Istim = 0

        #---------------------------------------------------------------------------------
        # Record spikes
        #---------------------------------------------------------------------------------

        mons = OrderedDict()
        var_list = ['G_AMPA', 'G_NMDA', 'G_GABA', 'V', 'G_AMPA_ext',
                    'I_AMPA', 'I_NMDA', 'I_GABA', 'Isyn']
        for pop in ['E', 'I']:
            mons['spike'+pop] = SpikeMonitor(net[pop], record=True)
            mons['pop'+pop]   = PopulationRateMonitor(net[pop], bin=0.1)
            for var in var_list:
                mons[var+pop] = StateMonitor(net[pop], var, record=True, clock=clocks['mons'])

        for pop, target in zip(['E','I'], [exc,inh]):
            for area in areas:
                mons['spike'+area+pop] = SpikeMonitor(target[area], record=True)
                mons['pop'+area+pop]   = PopulationRateMonitor(target[area], bin=0.1)

        pop = 'E'
        for var in ['x','sNMDA','Istim']:
            mons[var+pop] = StateMonitor(net[pop], var, record=True, clock=clocks['mons'])
        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.p          = p
        self.stimparams = stimparams
        self.net        = net
        self.exc        = exc
        self.inh        = inh
        self.mons       = mons
        self.clocks     = clocks

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects += self.net.values() + self.mons.values()
        self.contained_objects += [recurrent_GABA,recurrent_NMDA,recurrent_AMPA]
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
            self.net[pop].V = np.random.uniform(self.p['neuron']['Vreset'],
                                                self.p['neuron']['Vth'],
                                                size=len(self.net[pop].V))

        # Set synaptic variables to zero
        for var in ['sAMPA', 'x', 'sNMDA']:
            setattr(self.net['E'], var, 0)
        for var in ['sGABA']:
            setattr(self.net['I'], var, 0)

        # Set stimulus
        self.Istim = dict()
        for pop in ['E', 'I']:
            N = self.p['neuron']['N_'+pop]
            dtheta = deg2rad(stimparams['theta_stim'] - np.arange(N)/N*360.)
            self.Istim[pop]  = (stimparams['I0_'+pop] +
            stimparams['I1_'+pop] * np.exp(stimparams['mu']*(np.cos(dtheta)-1)))


#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    dt = 0.2*ms
    T  = 1.5*second

    # Setup the network
    model   = Model(modelparams, stimparams, dt)
    network = Network(model)
    
    # Setup the stimulus parameters for this trial (optional)
    model.stimparams['theta_stim']  = 180
    #model.stimparams['sigma_stim']  = 14
    #model.stimparams['Ipeak']  = 500*pA
    model.reinit(seed=1234)
    network.run(T, report='text')
    
    # Plot results
    plt.figure()
    spike_id, spike_time = zip(*model.mons['spikeE'].spikes)
    plt.plot(spike_time, spike_id, 'o', ms=2, mfc='k', mew=0)
    plt.ylim([min(spike_id),max(spike_id)])
    plt.xlabel('Time (second)')
    plt.ylabel('Neuron index')
    #plt.savefig('workingmemory_ringmodel.pdf')
    
#==============================================================================
# p = model.p
# w = dict()
# for area in areas:
#     JEE_plus = p[area]['JEE_plus'] # PFC and MT the same
#     sigma_EE = deg2rad(p[area]['sigma_EE'])
# 
#     tmp = (2*scipy.stats.norm.cdf(np.pi/sigma_EE)-1)/np.sqrt(2*np.pi)*sigma_EE
#     JEE_minus = (1-JEE_plus*tmp)/(1-tmp)
# 
#     dtheta = 2*np.pi*((np.arange(N_E)+1)/N_E-0.5)
#     w[area] = JEE_minus+((JEE_plus-JEE_minus)*np.exp(-dtheta**2/2./sigma_EE**2))
#==============================================================================

N0 = 16
N1 = 4
d0 = 2*np.pi*((np.arange(N0)+1)/N0-0.5)
d1 = 2*np.pi*((np.arange(N1)+1)/N1-0.5)