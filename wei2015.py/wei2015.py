"""
Role of the indirect pathway of the basal ganglia in perceptual decision making.
W Wei, JE Rubin, & X-J Wang, JNS 2015.

http://dx.doi.org/10.1523/JNEUROSCI.3611-14.2015

This example also demonstrates how to use Python's cPickle module to save and load
complex data.

"""
from __future__ import division

import cPickle as pickle
from collections import defaultdict, OrderedDict

import random as pyrand # Import before Brian floods the namespace

from scipy.sparse import csr_matrix

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

# sAMPA, x, sNMDA, sGABA are synaptic conductances stored pre-synatically
# S_AMPA, S_NMDA, S_GABA are synaptic conductances stored post-synaptically

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
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
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
    ''',

    Str = '''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : pA
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : pA
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA: 1
    S_GABA: 1
    ''',

    GPe = '''
    dV/dt         = (-(V - V_L) - I_T/gI - Isyn/gI) / tau_m_I : mV
    Isyn          = I_AMPA_ext + I_GABA_ext + I_AMPA + I_NMDA + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : pA
    I_GABA_ext    = gGABA_ext_I*sGABA_ext*(V - V_I) : pA
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : pA
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : pA
    I_T           = gT*h*(V>V_h)*(V-V_T) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA_ext/dt = -sGABA_ext/tauGABA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    dh/dt         = -h/tauhminus*(V>=V_h) + (1-h)/tauhplus*(V<V_h) : 1
    S_AMPA: 1
    S_NMDA: 1
    S_GABA: 1
    ''',

    STN = '''
    dV/dt         = (-(V - V_L) - I_T/gE - Isyn/gE) / tau_m_E : mV
    Isyn          = I_AMPA_ext + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : pA
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : pA
    I_T           = gT*h*(V>V_h)*(V-V_T) :pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
    dh/dt         = -h/tauhminus*(V>=V_h) + (1-h)/tauhplus*(V<V_h) :1
    S_GABA : 1
    ''',

    SNr = '''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : mV
    Isyn          = I_AMPA_ext + I_NMDA + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : pA
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_NMDA: 1
    S_GABA: 1
    ''',

    SCE = '''
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : mV
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : pA
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : pA
    I_AMPA        = gAMPA_E*S_AMPA*(V - V_E) : pA
    I_NMDA        = gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsNMDA/dt     = -sNMDA/tauNMDA : 1
    dF/dt         = -F/tauF :1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    ''',

    SCI = '''
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : mV
    Isyn          = I_AMPA_ext + I_NMDA: pA
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : pA
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : pA
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_NMDA: 1
    '''
    )

#=========================================================================================
# Parameters
#=========================================================================================

modelparams = {}

modelparams['Cx'] = dict(
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
    tauNMDA = 100*ms,
    alpha   = 0.63,
    tauGABA = 5*ms,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E = 2.1*nS, # This will be reduced to 2.0 to make the Cx recurrent connection not enough to support persisten activity
    gAMPA_ext_I = 1.62*nS,

    # Unscaled recurrent synaptic conductances (onto excitatory)
    gAMPA_E = 80.0*nS,
    gNMDA_E = 264.0*nS,
    gGABA_E = 520.0*nS,

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
    wp = 1.7,

    gNMDA_SCE_CxE = 0.05*nS, # From SCE to CxE
    gNMDA_SCE_CxI = 0.15*nS  # From SCE to CxE
    )

modelparams['Str'] = dict(
    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Projection LIF
    gI       = 25*nS,
    tau_m_I   = 20*ms,
    tau_ref_PJ = 0*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,


    # Synaptic time constants
    tauAMPA = 2*ms,
    tauGABA = 5*ms,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I = 4.0*nS,
    # Background Possion rate
    nu_ext = 0.8*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I = 3.0*nS, # From Cx
    gNMDA_I = 0.0*nS, # From Cx
    gGABA_I = 1.0*nS, # From within Str

    # Number of neurons
    N_PJ = 250*2
    )

modelparams['SNr'] = dict(
    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Projection LIF
    gI        = 25*nS,
    tau_m_I   = 20*ms,
    tau_ref_PJ = 0*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,

    # Synaptic time constants
    tauAMPA = 2*ms,
    tauGABA = 5*ms,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I = 14*nS,

    # Background Possion rate
    nu_ext = 0.8*kHz,

    # scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I = 0.0*nS,  # From STN
    gNMDA_I = 0.06*nS, # From STN
    gGABA_I = 3.0*nS,  # From Str

    gGABA_GPe_SNr=0.08*nS, # from GPe

     # Number of neurons
    N_PJ = 250*2
    )

modelparams['GPe'] = dict(
    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Projection LIF
    gI         = 25*nS,
    tau_m_I    = 20*ms,
    tau_ref_PJ = 0*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,

    # IFB model parameters
    V_T    = 120*mV,
    V_h    = -60*mV,
    gT     = 60*nS,
    tauhminus = 20*ms,
    tauhplus  = 100*ms,

    # Synaptic time constants
    tauAMPA = 2*ms,
    tauGABA = 5*ms,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I = 3.0*nS,

    # Background Possion rate
    nu_ext_AMPA = 3.2*kHz,

    # External synaptic conductances
    gGABA_ext_I = 2.0*nS,

    # Background Possion rate
    nu_ext_GABA = 2.0*kHz,

    # scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I = 0.05*nS, # From STN
    gNMDA_I = 2.0*nS,  # From STN
    gGABA_I = 4.0*nS,  # From Str

    gGABA_GPe_GPe=1.5*nS,

    # Number of neurons
    N_PJ = 2500*2
    )

modelparams['STN'] = dict(
    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Projection LIF
    gE         = 25*nS,
    tau_m_E    = 20*ms,
    tau_ref_PJ = 0*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,

   # IFB model parameters
    V_T    = 120*mV,
    V_h    = -60*mV,
    gT     = 60*nS,
    tauhminus = 20*ms,
    tauhplus  = 100*ms,

    # Synaptic time constants
    tauAMPA = 2*ms,
    tauNMDA = 100*ms,
    alpha   = 0.63,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E = 1.6*nS,

    # Background Possion rate
    nu_ext = 4.0*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gGABA_E = 0.6*nS, # From GPe

    # Number of neurons
    N_PJ = 2500*2
    )

modelparams['SCE'] = dict(
    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Projection LIF
    gE       = 25*nS,
    tau_m_E   = 20*ms,
    tau_ref_PJ = 0*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,

    # Synaptic time constants
    tauAMPA = 2*ms,
    tauNMDA = 100*ms,
    alpha   = 0.63,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_E = 0.19*nS,

    # Background Possion rate
    nu_ext = 1.28*kHz,

    # Scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_E = 3.5*nS,       # From Cx
    gGABA_E = 2.5*nS,       # From SNr
    gNMDA_E = 1.5*nS,       # From SCE to SCE
    gGABA_SCI_SCE = 2.5*nS, # From SCI to SCE

    # STF parameter
    alpha_F = 0.15,
    tauF    = 1000*ms,

    # Number of neurons
    N_PJ = 250*2
    )

modelparams['SCI'] = dict(
    # Common LIF
    V_L    = -70*mV,
    Vth    = -50*mV,
    Vreset = -55*mV,

    # Projection LIF
    gI       = 25*nS,
    tau_m_I   = 20*ms,
    tau_ref_PJ = 0*ms,

    # Reversal potentials
    V_E = 0*mV,
    V_I = -70*mV,

    # NMDA nonlinearity
    a = 0.062*mV**-1,
    b = 3.57,

    # Synaptic time constants
    tauAMPA = 2*ms,
    tauGABA = 5*ms,
    delay   = 0.2*ms,

    # External synaptic conductances
    gAMPA_ext_I = 2.0*nS,

    # Background Possion rate
    nu_ext = 1.28*kHz,

    # scaled recurrent synaptic conductances (onto projection neurons)
    gAMPA_I = 0.0*nS, # From SCE
    gNMDA_I = 0.7*nS, # From SCE
    gGABA_I = 0.0*nS, # No recurrent SCI -> SCI

    # Number of neurons
    N_PJ = 250
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
            params['Cx'][x] /= params['Cx']['N_E']
        for x in ['gGABA_E', 'gGABA_I']:
            params['Cx'][x] /= params['Cx']['N_I']

        # Make local variables for convenience
        N_E   = params['Cx']['N_E']
        fsel  = params['Cx']['fsel']
        wp    = params['Cx']['wp']
        delay = params['Cx']['delay']
        alpha = params['Cx']['alpha']

        # Subpopulation size
        N1 = int(fsel*N_E)
        N2 = N1
        N0 = N_E - (N1 + N2)
        params['Cx']['N0'] = N0
        params['Cx']['N1'] = N1
        params['Cx']['N2'] = N2

        # Hebb-weakened weight
        wm = (1 - wp*fsel)/(1 - fsel)
        params['Cx']['wm'] = wm

        # Synaptic weights between populations
        self.W = np.asarray([
            [1,  1,  1],
            [wm, wp, wm],
            [wm, wm, wp]
            ])

        #---------------------------------------------------------------------------------
        # Neuron populations
        #---------------------------------------------------------------------------------

        net      = OrderedDict() # Network objects
        netPJsub = OrderedDict() # Projection neuron subpopulations

        for x in ['E', 'I']:
            net['Cx'+x] = NeuronGroup(params['Cx']['N_'+x],
                                 Equations(equations[x], **params['Cx']),
                                 threshold=params['Cx']['Vth'],
                                 reset=params['Cx']['Vreset'],
                                 refractory=params['Cx']['tau_ref_'+x],
                                 clock=clock,
                                 order=2, freeze=True)
        # Excitatory subpopulations
        for x in xrange(3):
            netPJsub['Cx'+str(x)] = net['CxE'].subgroup(params['Cx']['N'+str(x)])

        for x in ['Str', 'SNr', 'GPe', 'STN', 'SCE', 'SCI']:
            net[x] = NeuronGroup(params[x]['N_PJ'],
                                 Equations(equations[x], **params[x]),
                                 threshold=params[x]['Vth'],
                                 reset=params[x]['Vreset'],
                                 refractory=params[x]['tau_ref_PJ'],
                                 clock=clock,
                                 order=2, freeze=True)
            for x1 in xrange(1,3):
                if x != 'SCI':
                   netPJsub[x+str(x1)] = net[x].subgroup(params[x]['N_PJ']//2)

        #---------------------------------------------------------------------------------
        # Background input (post-synaptic)
        #---------------------------------------------------------------------------------

        for x in ['E', 'I']:
            net['pg'+x] = PoissonGroup(params['Cx']['N_'+x], params['Cx']['nu_ext'], clock=clock)
            net['ic'+x] = IdentityConnection(net['pg'+x], net['Cx'+x], 'sAMPA_ext', delay=delay)

        for x in ['Str', 'SNr', 'STN', 'SCE','SCI']:
            net['pg'+x] = PoissonGroup(params[x]['N_PJ'], params[x]['nu_ext'], clock=clock)
            net['ic'+x] = IdentityConnection(net['pg'+x], net[x], 'sAMPA_ext', delay=delay)

        net['pg'+'GPe_AMPA'] = PoissonGroup(params['GPe']['N_PJ'], params['GPe']['nu_ext_AMPA'], clock=clock)
        net['ic'+'GPe_AMPA'] = IdentityConnection(net['pg'+'GPe_AMPA'], net['GPe'], 'sAMPA_ext', delay=delay)
        net['pg'+'GPe_GABA'] = PoissonGroup(params['GPe']['N_PJ'], params['GPe']['nu_ext_GABA'], clock=clock)
        net['ic'+'GPe_GABA'] = IdentityConnection(net['pg'+'GPe_GABA'], net['GPe'], 'sGABA_ext', delay=delay)

        #---------------------------------------------------------------------------------
        # Recurrent input
        #---------------------------------------------------------------------------------

        # Change pre-synaptic variables
        for x in ['CxI', 'Str', 'SNr', 'GPe','SCI']:
            net['icGABA_'+x] = IdentityConnection(net[x], net[x], 'sGABA', delay=delay)
        for x in ['CxE', 'STN']:
            net['icAMPA_NMDA_'+x] = Synapses(net[x], pre='''sAMPA+=1
                                                            sNMDA+=alpha*(1-sNMDA)''')
            net['icAMPA_NMDA_'+x].delay=delay
            net['icAMPA_NMDA_'+x].connect_one_to_one(net[x], net[x])

        # SCE
        alpha_F=params['SCE']['alpha_F']
        net['icNMDA_SCE'] = Synapses(net['SCE'], pre='''sNMDA+=alpha*(1-sNMDA)
                                                                    F+=alpha_F*(1-F)''')
        net['icNMDA_SCE'].delay=delay
        net['icNMDA_SCE'].connect_one_to_one(net['SCE'], net['SCE'])

        # sparse recurrent connection
        prob_GPe_GPe=0.05
        prob_GPe_STN=0.02  # GPe to STN
        prob_STN_GPe=0.05  # STN to GPe

        N_PJ1=params['GPe']['N_PJ']//2

        # here the seed 100 is choosed arbitarily
        rns = np.random.RandomState(100)

        conn_GPe_GPe = 1*(rns.random_sample((N_PJ1,N_PJ1))<prob_GPe_GPe)
        conn_GPe_STN = 1*(rns.random_sample((N_PJ1,N_PJ1))<prob_GPe_STN)
        conn_STN_GPe = 1*(rns.random_sample((N_PJ1,N_PJ1))<prob_STN_GPe)

        self.sconn_GPe_GPe = csr_matrix(conn_GPe_GPe)
        self.sconn_GPe_STN = csr_matrix(conn_GPe_STN)
        self.sconn_STN_GPe = csr_matrix(conn_STN_GPe)

        self.wGABA_GPe_SNr = params['SNr']['gGABA_GPe_SNr']/params['SNr']['gGABA_I']
        self.wGABA_GPe_GPe = params['GPe']['gGABA_GPe_GPe']/params['GPe']['gGABA_I']
        self.wNMDA_SCE_CxE = params['Cx']['gNMDA_SCE_CxE']/params['Cx']['gNMDA_E']
        self.wNMDA_SCE_CxI = params['Cx']['gNMDA_SCE_CxI']/params['Cx']['gNMDA_I']
        self.wGABA_SCI_SCE = params['SCE']['gGABA_SCI_SCE']/params['SCE']['gGABA_E']

        # Link pre-synaptic variables to post-synaptic variables
        @network_operation(when='start', clock=clock)
        def recurrent_input():
            SAMPA = defaultdict(dict)
            SNMDA = defaultdict(dict)
            SGABA = defaultdict(dict)

            for x in ['Cx', 'STN']:
                for i in ['1','2']:
                    SAMPA[i][x] = self.netPJsub[x+i].sAMPA.sum()
                    SNMDA[i][x] = self.netPJsub[x+i].sNMDA.sum()

            for x in ['Str', 'SNr', 'GPe']:
                for i in ['1','2']:
                    SGABA[i][x] = self.netPJsub[x+i].sGABA.sum()

            for i in ['1','2']:
                SNMDA[i]['SCE'] = self.netPJsub['SCE'+i].sNMDA.sum()


            SGABA['SCI'] = self.net['SCI'].sGABA.sum()

            SCx0_AMPA = self.netPJsub['Cx0'].sAMPA.sum()
            SCx0_NMDA = self.netPJsub['Cx0'].sNMDA.sum()

            S = self.W.dot([SCx0_AMPA, SAMPA['1']['Cx'], SAMPA['2']['Cx']])
            for i in xrange(3):
                self.netPJsub['Cx'+str(i)].S_AMPA = S[i]
            self.net['CxI'].S_AMPA = S[0]

            # NMDA
            S = self.W.dot([SCx0_NMDA, SNMDA['1']['Cx'], SNMDA['2']['Cx']])
            for i in xrange(3):
                self.netPJsub['Cx'+str(i)].S_NMDA = S[i]
            self.net['CxI'].S_NMDA = S[0] + self.wNMDA_SCE_CxI*(SNMDA['1']['SCE']+SNMDA['2']['SCE'])

            # GABA
            S = self.net['CxI'].sGABA.sum()
            self.net['CxE'].S_GABA = S
            self.net['CxI'].S_GABA = S

            for i in ['1','2']:
                # For SCE -> SCI
            	SNMDA[i]['SCE_F']=dot(self.netPJsub['SCE'+i].F, self.netPJsub['SCE'+i].sNMDA)

                self.netPJsub['Cx'+i].S_NMDA += self.wNMDA_SCE_CxE*(SNMDA['1']['SCE']+SNMDA['2']['SCE'])

                # Str
                self.netPJsub['Str'+i].S_AMPA = SAMPA[i]['Cx']
                self.netPJsub['Str'+i].S_GABA = SGABA[i]['Str']

                # SNr
                self.netPJsub['SNr'+i].S_NMDA = SNMDA[i]['STN']
                self.netPJsub['SNr'+i].S_GABA = SGABA[i]['Str'] + self.wGABA_GPe_SNr*SGABA[i]['GPe']

                # GPe
                self.netPJsub['GPe'+i].S_AMPA = self.sconn_STN_GPe.dot(self.netPJsub['STN'+i].sAMPA)
                self.netPJsub['GPe'+i].S_NMDA = self.sconn_STN_GPe.dot(self.netPJsub['STN'+i].sNMDA)
                self.netPJsub['GPe'+i].S_GABA = self.sconn_GPe_GPe.dot(self.netPJsub['GPe'+i].sGABA)*self.wGABA_GPe_GPe + SGABA[i]['Str']
                # STN:
                self.netPJsub['STN'+i].S_GABA = self.sconn_GPe_STN.dot(self.netPJsub['GPe'+i].sGABA)

                # SC
                self.netPJsub['SCE'+i].S_AMPA = SAMPA[i]['Cx']
                self.netPJsub['SCE'+i].S_NMDA = SNMDA[i]['SCE']
                self.netPJsub['SCE'+i].S_GABA = SGABA[i]['SNr']+ self.wGABA_SCI_SCE*SGABA['SCI']
            self.net['SCI'].S_NMDA = SNMDA['1']['SCE_F']+SNMDA['2']['SCE_F']

        #---------------------------------------------------------------------------------
        # External input (post-synaptic)
        #---------------------------------------------------------------------------------

        for i, stimulus in zip([1, 2], [stimulus.s1, stimulus.s2]):
            net['pg'+str(i)] = PoissonGroup(params['Cx']['N'+str(i)], stimulus, clock=clock)
            net['ic'+str(i)] = IdentityConnection(net['pg'+str(i)], netPJsub['Cx'+str(i)],
                'sAMPA_ext', delay=delay)

        #---------------------------------------------------------------------------------
        # Record rates
        #---------------------------------------------------------------------------------

        rates = OrderedDict()
        for x in netPJsub:
            rates[x] = PopulationRateMonitor(netPJsub[x], bin=2*ms)

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        self.params   = params
        self.net      = net
        self.netPJsub = netPJsub
        #self.mons     = mons
        self.rates    = rates

        # Add network objects and monitors to NetworkOperation's contained_objects
        self.contained_objects += self.net.values() + self.rates.values()
        self.contained_objects += [recurrent_input]

    def reinit(self):
        # Reset network components
        for n in self.net.values() + self.rates.values():
            n.reinit()

        # Randomly initialize membrane potentials
        for x in ['E', 'I']:
            self.net['Cx'+x].V = np.random.uniform(self.params['Cx']['Vreset'], self.params['Cx']['Vth'],
                                              size=self.params['Cx']['N_'+x])
        for x in ['Str', 'SNr', 'GPe', 'STN', 'SCE', 'SCI']:
            self.net[x].V = np.random.uniform(self.params[x]['Vreset'], self.params[x]['Vth'],
                                              size=self.params[x]['N_PJ'])

        # Set synaptic variables to zero
        for i in ['CxE', 'STN']:
            for x in ['sAMPA_ext', 'sAMPA', 'sNMDA']:
                setattr(self.net[i], x, 0)
        for i in ['CxI', 'Str', 'SNr', 'SCI']:
            for x in ['sAMPA_ext', 'sGABA']:
                setattr(self.net[i], x, 0)
        for x in ['sAMPA_ext', 'sNMDA']:
            setattr(self.net['SCE'], x, 0)
        for x in ['sAMPA_ext', 'sGABA_ext', 'sGABA']:
            setattr(self.net['GPe'], x, 0)
        setattr(self.net['SCE'], 'F', 0)
        setattr(self.net['GPe'], 'h', 1)
        setattr(self.net['STN'], 'h', 1)

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

    def saverates(self, filename, smooth=10*ms):
        time  = self.model.rates['Cx1'].times/ms
        rates = {}
        for name in ['Cx', 'Str', 'SNr', 'GPe', 'STN', 'SCE']:
            rates[name+'1'] = self.model.rates[name+'1'].smooth_rate(smooth)/Hz
            rates[name+'2'] = self.model.rates[name+'2'].smooth_rate(smooth)/Hz

        with open(filename, 'wb') as f:
            pickle.dump((time, rates), f)

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    stimparams = dict(
        Ton  = 0.5*second, # Stimulus onset
        Toff = 1.5*second, # Stimulus offset
        mu0  = 40*Hz,      # Input rate
        coh  = 0           # Percent coherence
        )

    dt = 0.05*ms
    T  = 2*second

    sim = Simulation(modelparams, stimparams, dt)
    sim.stimulus.set_coh(25.6) # Shows how coherence can be changed
    sim.run(T, seed=10)
    sim.saverates('rates.pkl')

    #-------------------------------------------------------------------------------------
    # Plot firing rates in different areas
    #-------------------------------------------------------------------------------------

    # Load firing rates
    with open('rates.pkl') as f:
        time, rates = pickle.load(f)

    # Align time to stimulus onset
    time -= stimparams['Ton']/ms

    import matplotlib.pyplot as plt

    w  = 0.23
    h  = 0.36
    dx = 0.08
    dy = 0.1
    x1 = 0.1
    x2 = x1+w+dx
    x3 = x2+w+dx
    y1 = 0.11
    y2 = y1+h+dy

    # Figure setup
    fig   = plt.figure()
    plots = {
        'GPe': fig.add_axes([x1, y1, w, h]),
        'STN': fig.add_axes([x2, y1, w, h]),
        'SCE': fig.add_axes([x3, y1, w, h]),
        'Cx':  fig.add_axes([x1, y2, w, h]),
        'Str': fig.add_axes([x2, y2, w, h]),
        'SNr': fig.add_axes([x3, y2, w, h])
        }
    for name, plot in plots.items():
        plot.set_title(name)
    plots['GPe'].set_xlabel('Time from stimulus (ms)')
    plots['GPe'].set_ylabel('Firing rate (Hz)')

    for name, plot in plots.items():
        plot.plot(time, rates[name+'1'], 'g', zorder=5)
        plot.plot(time, rates[name+'2'], 'b', zorder=5)
        plot.set_xlim(-100, 700)
        plot.set_xticks([0, 500])

    plots['Cx'].set_ylim(0, 20)
    plots['Str'].set_ylim(0, 35)
    plots['SNr'].set_ylim(0, 150)
    plots['GPe'].set_ylim(0, 100)
    plots['STN'].set_ylim(0, 100)
    plots['SCE'].set_ylim(0, 250)

    print("Saving plot to wei2015.pdf")
    plt.savefig('wei2015.pdf')
