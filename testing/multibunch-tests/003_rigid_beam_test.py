'''
@authors: Jani Komppula
@date:    23/01/2018

Compares results from a PyHEADTAIL simulation without longitudinal tracking
to results from the independent minimalistic rigid bunch/beam code with wakes.
'''

from __future__ import division

import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

import numpy as np
import seaborn as sns
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p
from collections import deque

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResistiveWall, CircularResonator, WakeField
from PyHEADTAIL.machines.synchrotron import Synchrotron


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
np.random.seed(rank)


def gather_beam_init_data(allbunches, slicer):
    # Gathers PyHEADTAIL bunch data for all processors for the rigid beam
    # initialization
    
    comm = MPI.COMM_WORLD
    
    beam = comm.allgather(allbunches)
    if rank == 0:
        beam = sum(beam)
        beam = beam.split()
        slice_sets = []
        for b in beam:
            s = b.get_slices(slicer, statistics=['mean_x', 'mean_y', 'mean_xp', 'mean_yp'])
            slice_sets.append(s)
            
        return beam, slice_sets
    else:
        return 0, 0


def gather_beam_ref_data(allbunches, slicer, filling_sceme, circumference,
                         h_bunch, n_slices_per_bunch, slicing_fraction):
    # Gathers PyHEADTAIL bunch data for all processors for plotting
    
    comm = MPI.COMM_WORLD
    
    beam = comm.allgather(allbunches)
    if rank == 0:
        beam = sum(beam)
        beam = beam.split()
        slice_sets = []
        for b in beam:
            s = b.get_slices(slicer, statistics=['mean_x', 'mean_y', 'mean_xp', 'mean_yp'])
            slice_sets.append(s)
            
        n_total_slices = int(float(h_bunch)*float(n_slices_per_bunch)/float(slicing_fraction))
        n_slices_per_bucket = int(n_slices_per_bunch/float(slicing_fraction))
        
        slice_width = circumference/float(n_total_slices)
        z = np.linspace(0, circumference-slice_width, n_total_slices)
        
        x = np.zeros(n_total_slices)
        y = np.zeros(n_total_slices)
    
        for idx, slice_set, bunch in zip(filling_scheme, slice_sets, beam):
            i_from = int(idx*n_slices_per_bucket + (n_slices_per_bucket-n_slices_per_bunch)/2)
            i_to = i_from + n_slices_per_bunch
            
            np.copyto(x[i_from:i_to],slice_set.mean_x)
            np.copyto(y[i_from:i_to],slice_set.mean_y)
        
        return z, x, y
    else:
        return 0, 0, 0

def generate_point_beam(filling_sceme, beam, slice_sets, circumference, h_bunch, n_slices_per_bunch, slicing_fraction):
    # generates rigid beam object from PyHEADTAIL bunches and slice sets
    
    n_total_slices = int(float(h_bunch)*float(n_slices_per_bunch)/float(slicing_fraction))
    n_slices_per_bucket = int(n_slices_per_bunch/float(slicing_fraction))
    
    slice_width = circumference/float(n_total_slices)
    z = np.linspace(0, circumference-slice_width, n_total_slices)
    x = np.zeros(n_total_slices)
    y = np.zeros(n_total_slices)
    xp = np.zeros(n_total_slices)
    yp = np.zeros(n_total_slices)
    n_mp = np.zeros(n_total_slices)

    for idx, slice_set, bunch in zip(filling_scheme, slice_sets, beam):
        i_from = int(idx*n_slices_per_bucket + (n_slices_per_bucket-n_slices_per_bunch)/2)
        i_to = i_from + n_slices_per_bunch
        
        np.copyto(x[i_from:i_to],slice_set.mean_x)
        np.copyto(xp[i_from:i_to],slice_set.mean_xp)
        np.copyto(y[i_from:i_to],slice_set.mean_y)
        np.copyto(yp[i_from:i_to],slice_set.mean_yp)
        np.copyto(n_mp[i_from:i_to],slice_set.n_macroparticles_per_slice)

    beam_x = RigidBeam(z,x,xp,n_mp)
    beam_y = RigidBeam(z,y,yp,n_mp)
    
    return beam_x, beam_y

class RigidBeam(object):
    # A minimalistic point-like-bunch/rigid-slice simulations which results are
    # compared into a PyHEADTAIL simulation without longitudinal tracking
    #
    # In this RigidBeam simulation the entire synchrotron is sliced uniformally
    # and each slice has z, x, xp and number of macroparticle properties. On
    # each turn x and xp coordinates have been changed by applying linear
    # transverse betatron motion. Wakes are applied by calculating a circular 
    # covolution between a beam and one turn lengh wake function
    
    def __init__(self, z, x, xp, n_mp):
        
        # slice coordinates and charge distribution
        self.z = z
        self.x = x
        self.xp = xp
        self.n_macroparticles_per_slice = n_mp
        
        # A map for the wakes, i.e. xp coordinates are changed only for slices
        # which includes "macroparticles"
        self.filled_slices = (self.n_macroparticles_per_slice>0)
    
        
    def init_accelerator_parameters(self, p0, circumference, Q_x, beta_x):
        # init accelerator parameters required by transverse tracking
        
        self.circumference = circumference
        self.Q_x = Q_x
        self.beta_x = beta_x
        
        self.gamma = np.sqrt(1 + (p0 / (m_p * c))**2)
        self.beta_beam = np.sqrt(1 - self.gamma**-2)
        
    def initPyHEADTAILwakes(self, wakes, n_turn_wakes, intensity, n_macroparticles, n_slices_per_bunch, slicing_fraction):
        # init the wake functions for the convolutions by extracting them from
        # a PyHEADTAIL wake object. The input argument intensity is a number
        # of protons per bunch and n_macroparticles is the total number of
        # macroparticles per bunch
        
        n_slices_per_bucket = int(n_slices_per_bunch/slicing_fraction)
        n_roll = int(n_slices_per_bucket/2)
        
        # The memory optimized version can handle low beta-beam wakes
        # automatically but they are not active at the moment. In order to get
        # FFT convolution work correctly, the wake values for the negative
        # time vakues must rolled to the end of the array. This small hack has
        # negligible effects to the physics but it allows an order of 1e-12 
        # value comparison after 2000 turns between, this rigid beam code and
        # the PyHEADTAIL wakes ('mpi_full_ring_fft' and 'memory_optimized').
        z_values = np.roll(self.z,-n_roll)
        z_values = z_values - z_values[0]
        
        self.n_turn_wakes = n_turn_wakes
        wf = wakes.function_transverse(1)
        
        # the classical PyHEADTAIL wake factor
        self.wake_factor = intensity/float(n_macroparticles)*(-(e)**2 / (m_p * self.gamma * (self.beta_beam * c)**2))
        
        self._turn_by_turn_wake_functions = []
        turn_length = self.circumference/c
        
        self._acculumated_kicks = deque(maxlen=self.n_turn_wakes)
        for i in range(self.n_turn_wakes):
            self._acculumated_kicks.append(np.zeros(len(self.z)))
            t_values = z_values/c + float(i)*turn_length
            self._turn_by_turn_wake_functions.append(np.array(wf(-t_values, beta=self.beta_beam)))
            
            
            if i == 0:
                # Low beta beam wakes are not implemented at the moment
                # thus values for the negative times are set to zero
                negative_map = (t_values < 0.)
                self._turn_by_turn_wake_functions[0][negative_map] = 0.
        
    def rotate(self):
        # linear transverse tracking
        ss = np.sin(2.*np.pi*self.Q_x)
        cc = np.cos(2.*np.pi*self.Q_x)
        
        new_x = cc * self.x + self.beta_x * ss * self.xp
        new_xp = (-1. / self.beta_x) * ss * self.x + cc * self.xp
        
        np.copyto(self.x, new_x)
        np.copyto(self.xp, new_xp)

    def apply_wakes(self):       
        source = np.copy(self.x*self.n_macroparticles_per_slice)
        source = source[::-1]

        # calculates turn by turn circular convolutions for wake kicks
        for i, wake in enumerate(self._turn_by_turn_wake_functions):
            kick = np.real(np.fft.ifft(np.fft.fft(source) * np.fft.fft(wake)))
            # the total wake kick is accumulated by adding the calculated values to
            # the values from previous tracking turns. The index i+1 comes from the property
            # of a deque object, i.e. the last kick is added to the end of the deque
            # which pops out the first value.
            if i < (self.n_turn_wakes-1):
                self._acculumated_kicks[i+1] += kick
            else:
                self._acculumated_kicks.append(kick)
        
        self.xp[self.filled_slices] = self.xp[self.filled_slices] + self.wake_factor*self._acculumated_kicks[0][::-1][self.filled_slices]



# MACHINE AND SIMULATION SETTINGS
#================================

n_turns = 1500

n_macroparticles = 1000 # per bunch 
intensity = 2.3e11
intensity = 3e14

alpha = 53.86**-2

p0 = 7000e9 * e / c

accQ_x = 62.31
accQ_y = 60.32
Q_s = 2.1e-3
chroma=0

h_bunch = 53
h_RF = h_bunch*10

circumference = 25e-9*c*h_bunch

beta_x = circumference / (2.*np.pi*accQ_x)
beta_y = circumference / (2.*np.pi*accQ_y)

epsn_x = 2e-6
epsn_y = 2e-6
sigma_z = 0.09

machine = Synchrotron(
        optics_mode='smooth', circumference=circumference,
        n_segments=1, s=None, name=None,
        alpha_x=None, beta_x=beta_x, D_x=0,
        alpha_y=None, beta_y=beta_y, D_y=0,
        accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=chroma, Qp_y=chroma,
        app_x=0, app_y=0, app_xy=0,
        alpha_mom_compaction=alpha, longitudinal_mode='linear',
        h_RF=np.atleast_1d(h_RF), p0=p0,
        charge=e, mass=m_p, wrap_z=False, Q_s=Q_s)

# Removes the longitudinal map from the one turn map
machine.one_turn_map = machine.one_turn_map[1:]


# FILLING SCHEME
#===============
filling_scheme = [] # A list of filled buckets

# -- Option 1: Fully filled
for i in range(h_bunch):
    filling_scheme.append(i)

# -- Option 2: filling scheme
#for i in range(1):
#    for j in range(42):
#        filling_scheme.append(i*50+j)

# SLICING OPTIONS
#=================
bunch_scaping = circumference/float(h_bunch)

# -- Option 1: Point like bunches
slicing_fraction = 1./1. # a fraction of bunch spacing sliced
n_slices_per_bunch = 1 # a number of PyHEADTAIL slices per bunch

# -- Option 2: Multiple slices per bunch
#slicing_fraction = 1./10. # a fraction of bunch spacing sliced
#n_slices_per_bunch = 8 # a number of PyHEADTAIL slices per bunch


allbunches = machine.generate_6D_Gaussian_bunch(n_macroparticles, intensity,
                                                epsn_x, epsn_y, sigma_z=sigma_z,
                                                filling_scheme=filling_scheme,
                                                matched=False)

slicer = UniformBinSlicer(n_slices_per_bunch, z_cuts=(-0.5*bunch_scaping*slicing_fraction, 0.5*bunch_scaping*slicing_fraction),
                               circumference=machine.circumference, h_bunch=h_bunch)


# PyHEADTAIL WAKES
#=================

mpi_settings = 'mpi_full_ring_fft'
#mpi_settings = 'memory_optimized'
n_turns_wake = 2

# pipe radius [m]
b = 13.2e-3
# length of the pipe [m]
L=100000.
# conductivity of the pipe 1/[Ohm m]
sigma = 1./(7.88e-10)

wakes = CircularResistiveWall(b,L,sigma,b/c,beta_beam=machine.beta, n_turns_wake=n_turns_wake)
#wakes = CircularResonator(135e6, 1.97e5, 31000, n_turns_wake=n_turns_wake)

wake_field = WakeField(slicer, wakes, mpi=mpi_settings)
machine.one_turn_map.append(wake_field)

# TRACKING AND PLOTTING
#======================

beam, slice_sets = gather_beam_init_data(allbunches, slicer)
if rank == 0:
    # inits beams and figures only on rank 0
    beam_x, beam_y = generate_point_beam(filling_scheme, beam, slice_sets,
                                         circumference, h_bunch, 
                                         n_slices_per_bunch, slicing_fraction)
    beam_x.init_accelerator_parameters(p0, circumference, accQ_x, beta_x)
    beam_y.init_accelerator_parameters(p0, circumference, accQ_y, beta_y)
    
    # Wake function values for the rigid beam are riden from the PyHEADTAIL
    # wake object
    beam_x.initPyHEADTAILwakes(wakes, n_turns_wake, intensity, n_macroparticles,
                               n_slices_per_bunch, slicing_fraction)
    beam_y.initPyHEADTAILwakes(wakes, n_turns_wake, intensity, n_macroparticles,
                               n_slices_per_bunch, slicing_fraction)


    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("X-plane")
    ax1.set_xlabel('Z-location [m]')
    ax1.set_ylabel('Bunch mean_x')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Y-plane")
    ax2.set_xlabel('Z-location [m]')
    ax2.set_ylabel('Bunch mean_y')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlabel('Z-location [m]')
    ax3.set_ylabel('Slice-by-slice difference [%]')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlabel('Z-location [m]')
    ax4.set_ylabel('Slice-by-slice difference [%]')
  


for i in range(n_turns):

    if (i == 0) or (i == n_turns - 1):
        # Beam oscillations from the first and last turns are plotted
        
        # Gathers bunch postion data from all processors
        z, x, y = gather_beam_ref_data(allbunches, slicer, filling_scheme, circumference,
                                 h_bunch, n_slices_per_bunch, slicing_fraction)
        
        if (rank == 0) and (i == 0):
            print('Plotting the first turn')
            ax1.plot(z,x, 'b-', label='PyHEADTAIL, turn 0')
            ax1.plot(beam_x.z, beam_x.x, 'r--', label='Rigid beam, turn 0')
            ax2.plot(z,y, 'b-', label='PyHEADTAIL, turn 0')
            ax2.plot(beam_y.z, beam_y.x, 'r--', label='Rigid beam, turn 0')
            
            ax3.plot(z, (beam_x.x-x)/np.max(x), label='Difference, turn 0')
            ax4.plot(z, (beam_y.x-y)/np.max(y), label='Difference, turn 0')
        elif (rank == 0) and (i == n_turns-1):
            print('Plotting the last turn')
            ax1.plot(z,x, 'g-', label='PyHEADTAIL, turn ' + str(n_turns))
            ax1.plot(beam_x.z, beam_x.x, '--', color='orange', label='Rigid beam, turn ' + str(n_turns))
            ax2.plot(z,y, 'g-', label='PyHEADTAIL, turn ' + str(n_turns))
            ax2.plot(beam_y.z, beam_y.x, '--', color='orange', label='Rigid beam, turn ' + str(n_turns))
            
            ax3.plot(z, (beam_x.x-x)/np.max(x)*100., label='Difference, turn ' + str(n_turns))
            ax4.plot(z, (beam_y.x-y)/np.max(y)*100., label='Difference, turn ' + str(n_turns))
    
    # Normal PyHEADTAIL tracking
    machine.track(allbunches)
    
    if rank == 0:
        # Rigid beam tracking on the rank 0
        beam_x.rotate()
        beam_y.rotate()
        beam_x.apply_wakes()
        beam_y.apply_wakes()
        
if rank == 0:
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    plt.show()