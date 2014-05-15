from __future__ import division
import cProfile, itertools, time, timeit
import numpy as np


from beams.bunch import *
from beams import slices
from monitors.monitors import *
from aperture.aperture import *
from impedances.wake_fields  import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *
from plots import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants


# simulation setup

# Variables
#  beta_x,y                                   average transverse beta functions along turn.
#  bunch_length
#  momentum spread
#  epsn_x,y                                   transverse (x,y) beam size in [um]
#  gamma_t                                    gamma transition
#  C                                          circumference of accel. (1 turn)
#  energy                                     total beam energy
#  n_macroparticles                           number of simulated particle 'clouds'
#  n_particles                                actual number of particles in bunch
#  n_slices                                   longitudinal binning (# of bins)
#  Qx,y                                       in R.R.: nu_x,y. Number of betatron osc. in one turn.
#                                             (aka betatron tune)
#  Qs                                         synchrotron tune. Number of synchrotron osc. in one turn.
#  Qp_x,y                                     chromaticity (??)
charge = e
mass = m_p
#~ n_particles = 1.15e11 
beta_x = 54.6408 # [m]
beta_y = 54.5054 # [m]
bunch_length = 0.192368 # [m]
momentum_spread = 0.00166945
epsn_x = 20.0 # orig. 2 [um]
epsn_y = 20.0 # orig. 2 [um]
gamma_t = 1/np.sqrt(0.0031)
C = 6911. # orig. 6911. # [m] # circumference of acc. (reference trajectory)
energy = 26e9 # total [eV]
n_turns = 128
nsigmaz = 3
Qx = 20.15 # orig 20.15
Qy = 20.15 # orig 20.15
Qs = 0.017 # orig 0.017
Qp_x = 0
Qp_y = 500.
n_macroparticles = 10000
n_particles = 1.15e11
n_slices = 100
R_frequency = 1.0e9 # [Hz]
Q = 1.
R_shunt = 10e6 # [Ohm/m]
initial_kick_x = 0.1*np.sqrt(beta_x * epsn_x*1.e-6 / (energy / 0.938e9))
initial_kick_y = 1*np.sqrt(beta_y * epsn_y*1.e-6 / (energy / 0.938e9))
RF_voltage = 4e6 # [V]
harmonic_number = 4620
Yokoya_X1 = np.pi**2/24
Yokoya_X2 = 0. #-np.pi**2/24
Yokoya_Y1 = 0. #np.pi**2/12
Yokoya_Y2 = 0. #np.pi**2/24


# Betatron
n_segments = 1
s = np.arange(1, n_segments + 1) * C / n_segments
linear_map = TransverseTracker.from_copy(s,
                               np.zeros(n_segments),
                               np.ones(n_segments) * beta_x,
                               np.zeros(n_segments),
                               np.zeros(n_segments),
                               np.ones(n_segments) * beta_y,
                               np.zeros(n_segments),
                               Qx, Qp_x, 0, Qy, Qp_y, 0)


# Synchrotron motion
# cavity = CSCavity(C, gamma_t, Qs)
# cavity = RFCavity(C, C, gamma_t, harmonic_number, RF_voltage, 0, integrator='rk4')


# Bunch
bunch = bunch_matched_and_sliced(n_macroparticles, n_particles, charge, energy, mass,
                                 epsn_x, epsn_y, linear_map[0], bunch_length, 0.35, matching=None,
                                 n_slices=n_slices, nsigmaz=nsigmaz, slicemode='cspace')

# bunch =  bunch_unmatched_inbucket_sliced(n_macroparticles, n_particles, charge, energy, mass,
#                                          epsn_x, epsn_y, linear_map[0], bunch_length, momentum_spread, bucket=cavity,
#                                          n_slices=n_slices, nsigmaz=nsigmaz, slicemode='cspace')                       

# initial transverse kicks
# bunch.x += initial_kick_x
# bunch.x += 0.01
# bunch.y += 0.01

# # save initial distribution
# ParticleMonitor('initial_distribution').dump(bunch)

# distribution from file
# bunch = bunch_from_file('initial_distribution', 0, n_particles, charge, energy, mass, n_slices, nsigmaz, slicemode='cspace')


# Monitors
bunchmonitor = BunchMonitor('bunch', n_turns)
# particlemonitor = ParticleMonitor('particles', 200, simulation_parameters_dict, 1024)


# Resonator wakefields
# wakes = BB_Resonator_transverse(R_shunt=R_shunt, frequency=R_frequency, Q=Q, Yokoya_X1=Yokoya_X1, Yokoya_Y1=Yokoya_Y1, Yokoya_X2=Yokoya_X2, Yokoya_Y2=Yokoya_Y2)
#~ wakes = BB_Resonator_Circular(R_shunt=R_shunt, frequency=R_frequency, Q=Q)


# accelerator map
# map_ = linear_map + [cavity]
map_ = linear_map

plt.ion()
for i in range(n_turns):
    bunch.compute_statistics()
    t0 = time.clock() 
    for m in map_:
        m.track(bunch)

    plt.cla()
    # plt.scatter(bunch.dz, bunch.dp, marker='.')   # longitudinal phase space
    # plt.scatter(bunch.y, bunch.yp, marker = '.')  # transverse phase space (y)
    # plt.scatter(bunch.y[-1], bunch.yp[-1], c='r')  # mark specific macroparticle in bunch
    plt.scatter(bunch.x, bunch.y, marker = '.')  # transverse space (x,y)
    plt.scatter(bunch.x[-1], bunch.y[-1], c='r') # mark specific macroparticle in bunch

    plt.xlabel('x [$\mu m$]')
    plt.ylabel('y [$\mu m$]')
    
    # transverse y phase space limits
    # plt.xlim(-1e-2, 1e-2)
    # plt.ylim(-2e-4, 2e-4)
    
    # transverse (x,y) space limits
    plt.xlim(-2.e-2, 2.e-2)
    plt.ylim(-2.e-2, 2.e-2)

    plt.draw()
    bunchmonitor.dump(bunch)
    # if i < 1024:
        # particlemonitor.dump(bunch)

    print '{0:4d} \t {1:+3e} \t {2:+3e} \t {3:+3e} \t {4:3e} \t {5:3e} \t {6:3f} \t {7:3f} \t {8:3f} \t {9:4e} \t {10:3s}'.format(i, bunch.slices.mean_x[-2], bunch.slices.mean_y[-2], bunch.slices.mean_dz[-2], bunch.slices.epsn_x[-2], bunch.slices.epsn_y[-2], bunch.slices.epsn_z[-2], bunch.slices.sigma_dz[-2], bunch.slices.sigma_dp[-2], bunch.slices.n_macroparticles[-2] / bunch.n_macroparticles * bunch.n_particles, str(time.clock() - t0))


# # dictionary of simulation parameters
# simulation_parameters_dict = {'comment': 'This is a broadband resonator with only a horizontal dipole wake',\
#                               'charge': charge,\
#                               'mass': mass,\
#                               'n_particles': n_particles,\
#                               'beta_x': beta_x,\
#                               'beta_y': beta_y,\
#                               'bunch_length': bunch_length,\
#                               'momentum_spread': momentum_spread,\
#                               'epsn_x': epsn_x,\
#                               'epsn_y': epsn_y,\
#                               'gamma_t': gamma_t,\
#                               'C': C,\
#                               'energy': energy,\
#                               'n_turns': n_turns,\
#                               'nsigmaz': nsigmaz,\
#                               'Qx': Qx,\
#                               'Qy': Qy,\
#                               'Qs': Qs,\
#                               'Qp_x': Qp_x,\
#                               'Qp_y': Qp_y,\
#                               'n_macroparticles': bunch.n_macroparticles,\
#                               'n_slices': n_slices,\
#                               'R_frequency': R_frequency,\
#                               'Q': Q,\
#                               'R_shunt': R_shunt,\
#                               'initial_kick_x': initial_kick_x,\
#                               'initial_kick_y': initial_kick_y,\
#                               'RF_voltage': RF_voltage,\
#                               'harmonic_number': harmonic_number,\
#                               'Yokoya_X1': Yokoya_X1,\
#                               'Yokoya_X2': Yokoya_X2,\
#                               'Yokoya_Y1': Yokoya_Y1,\
#                               'Yokoya_Y2': Yokoya_Y2}
