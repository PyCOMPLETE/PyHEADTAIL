from __future__ import division
import cProfile, itertools, time, timeit
import numpy as np

from trackers.detuners import *
#from rfq.rfq import *
from particles.particles import *
from particles.slicer import *
from monitors.monitors import *
from aperture.aperture import *
from impedances.wake_fields  import *
from trackers.transverse_tracker_2 import *
from trackers.longitudinal_tracker import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants

#import pylab as plt

#@profile
def main():
    # ==============================================================================================
    # SIMULATION SETUP.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # LHC @4TeV, parameters taken from list LHC_4TeV_2012_v2_lin.cfg

    i_oct_fd = 100

    # PHYSICS AND MACHINE PARAMETERS.
    intensity = 4.e11                           # Number of particles (protons) per bunch.
    charge    = e                               # Charge of a proton.
    mass      = m_p                             # Mass of a proton.

    sigma_z   = 0.0936851405476                 # Bunch length (RMS) [m].
    gamma     = 4263.15613303                   # Relativistic gamma.
    alpha_0   = 0.0003225                       # Momentum compaction factor.
    eta       = alpha_0 - 1./gamma**2           # Slippage factor.
    gamma_t   = 1./np.sqrt(alpha_0)             # Transition gamma.

    p0 = np.sqrt(gamma**2 - 1) * mass * c       # Momentum.
    
    Q_s       = 0.00234243399047                # Synchrotron tune.
    Q_x       = 64.31                           # Betatron tune (horizontal).
    Q_y       = 59.32                           # Betatron tune (vertical).
    
    C         = 26658.883                       # Ring circumference [m].
    R         = C/(2.*np.pi)                    # Ring radius [m].

    alpha_x   = 0.
    alpha_y   = 0.

    Qp_x      = 0.                              # Horizontal chromaticity.
    Qp_y      = 0.                              # Vertical chromaticity.    
        
    beta_x    = 65.9756337546                   # Horizontal beta function [m].
    beta_y    = 71.5255058456                   # Vertical beta function [m].
    beta_z    = eta*R/Q_s                       # Longitudinal beta function [m].

    epsn_x    = 2.0                             # Horizontal emittance [um].
    epsn_y    = 2.0                             # Vertical emittance [um].
    epsn_z    = 4.*np.pi*sigma_z**2 * p0 / (beta_z * e)

    i_oct_f   = -i_oct_fd                       # Octupole current (focusing).
    i_oct_d   = i_oct_fd                        # Octupole current (defocusing).
    
    initial_kick_x = 0.                         # Initial horizontal kick of beam.
    initial_kick_y = 0.                         # Initial vertical kick of beam.

    # SIMULATION PARAMETERS.
    n_macroparticles = 200000                    # Number of macroparticles per bunch (go to 1e6).
    n_turns          = 300                      # Number of turn (set to 2e5 later)

    
    # ==============================================================================================
    # GENERATE BUNCH AND CREATE VARIOUS SLICINGS.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Create bunch (matched) and apply initial kicks.
    bunch = Particles.as_gaussian(n_macroparticles, charge, gamma, intensity, mass,
                                  alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                                  beta_z, epsn_z, generator_seed=10)

    bunch.x += initial_kick_x
    bunch.y += initial_kick_y

    # SLICINGS
    # Use same slicing for wakefields as for monitors for now. 
    n_slices_for_monitor = 100                     # Number of slices per bunch for monitor slicing.
    n_slices_for_wakes   = n_slices_for_monitor     # Number of slices per bunch for wake fields.
    
    # slices_for_monitor = Slicer(n_slices_for_monitor, nsigmaz=3, mode='const_space')
    slices_for_wakes   = Slicer(n_slices_for_wakes, nsigmaz=3, mode='const_space')
    # slices_for_monitor.update_slices(bunch)
    slices_for_wakes.update_slices(bunch)
    

    # ==============================================================================================
    # SET UP SYNCHROTRON AND BETATRON MOTION.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    n_segments = 1                # Number of segments per turn.
    s = np.arange(0, n_segments + 1) * C/n_segments

    # BETATRON
    # Loop on number of segments and create the TransverseSegmentMap for each segment.
    alpha_x *= np.ones(n_segments)
    beta_x  *= np.ones(n_segments)
    D_x      = np.zeros(n_segments)
    alpha_y *= np.ones(n_segments)
    beta_y  *= np.ones(n_segments)
    D_y      = np.zeros(n_segments)

    # Create detuning elements.
    chromaticity       = Chromaticity(Qp_x, Qp_y)
    amplitude_detuning = AmplitudeDetuning.from_octupole_currents_LHC(i_oct_f, i_oct_d)
    
    # Generate transverse map.
    transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
                                   chromaticity, amplitude_detuning)

    
    # SYNCHROTRON
    cavity = LinearMap(C, alpha_0, Q_s)
    
    # ==============================================================================================
    # SET UP WAKE FIELDS USING PyZBASE WAKE TABLE.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Load data from file.
    # 7 columns in file: time (z), dipolar_x, dipolar_y, quadrupolar_x, quadrupolar_y, 2 cross terms.
    # Cross terms are not considered for now.
    # wakepath = '/data/TECH/Experiments/LHC/4TeV/'
    #wakepath = '/data/TECH/Experiments/LHC/4TeV/PyHT/scripts_old/'
    #wakefile = wakepath + 'wakeforhdtl_PyZbase_Allthemachine_4TeV_B1_2012_v2.dat'

    #keys = ['time', 'dipolar_x', 'dipolar_y', 'quadrupolar_x', 'quadrupolar_y',
    #        'x_terms_1', 'x_terms_2'] 
    #wakes = Wake_table.from_ASCII(wakefile, keys, slices_for_wakes)


    # ==============================================================================================
    # SET UP PARAMETERS DICTIONARY AND BEAM MONITORS
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    simulation_parameters_dict = {'comment': 'LHC at 4TeV without wakefields. Pure octupoles.',\
                                  'charge':           charge,\
                                  'mass':             mass,\
                                  'intensity':        intensity,\
                                  'beta_x':           beta_x,\
                                  'beta_y':           beta_y,\
                                  'beta_z':           beta_z,\
                                  'sigma_z':          sigma_z,\
                                  # 'momentum_spread': momentum_spread,\
                                  'epsn_x':           epsn_x,\
                                  'epsn_y':           epsn_y,\
                                  'gamma_t':          gamma_t,\
                                  'C':                C,\
                                  # 'energy': energy,\
                                  'n_turns':          n_turns,\
                                  # 'nsigmaz': nsigmaz,\
                                  'Q_x':              Q_x,\
                                  'Q_y':              Q_y,\
                                  'Q_s':              Q_s,\
                                  'Qp_x':             Qp_x,\
                                  'Qp_y':             Qp_y,\
                                  'i_oct_f':          i_oct_f,\
                                  'i_oct_d':          i_oct_d,\
                                  'n_macroparticles': n_macroparticles,\
                                  'n_slices':         n_slices_for_monitor,\
                                  # 'R_frequency': R_frequency,\
                                  # 'Q': Q,\
                                  # 'R_shunt': R_shunt,\
                                  'initial_kick_x':   initial_kick_x,\
                                  'initial_kick_y':   initial_kick_y,\
                                  # 'RF_voltage': RF_voltage,\
                                  # 'harmonic_number': harmonic_number,\
                                  # 'Yokoya_X1': Yokoya_X1,\
                                  # 'Yokoya_X2': Yokoya_X2,\
                                  # 'Yokoya_Y1': Yokoya_Y1,\
                                  # 'Yokoya_Y2': Yokoya_Y2
                                  }

    #particles_stride = 2
    #filename_monitor = 'I' + str(int(intensity/1e10)) + 'e10_' + 'Of' + str(int(i_oct_f)) + \
    #                   '_Od' + str(int(i_oct_d)) + '_T' + str(int(n_turns)) + '_pureOCTUPOLES'
    #particle_monitor = ParticleMonitor('particles_' + filename_monitor, particles_stride,
    #                                   simulation_parameters_dict, slices_for_monitor)
    #slice_monitor = SliceMonitor('slices_' + filename_monitor, n_turns,
    #                             simulation_parameters_dict, slices_for_monitor)
    # bunch_monitor = BunchMonitor('bunch', n_turns, simulation_parameters_dict)

    # ==============================================================================================
    # SET UP ACCELERATOR MAP AND START TRACKING
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Accelerator map.
    map_ = transverse_map + [cavity] #[wakes]

    # Start tracking.
    for i in range(n_turns):
        t0 = time.clock()
	
        for m in map_:
            m.track(bunch)

        # slice_monitor.dump(bunch)
        # bunch_monitor.dump(bunch)
        # particle_monitor.dump(bunch)
      
        print 'turn', i    

        #print '{0:4d} \t {1:+3e} \t {2:+3e} \t {3:+3e} \t {4:3e} \t {5:3e} \t {6:3f} \t {7:3f} \t {8:3f} \t {9:4e} \t {10:3s}'.format(i, bunch.mean_x, bunch.mean_y, bunch.mean_z, bunch.epsn_x, bunch.epsn_y, bunch.epsn_z, bunch.sigma_z, bunch.sigma_dp, bunch.n_macroparticles / bunch.n_macroparticles * bunch.intensity, str(time.clock() - t0))

    # particle_monitor.close()
    # slice_monitor.close()
    # bunch_monitor.close()


if __name__ == '__main__':
    main()
