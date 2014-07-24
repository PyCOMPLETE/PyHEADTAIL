from __future__ import division
import numpy as np
import pylab as pl
import time



import ecloud.PyECLOUD_for_PyHEADTAIL as pyecl
from particles.particles import *
from scipy.constants import e, m_e
import numpy as np
from particles.slicer import *
from trackers.transverse_tracker_2 import *
from trackers.longitudinal_tracker import *
from monitors.monitors import *

# ==============================================================================================
# SIMULATION SETUP.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LHC @4TeV, parameters taken from list LHC_4TeV_2012_v2_lin.cfg

# PHYSICS AND MACHINE PARAMETERS.
intensity = 2.5e11                           # Number of particles (protons) per bunch.
charge    = e                               # Charge of a proton.
mass      = m_p                             # Mass of a proton.

sigma_z   = .23				                # Bunch length (RMS) [m].
gamma     = 27.7		                    # Relativistic gamma.
alpha_0   = 0.00308642                      # Momentum compaction factor.
eta       = alpha_0 - 1./gamma**2           # Slippage factor.
gamma_t   = 1./np.sqrt(alpha_0)             # Transition gamma.

p0 = np.sqrt(gamma**2 - 1) * mass * c       # Momentum.

Q_s       = 0.017			                # Synchrotron tune.
Q_x       = 20.13                           # Betatron tune (horizontal).
Q_y       = 20.18                           # Betatron tune (vertical).

C         = 6911.                           # Ring circumference [m].
R         = C/(2.*np.pi)                    # Ring radius [m].

alpha_x   = 0.
alpha_y   = 0.

#Qp_x      = 0.                              # Horizontal chromaticity.
#Qp_y      = 0.                              # Vertical chromaticity.    
	
beta_x    = 54.6		                    # Horizontal beta function [m].
beta_y    = 54.6                            # Vertical beta function [m].
beta_z    = eta*R/Q_s                       # Longitudinal beta function [m].

epsn_x    = 2.5                             # Horizontal emittance [um].
epsn_y    = 2.5                             # Vertical emittance [um].
epsn_z    = 4.*np.pi*sigma_z**2 * p0 / (beta_z * e)

#i_oct_f   = -i_oct_fd                       # Octupole current (focusing).
#i_oct_d   = i_oct_fd                        # Octupole current (defocusing).

initial_kick_x = 0.                         # Initial horizontal kick of beam.
initial_kick_y = 0.                         # Initial vertical kick of beam.

# SIMULATION PARAMETERS.
n_macroparticles = 200000                    # Number of macroparticles per bunch (go to 1e6).
n_turns          = 512                       # Number of turn (set to 2e5 later)

n_segments = 5#57


simulation_parameters_dict = {'comment': 'test instab ecloud',\
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
                                  #~ 'Qp_x':             Qp_x,\
                                  #~ 'Qp_y':             Qp_y,\
                                  #~ 'i_oct_f':          i_oct_f,\
                                  #~ 'i_oct_d':          i_oct_d,\
                                  'n_macroparticles': n_macroparticles,\
                                  #'n_slices':         n_slices_for_monitor,\
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






L_ecloud = C/n_segments



# Beam
bunch = Particles.as_gaussian(n_macroparticles, e, gamma, intensity, m_p, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, epsn_z)

#ecloud
beamslicer = Slicer(50, nsigmaz=2)
ecloud = pyecl.Ecloud(L_ecloud, beamslicer, Dt_ref = 25e-12, pyecl_input_folder='drift_for_instab_SPS')

s = np.arange(0, n_segments + 1) * C/n_segments

# BETATRON
# Loop on number of segments and create the TransverseSegmentMap for each segment.
alpha_x *= np.ones(n_segments)
beta_x  *= np.ones(n_segments)
D_x      = np.zeros(n_segments)
alpha_y *= np.ones(n_segments)
beta_y  *= np.ones(n_segments)
D_y      = np.zeros(n_segments)

#~ # Create detuning elements.
#~ chromaticity       = Chromaticity(Qp_x, Qp_y)
#~ amplitude_detuning = AmplitudeDetuning.from_octupole_currents_LHC(i_oct_f, i_oct_d)

# Generate transverse map.
transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)#,
							   #~ chromaticity, amplitude_detuning)


# SYNCHROTRON
cavity = LinearMap(C, alpha_0, Q_s)

# build ring
elements=[]
for l in transverse_map:
    elements+=[l, ecloud]
elements.append(cavity)
bunch_monitor = BunchMonitor('bunch', n_turns, simulation_parameters_dict)

for i in range(n_turns):
        t0 = time.clock()
	
        for ind, m in enumerate(elements):
			m.track(bunch)
			print ind, m
			

        # slice_monitor.dump(bunch)
        bunch_monitor.dump(bunch)
        # particle_monitor.dump(bunch)
      
        print '{0:4d} \t {1:+3e} \t {2:+3e} \t {3:+3e} \t {4:3e} \t {5:3e} \t {6:3f} \t {7:3f} \t {8:3f} \t {9:4e} \t {10:3s}'.format(i, bunch.mean_x, bunch.mean_y, bunch.mean_z, bunch.epsn_x, bunch.epsn_y, bunch.epsn_z, bunch.sigma_z, bunch.sigma_dp, bunch.n_macroparticles / bunch.n_macroparticles * bunch.intensity, str(time.clock() - t0))

bunch_monitor.close()          
