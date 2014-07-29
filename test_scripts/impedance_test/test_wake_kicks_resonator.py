from __future__ import division
import time
import numpy as np

from particles.particles import *
from particles.slicer import *
from impedances.wake_fields_3  import *
from trackers.transverse_tracker_2 import *
from trackers.longitudinal_tracker import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants

import pylab as plt


def main():

    # PHYSICS AND MACHINE PARAMETERS.
    intensity = 4.5e11                           # Number of particles (protons) per bunch.
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

    # SIMULATION PARAMETERS.
    n_macroparticles = 20000                    # Number of macroparticles per bunch (go to 1e6).
    n_turns          = 5                         # Number of turn (set to 2e5 later)


    # ==============================================================================================
    # GENERATE BUNCH AND CREATE VARIOUS SLICINGS.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Create bunch (matched) and apply initial kicks.
    bunch = Particles.as_gaussian(n_macroparticles, charge, gamma, intensity, mass,
                                  alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y,
                                  beta_z, epsn_z, generator_seed=10)

    # Import bunch from file.
    bunch.x = np.loadtxt('./initBunch/x.dat')
    bunch.y = np.loadtxt('./initBunch/y.dat')
    bunch.z = np.loadtxt('./initBunch/z.dat')
    bunch.xp = np.loadtxt('./initBunch/xp.dat')
    bunch.yp = np.loadtxt('./initBunch/yp.dat')
    bunch.dp = np.loadtxt('./initBunch/dp.dat')
    
    # SLICINGS
    n_slices_for_wakes = 20
    slices_for_wakes = Slicer(n_slices_for_wakes, nsigmaz=3, mode='const_space')
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

    transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)

    # SYNCHROTRON
    cavity = LinearMap(C, alpha_0, Q_s, (slices_for_wakes, ))


    # ==============================================================================================
    # SET UP WAKE FIELDS USING RESONATOR MODEL.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    R_shunt   = 10e6
    frequency = 2e9
    Q         = 0.9
    Yokoya_X1 = 0.5
    Yokoya_Y1 = 0.5
    Yokoya_X2 = 0.5
    Yokoya_Y2 = 0.5
    Yokoya_Z  = 0
    wakes = Wakes.resonator(R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2,
                            Yokoya_Z, slices_for_wakes)


    # ==============================================================================================
    # SET UP ACCELERATOR MAP AND START TRACKING
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Accelerator map.
    map_ = transverse_map + [cavity]
    wakes = [wakes]
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    
    # Read data from files created with old PyHT version.
    # (where wakes were benchmarked against HT -> see pres. M. Schenk, 22.07.2014, HTWG Meeting).
    dipX_kick_old_version = plt.loadtxt('./resultsOldPyHT/kicks_dipX_resonator.dat')
    dipY_kick_old_version = plt.loadtxt('./resultsOldPyHT/kicks_dipY_resonator.dat')
    quadX_kick_old_version = plt.loadtxt('./resultsOldPyHT/kicks_quadX_resonator.dat')
    quadY_kick_old_version = plt.loadtxt('./resultsOldPyHT/kicks_quadY_resonator.dat')

    # For new PyHT version.
    dipX_kick = np.zeros(n_macroparticles)
    dipY_kick = np.zeros(n_macroparticles)
    quadX_kick = np.zeros(n_macroparticles)
    quadY_kick = np.zeros(n_macroparticles)
    
    # Start tracking.
    for i in range(n_turns):
        t0 = time.clock()

        for m in map_:
            m.track(bunch)

        if i == 4:            
            # Dip kicks.
            xp_old = np.zeros(n_macroparticles)
            xp_old[:] = bunch.xp
            wakes[0].wake_kicks[0].apply(bunch, slices_for_wakes)
            dipX_kick[:] = bunch.xp - xp_old

            yp_old = np.zeros(n_macroparticles)
            yp_old[:] = bunch.yp
            wakes[0].wake_kicks[2].apply(bunch, slices_for_wakes)
            dipY_kick[:] = bunch.yp - yp_old

            # Quad kicks.
            wakes[0].wake_kicks[1].apply(bunch, slices_for_wakes)
            quadX_kick[:] = bunch.xp - xp_old

            wakes[0].wake_kicks[3].apply(bunch, slices_for_wakes)
            quadY_kick[:] = bunch.yp - yp_old

            # Compare results for old and new versions.       
            ax1.plot(dipX_kick, c='r', label='new version')
            ax1.plot(dipX_kick_old_version, 'b', linestyle='dashed', label='old version')
            ax1.set_ylabel('dipX kick')
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels)
                        
            ax2.plot(dipY_kick, c='r', label='new version')
            ax2.plot(dipY_kick_old_version, c='b', linestyle='dashed', label='old version')
            ax2.set_ylabel('dipY kick')
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels)
            
            ax3.plot(quadX_kick, c='r', label='new version')
            ax3.plot(quadX_kick_old_version, c='b', linestyle='dashed', label='old version')
            ax3.set_ylabel('quadX kick')
            handles, labels = ax3.get_legend_handles_labels()
            ax3.legend(handles, labels)

            ax4.plot(quadY_kick, c='r', label='new version')
            ax4.plot(quadY_kick_old_version, c='b', linestyle='dashed', label='old version')
            ax4.set_ylabel('quadY kick')
            ax4.set_xlabel('particle no.')
            handles, labels = ax4.get_legend_handles_labels()
            ax4.legend(handles, labels)

        else:
            wakes[0].track(bunch)
            
        print i

    plt.show()
    
if __name__ == '__main__':
    main()
