from __future__ import division
import cProfile, itertools, time, timeit
import numpy as np

from pylab import *
import h5py as hp

from particles.particles import *
from particles.slicer import *
from trackers.transverse_tracker import *
from trackers.simple_long_linear_map import *
from monitors.monitors import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants

def main():
    # ==============================================================================================
    # SIMULATION SETUP.
    #
    # DON'T CHANGE! Reference data were generated with these settings.
    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # LHC @3.5TeV
    i_oct_fd = 0.

    intensity = 1.05e11                         # Number of particles (protons) per bunch.
    charge    = e                               # Charge of a proton.
    mass      = m_p                             # Mass of a proton.

    sigma_z   = 0.059958                        # Bunch length (RMS) [m].
    gamma     = 3730.26                         # Relativistic gamma.
    alpha_0   = 0.0003225                       # Momentum compaction factor.
    eta       = alpha_0 - 1./gamma**2           # Slippage factor.
    gamma_t   = 1./np.sqrt(alpha_0)             # Transition gamma.

    p0 = np.sqrt(gamma**2 - 1) * mass * c	    # Momentum.

    epsn_x    = 3.75                            # Horizontal emittance [um].
    epsn_y    = 3.75                            # Vertical emittance [um].
    
    Q_s       = 0.0020443                       # Synchrotron tune.
    Q_x       = 64.28                           # Betatron tune (horizontal).
    Q_y       = 59.31                           # Betatron tune (vertical).
    
    C         = 26658.883                       # Ring circumference [m].
    R         = C/(2.*np.pi)                    # Ring radius [m].

    alpha_x   = 0.
    alpha_y   = 0.
    
    beta_x    = 66.0064                         # Horizontal beta function [m].
    beta_y    = 71.5376                         # Vertical beta function [m].
    beta_z    = eta*R/Q_s                       # Longitudinal beta function [m].
    Qp_x      = 0.                              # Horizontal chromaticity.
    Qp_y      = 0.                              # Vertical chromaticity.

    i_oct_f   = -i_oct_fd                       # Octupole current [A] focusing (< 0)
    i_oct_d   = i_oct_fd                        # Octupole current [A] defocusing (> 0 and i_oct_d = -i_oct_f).

    initial_kick_x = 0.000                      # Initial horizontal kick of beam.
    initial_kick_y = 0.000                      # Initial vertical kick of beam.

    epsn_z    = 4.*np.pi*sigma_z**2 * p0 / (beta_z * e)
    
    n_macroparticles = 200000                   # Number of macroparticles per bunch (go to 1e6).
    n_turns          = 200                      # Number of turn (set to 2e5 later)
    n_segments       = 1                        # Number of segments per turn.
    

    # ============================================================================
    # Plotting environment.
    fig,  (ax1, ax2) = plt.subplots(1, 2, figsize=(25,7))
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(25,7))
    
    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('# macroparticles')
    ax2.set_xlabel('z [m]')

    ax3.set_xlabel('z [m]')
    ax3.set_ylabel('# macroparticles')
    ax4.set_xlabel('z [m]')

    fig.suptitle('nsigmaz = None')
    fig2.suptitle('nsigmaz = 2')

    # Import ref. init bunch.
    bunch_init_path = '../bunch_init_data/'
    f = hp.File(bunch_init_path + 'init_bunch_2e5particles.h5', 'r')
    x = np.zeros(n_macroparticles)
    y = np.zeros(n_macroparticles)
    z = np.zeros(n_macroparticles)
    xp = np.zeros(n_macroparticles)
    yp = np.zeros(n_macroparticles)
    dp = np.zeros(n_macroparticles)
       
    # ============================================================================
    # 1ST TEST:
    # For nsigmaz = None
    print '\n======= Running part 1 ======='
    print '  nsigmaz = None'
    x = f['x'][:]
    y = f['y'][:]
    z = f['z'][:]
    xp = f['xp'][:]
    yp = f['yp'][:]
    dp = f['dp'][:]

    bunch = Particles.as_import(n_macroparticles, charge, mass, gamma, intensity, x, xp, y, yp, z, dp)

    n_slices = 20
    
    slices_const_space = Slicer(n_slices, mode='const_space')
    slices_const_space.update_slices(bunch)

    slices_const_charge = Slicer(n_slices, mode='const_charge')
    slices_const_charge.update_slices(bunch)
        
    s = np.arange(0, n_segments + 1) * C/n_segments

    alpha_x *= np.ones(n_segments)
    beta_x  *= np.ones(n_segments)
    D_x      = np.zeros(n_segments)
    alpha_y *= np.ones(n_segments)
    beta_y  *= np.ones(n_segments)
    D_y	     = np.zeros(n_segments)

    transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    cavity         = LinearMap(C, alpha_0, Q_s, (slices_const_space, slices_const_charge))

    # Track for a few hundred turns (cover synchrotron half-period).
    for i in xrange(n_turns):

        if i%50 == 0:
            print '\t turn', i

        transverse_map[0].track(bunch)
        cavity.track(bunch)

        
    ax1.hist(bunch.z, slices_const_space.n_slices, color='r', alpha=0.6, label='matplotlib hist')
    [ax1.axvline(z, c='b', lw=2) for z in slices_const_space.z_bins[1:]]
    ax1.axvline(slices_const_space.z_bins[0], c='b', lw=2, label='z_bins PyHT const space slicer')
    ax1.plot(slices_const_space.z_centers, slices_const_space.n_macroparticles, 'x', color='black', ms=10, mew=2,
             label='z_centers vs. n_macroparticles PyHT const space slicer')
    y_up =  max(slices_const_space.n_macroparticles)
    y_up += 0.1 * y_up
    ax1.set_ylim((0,y_up))

    [ax2.axvline(z, c='b', lw=2) for z in slices_const_charge.z_bins[1:]]
    ax2.axvline(slices_const_charge.z_bins[0], c='b', lw=2, label='z_bins, PyHT const charge slicer')
    ax2.axhline(n_macroparticles / n_slices, c='r', ls='dashed', lw=2, label='expected number of particles per slice')
    ax2.plot(slices_const_charge.z_centers, slices_const_charge.n_macroparticles, 'x', color='black', ms=10, mew=2,
             label='z_centers vs. n_macroparticles, PyHT const charge slicer')
    y_up =  max(slices_const_charge.n_macroparticles)
    y_up += 0.1 * y_up
    ax2.set_ylim((0,y_up))

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')


    # ============================================================================
    # 2ND TEST:
    # For nsigmaz = 2. Use reference data.
    print '\n======= Running part 2 ======='
    print '  nsigmaz = 2'

    x = f['x'][:]
    y = f['y'][:]
    z = f['z'][:]
    xp = f['xp'][:]
    yp = f['yp'][:]
    dp = f['dp'][:]

    bunch = Particles.as_import(n_macroparticles, charge, mass, gamma, intensity, x, xp, y, yp, z, dp)

    n_slices = 20
    
    slices_const_space = Slicer(n_slices, nsigmaz=2, mode='const_space')
    slices_const_space.update_slices(bunch)

    slices_const_charge = Slicer(n_slices, nsigmaz=2, mode='const_charge')
    slices_const_charge.update_slices(bunch)
    
    s = np.arange(0, n_segments + 1) * C/n_segments

    alpha_x *= np.ones(n_segments)
    beta_x  *= np.ones(n_segments)
    D_x      = np.zeros(n_segments)
    alpha_y *= np.ones(n_segments)
    beta_y  *= np.ones(n_segments)
    D_y	     = np.zeros(n_segments)

    transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    cavity         = LinearMap(C, alpha_0, Q_s, (slices_const_space, slices_const_charge))

    # Track for a few hundred turns (cover synchrotron half-period).
    for i in xrange(n_turns):
        if i%50 == 0:
            print '\t turn', i

        transverse_map[0].track(bunch)
        cavity.track(bunch)
        
    [ax3.axvline(z, c='b', lw=2) for z in slices_const_space.z_bins[1:]]
    ax3.axvline(slices_const_space.z_bins[0], c='b', lw=2, label='z_bins, PyHT const space slicer')
    ax3.plot(slices_const_space.z_centers, slices_const_space.n_macroparticles, 'x', color='black', ms=10, mew=2,
             label='z_centers vs. n_macroparticles, PyHT const space slicer')
    y_up =  max(slices_const_space.n_macroparticles)
    y_up += 0.1 * y_up
    ax3.set_ylim((0,y_up))

    [ax4.axvline(z, c='b', lw=2) for z in slices_const_charge.z_bins[1:]]
    ax4.axvline(slices_const_charge.z_bins[0], c='b', lw=2, label='z_bins, PyHT const charge slicer')
    ax4.plot(slices_const_charge.z_centers, slices_const_charge.n_macroparticles, 'x', color='black', ms=10, mew=2,
             label='z_centers vs. n_macroparticles, PyHT const charge slicer')
        
    # Get reference slicer data.
    ref_data_path = './reference_data/'
    f_sl = hp.File(ref_data_path + 'ref_slices_20_with_tracking.h5')
    f_sl_grp = f_sl['const_space_slicer']

    sl_const_space_n_macroparticles = np.zeros(n_slices)
    sl_const_space_z_bins           = np.zeros(n_slices + 1)
    sl_const_space_z_centers        = np.zeros(n_slices)
    sl_const_space_n_macroparticles = f_sl_grp['n_macroparticles_slice'][:]
    sl_const_space_z_bins           = f_sl_grp['z_bins'][:]
    sl_const_space_z_centers        = f_sl_grp['z_centers'][:]

    f_sl_grp = f_sl['const_charge_slicer']
    sl_const_charge_n_macroparticles = np.zeros(n_slices)
    sl_const_charge_z_bins           = np.zeros(n_slices + 1)
    sl_const_charge_z_centers        = np.zeros(n_slices)
    sl_const_charge_n_macroparticles = f_sl_grp['n_macroparticles_slice'][:]
    sl_const_charge_z_bins           = f_sl_grp['z_bins'][:]
    sl_const_charge_z_centers        = f_sl_grp['z_centers'][:]

    [ax3.axvline(z, c='r', ls='dashed', lw=2) for z in sl_const_space_z_bins[1:]]
    ax3.axvline(sl_const_space_z_bins[0], c='r', ls='dashed', lw=2, label='z_bins reference data, const charge slicer')
    ax3.plot(sl_const_space_z_centers, sl_const_space_n_macroparticles, 'sr', mfc='None', mew=2, mec='g', ms=10,
             label='z_centers vs. n_macroparticles reference data, const charge slicer')

    [ax4.axvline(z, c='r', ls='dashed', lw=2) for z in sl_const_charge_z_bins[1:]]
    ax4.axvline(sl_const_charge_z_bins[0], c='r', ls='dashed', lw=2, label='z_bins reference data')
    ax4.plot(sl_const_charge_z_centers, sl_const_charge_n_macroparticles, 'sr', mfc='None', mew=2, mec='g', ms=10,
             label='z_centers vs. n_macroparticles reference data, const charge slicer')
    y_up =  max(sl_const_charge_n_macroparticles)
    y_up += 0.1 * y_up
    ax4.set_ylim((0,y_up))

    ax3.legend(loc='lower left')
    ax4.legend(loc='lower left')
        
    plt.show()


    f.close()
    f_sl.close()


if __name__ == "__main__":
    main()
