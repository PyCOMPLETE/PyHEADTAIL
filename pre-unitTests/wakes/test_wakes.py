from __future__ import division
import cProfile, itertools, time, timeit
import numpy as np

from pylab import *
import h5py as hp

from particles.particles import *
from particles.slicer import *
from trackers.transverse_tracker import *
from trackers.simple_long_linear_map import *
from impedances.wake_fields import *
import impedances.wake_sources as WakeSources
from monitors.monitors import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants

import cobra_functions.stats as cp


def run_test(nsigmaz, ax11, ax12, ax13, ax21):

    # ==============================================================================================
    # SIMULATION SETUP.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # LHC @3.5TeV.
    i_oct_fd = 0.

    intensity = 1.05e11                         # Number of particles (protons) per bunch.
    charge    = e                               # Charge of a proton.
    mass      = m_p                             # Mass of a proton.

    sigma_z   = 0.059958                        # Bunch length (RMS) [m].
    gamma     = 3730.26                         # Relativistic gamma.
    alpha_0   = 0.0003225                       # Momentum compaction factor.
    eta       = alpha_0 - 1./gamma**2           # Slippage factor.
    gamma_t   = 1./np.sqrt(alpha_0)             # Transition gamma.

    p0 = np.sqrt(gamma**2 - 1) * mass * c       # Momentum.

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

    n_macroparticles = 20000                    # Number of macroparticles per bunch (go to 1e6).
    n_turns          = 200                      # Number of turn (set to 2e5 later)
    n_segments       = 1                        # Number of segments per turn.

    bunch = Particles.as_gaussian(macroparticlenumber=n_macroparticles, charge=charge, gamma_reference=gamma,
                                  intensity=intensity, mass=mass,
                                  alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
                                  alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
                                  beta_z=beta_z, epsn_z=epsn_z, generator_seed=10)

    # Slicer.
    n_slices = 15
    slices_const_space = Slicer(n_slices, nsigmaz=nsigmaz, mode='const_space')
    slices_const_space.update_slices(bunch)

    # Map.
    s = np.arange(0, n_segments + 1) * C/n_segments

    alpha_x *= np.ones(n_segments)
    beta_x  *= np.ones(n_segments)
    D_x      = np.zeros(n_segments)
    alpha_y *= np.ones(n_segments)
    beta_y  *= np.ones(n_segments)
    D_y      = np.zeros(n_segments)

    transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    cavity         = LinearMap(C, alpha_0, Q_s, (slices_const_space,))

    # Wakes.
    wakepath = './'
    wakefile = wakepath + 'wakeforhdtl_PyZbase_Allthemachine_3p5TeV_B2_2010.dat'
    wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y',
                         'No_dipole_xy', 'No_dipole_yx']
    wake_table = WakeSources.WakeTable(wakefile, wake_file_columns, slices_const_space)
    wakes = Wakes((wake_table,), slices_const_space)

    # Track for a few hundred turns.
    xp_diff_quad = np.zeros(n_macroparticles)
    xp_diff_dip  = np.zeros(n_macroparticles)

    for i in xrange(n_turns):
        transverse_map[0].track(bunch)
        cavity.track(bunch)

        # Dipole X kick.
        if i == (n_turns - 1):
            xp_old = bunch.xp.copy()
        wakes.wake_kicks[0].apply(bunch, slices_const_space)

        if i == (n_turns - 1):
            xp_diff_dip[:] = bunch.xp[:] - xp_old[:]

        # Quadrupole X kick.
        if i == (n_turns - 1):
            xp_old = bunch.xp.copy()
        wakes.wake_kicks[1].apply(bunch, slices_const_space)

        if i == (n_turns - 1):
            xp_diff_quad[:] = bunch.xp[:] - xp_old[:]

    # Plot bunch.z vs. slice index of particle. Mark particles within
    # z cuts in green.
    nsigmaz_lbl = ' (nsigmaz =' + str(nsigmaz) + ')'

    pidx = slices_const_space.particles_within_cuts
    slidx = slices_const_space.slice_index_of_particle

    z_cut_tail, z_cut_head = slices_const_space._set_longitudinal_cuts(bunch)

    ax11.plot(slidx, bunch.z, '.r', ms=10, label='all particles')
    ax11.plot(slidx.take(pidx), bunch.z.take(pidx), '.g', label='particles within z cuts')
    ax11.axhline(z_cut_tail, color='m', linestyle='dashed', label='slicer boundaries ' + nsigmaz_lbl)
    ax11.axhline(z_cut_head, color='m', linestyle='dashed')
    ax11.axvline(0, color='b', linestyle='dashed', label='first and last slices' + nsigmaz_lbl)
    ax11.axvline(n_slices-1, color='b', linestyle='dashed')
    [ ax11.axhline(z, color='m', linestyle='dashed') for z in slices_const_space.z_bins ]
    ax11.legend(loc='upper left')

    # Show dipole and qudrupole kicks applied for each particle for the
    # last turn.
    ax13.plot(slidx, xp_diff_quad, '.g', ms=10, label='quad x kicks')
    ax12.plot(slidx, xp_diff_dip, '.r', label='dip x kicks')
    ax12.axvline(0, color='b', linestyle='dashed', label='first and last slices' + nsigmaz_lbl)
    ax12.axvline(n_slices-1, color='b', linestyle='dashed')
    ax12.axhline(0, color='black', ls='dashed')
    ax13.axvline(0, color='b', linestyle='dashed', label='first and last slices' + nsigmaz_lbl)
    ax13.axvline(n_slices-1, color='b', linestyle='dashed')
    ax12.legend(loc='lower right')
    ax13.legend(loc='lower right')

    xmax = max(slidx)
    xmax += 2
    xmin = min(slidx)
    xmin -= 2

    ymax = max(xp_diff_dip)
    ymax += ymax*0.2
    ymin = min(xp_diff_dip)
    ymin += ymin*0.2

    ax11.set_xlim((xmin, xmax))
    ax12.set_xlim((xmin, xmax))
    ax13.set_xlim((xmin, xmax))

    # Plot wake table and show where wake is sampled (values obtained from interp1d)
    # for dipole and quadrupole X kicks.
    ax21.plot(wake_table.wake_table['time'][:-1], abs(wake_table.wake_table['dipole_x'][:-1]),
              color='b', label='dipole x')
    ax21.plot(wake_table.wake_table['time'][:-1], abs(wake_table.wake_table['quadrupole_x'][:-1]),
              color='r', label='quadrupole x')

    sampled_wake = wake_table._function_transverse('dipole_x')
    dz = np.concatenate((slices_const_space.z_centers - slices_const_space.z_centers[-1],
                        (slices_const_space.z_centers - slices_const_space.z_centers[0])[1:]))

    ax21.plot(abs(dz / (bunch.beta * c)), abs(sampled_wake(bunch.beta, dz)), '.g', ms=15,
              label='sampled and interpolated dipole x wake')

    sampled_wake = wake_table._function_transverse('quadrupole_x')
    ax21.plot(abs(dz / (bunch.beta * c)), abs(sampled_wake(bunch.beta, dz)), '.m', ms=15,
              label='sampled and intepolated quadrupole x wake')

    slice_width = (z_cut_head - z_cut_tail) / n_slices
    dzz = np.arange(0, n_slices*slice_width, slice_width)
    [ ax21.axvline(z / (bunch.beta * c), color='black', ls='dashed') for z in dzz[1:] ]
    ax21.axvline(dzz[0] / (bunch.beta * c), color='black', ls='dashed', label='slice widths' + nsigmaz_lbl)

    ax21.set_xscale('log')
    ax21.set_yscale('log')
    ax21.set_xlabel('time [s]')
    ax21.set_ylabel('abs. wake strength [V/C/m]')
    ax21.legend(loc='upper right')

    ax11.set_ylabel('z [m]')
    ax11.set_xlabel('slices index')
    ax12.set_xlabel('slices index')
    ax12.set_ylabel(r'$\Delta$xp')
    ax13.set_xlabel('slices index')
    ax13.set_ylabel(r'$\Delta$xp')

    ax21.set_xlim((wake_table.wake_table['time'][0], wake_table.wake_table['time'][-2]))


def main():

    # Plotting environment.
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(30,12))
    fig2, ((ax12, ax22)) = plt.subplots(2,1, figsize=(20,10), sharex=True)

    fig.text(0., 0.75, 'nsigmaz = None', fontsize='large', weight='bold', color='r')
    fig.text(0., 0.25, 'nsigmaz = 2', fontsize='large', weight='bold', color='r')
    fig2.text(0., 0.75, 'nsigmaz = None', fontsize='large', weight='bold', color='r')
    fig2.text(0., 0.25, 'nsigmaz = 2', fontsize='large', weight='bold', color='r')

    # Perform two tests with different slicer cuts.
    print '\n======= Running part 1 ======='
    print '  nsigmaz = None'
    run_test(nsigmaz=None, ax11=ax1, ax12=ax2, ax13=ax5, ax21=ax12)
    print '\n======= Running part 2 ======='
    print '  nsigmaz = 2'
    run_test(nsigmaz=2, ax11=ax3, ax12=ax4, ax13=ax6, ax21=ax22)

    plt.show()


if __name__ == "__main__":
    main()
