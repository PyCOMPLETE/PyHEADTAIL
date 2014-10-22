#!/usr/bin/python
from __future__ import division

# include PyHEADTAIL directory
import sys
import os
#BIN = os.path.dirname(__file__)
#BIN = os.path.abspath( BIN )
#BIN = os.path.dirname( BIN )
#BIN = "/home/oeftiger/cern/git/PyHEADTAIL-new/"
BIN = "/home/michael/TECH/PyHEADTAIL/"
sys.path.append(BIN)

import numpy as np
import pylab as plt

from PyHEADTAIL.particles.particles import *
from PyHEADTAIL.particles.slicing import *
from scipy.constants import c, e, m_p
from scipy.constants import physical_constants

def generate_bunch(n_macroparticles):
    '''Generate standard bunch for slicing tests.'''
    intensity = 1.05e11
    charge    = e
    mass      = m_p

    sigma_z   = 0.059958
    gamma     = 3730.26
    alpha_0   = 0.0003225
    eta       = alpha_0 - 1./gamma**2
    gamma_t   = 1./np.sqrt(alpha_0)

    p0 = np.sqrt(gamma**2 - 1) * mass * c

    epsn_x    = 3.75
    epsn_y    = 3.75

    Q_s       = 0.0020443
    Q_x       = 64.28
    Q_y       = 59.31

    C         = 26658.883
    R         = C/(2.*np.pi)

    beta_x    = 66.0064
    beta_y    = 71.5376
    beta_z    = eta*R/Q_s

    epsn_z    = 4.*np.pi*sigma_z**2 * p0 / (beta_z * e)

    bunch = Particles.as_gaussian_linear(macroparticlenumber=n_macroparticles,
        intensity=intensity, charge=charge,
        gamma_reference=gamma, mass=mass, circumference=C,
        alpha_x=0, beta_x=beta_x, epsn_x=epsn_x,
        alpha_y=0, beta_y=beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z,
        generator_seed=None)
    
    return bunch

def plot_slice_set_structure(axes, bunch, slice_set):
    '''Plot basic structure of SliceSet - z_bins, z_centers,
    n_macroparticles_per_slice.
    '''
    [axes.axvline(z, c='b', lw=2) for z in slice_set.z_bins[1:]]
    axes.axvline(slice_set.z_bins[0], c='b', lw=2,
                 label='z_bins')
    [axes.axvline(z, c='b', ls='dashed') for z in
        slice_set.z_centers[1:]]
    axes.axvline(slice_set.z_centers[0], c='b', ls='dashed',
                 label='z_centers')
    axes.plot(slice_set.z_centers,
              slice_set.n_macroparticles_per_slice,
              'x', color='black', ms=10, mew=2,
              label='z_centers vs. n_macroparticles')
    y_up =  max(slice_set.n_macroparticles_per_slice)
    y_up += 0.1 * y_up
    axes.set_ylim((0,y_up))
    axes.set_xlim(((1+0.1) * min(bunch.z), (1+0.1) * max(bunch.z)))

def plot_particle_indices_of_slice(axes, bunch, slice_set):
    '''Show all particles in a z vs. slice_index plot. Add SliceSet
    and slice boundaries. particles_within_cuts are overlayed
    to see if they have been correctly determined.
    '''
    z_cut_tail = slice_set.z_cut_tail
    z_cut_head = slice_set.z_cut_head
    part_in_cuts = slice_set.particles_within_cuts
    six = slice_set.slice_index_of_particle

    axes.plot(six, bunch.z, '.r', ms=12, label='All particles')
    axes.plot(six.take(part_in_cuts), bunch.z.take(part_in_cuts), '.g',
             label='particles_within_cuts')
    axes.axhline(z_cut_tail, color='b', ls='dashed',
                label='SliceSet boundaries')
    axes.axhline(z_cut_head, color='b', ls='dashed')
    [axes.axhline(z, color='b', ls='dashed') for z in
        slice_set.z_bins]
    axes.axvline(0, color='m', label='slices 0 and n-1')
    axes.axvline(slice_set.n_slices-1, color='m')
    axes.set_xlim((min(slice_set.slice_index_of_particle)-1,
                   max(slice_set.slice_index_of_particle)+1))
    axes.legend(loc='lower right')

def test_particle_indices_of_slice(bunch, slice_set):
    '''Get particle_indices_of_slice for specific slice index. Apply
    'inverse function' slice_index_of_particle to get back slice_index
    if everything works correctly.
    '''
    all_pass = True
    for i in xrange(slice_set.n_slices):
        pix_slice = slice_set.particle_indices_of_slice(i)
        six_pix = slice_set.slice_index_of_particle[pix_slice]
        if (six_pix != i).any():
            all_pass = False

    if all_pass:
        print ('  Particle_indices_of_slice <-> slice_index_of_particle PASSED')
    if not all_pass:
        print ('  Particle_indices_of_slice and slice_index_of_particle FAILED')

def slice_set_statistics(bunch, slice_set):
    '''Test if statistics functions are executable. No value
    checking
    '''
    slice_set.mean_x(bunch)
    slice_set.sigma_x(bunch)
    slice_set.epsn_x(bunch)
    slice_set.mean_y(bunch)
    slice_set.sigma_y(bunch)
    slice_set.epsn_y(bunch)
    slice_set.mean_z(bunch)
    slice_set.sigma_z(bunch)
    slice_set.epsn_z(bunch)


def main():
    bunch = generate_bunch(n_macroparticles=400)
    n_slices = 10
    n_sigma_z = None

    # Plot environment
    fig, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2, 2, figsize=(25,18))

    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('# macroparticles')
    ax2.set_xlabel('z [m]')
    ax5.set_xlabel('slice index')
    ax5.set_ylabel('z [m]')
    ax6.set_xlabel('slice index')

    fig.suptitle('n_sigma_z = ' + str(n_sigma_z))

    # ==================================================================
    # UNIFORM BIN SLICING
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)
    uniform_bin_slice_set = uniform_bin_slicer.slice(bunch)

    # SliceSet structure
    plot_slice_set_structure(axes=ax1, bunch=bunch,
                             slice_set=uniform_bin_slice_set)
    if n_sigma_z == None:
        ax1.hist(bunch.z, uniform_bin_slice_set.n_slices, color='r', alpha=0.6,
                 label='matplotlib hist')
    ax1.legend(loc='lower left')

    # slice_index_of_particle and particle_indices_of_slice visual tests.
    plot_particle_indices_of_slice(axes=ax5, bunch=bunch,
                                   slice_set=uniform_bin_slice_set)

    test_particle_indices_of_slice(bunch=bunch,
                                   slice_set=uniform_bin_slice_set)

    # SliceSet statistics.
    slice_set_statistics(bunch, slice_set=uniform_bin_slice_set)


    # ==================================================================
    # UNIFORM CHARGE SLICING
    uniform_charge_slicer = UniformChargeSlicer(n_slices, n_sigma_z)
    uniform_charge_slice_set = uniform_charge_slicer.slice(bunch)

    # SliceSet structure
    plot_slice_set_structure(axes=ax2, bunch=bunch,
                             slice_set=uniform_charge_slice_set)
    n_particles_within_cuts = len(
        uniform_charge_slice_set.particles_within_cuts)
    ax2.axhline(n_particles_within_cuts / uniform_charge_slice_set.n_slices,
                c='r', ls='dashed', lw=2,
                label='expected number of particles per slice')
    ax2.legend(loc='lower left')

    # slice_index_of_particle and particle_indices_of_slice visual tests.
    plot_particle_indices_of_slice(axes=ax6, bunch=bunch,
                                   slice_set=uniform_charge_slice_set)

    test_particle_indices_of_slice(bunch=bunch,
                                   slice_set=uniform_charge_slice_set)

    # SliceSet statistics.
    slice_set_statistics(bunch, slice_set=uniform_charge_slice_set)

    plt.show()


if __name__ == "__main__":
    main()
