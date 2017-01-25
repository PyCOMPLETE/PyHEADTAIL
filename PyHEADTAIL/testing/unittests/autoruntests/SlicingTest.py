
# coding: utf-8

# In[1]:

import sys, os
BIN = os.path.expanduser("../../../..")
sys.path.append(BIN)


# In[2]:

import numpy as np
from scipy.constants import m_p, c, e
from scipy.constants import e as ee

from PyHEADTAIL.particles.particles import Particles
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer, ModeIsUniformCharge


# In[3]:
def run():
    #HELPERS
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
        if not all_pass:
            print ('  Particle_indices_of_slice and slice_index_of_particle FAILED')

    def slice_check_statistics(slice_set):
        '''Test if statistics functions are executable. No value
        checking
        '''
        slice_set.mean_x
        slice_set.sigma_x
        slice_set.epsn_x
        slice_set.mean_y
        slice_set.sigma_y
        slice_set.epsn_y
        slice_set.mean_z
        slice_set.sigma_z
        slice_set.epsn_z
        slice_set.mean_xp
        slice_set.mean_yp
        slice_set.mean_dp
        slice_set.sigma_dp


    def call_slice_set_attributes(bunch, slice_set, line_density_testing=True):
        # Call all the properties / attributes / methods.
        slice_set.z_cut_head
        slice_set.z_cut_tail
        slice_set.z_centers
        slice_set.n_slices
        slice_set.slice_widths
        slice_set.slice_positions
        slice_set.n_macroparticles_per_slice
        slice_set.particles_within_cuts
        slice_set.particle_indices_by_slice


        # if line_density_testing:
        #     slice_set.line_density_derivative_gauss()
        #     slice_set.line_density_derivative()


    def call_slicer_attributes():
        pass

    def clean_bunch(bunch):
        bunch.clean_slices()


# In[4]:
    # Basic parameters.
    n_macroparticles = 500

    Q_x = 64.28
    Q_y = 59.31
    Q_s = 0.0020443

    C = 26658.883
    R = C / (2.*np.pi)

    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj = 66.0064
    beta_y_inj = 71.5376
    alpha_0 = [0.0003225]


    # In[5]:

    # general simulation parameters
    n_particles = 10000

    # machine parameters
    circumference = 157.
    inj_alpha_x = 0
    inj_alpha_y = 0
    inj_beta_x = 5.9 # in [m]
    inj_beta_y = 5.7 # in [m]
    Qx = 5.1
    Qy = 6.1
    gamma_tr = 4.05
    alpha_c_array = [gamma_tr**-2]
    V_rf = 8e3 # in [V]
    harmonic = 1
    phi_offset = 0 # measured from aligned focussing phase (0 or pi)
    pipe_radius = 5e-2

    # beam parameters
    Ekin = 1.4e9 # in [eV]
    intensity = 1.684e12
    epsn_x = 2.5e-6 # in [m*rad]
    epsn_y = 2.5e-6 # in [m*rad]
    epsn_z = 1.2 # 4pi*sig_z*sig_dp (*p0/e) in [eVs]

    # calculations
    gamma = 1 + ee * Ekin / (m_p * c**2)
    beta = np.sqrt(1 - gamma**-2)
    eta = alpha_c_array[0] - gamma**-2
    if eta < 0:
        phi_offset = np.pi - phi_offset
    Etot = gamma * m_p * c**2 / ee
    p0 = np.sqrt(gamma**2 - 1) * m_p * c
    Q_s = np.sqrt(np.abs(eta) * V_rf / (2 * np.pi * beta**2 * Etot))
    beta_z = np.abs(eta) * circumference / (2 * np.pi * Q_s)
    turn_period = circumference / (beta * c)

    bunch = generators.generate_Gaussian6DTwiss( # implicitly tests Gaussian and Gaussian2DTwiss as well
        n_particles, intensity, ee, m_p, circumference, gamma=gamma,
        alpha_x=inj_alpha_x, beta_x=inj_beta_x, epsn_x=epsn_x,
        alpha_y=inj_alpha_y, beta_y=inj_beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z
        )



    # In[6]:

    # Uniform bin slicer
    n_slices = 10
    n_sigma_z = 2
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)

    # Request slice_set from bunch with the uniform_bin_slicer config.
    bunch._slice_sets
    uniform_bin_slice_set = bunch.get_slices(uniform_bin_slicer, statistics=True)
    bunch._slice_sets

    uniform_bin_slicer.config
    call_slice_set_attributes(bunch, uniform_bin_slice_set)
    #call_slicer_attributes(uniform_bin_slice_set)

    # Let bunch remove the slice_set.
    bunch.clean_slices()
    bunch._slice_sets


    # In[7]:

    # Uniform charge slicer
    n_slices = 10
    n_sigma_z = 2

    clean_bunch(bunch)

    uniform_charge_slicer = UniformChargeSlicer(n_slices, n_sigma_z)
    uniform_charge_slice_set = bunch.get_slices(uniform_charge_slicer, statistics=True)
    uniform_charge_slice_set.mode
    uniform_charge_slicer.config
    call_slice_set_attributes(bunch, uniform_charge_slice_set, line_density_testing=False)

    try:
        call_slice_set_attributes(bunch, uniform_charge_slice_set, line_density_testing=True)
    except ModeIsNotUniformBin as e:
        pass



    # In[8]:

    # Other cases. When are slicers equal?
    n_slices = 10
    n_sigma_z = 2
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)
    uniform_charge_slicer = UniformChargeSlicer(n_slices, n_sigma_z)


    # In[9]:

    # Other cases. When are slicers equal?
    n_slices = 10
    n_sigma_z = 2
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)
    uniform_bin_slicer_2 = UniformBinSlicer(n_slices, n_sigma_z)


    # In[10]:

    # Does bunch slice_set management work?
    n_slices = 10
    n_sigma_z = 2

    clean_bunch(bunch)

    uniform_charge_slicer = UniformChargeSlicer(n_slices, n_sigma_z)
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)

    uniform_charge_slice_set = bunch.get_slices(uniform_charge_slicer)
    uniform_bin_slice_set = bunch.get_slices(uniform_bin_slicer)
    uniform_charge_slice_set = bunch.get_slices(uniform_charge_slicer)

    bunch.clean_slices()


    # In[11]:

    # Old method update_slices should give RuntimeError.
    n_slices = 10
    n_sigma_z = 2

    clean_bunch(bunch)
    uniform_charge_slicer = UniformChargeSlicer(n_slices, n_sigma_z)


    # In[12]:

    # beam parameters attached to SliceSet?
    n_slices = 10
    n_sigma_z = 2

    clean_bunch(bunch)

    slicer = UniformBinSlicer(n_slices, n_sigma_z)
    slices = bunch.get_slices(slicer)

    beam_parameters = slicer.extract_beam_parameters(bunch)

    for p_name, p_value in beam_parameters.iteritems():
        pass

    # In[14]:

    # CASE I
    # UniformBinSlicer, no longitudinal cut.
    n_slices = 10
    n_sigma_z = None
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)

    clean_bunch(bunch)

    bunch._slice_sets

    # Request slice_set from bunch with the uniform_bin_slicer config.
    uniform_bin_slice_set = bunch.get_slices(uniform_bin_slicer)
    bunch._slice_sets



    # In[15]:

    # CASE II
    # UniformBinSlicer, n_sigma_z = 1
    n_slices = 10
    n_sigma_z = 1
    uniform_bin_slicer = UniformBinSlicer(n_slices, n_sigma_z)

    clean_bunch(bunch)

    bunch._slice_sets

    # Request slice_set from bunch with the uniform_bin_slicer config.
    uniform_bin_slice_set = bunch.get_slices(uniform_bin_slicer)
    bunch._slice_sets



    # In[16]:

    # CASE II b.
    # UniformBinSlicer, set z_cuts
    n_slices = 10
    z_cuts = (-0.05, 0.15)
    uniform_bin_slicer = UniformBinSlicer(n_slices, z_cuts=z_cuts)

    clean_bunch(bunch)

    bunch._slice_sets

    # Request slice_set from bunch with the uniform_bin_slicer config.
    uniform_bin_slice_set = bunch.get_slices(uniform_bin_slicer)
    bunch._slice_sets




    # In[18]:

    # CASE III
    # UniformChargeSlicer, no longitudinal cut.
    n_slices = 10
    n_sigma_z = None
    uniform_charge_slicer = UniformChargeSlicer(n_slices, n_sigma_z)

    clean_bunch(bunch)

    bunch._slice_sets

    # Request slice_set from bunch with the uniform_charge_slicer config.
    bunch.get_slices(uniform_charge_slicer)
    bunch._slice_sets


if __name__ == '__main__':
    run()
