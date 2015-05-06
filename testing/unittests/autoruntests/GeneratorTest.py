
# coding: utf-8

# In[1]:

import sys, os
BIN = os.path.dirname(__file__) # ./PyHEADTAIL/testing/unittests/
BIN = os.path.abspath( BIN ) # absolute path to unittests
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/testing/
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/
BIN = os.path.dirname( BIN ) # ../ -->  ./
BIN = os.path.dirname( BIN ) # ../ -->  ./

sys.path.append(BIN)


# In[2]:

import numpy as np
from scipy.constants import m_p, c, e


# In[3]:

from PyHEADTAIL.particles.particles import Particles
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap


# In[4]:

# In[5]:
def run():
# general simulation parameters
    n_turns = 3
    n_particles = 3
    n_segments = 1

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
    gamma = 1 + e * Ekin / (m_p * c**2)
    beta = np.sqrt(1 - gamma**-2)
    eta = alpha_c_array[0] - gamma**-2
    if eta < 0:
        phi_offset = np.pi - phi_offset
    Etot = gamma * m_p * c**2 / e
    p0 = np.sqrt(gamma**2 - 1) * m_p * c
    Qs = np.sqrt(np.abs(eta) * V_rf / (2 * np.pi * beta**2 * Etot))
    beta_z = np.abs(eta) * circumference / (2 * np.pi * Qs)
    turn_period = circumference / (beta * c)

    # BETATRON
    # Loop on number of segments and create the TransverseSegmentMap
    # for each segment.
    s = np.arange(0, n_segments + 1) * circumference / n_segments
    alpha_x = inj_alpha_x * np.ones(n_segments)
    beta_x  = inj_beta_x * np.ones(n_segments)
    D_x     = np.zeros(n_segments)
    alpha_y = inj_alpha_y * np.ones(n_segments)
    beta_y  = inj_beta_y * np.ones(n_segments)
    D_y     = np.zeros(n_segments)





    # In[7]:

    bunch = generators.Gaussian6DTwiss( # implicitly tests Gaussian and Gaussian2DTwiss as well
        n_particles, intensity, e, m_p, circumference, gamma,
        inj_alpha_x, inj_beta_x, epsn_x,
        inj_alpha_y, inj_beta_y, epsn_y,
        beta_z, epsn_z
        ).generate()


    # In[8]:

    # Gaussian6D


    # In[9]:

    bunch = generators.ImportDistribution(
        n_particles, intensity, e, m_p, circumference, gamma,
        bunch.get_coords_n_momenta_dict()
        ).generate()


    # In[10]:

    bunch = generators.Uniform3D(
        n_particles, intensity, e, m_p, circumference, gamma,
        2e-3, 2e-3, 30
        ).generate()

    # In[11]:

    transverse = TransverseMap(circumference, s, alpha_x, beta_x, D_x, alpha_y,
                               beta_y, D_y, Qx, Qy)
    longitudinal = LinearMap(alpha_c_array, circumference, Qs)

    # In[12]:

    bunch = generators.MatchGaussian6D( # implicitly tests MatchLinearLongMap and MatchTransverseMap
        n_particles, intensity, e, m_p, circumference, gamma,
        transverse, epsn_x, epsn_y, longitudinal, epsn_z
        ).generate()


    # In[13]:

    rfsystems = RFSystems(circumference, [harmonic], [V_rf], [phi_offset],
                          alpha_c_array, gamma)


    # In[14]:

    bunch = generators.MatchRFBucket6D( # implicitly tests MatchRFBucket2D and MatchTransverseMap
        n_particles, intensity, e, m_p, circumference, gamma,
        transverse, epsn_x, epsn_y, rfsystems.get_bucket(gamma=gamma), epsn_z
        ).generate()


    # In[15]:

    bunch = generators.Gaussian6DTwiss(
        n_particles, intensity, e, m_p, circumference, gamma,
        inj_alpha_x, inj_beta_x, epsn_x,
        inj_alpha_y, inj_beta_y, epsn_y, beta_z, epsn_z
        ).generate()
    generators.MatchRFBucket2D(
        n_particles, intensity, e, m_p, circumference, gamma,
        rfsystems.get_bucket(gamma=gamma), epsn_z=epsn_z
        ).update(bunch)


    # In[16]:

    sigma_z = np.sqrt(beta_z * e * epsn_z / (4. * np.pi * p0))
    sigma_dp = sigma_z / beta_z

    bunch = generators.CutRFBucket6D( # implicitly tests CutRFBucket2D as well.
        n_particles, intensity, e, m_p, circumference, gamma,
        transverse, epsn_x, epsn_y, sigma_z, sigma_dp,
        rfsystems.get_bucket(gamma=gamma).make_is_accepted(0.6)
        ).generate()


    # In[17]:

    coords_n_momenta_dict = bunch.get_coords_n_momenta_dict()


    # In[19]:

    # Test particles.update method.
    bunch = generators.Gaussian6DTwiss(
        n_particles, intensity, e, m_p, circumference, gamma,
        inj_alpha_x, inj_beta_x, epsn_x,
        inj_alpha_y, inj_beta_y, epsn_y,
        beta_z, epsn_z
        ).generate()
    x = np.random.randn(bunch.macroparticlenumber)
    xp = np.random.randn(bunch.macroparticlenumber)
    xx = np.random.randn(bunch.macroparticlenumber)
    xxp = np.random.randn(bunch.macroparticlenumber)
    coords_momenta_dict_to_update = {'x': x, 'xp': xp, 'xx': xx, 'xxp': xxp}
    bunch.update(coords_momenta_dict_to_update)


if __name__ == '__main__':
    run()


# In[ ]:
