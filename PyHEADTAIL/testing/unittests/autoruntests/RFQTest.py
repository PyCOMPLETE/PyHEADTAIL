
# coding: utf-8

# In[1]:

import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)


# In[2]:

import numpy as np
from scipy.constants import m_p, c, e
import matplotlib.pyplot as plt

np.random.seed(0)

from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.rfq.rfq import (
    RFQLongitudinalKick, RFQTransverseKick, RFQTransverseDetuner)
import PyHEADTAIL.particles.generators as generators


# In[3]:

# HELPERS
def run():
    def track(bunch, map_):
        for i in range(n_turns):
            for m in map_:
                m.track(bunch)

    def generate_bunch(n_macroparticles, alpha_x, alpha_y, beta_x, beta_y, linear_map):

        intensity = 1.05e11
        sigma_z = 0.059958
        gamma = 3730.26
        gamma_t = 1. / np.sqrt(alpha_0)
        p0 = np.sqrt(gamma**2 - 1) * m_p * c

        beta_z = (linear_map.eta(dp=0, gamma=gamma) * linear_map.circumference /
                  (2 * np.pi * linear_map.Q_s))

        epsn_x = 3.75e-6 # [m rad]
        epsn_y = 3.75e-6 # [m rad]
        epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)

        bunch = generators.generate_Gaussian6DTwiss(
            macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
            gamma=gamma, mass=m_p, circumference=C,
            alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
            alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
            beta_z=beta_z, epsn_z=epsn_z)

        return bunch


    # In[4]:

    # Basic parameters.
    n_turns = 3
    n_segments = 5
    n_macroparticles = 10

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

    # Parameters for transverse map.
    s = np.arange(0, n_segments + 1) * C / n_segments

    alpha_x = alpha_x_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)

    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_y = np.zeros(n_segments)


    # In[6]:

    # TEST CASE SETUP
    def gimme(*detuners):
        trans_map = TransverseMap(
            s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, *detuners)
        long_map = LinearMap(alpha_0, C, Q_s)
        bunch = generate_bunch(
            n_macroparticles, alpha_x_inj, alpha_y_inj, beta_x_inj, beta_y_inj,
            long_map)
        return bunch, trans_map, long_map


    # In[7]:

    # CASE I
    # With RFQ transverse as detuner

    # EXPECTED TUNE SPREADS AT THE GIVEN SETTINGS ARE 3.4e-4 FOR HORIZONTAL
    # AND 1.7e-4 FOR VERTICAL.
    rfq_t = RFQTransverseDetuner(v_2=2e9, omega=800e6*2.*np.pi, phi_0=0.,
                                 beta_x_RFQ=200., beta_y_RFQ=100.)
    bunch, trans_map, long_map = gimme(rfq_t)

    trans_one_turn = [ m for m in trans_map ]
    map_ = trans_one_turn + [long_map]

    track(bunch, map_)


    # In[8]:

    # CASE II
    # With RFQ transverse as kick

    # EXPECTED TUNE SPREADS AT THE GIVEN SETTINGS ARE ROUGHLY 1.2e-4 FOR HORIZONTAL
    # AND FOR VERTICAL.
    rfq_t = RFQTransverseKick(v_2=2e9, omega=800e6*2.*np.pi, phi_0=0.)
    bunch, trans_map, long_map = gimme()

    trans_one_turn = [ m for m in trans_map ]
    map_ = trans_one_turn + [long_map] + [rfq_t]

    track(bunch, map_)


    # In[9]:

    # CASE III
    # With RFQ longitudinal Kick.

    # NEGLIGIBLE TUNE SPREADS EXPECTED.
    rfq_long = RFQLongitudinalKick(v_2=2e9, omega=800e6*2.*np.pi, phi_0=0.)

    bunch, trans_map, long_map = gimme()
    trans_one_turn = [ m for m in trans_map ]
    map_ = trans_one_turn + [long_map] + [rfq_long]

    track(bunch, map_)


    # In[ ]:


if __name__ == '__main__':
    run()

