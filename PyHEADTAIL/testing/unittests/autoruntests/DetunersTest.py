
# coding: utf-8

# In[1]:

import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)


# In[2]:

import numpy as np
from scipy.constants import m_p, c, e
import matplotlib.pyplot as plt

from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
import PyHEADTAIL.particles.generators as generators


# In[3]:

# HELPERS
def run():
    def track(bunch, map_):
        for i in range(n_turns):
            for m in map_:
                m.track(bunch)

    def generate_bunch(n_macroparticles, alpha_x, alpha_y, beta_x, beta_y, alpha_0, Q_s, R):
        intensity = 1.05e11
        sigma_z = 0.059958
        gamma = 3730.26
        eta = alpha_0 - 1. / gamma**2
        gamma_t = 1. / np.sqrt(alpha_0)
        p0 = np.sqrt(gamma**2 - 1) * m_p * c

        beta_z = eta * R / Q_s

        epsn_x = 3.75e-6 # [m rad]
        epsn_y = 3.75e-6 # [m rad]
        epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)

        bunch = generators.generate_Gaussian6DTwiss(
            macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
            gamma=gamma, mass=m_p, circumference=C,
            alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
            alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
            beta_z=beta_z, epsn_z=epsn_z)
        #print bunch.sigma_z()

        return bunch


    # In[4]:

    # Basic parameters.
    n_turns = 3
    n_segments = 1
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
    alpha_0 = 0.0003225


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

    # CASE I
    # With amplitude detuning (python implementation)

    # EXPECTED TUNE SPREADS AT THE GIVEN SETTINGS ARE 5e-4 FOR HORIZONTAL
    # AND VERTICAL.
    bunch = generate_bunch(
        n_macroparticles, alpha_x_inj, alpha_y_inj, beta_x_inj, beta_y_inj,
        alpha_0, Q_s, R)

    ampl_det = AmplitudeDetuning.from_octupole_currents_LHC(i_focusing=400, i_defocusing=-400)
    trans_map = TransverseMap(
        s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, [ampl_det])

    trans_one_turn = [ m for m in trans_map ]
    map_ = trans_one_turn

    track(bunch, map_)


    # In[7]:

    # CASE II
    # With first order Chromaticity (python implementation)
    bunch = generate_bunch(
        n_macroparticles, alpha_x_inj, alpha_y_inj, beta_x_inj, beta_y_inj,
        alpha_0, Q_s, R)

    chroma = Chromaticity(Qp_x=[6], Qp_y=[3])
    trans_map = TransverseMap(
        s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, [chroma])

    trans_one_turn = [ m for m in trans_map ]
    map_ = trans_one_turn

    track(bunch, map_)


    # In[8]:

    # CASE III
    # With higher order Chromaticity (python implementation)
    bunch = generate_bunch(
        n_macroparticles, alpha_x_inj, alpha_y_inj, beta_x_inj, beta_y_inj,
        alpha_0, Q_s, R)

    chroma = Chromaticity(Qp_x=[6., 4e4], Qp_y=[3., 0., 2e8])
    trans_map = TransverseMap(
        s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, [chroma])

    trans_one_turn = [ m for m in trans_map ]
    map_ = trans_one_turn

    track(bunch, map_)



    # In[ ]:


if __name__ == '__main__':
    run()

