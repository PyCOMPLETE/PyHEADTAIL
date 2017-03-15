import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)


import numpy as np
from scipy.constants import m_p, c, e


from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss
from PyHEADTAIL.trackers.longitudinal_tracking import RFSystems
import PyHEADTAIL.aperture.aperture as aperture


# In[5]:

import matplotlib.pyplot as plt


# In[6]:

def run():

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
    phi_offset = 0 # measured from aligned focusing phase (0 or pi)
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
    Q_s = np.sqrt(np.abs(eta) * V_rf / (2 * np.pi * beta**2 * Etot))
    beta_z = np.abs(eta) * circumference / (2 * np.pi * Q_s)


    # In[7]:

    def plot_phase_space(bunch, ax0, ax1, ax2, col):
        # phase spaces
        ax0.scatter(bunch.x, bunch.xp, color=col)
        ax1.scatter(bunch.y, bunch.yp, color=col)
        ax2.scatter(bunch.z, bunch.dp, color=col)
        # statistical quantities
        ax0.scatter(bunch.mean_x(), bunch.mean_xp(), color='red')
        ax1.scatter(bunch.mean_y(), bunch.mean_yp(), color='red')
        ax2.scatter(bunch.mean_z(), bunch.mean_dp(), color='red')


    # In[8]:

    def generate_bunch(n_particles):
        bunch = generate_Gaussian6DTwiss(
            n_particles, intensity, e, m_p, circumference, gamma,
            inj_alpha_x, inj_alpha_y, inj_beta_x, inj_beta_y, beta_z,
            epsn_x, epsn_y, epsn_z
            )
        return bunch



    # In[15]:

    # (I) RectangularApertureX
    bunch = generate_bunch(5)
    apt_x = aperture.RectangularApertureX(x_low=-0.004, x_high=0.005)
    apt_x.track(bunch)

    # In[10]:

    # (II) RectangularApertureY
    bunch = generate_bunch(5)

    apt_y = aperture.RectangularApertureY(y_low=-0.005, y_high=0.005)
    apt_y.track(bunch)


    # In[11]:

    # (III) RectangularApertureZ
    bunch = generate_bunch(5)
    apt_z = aperture.RectangularApertureZ(z_low=-15, z_high=25)
    apt_z.track(bunch)


    # In[12]:

    # (IV) CircularApertureXY
    bunch = generate_bunch(5)
    apt_xy = aperture.CircularApertureXY(radius=0.005)
    apt_xy.track(bunch)


    # In[13]:
    #errorgenerator
    # (V) EllipticalApertureXY

    bunch = generate_bunch(5)
    x_aper = 5e-3
    y_aper = 2e-3

    apt_xy = aperture.EllipticalApertureXY(x_aper=x_aper, y_aper=y_aper)
    apt_xy.track(bunch)

if __name__ == '__main__':
    run()
