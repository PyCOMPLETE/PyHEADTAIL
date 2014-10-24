#!/usr/bin/python
from __future__ import division

import unittest

import numpy as np
from scipy.constants import c, e, m_p

from ..particles.generators import (
    ParticleGenerator,
    Gaussian6D
    )
from ..trackers.rf_bucket import RFBucket
from ..trackers.transverse_tracking import TransverseMap
from ..trackers.simple_long_tracking import LinearMap, RFSystems

class TestParticleGenerators(unittest.TestCase):

    def setUp(self):
        # general simulation parameters
        self.macroparticlenumber = 10000
        self.n_segments = 1

        # machine parameters
        self.C = 157.
        self.inj_alpha_x = 0
        self.inj_alpha_y = 0
        self.inj_beta_x = 5.9 # in [m]
        self.inj_beta_y = 5.7 # in [m]
        self.Qx = 5.1
        self.Qy = 6.1
        self.gamma_tr = 4.05
        self.alpha_array = [self.gamma_tr**-2]
        self.V_rf = 8e3 # in [V]
        self.harmonic = 1
        self.phi_offset = 0 # measured from aligned focussing phase (0 or pi)

        # beam parameters
        self.Ekin = 1.4e9 # in [eV]
        self.intensity = 1.684e12
        self.epsn_x = 2.5e-6 # in [m*rad]
        self.epsn_y = 2.5e-6 # in [m*rad]
        self.epsn_z = 1.2 # 4pi*sig_z*sig_dp (*p0/e) in [eVs]

        # calculations
        self.gamma = 1 + e * self.Ekin / (m_p * c**2)
        self.beta = np.sqrt(1 - self.gamma**-2)
        self.betagamma = np.sqrt(self.gamma**2 - 1)
        self.eta = self.alpha_c_array[0] - self.gamma**-2
        if self.eta < 0:
            self.phi_offset = np.pi - self.phi_offset
        self.Etot = self.gamma * m_p * c**2 / e
        self.p0 = np.sqrt(self.gamma**2 - 1) * m_p * c
        self.Qs = np.sqrt(np.abs(self.eta) * self.V_rf /
                          (2 * np.pi * self.beta**2 * self.Etot))
        self.beta_z = np.abs(self.eta) * self.C / (2 * np.pi * self.Qs)
        self.turn_period = self.C / (self.beta * c)

        self.sigma_x = np.sqrt(self.epsn_x * self.inj_beta_x/ self.betagamma)
        self.sigma_y = np.sqrt(self.epsn_y * self.inj_beta_y/ self.betagamma)
        self.sigma_z = np.sqrt(self.epsn_z * self.beta_z /
                               (4 * np.pi * self.p0/e))
        self.sigma_xp = self.sigma_x / self.inj_beta_x
        self.sigma_yp = self.sigma_y / self.inj_beta_y
        self.sigma_dp = self.sigma_z / self.beta_z

        # BETATRON
        # Loop on number of segments and create the TransverseSegmentMap
        # for each segment.
        self.s = np.arange(0, self.n_segments + 1) * self.C / self.n_segments
        self.alpha_x = self.inj_alpha_x * np.ones(self.n_segments)
        self.beta_x  = self.inj_beta_x * np.ones(self.n_segments)
        self.D_x     = np.zeros(self.n_segments)
        self.alpha_y = self.inj_alpha_y * np.ones(self.n_segments)
        self.beta_y  = self.inj_beta_y * np.ones(self.n_segments)
        self.D_y     = np.zeros(self.n_segments)

    def create_Gaussian6D(self):
        return generators.Gaussian6D(
            self.n_particles, self.intensity, self.e, self.m_p,
            self.circumference, self.gamma,
            self.sigma_x, self.sigma_xp,
            self.sigma_y, self.sigma_yp,
            self.sigma_z, self.sigma_dp
            )

    def test_Gaussian6D_macroparticlenumber(self):
        bunch = create_Gaussian6D().generate()
        self.assertTrue(bunch.macroparticlenumber == self.macroparticlenumber)

if __name__ == '__main__':
    unittest.main()
