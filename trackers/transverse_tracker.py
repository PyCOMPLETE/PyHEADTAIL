'''
Created on 07.01.2014

@author: Kevin Li, Michael Schenk
'''

from __future__ import division
import numpy as np
from scipy.constants import e, c

sin = np.sin
cos = np.cos


# The dependence of app_x,y,xy,yx on p0 is put explicitly in the detune function only.
# Variables dQ_x,y, dQp_x,y, dapp_x,y and dapp_xy,yx refer to segment [s_i, s_i+1], ie. are in general not 1-turn integrated values.
class LinearPeriodicMap(object):
    def __init__(self, I, J, beta_x, dQ_x, dQp_x, dapp_x, dapp_xy,
                             beta_y, dQ_y, dQp_y, dapp_y, dapp_yx):
        self.I = I
        self.J = J

        self.beta_x  = beta_x
        self.dQ_x    = dQ_x
        self.dQp_x   = dQp_x
        self.dapp_x  = dapp_x
        self.dapp_xy = dapp_xy

        self.beta_y  = beta_y
        self.dQ_y    = dQ_y
        self.dQp_y   = dQp_y
        self.dapp_y  = dapp_y
        self.dapp_yx = dapp_yx

    #~ @profile
    def track(self, beam):
        dphi_x, dphi_y = self.detune(beam)

        cos_dphi_x = cos(dphi_x)
        cos_dphi_y = cos(dphi_y)
        sin_dphi_x = sin(dphi_x)
        sin_dphi_y = sin(dphi_y)

        M00 = self.I[0, 0] * cos_dphi_x + self.J[0, 0] * sin_dphi_x
        M01 = self.I[0, 1] * cos_dphi_x + self.J[0, 1] * sin_dphi_x
        M10 = self.I[1, 0] * cos_dphi_x + self.J[1, 0] * sin_dphi_x
        M11 = self.I[1, 1] * cos_dphi_x + self.J[1, 1] * sin_dphi_x
        M22 = self.I[2, 2] * cos_dphi_y + self.J[2, 2] * sin_dphi_y
        M23 = self.I[2, 3] * cos_dphi_y + self.J[2, 3] * sin_dphi_y
        M32 = self.I[3, 2] * cos_dphi_y + self.J[3, 2] * sin_dphi_y
        M33 = self.I[3, 3] * cos_dphi_y + self.J[3, 3] * sin_dphi_y

        beam.x, beam.xp = M00 * beam.x + M01 * beam.xp, M10 * beam.x + M11 * beam.xp
        beam.y, beam.yp = M22 * beam.y + M23 * beam.yp, M32 * beam.y + M33 * beam.yp

    #~ @profile
    def detune(self, beam):
        Jx = (beam.x ** 2 + (self.beta_x * beam.xp) ** 2) / (2. * self.beta_x)
        Jy = (beam.y ** 2 + (self.beta_y * beam.yp) ** 2) / (2. * self.beta_y)

        dphi_x = 2 * np.pi * (self.dQ_x
                            + self.dQp_x * beam.dp
                            + self.dapp_x/beam.p0 * Jx
                            + self.dapp_xy/beam.p0 * Jy)
        dphi_y = 2 * np.pi * (self.dQ_y
                            + self.dQp_y * beam.dp
                            + self.dapp_y/beam.p0 * Jy
                            + self.dapp_yx/beam.p0 * Jx)

        return dphi_x, dphi_y


class TransverseTracker(object):
    '''
    classdocs
    '''
    def __init__(self, s, alpha_x, beta_x, D_x, Q_x, Qp_x, app_x, app_xy,
                          alpha_y, beta_y, D_y, Q_y, Qp_y, app_y, app_yx):
        '''
        Most minimalistic constructor. Pure python name binding.
        '''
        assert((len(s)-1) == len(alpha_x) == len(beta_x) == len(D_x)
                          == len(alpha_y) == len(beta_y) == len(D_y))

        self.s = s

        self.alpha_x = alpha_x
        self.beta_x  = beta_x
        self.D_x     = D_x
        self.alpha_y = alpha_y
        self.beta_y  = beta_y
        self.D_y     = D_y

        # Segment the 1-turn integrated quantities Q_x,y, Qp_x,y, app_x,y and app_xy,yx.
        # Scaling assumes the effect of app_x,y, ... to be uniform around the accelerator ring.
        # self.s[-1] = C.
        scale_to_segment = np.diff(s) / s[-1]

        self.dQ_x    = scale_to_segment*Q_x
        self.dQp_x   = scale_to_segment*Qp_x
        self.dapp_x  = scale_to_segment*app_x
        self.dapp_xy = scale_to_segment*app_xy
        self.dQ_y    = scale_to_segment*Q_y
        self.dQp_y   = scale_to_segment*Qp_y
        self.dapp_y  = scale_to_segment*app_y
        self.dapp_yx = scale_to_segment*app_yx


    @classmethod
    def default(cls, n_segments, C, beta_x, Q_x, Qp_x, app_x, app_xy,
                                    beta_y, Q_y, Qp_y, app_y, app_yx):

        s = np.arange(0, n_segments + 1) * C / n_segments

        alpha_x = np.zeros(n_segments)
        beta_x  = np.ones(n_segments) * beta_x
        D_x     = np.zeros(n_segments)
        alpha_y = np.zeros(n_segments)
        beta_y  = np.ones(n_segments) * beta_y
        D_y     = np.zeros(n_segments)

        self   = cls(s, alpha_x, beta_x, D_x, Q_x, Qp_x, app_x, app_xy,
                        alpha_y, beta_y, D_y, Q_y, Qp_y, app_y, app_yx)
        self.M = self.build_maps()

        return self.M


    @classmethod
    def from_copy(cls, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y,
                  Q_x, Qp_x, app_x, app_xy, Q_y, Qp_y, app_y, app_yx):

        s       = np.copy(s)
        alpha_x = np.copy(alpha_x)
        beta_x  = np.copy(beta_x)
        D_x     = np.copy(D_x)
        alpha_y = np.copy(alpha_y)
        beta_y  = np.copy(beta_y)
        D_y     = np.copy(D_y)

        self   = cls(s, alpha_x, beta_x, D_x, Q_x, Qp_x, app_x, app_xy,
                        alpha_y, beta_y, D_y, Q_y, Qp_y, app_y, app_yx)
        self.M = self.build_maps()

        return self.M


    def build_maps(self):
        n_segments = len(self.s) - 1

        # Allocate coefficient matrices
        I = [np.zeros((4, 4)) for i in xrange(n_segments)]
        J = [np.zeros((4, 4)) for i in xrange(n_segments)]

        for i in range(n_segments):
            s0 = i % n_segments
            s1 = (i + 1) % n_segments
            # sine component
            I[i][0, 0] = np.sqrt(self.beta_x[s1] / self.beta_x[s0])
            I[i][0, 1] = 0
            I[i][1, 0] = np.sqrt(1 / (self.beta_x[s0] * self.beta_x[s1])) \
                       * (self.alpha_x[s0] - self.alpha_x[s1])
            I[i][1, 1] = np.sqrt(self.beta_x[s0] / self.beta_x[s1])
            I[i][2, 2] = np.sqrt(self.beta_y[s1] / self.beta_y[s0])
            I[i][2, 3] = 0
            I[i][3, 2] = np.sqrt(1 / (self.beta_y[s0] * self.beta_y[s1])) \
                       * (self.alpha_y[s0] - self.alpha_y[s1])
            I[i][3, 3] = np.sqrt(self.beta_y[s0] / self.beta_y[s1])
            # cosine component
            J[i][0, 0] = np.sqrt(self.beta_x[s1] / self.beta_x[s0]) \
                       * self.alpha_x[s0]
            J[i][0, 1] = np.sqrt(self.beta_x[s0] * self.beta_x[s1])
            J[i][1, 0] = -np.sqrt(1 / (self.beta_x[s0] * self.beta_x[s1])) \
                       * (1 + self.alpha_x[s0] * self.alpha_x[s1])
            J[i][1, 1] = -np.sqrt(self.beta_x[s0] / self.beta_x[s1]) \
                       * self.alpha_x[s1]
            J[i][2, 2] = np.sqrt(self.beta_y[s1] / self.beta_y[s0]) \
                       * self.alpha_y[s0]
            J[i][2, 3] = np.sqrt(self.beta_y[s0] * self.beta_y[s1])
            J[i][3, 2] = -np.sqrt(1 / (self.beta_y[s0] * self.beta_y[s1])) \
                       * (1 + self.alpha_y[s0] * self.alpha_y[s1])
            J[i][3, 3] = -np.sqrt(self.beta_y[s0] / self.beta_y[s1]) \
                       * self.alpha_y[s1]

        # Generate a linear periodic map for every segment.
        M = [LinearPeriodicMap(I[i], J[i], self.beta_x[i], self.dQ_x[i], self.dQp_x[i], self.dapp_x[i], self.dapp_xy[i],
                                           self.beta_y[i], self.dQ_y[i], self.dQp_y[i], self.dapp_y[i], self.dapp_yx[i])
             for i in xrange(n_segments)]

        return M

#         cls.R = [np.kron(np.eye(2), np.ones((2, 2)))
#                  for i in xrange(cls.n_segments)]
#         cls.N0 = [np.kron(np.eye(2), np.ones((2, 2)))
#                   for i in xrange(cls.n_segments)]
#         cls.N1 = [np.kron(np.eye(2), np.ones((2, 2)))
#                   for i in xrange(cls.n_segments)]
#
#         dmu_x = np.diff(cls.mu_x)
#         dmu_y = np.diff(cls.mu_y)
#
#         for i in range(cls.n_segments):
#             s0 = i % cls.n_segments
#             s1 = (i + 1) % cls.n_segments
#
#             cls.R[i][0, 0] *= np.cos(dmu_x[s0])
#             cls.R[i][0, 1] *= np.sin(dmu_x[s0])
#             cls.R[i][1, 0] *= -np.sin(dmu_x[s0])
#             cls.R[i][1, 1] *= np.cos(dmu_x[s0])
#             cls.R[i][2, 2] *= np.cos(dmu_y[s0])
#             cls.R[i][2, 3] *= np.sin(dmu_y[s0])
#             cls.R[i][3, 2] *= -np.sin(dmu_y[s0])
#             cls.R[i][3, 3] *= np.cos(dmu_y[s0])
#
#             cls.N0[i][0, 0] = cls.N0[i][0, 0] * 1. / np.sqrt(cls.beta_x[s0])
#             cls.N0[i][0, 1] *= 0
#             cls.N0[i][1, 0] *= cls.alpha_x[s0] / np.sqrt(cls.beta_x[s0])
#             cls.N0[i][1, 1] *= np.sqrt(cls.beta_x[s0])
#             cls.N0[i][2, 2] *= 1 / np.sqrt(cls.beta_y[s0])
#             cls.N0[i][2, 3] *= 0
#             cls.N0[i][3, 2] *= cls.alpha_y[s0] / np.sqrt(cls.beta_y[s0])
#             cls.N0[i][3, 3] *= np.sqrt(cls.beta_y[s0])
#
#             cls.N1[i][0, 0] *= np.sqrt(cls.beta_x[s1])
#             cls.N1[i][0, 1] *= 0
#             cls.N1[i][1, 0] *= -cls.alpha_x[s1] / np.sqrt(cls.beta_x[s1])
#             cls.N1[i][1, 1] *= 1 / np.sqrt(cls.beta_x[s1])
#             cls.N1[i][2, 2] *= np.sqrt(cls.beta_y[s1])
#             cls.N1[i][2, 3] *= 0
#             cls.N1[i][3, 2] *= -cls.alpha_y[s1] / np.sqrt(cls.beta_y[s1])
#             cls.N1[i][3, 3] *= 1 / np.sqrt(cls.beta_y[s1])
#
#         cls.M = [#(cls.N1[i] * cls.R[i] * cls.N0[i])
#              np.dot(cls.N1[i], np.dot(cls.R[i], cls.N0[i]))
#              for i in range(cls.n_segments)]



# CONVENIENCE FUNCTIONS
def get_LHC_octupole_parameters_from_currents(i_f, i_d):
    # Calculate app_x, app_y, app_xy = app_yx on the basis of formulae (3.6) in
    # 'THE LHC TRANSVERSE COUPLED-BUNCH INSTABILITY', N. Mounet, 2012 from
    # LHC octupole currents i_f, i_d [A].
    app_x  = 7000.*(267065.*i_f/550. - 7856.*i_d/550.)
    app_y  = 7000.*(9789.*i_f/550. - 277203.*i_d/550.)
    app_xy = 7000.*(-102261.*i_f/550. + 93331.*i_d/550.)

    convert_to_SI_units = e/(1e-9*c)
    app_x  *= convert_to_SI_units
    app_y  *= convert_to_SI_units
    app_xy *= convert_to_SI_units

    app_yx = app_xy

    return app_x, app_xy, app_y, app_yx
