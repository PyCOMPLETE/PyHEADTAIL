import numpy as np
from scipy.constants import c, e

from simple_long_tracker_2 import LongitudinalMap, clean_slices

sin = np.sin
cos = np.cos


class LinearMap(LongitudinalMap):
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    self.alpha is the linear momentum compaction factor.
    '''

    def __init__(self, circumference, alpha, Qs):
        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        self.circumference = circumference
        self.alpha = alpha
        self.Qs = Qs

    @clean_slices
    def track(self, beam):

        eta = self.alpha - beam.gamma ** -2

        omega_0 = 2 * np.pi * beam.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = cos(dQs)
        sindQs = sin(dQs)

        z0 = beam.z
        dp0 = beam.dp

        beam.z = z0 * cosdQs - eta * beam.beta * c / omega_s * dp0 * sindQs
        beam.dp = dp0 * cosdQs + omega_s / eta / (beam.beta * c) * z0 * sindQs

        # sz[kmain] = csa * szz0 - eta * betar * C/omegas * ssa * dpp0;
        # dp[kmain] = csa * dpp0 + omegas/eta/betar/C * ssa * szz0


# class LinearMap2(object):
#     '''
#     Linear Map represented by a Courant-Snyder transportation matrix.
#     self.alpha is the linear momentum compaction factor.
#     '''

#     def __init__(self, circumference, alpha, Qs, slices_tuple):
#         """alpha is the linear momentum compaction factor,
#         Qs the synchroton tune."""
#         self.circumference = circumference
#         self.alpha = alpha
#         self.Qs = Qs
#         self.slices_tuple = slices_tuple

#     def track(self, beam):

#         eta = self.alpha - beam.gamma ** -2

#         omega_0 = 2 * np.pi * beam.beta * c / self.circumference
#         omega_s = self.Qs * omega_0

#         dQs = 2 * np.pi * self.Qs
#         cosdQs = cos(dQs)
#         sindQs = sin(dQs)

#         z0 = beam.z
#         dp0 = beam.dp

#         beam.z = z0 * cosdQs - eta * c / omega_s * dp0 * sindQs
#         beam.dp = dp0 * cosdQs + omega_s / eta / c * z0 * sindQs

#         for slices in self.slices_tuple:
#             slices.update_slices(beam)
