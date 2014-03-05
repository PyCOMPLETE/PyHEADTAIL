from __future__ import division
'''
@file matching
@author Kevin Li
@date February 2014
@brief Module for matching transverse and longitudinal distributions
@copyright CERN
'''


import numpy as np


from configuration import *


def match_transverse(epsn_x, epsn_y, ltm=None):

    beta_x, beta_y = 100, 100

    def match(bunch):
        sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (bunch.gamma * bunch.beta))
        sigma_xp = sigma_x / beta_x
        sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (bunch.gamma * bunch.beta))
        sigma_yp = sigma_y / beta_y

        bunch.x *= sigma_x
        bunch.xp *= sigma_xp
        bunch.y *= sigma_y
        bunch.yp *= sigma_yp

    return match

def match_longitudinal(length, bucket, matching=None):

    def match(bunch):
        p0 = bunch.mass * bunch.gamma * bunch.beta * c / e

        sigma_dz = length
        sigma_dp = epsn_z / (4 * np.pi * sigma_dz) / p0

        bunch.dz *= sigma_dz
        bunch.dp *= sigma_dp

    return match

# if cavity:
#     if matching == 'simple':
#         match_simple(self, cavity)
#     elif matching == 'full':
#         match_full(self, cavity)
#     else:
#         raise ValueError("Unknown matching " + matching)
# else:
#     pass
