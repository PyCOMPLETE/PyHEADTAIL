'''
@class WakeKick
@author Kevin Li & Michael Schenk
@date July 2014
@Class Collection of wake kicks.
@copyright CERN
'''
from __future__ import division

import numpy as np
from scipy.constants import c


class WakeKick(object):
    '''
    Base class for wake kicks (constant, dipolar, quadrupolar, ...).
    '''
    def __init__(self, wake_function, slices):

        if slices.mode == 'constant_charge':
            self._convolution = self._convolution_dot_product
        else:
            self._convolution = self._convolution_numpy

        self.wake_function = wake_function

        
    def _wake_factor(self, bunch):
        
        particles_per_macroparticle = bunch.intensity / bunch.n_macroparticles
        wake_factor = -(bunch.charge) ** 2 / (bunch.mass * bunch.gamma * (bunch.beta * c) ** 2) * particles_per_macroparticle

        return wake_factor
            
                
    def _convolution_dot_product(self, bunch, slices, f, g):

        dz_to_target_slice = [slices.z_centers] - np.transpose([slices.z_centers])
        wake = f(bunch.beta, dz_to_target_slice)
        beam_profile = g

        return np.dot(beam_profile, wake)

    
    def _convolution_numpy(self, bunch, slices, f, g):

        dz_to_target_slice = np.concatenate((slices.z_centers - slices.z_centers[-1],
                                            (slices.z_centers - slices.z_centers[0])[1:]))
        wake = f(bunch.beta, dz_to_target_slice)
        beam_profile = g

        return np.convolve(beam_profile, wake, 'valid')


'''
Constant wake kicks.
'''
class ConstantWakeKickX(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        constant_kick = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.xp += constant_kick[ix]

        
class ConstantWakeKickY(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        constant_kick = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.yp += constant_kick[ix]


class ConstantWakeKickZ(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        constant_kick = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.dp += constant_kick[ix]


'''
Dipole wake kicks.
'''
class DipoleWakeKickX(WakeKick):

    def apply(self, bunch, slices):

        first_moment_x = slices.n_macroparticles * slices.mean_x(bunch)
        dipole_kick_x = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, first_moment_x)
        ix = slices.slice_index_of_particle

        bunch.xp += dipole_kick_x[ix]


class DipoleWakeKickXY(WakeKick):

    def apply(self, bunch, slices):

        first_moment_y = slices.n_macroparticles * slices.mean_y(bunch)
        dipole_kick_xy = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, first_moment_y)
        ix = slices.slice_index_of_particle

        bunch.xp += dipole_kick_xy[ix]

        
class DipoleWakeKickY(WakeKick):

    def apply(self, bunch, slices):

        first_moment_y = slices.n_macroparticles * slices.mean_y(bunch)
        dipole_kick_y = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, first_moment_y)
        ix = slices.slice_index_of_particle

        bunch.yp += dipole_kick_y[ix]


class DipoleWakeKickYX(WakeKick):

    def apply(self, bunch, slices):

        first_moment_x = slices.n_macroparticles * slices.mean_x(bunch)
        dipole_kick_yx = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, first_moment_x)
        ix = slices.slice_index_of_particle

        bunch.yp += dipole_kick_yx[ix]

        
'''
Quadrupole wake kicks.
'''
class QuadrupoleWakeKickX(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        quadrupole_kick_x = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.xp += quadrupole_kick_x[ix] * bunch.x


class QuadrupoleWakeKickXY(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        quadrupole_kick_xy = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.xp += quadrupole_kick_xy[ix] * bunch.y

        
class QuadrupoleWakeKickY(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        quadrupole_kick_y = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.yp += quadrupole_kick_y[ix] * bunch.y


class QuadrupoleWakeKickYX(WakeKick):

    def apply(self, bunch, slices):

        zeroth_moment = slices.n_macroparticles
        quadrupole_kick_yx = self._wake_factor(bunch) * self._convolution(bunch, slices, self.wake_function, zeroth_moment)
        ix = slices.slice_index_of_particle

        bunch.yp += quadrupole_kick_yx[ix] * bunch.x
