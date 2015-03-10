"""
@class WakeKick
@author Kevin Li, Michael Schenk
@date July 2014
@brief Implementation of the wake kicks, i.e. of the elementary objects
       describing the effects of a wake field.
@copyright CERN
"""
from __future__ import division

import numpy as np
from scipy.constants import c
from abc import ABCMeta, abstractmethod


class WakeKick(object):
    """ Abstract base class for wake kick classes, like e.g. the
    DipoleWakeKickX.
    Provides the basic and universal methods to calculate the strength
    of a wake kick. Two implementations of the convolution are
    available. Based on what slicer mode (uniform_bin, uniform_charge)
    is used, the self._convolution method is bound to one or the other.
    The self.apply(bunch, slice_set) method calculates and applies the
    corresponding kick to the particles of the bunch that are located
    inside the slicing region defined by a slice_set. This should be
    the only method to be implemented for a child class inheriting from
    the WakeKick class. """

    __metaclass__ = ABCMeta

    def __init__(self, wake_function, slicer_mode):
        """ Universal constructor for WakeKick objects. The slicer_mode
        is passed only to decide about which of the two implementations
        of the convolution the self._convolution method is bound to. """
        self.wake_function = wake_function

        if slicer_mode == 'uniform_charge':
            self._convolution = self._convolution_dot_product
        elif slicer_mode == 'uniform_bin':
            self._convolution = self._convolution_numpy
        else:
            raise ValueError("Unknown slicer_mode. Must either be \n" +
                             "'uniform_bin' or 'uniform_charge'. \n")

    @abstractmethod
    def apply(self, bunch, slice_set):
        """ Calculates and applies the corresponding wake kick to the
        bunch conjugate momenta using the given slice_set. Only
        particles within the slicing region, i.e particles_within_cuts
        (defined by the slice_set) experience a kick. """
        pass

    @staticmethod
    def _wake_factor(bunch):
        """ Universal scaling factor for the strength of a wake field
        kick. """
        wake_factor = (-(bunch.charge)**2 / (bunch.mass * bunch.gamma *
                       (bunch.beta * c)**2) * bunch.particlenumber_per_mp)
        return wake_factor

    def _convolution_dot_product(self, bunch, slice_set, beam_profile):
        """ Implementation of the convolution of wake_field and
        beam_profile using the numpy dot product. To be used with the
        'uniform_charge' slicer mode. """
        dz_to_target_slice = ([slice_set.z_centers] -
                              np.transpose([slice_set.z_centers]))
        wake = self.wake_function(bunch.beta, dz_to_target_slice)

        return np.dot(beam_profile, wake)

    def _convolution_numpy(self, bunch, slice_set, beam_profile):
        """ Implementation of the convolution of wake_field and
        beam_profile using the numpy built-in numpy.convolve method.
        Recommended use with the 'uniform_bin' slicer mode for higher
        performance. """
        dz_to_target_slice = np.concatenate(
            (slice_set.z_centers - slice_set.z_centers[-1],
            (slice_set.z_centers - slice_set.z_centers[0])[1:]))
        wake = self.wake_function(bunch.beta, dz_to_target_slice)

        return np.convolve(beam_profile, wake, 'valid')


""" Constant wake kicks """

class ConstantWakeKickX(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a constant wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        constant_kick = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] += constant_kick.take(s_idx)


class ConstantWakeKickY(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a constant wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        constant_kick = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.yp[p_idx] += constant_kick.take(s_idx)


class ConstantWakeKickZ(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a constant wake kick to bunch.dp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        constant_kick = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.dp[p_idx] += constant_kick.take(s_idx)


""" Dipolar wake kicks """

class DipoleWakeKickX(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a dipolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        first_moment_x = (
            slice_set.n_macroparticles_per_slice * slice_set.mean_x)
        dipole_kick_x = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, first_moment_x))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] += dipole_kick_x.take(s_idx)


class DipoleWakeKickXY(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a dipolar (cross term x-y) wake kick
        to bunch.xp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick. """
        first_moment_y = (
            slice_set.n_macroparticles_per_slice * slice_set.mean_y)
        dipole_kick_xy = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, first_moment_y))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] += dipole_kick_xy.take(s_idx)


class DipoleWakeKickY(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a dipolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        first_moment_y = (
            slice_set.n_macroparticles_per_slice * slice_set.mean_y)
        dipole_kick_y = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, first_moment_y))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.yp[p_idx] += dipole_kick_y.take(s_idx)


class DipoleWakeKickYX(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a dipolar (cross term y-x) wake kick
        to bunch.yp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick. """
        first_moment_x = (
            slice_set.n_macroparticles_per_slice * slice_set.mean_x)
        dipole_kick_yx = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, first_moment_x))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.yp[p_idx] += dipole_kick_yx.take(s_idx)


""" Quadrupolar wake kicks """

class QuadrupoleWakeKickX(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a quadrupolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        quadrupole_kick_x = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] += quadrupole_kick_x.take(s_idx) * bunch.x.take(p_idx)


class QuadrupoleWakeKickXY(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a quadrupolar (cross term x-y) wake
        kick to bunch.xp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        quadrupole_kick_xy = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] += quadrupole_kick_xy.take(s_idx) * bunch.y.take(p_idx)


class QuadrupoleWakeKickY(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a quadrupolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        quadrupole_kick_y = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.yp[p_idx] += quadrupole_kick_y.take(s_idx) * bunch.y.take(p_idx)


class QuadrupoleWakeKickYX(WakeKick):

    def apply(self, bunch, slice_set):
        """ Calculates and applies a quadrupolar (cross term y-x) wake
        kick to bunch.yp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick. """
        zeroth_moment = slice_set.n_macroparticles_per_slice
        quadrupole_kick_yx = (WakeKick._wake_factor(bunch) *
            self._convolution(bunch, slice_set, zeroth_moment))

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.yp[p_idx] += quadrupole_kick_yx.take(s_idx) * bunch.x.take(p_idx)
