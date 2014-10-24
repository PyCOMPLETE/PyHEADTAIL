"""
This module includes the description of a WakeField as well as the
implementation of the WakeSource objects.

A WakeField is defined as a composition of the elementary WakeKick
objects (see .wake_kicks module). They originate from WakeSources,
e.g. a WakeTable, Resonator and/or a ResistiveWall. The WakeField does
not directly accept the WakeKick objects, but takes a list of
WakeSources first (can be of different kinds), each of which knows how
to generate its WakeKick objects via the factory method
WakeSource.get_wake_kicks(..). The collection of WakeKicks from all the
WakeSources define the WakeField and are the elementary objects that are
stored, (i.e. the WakeField forgets about the origin of the WakeKicks
once the latter have been created).

@author Hannes Bartosik, Kevin Li, Giovanni Rumolo, Michael Schenk
@date March 2014
@brief Implementation of a WakeField as a composition of WakeKicks
       originating from different WakeSources.
@copyright CERN
"""
from __future__ import division

import numpy as np
from scipy.constants import c, physical_constants
from scipy.interpolate import interp1d
from abc import ABCMeta, abstractmethod

from wake_kicks import *
from . import Element

sin = np.sin
cos = np.cos


class WakeField(Element):
    """ A WakeField is defined by elementary WakeKick objects that may
    originate from different WakeSource objects. Usually, there is
    no need for the user to define more than one instance of the
    WakeField class in a simulation - except if one wants to use
    different slicing configurations (one WakeField object is allowed
    to have exactly one slicing configuration, i.e. only one instance
    of the Slicer class). """

    def __init__(self, slicer, *wake_sources):
        """ Accepts a list of WakeSource objects. Each WakeSource object
        knows how to generate its corresponding WakeKick objects. The
        collection of all the WakeKick objects of each of the passed
        WakeSource objects defines the WakeField.
        When instantiating the WakeField object, the WakeKick objects
        for each WakeSource defined in wake_sources are requested. The
        returned WakeKick lists are all stored in the
        WakeField.wake_kicks list. The WakeField itself forgets about
        the origin (WakeSource) of the kicks as soon as they have been
        generated.
        Exactly one instance of the Slicer class must be passed to the
        WakeField constructor. All the wake field components (kicks)
        hence use the same slicing and thus the same slice_set to
        calculate the strength of the kicks. """
        self.slicer = slicer

        self.wake_kicks = []
        for source in wake_sources:
            kicks = source.get_wake_kicks(self.slicer.mode)
            self.wake_kicks.extend(kicks)

    def track(self, bunch):
        """ Calls the WakeKick.apply(bunch, slice_set) method of each of
        the WakeKick objects stored in self.wake_kicks. A slice_set is
        necessary to perform this operation. It is requested from the
        bunch (instance of the Particles class) using the
        Particles.get_slices(self.slicer) method, where self.slicer is
        the instance of the Slicer class used for this particluar
        WakeField object. A slice_set is returned according to the
        self.slicer configuration. """
        slice_set = bunch.get_slices(self.slicer)

        for kick in self.wake_kicks:
            kick.apply(bunch, slice_set)


''' WakeSource classes. '''

class WakeSource(object):
    """ Abstract base class for wake sources, such as WakeTable,
    Resonator or ResistiveWall. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_wake_kicks(self, slicer_mode):
        """ Factory method. Creates instances of the WakeKick objects
        for the given WakeSource and returns them as a list wake_kicks.
        This method is usually only called by a WakeField object to
        collect and create all the WakeKick objects originating from the
        different sources. (The slicer mode Slicer.mode must be passed
        at instantiation of a WakeKick object only to set the
        appropriate convolution method. See docstrings of WakeKick
        class.) """
        pass


class WakeTable(WakeSource):
    """ Class to define wake functions and WakeKick objects using data
    from a table. """

    def __init__(self, wake_file, wake_file_columns):
        """ Load data from the wake_file and store them in a dictionary
        self.wake_table. Keys are the names specified by the user in
        wake_file_columns and describe the names of the wake field
        components (e.g. dipole_x or dipole_yx). The dict values are
        given by the corresponding data read from the table. The
        nomenclature of the wake components must be strictly obeyed.
        Valid names for wake components are:

        'constant_x', 'constant_y', 'dipole_x', 'dipole_y', 'dipole_xy',
        'dipole_yx', 'quadrupole_x', 'quadrupole_y', 'quadrupole_xy',
        'quadrupole_yx', 'longitudinal'.

        The order of wake_file_columns is relevant and must correspond
        to the one in the wake_file. There is no way to check this here
        and it is in the responsibility of the user to ensure it is
        correct. Two checks made here are whether the length of
        wake_file_columns corresponds to the number of columns in the
        wake_file and whether a column 'time' is specified.

        The units and signs of the wake table data are assumed to follow
        the HEADTAIL conventions, i.e.
          time: [ns]
          transverse wake components: [V/pC/mm]
          longitudinal wake component: [V/pC]. """
        self.wake_table = {}

        wake_data = np.loadtxt(wake_file, delimiter="\t")
        if len(wake_file_columns) != wake_data.shape[1]:
            raise ValueError('Length of wake_file_columns list does not \n' +
                             'correspond to the number of columns in \n' +
                             'the specified wake_file. \n')
        if 'time' not in wake_file_columns:
            raise ValueError("No wake_file_column with name 'time' has \n" +
                             "been specified. \n")

        for i, column_name in enumerate(wake_file_columns):
            self.wake_table.update({ column_name : wake_data[:,i] })

    def get_wake_kicks(self, slicer_mode):
        """ Factory method. Creates instances of the appropriate
        WakeKick objects for all the wake components provided by the
        user (and the wake table data). The WakeKick objects are
        returned as a list wake_kicks. """
        wake_kicks = []

        # Constant wake kicks.
        if self._is_provided('constant_x'):
            wake_function = self._function_transverse('constant_x')
            wake_kicks.append(ConstantWakeKickX(wake_function, slicer_mode))

        if self._is_provided('constant_y'):
            wake_function = self._function_transverse('constant_y')
            wake_kicks.append(ConstantWakeKickY(wake_function, slicer_mode))

        if self._is_provided('longitudinal'):
            wake_function = self._function_longitudinal('longitudinal')
            wake_kicks.append(ConstantWakeKickZ(wake_function, slicer_mode))

        # Dipolar wake kicks.
        if self._is_provided('dipole_x'):
            wake_function = self._function_transverse('dipole_x')
            wake_kicks.append(DipoleWakeKickX(wake_function, slicer_mode))

        if self._is_provided('dipole_y'):
            wake_function = self._function_transverse('dipole_y')
            wake_kicks.append(DipoleWakeKickY(wake_function, slicer_mode))

        if self._is_provided('dipole_xy'):
            wake_function = self._function_transverse('dipole_xy')
            wake_kicks.append(DipoleWakeKickXY(wake_function, slicer_mode))

        if self._is_provided('dipole_yx'):
            wake_function = self._function_transverse('dipole_yx')
            wake_kicks.append(DipoleWakeKickYX(wake_function, slicer_mode))

        # Quadrupolar wake kicks.
        if self._is_provided('quadrupole_x'):
            wake_function = self._function_transverse('quadrupole_x')
            wake_kicks.append(QuadrupoleWakeKickX(wake_function, slicer_mode))

        if self._is_provided('quadrupole_y'):
            wake_function = self._function_transverse('quadrupole_y')
            wake_kicks.append(QuadrupoleWakeKickY(wake_function, slicer_mode))

        if self._is_provided('quadrupole_xy'):
            wake_function = self._function_transverse('quadrupole_xy')
            self.kicks.append(QuadrupoleWakeKickXY(wake_function, slicer_mode))

        if self._is_provided('quadrupole_yx'):
            wake_function = self._function_transverse('quadrupole_yx')
            wake_kicks.append(QuadrupoleWakeKickYX(wake_function, slicer_mode))

        return wake_kicks

    def _is_provided(self, wake_component):
        """ Check whether wake_component is a valid name and available
        in wake table data. Return 'True' if yes and 'False' if no. """
        if wake_component in self.wake_table.keys():
            return True
        else:
            print (wake_component + ' \n' +
                  'Wake component is either not provided or does not \n'+
                  'use correct nomenclature. See docstring of WakeTable \n' +
                  'constructor to display valid names. \n')
            return False

    def _function_transverse(self, wake_component):
        """ Defines and returns the wake(beta, dz) function for the
        given wake_component (transverse). Data from the wake table are
        used, but first converted to SI units assuming that time is
        specified in [ns] and transverse wake field strengths in
        [V/pC/mm]. Sign conventions are applied (HEADTAIL conventions).
        dz is related to wake table time data by dz = beta c dt (dz < 0
        for the ultrarelativistic case).
        The wake(beta, dz) uses the scipy.interpolate.interp1d linear
        interpolation to calculate the wake strength at an arbitrary
        value of dz (provided it is in the valid range). The valid range
        of dz is given by the time range from the wake table. If values
        of wake(beta, dz) are requested for dz outside the valid range,
        a ValueError is raised by interp1d.
        Very basic conformity checks for the wake table data are already
        performed at definition time of the wake(beta, dz) method. E.g.
        whether the specified wake is valid only for ultrarelativistic
        cases or low beta cases. In the former case, the wake strength
        at time 0 must be defined by the user! """
        convert_to_s = 1e-9
        convert_to_V_per_Cm = 1e15

        time = convert_to_s * self.wake_table['time']
        wake_strength = -convert_to_V_per_Cm * self.wake_table[wake_component]

        if (time[0] == 0) and (wake_strength[0] == 0):
            def wake(beta, dz):
                dz = dz.clip(max=0)
                return interp1d(time, wake_strength)(- dz / (beta * c))
            print (wake_component + ' \n' +
                  'Assuming ultrarelativistic wake. \n')

        elif (time[0] < 0) and (wake_strength[0] != 0):
            def wake(beta, dz):
                return interp1d(time, wake_strength)(- dz / (beta * c))
            print (wake_component + ' \n' + 'Found low beta wake.')

        elif (time[0] > 0) and (wake_strength[0] != 0):
            raise ValueError(wake_component + ' \n' +
                             'does not meet requirements. \n')
        return wake

    def _function_longitudinal(self, wake_component):
        """ Defines and returns the wake(beta, dz) function for the
        given wake_component (longitudinal). Data from the wake table
        are used, but first converted to SI units assuming that time is
        specified in [ns] and longitudinal wake field strength in
        [V/pC]. Sign conventions are applied (HEADTAIL conventions). dz
        is related to wake table time data by dz = -beta c dt (dz < 0
        for the ultrarelativistic case).
        The wake(beta, dz) uses the scipy.interpolate.interp1d linear
        interpolation to calculate the wake strength at an arbitrary
        value of dz (provided it is in the valid range). The valid range
        of dz is given by the time range from the wake table. If values
        of wake(beta, dz) are requested for dz outside the valid range,
        a ValueError is raised by interp1d.
        The beam loading theorem is respected and applied for dz=0. """
        convert_to_s = 1e-9
        convert_to_V_per_C = 1e12

        time = convert_to_s * self.wake_table['time']
        wake_strength = -convert_to_V_per_C * self.wake_table[wake_component]

        def wake(beta, dz):
            wake_interpolated = interp1d(time, wake_strength)(-dz / (beta * c))
            if time[0] == 0:
                # Beam loading theorem: Half value of wake strength at
                # dz = 0.
                return (np.sign(-dz) + 1) / 2 * wake_interpolated
            elif time[0] < 0:
                return wake_interpolated
            elif (time[0] > 0):
                raise ValueError(wake_component + ' \n' +
                                 'does not meet requirements.')
        return wake


class Resonator(WakeSource):
    """ Class to describe the wake functions originating from a
    resonator impedance. Alex Chao's resonator model (Eq. 2.82) is used
    as well as the definitions from HEADTAIL. """

    def __init__(self, R_shunt, frequency, Q,
                 Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, Yokoya_Z):
        """ General constructor to create a Resonator WakeSource object
        describing the wake functions of a resonator impedance. Alex
        Chao's resonator model (Eq. 2.82) is used as well as definitions
        from HEADTAIL.
        Note that it is no longer allowed to pass a LIST of parameters
        to generate a number of resonators with different parameters
        within the same Resonator object. Instead, create the Resonator
        objects and pass all of them to the WakeField constructor. """
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_Y2 = Yokoya_Y2
        self.Yokoya_Z = Yokoya_Z

    @classmethod
    def circular(cls, R_shunt, frequency, Q):
        """ Special case of circular resonator. """
        Yokoya_X1 = 1.
        Yokoya_Y1 = 1.
        Yokoya_X2 = 0.
        Yokoya_Y2 = 0.
        Yokoya_Z = 0.

        self = cls(R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1,
                   Yokoya_X2, Yokoya_Y2, Yokoya_Z)
        return self

    @classmethod
    def parallel_plates(cls, R_shunt, frequency, Q):
        """ Special case of parallel plate resonator. """
        Yokoya_X1 = np.pi**2 / 24.
        Yokoya_Y1 = np.pi**2 / 12.
        Yokoya_X2 = -np.pi**2 / 24.
        Yokoya_Y2 = np.pi**2 / 24.
        Yokoya_Z = 0

        self = cls(R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1,
                   Yokoya_X2, Yokoya_Y2, Yokoya_Z)
        return self

    def get_wake_kicks(self, slicer_mode):
        """ Factory method. Creates instances of the appropriate
        WakeKick objects for a Resonator WakeSource with the specified
        parameters. A WakeKick object is instantiated only if the
        corresponding Yokoya factor is non-zero. The WakeKick objects
        are returned as a list wake_kicks. """
        wake_kicks = []

        # Dipole wake kick x.
        if self.Yokoya_X1:
            wake_function = self._function_transverse(
                self.R_shunt, self.frequency, self.Q, self.Yokoya_X1)
            wake_kicks.append(DipoleWakeKickX(wake_function, slicer_mode))

        # Quadrupole wake kick x.
        if self.Yokoya_X2:
            wake_function = self._function_transverse(
                self.R_shunt, self.frequency, self.Q, self.Yokoya_X2)
            wake_kicks.append(QuadrupoleWakeKickX(wake_function, slicer_mode))

        # Dipole wake kick y.
        if self.Yokoya_Y1:
            wake_function = self._function_transverse(
                self.R_shunt, self.frequency, self.Q, self.Yokoya_Y1)
            wake_kicks.append(DipoleWakeKickY(wake_function, slicer_mode))

        # Quadrupole wake kick y.
        if self.Yokoya_Y2:
            wake_function = self._function_transverse(
                self.R_shunt, self.frequency, self.Q, self.Yokoya_Y2)
            wake_kicks.append(QuadrupoleWakeKickY(wake_function, slicer_mode))

        # Constant wake kick z.
        if self.Yokoya_Z:
            wake_function = self._function_longitudinal(
                self.R_shunt, self.frequency, self.Q, self.Yokoya_Z)
            wake_kicks.append(ConstantWakeKickZ(wake_function, slicer_mode))

        return wake_kicks

    def _function_transverse(self, R_shunt, frequency, Q, Yokoya_factor):
        """ Define the wake function (transverse) of a resonator with
        the given parameters according to Alex Chao's resonator model
        (Eq. 2.82) and definitions of the resonator in HEADTAIL. """
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega**2 - alpha**2))

        def wake(beta, dz):
            dt = dz.clip(max=0) / (beta * c)
            if Q > 0.5:
                y = (Yokoya_factor * R_shunt * omega**2 / (Q*omegabar) *
                      np.exp(alpha*dt) * sin(omegabar*dt))
            elif Q == 0.5:
                y = (Yokoya_factor * R_shunt * omega**2 / Q *
                      np.exp(alpha*dt) * dt)
            else:
                y = (Yokoya_factor * R_shunt * omega**2 / (Q*omegabar) *
                     np.exp(alpha*dt) * np.sinh(omegabar*dt))
            return y
        return wake

    def _function_longitudinal(self, R_shunt, frequency, Q, Yokoya_factor):
        """ Define the wake function (longitudinal) of a resonator with
        the given parameters according to Alex Chao's resonator model
        (Eq. 2.82) and definitions of the resonator in HEADTAIL. """
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega ** 2 - alpha ** 2))

        def wake(beta, dz):
            dt = dz.clip(max=0) / (beta * c)
            if Q > 0.5:
                y = (-Yokoya_factor * (np.sign(dt) - 1) * R_shunt * alpha *
                      np.exp(alpha * dt) * (cos(omegabar * dt) +
                      alpha / omegabar * sin(omegabar*dt)))
            elif Q == 0.5:
                y = (-Yokoya_factor * (np.sign(dt) - 1) * R_shunt * alpha *
                      np.exp(alpha * dt) * (1. + alpha * dt))
            elif Q < 0.5:
                y = (-Yokoya_factor * (np.sign(dt) - 1) * R_shunt * alpha *
                     np.exp(alpha * dt) * (np.cosh(omegabar * dt) +
                     alpha / omegabar * np.sinh(omegabar * dt)))
            return y
        return wake


class ResistiveWall(WakeSource):
    """ Class to describe the wake functions originating from a
    resistive wall impedance. """

    def __init__(self, pipe_radius, resistive_wall_length, conductivity,
                 dz_min, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2):
        """ General constructor to create a ResistiveWall WakeSource
        object describing the wake functions of a resistive wall
        impedance. """
        self.pipe_radius = np.array([pipe_radius]).flatten()
        self.resistive_wall_length = resistive_wall_length
        self.conductivity = conductivity
        self.dz_min = dz_min

        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y2 = Yokoya_Y2

    @classmethod
    def circular(cls, pipe_radius, resistive_wall_length, conductivity,
                 dz_min):
        """ Special case of a circular resistive wall. """
        Yokoya_X1 = 1.
        Yokoya_Y1 = 1.
        Yokoya_X2 = 0.
        Yokoya_Y2 = 0.

        self = cls(pipe_radius, resistive_wall_length, conductivity, dz_min,
                   Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2)
        return self

    @classmethod
    def parallel_plates(cls, pipe_radius, resistive_wall_length, conductivity,
                        dz_min):
        """ Special case of a parallel plates resistive wall. """
        Yokoya_X1 = np.pi**2 / 24.
        Yokoya_Y1 = np.pi**2 / 12.
        Yokoya_X2 = -np.pi**2 / 24.
        Yokoya_Y2 = np.pi**2 / 24.

        self = cls(pipe_radius, resistive_wall_length, conductivity, dz_min,
                   Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2)
        return self

    def get_wake_kicks(self, slicer_mode):
        """ Factory method. Creates instances of the appropriate
        WakeKick objects for the ResistiveWall WakeSource with the
        specified parameters. A WakeKick object is instantiated only if
        the corresponding Yokoya factor is non-zero. The WakeKick
        objects are returned as a list wake_kicks. """
        wake_kicks = []

        # Dipole wake kick x.
        if self.Yokoya_X1:
            wake_function = self._function_transverse(self.Yokoya_X1)
            wake_kicks.append(DipoleWakeKickX(wake_function, slicer_mode))

        # Quadrupole wake kick x.
        if self.Yokoya_X2:
            wake_function = self._function_transverse(self.Yokoya_X2)
            wake_kicks.append(QuadrupoleWakeKickX(wake_function, slicer_mode))

        # Dipole wake kick y.
        if self.Yokoya_Y1:
            wake_function = self._function_transverse(self.Yokoya_Y1)
            wake_kicks.append(DipoleWakeKickY(wake_function, slicer_mode))

        # Quadrupole wake kick y.
        if self.Yokoya_Y2:
            wake_function = self._function_transverse(self.Yokoya_Y2)
            wake_kicks.append(QuadrupoleWakeKickY(wake_function, slicer_mode))

        return wake_kicks

    def _function_transverse(self, Yokoya_factor):
        """ Define the wake function (transverse) of a resistive wall
        with the given parameters. """
        Z0 = physical_constants['characteristic impedance of vacuum'][0]
        lambda_s = 1. / (Z0 * self.conductivity)
        mu_r = 1

        def wake(beta, dz):
            y = (Yokoya_factor * (np.sign(dz + np.abs(self.dz_min)) - 1) / 2 *
                 beta * c * Z0 * self.resistive_wall_length / np.pi /
                 self.pipe_radius**3 * np.sqrt(-lambda_s * mu_r / np.pi /
                 dz.clip(max=-abs(self.dz_min))))
            return y
        return wake
