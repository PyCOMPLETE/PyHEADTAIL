'''
@class WakeSources
@author Hannes Bartosik & Kevin Li & Giovanni Rumolo & Michael Schenk
@date July 2014
@Class for creation and management of impedance sources
@copyright CERN
'''
from __future__ import division

from wake_kicks import *

import numpy as np
from scipy.constants import c
from scipy.constants import physical_constants
from scipy.interpolate import interp1d

sin = np.sin
cos = np.cos


class WakeTable(object):

    def __init__(self, wake_file, wake_file_columns, slices):

        table = np.loadtxt(wake_file, delimiter="\t")
        self.wake_table = dict(zip(wake_file_columns, np.array(zip(*table))))
        self._unit_conversion()

        # Generate wake kicks and append to list of kicks.                       
        self.kicks = []
        self._generate_wake_kicks(slices)

                
    def _unit_conversion(self):

        print 'Converting wake table to correct units ... '
        
        allowed_transverse_wakefield_keys   = ['constant_x', 'constant_y',
                                               'dipole_x',   'dipole_y',   'quadrupole_x',  'quadrupole_y',
                                               'dipole_xy',  'dipole_yx',  'quadrupole_xy', 'quadrupole_yx']
        allowed_longitudinal_wakefield_keys = ['longitudinal']

        self.wakefield_keys   = []
        self.wake_table['time'] *= 1e-9 # unit convention [ns]
        print '\t converted time from [ns] to [s]'

        for wake_component in allowed_transverse_wakefield_keys:
            try:
                self.wake_table[wake_component] *= - 1.e15 # unit convention [V/pC/mm] and sign convention !!
                print '\t converted "' + wake_component + '" wake from [V/pC/mm] to [V/C/m] and inverted sign'
                self.wakefield_keys += [wake_component]
            except:
                print '\t "' + wake_component + '" wake not provided'

        for wake_component in allowed_longitudinal_wakefield_keys:
            try:
                self.wake_table[wake_component] *= - 1.e12 # unit convention [V/pC] and sign convention !!
                print '\t converted "' + wake_component + '" wake from [V/pC] to [V/C]'
                self.wakefield_keys += [wake_component]
            except:
                print '\t "' + wake_component + '" wake not provided'

                
    def _generate_wake_kicks(self, slices):
        
        # X wakes.
        if 'constant_x' in self.wakefield_keys:
            wake_function = self._function_transverse('constant_x')
            self.kicks.append(ConstantWakeKickX(wake_function, slices))
                    
        if 'dipole_x' in self.wakefield_keys:
            wake_function = self._function_transverse('dipole_x')
            self.kicks.append(DipoleWakeKickX(wake_function, slices))

        if 'quadrupole_x' in self.wakefield_keys:
            wake_function = self._function_transverse('quadrupole_x')
            self.kicks.append(QuadrupoleWakeKickX(wake_function, slices))
            
        if 'dipole_xy' in self.wakefield_keys:
            wake_function = self._function_transverse('dipole_xy')
            self.kicks.append(DipoleWakeKickXY(wake_function, slices))

        if 'quadrupole_xy' in self.wakefield_keys:
            wake_function = self._function_transverse('quadrupole_xy')
            self.kicks.append(QuadrupoleWakeKickXY(wake_function, slices))

        # Y wakes.
        if 'constant_y' in self.wakefield_keys:
            wake_function = self._function_transverse('constant_y')
            self.kicks.append(ConstantWakeKickY(wake_function, slices))
                    
        if 'dipole_y' in self.wakefield_keys:
            wake_function = self._function_transverse('dipole_y')
            self.kicks.append(DipoleWakeKickY(wake_function, slices))

        if 'quadrupole_y' in self.wakefield_keys:
            wake_function = self._function_transverse('quadrupole_y')
            self.kicks.append(QuadrupoleWakeKickY(wake_function, slices))
            
        if 'dipole_yx' in self.wakefield_keys:
            wake_function = self._function_transverse('dipole_yx')
            self.kicks.append(DipoleWakeKickYX(wake_function, slices))

        if 'quadrupole_yx' in self.wakefield_keys:
            wake_function = self._function_transverse('quadrupole_yx')
            self.kicks.append(QuadrupoleWakeKickYX(wake_function, slices))

        # Z wakes.
        if 'longitudinal' in self.wakefield_keys:
            wake_function = self._function_longitudinal('longitudinal')
            self.kicks.append(ConstantWakeKickZ(wake_function, slices))

    def _function_transverse(self, key):
        time          = np.array(self.wake_table['time'])
        wake_strength = np.array(self.wake_table[key])
        # insert zeros at origin if wake functions at (or below) zero not provided
        if time[0] > 0:
            time          = np.append(0, time)
            wake_strength = np.append(0, wake_strength)
        # TODO: check this (commented; diff has a problem here -- KL 30.08.2014)
        # This should only be true for ultrarelativistic wakes? Perhaps this should be left to the wakefield maker...
        # insert zero value of wake field if provided wake begins with a finite value
        # if wake_strength[0] != 0:
        #     time          = np.append(time[0] - np.diff(time[1], time[0]), time)
        #     wake_strength = np.append(0, wake_strength)

        def wake(beta, z):
            # interp1d(time, wake_strength)(-z/(beta*c))
            return np.interp(- z / (beta * c), time, wake_strength, left=0, right=0)

        return wake

    def _function_longitudinal(self, key):
        time          = np.array(self.wake_table['time'])
        wake_strength = np.array(self.wake_table[key])

        def wake(beta, z):
            wake_interpolated = np.interp(- z / (beta * c), time, wake_strength, left=0, right=0)
            if time[0] < 0:
                return wake_interpolated
            elif time[0] == 0:
                # beam loading theorem: half value of wake at z=0;
                return (np.sign(-z) + 1) / 2 * wake_interpolated

        return wake


class Resonator(object):

    def __init__(self, R_shunt, frequency, Q,
                 Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, Yokoya_Z, slices):

        # Handle single-element inputs. Is there a better option?
        if not isinstance(R_shunt, list):   R_shunt   = [R_shunt]
        if not isinstance(frequency, list): frequency = [frequency]
        if not isinstance(Q, list):         Q         = [Q]
        if not isinstance(Yokoya_X1, list): Yokoya_X1 = [Yokoya_X1]
        if not isinstance(Yokoya_X2, list): Yokoya_X2 = [Yokoya_X2]
        if not isinstance(Yokoya_Y1, list): Yokoya_Y1 = [Yokoya_Y1]
        if not isinstance(Yokoya_Y2, list): Yokoya_Y2 = [Yokoya_Y2]
        if not isinstance(Yokoya_Z, list):  Yokoya_Z  = [Yokoya_Z]

        assert(len(R_shunt)   == len(frequency) == len(Q) == len(Yokoya_X1) == len(Yokoya_X2) == \
               len(Yokoya_Y1) == len(Yokoya_Y2) == len(Yokoya_Z))

        self.R_shunt   = R_shunt
        self.frequency = frequency
        self.Q         = Q
        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_Y2 = Yokoya_Y2
        self.Yokoya_Z  = Yokoya_Z 

        # Generate wake kicks and append to list of kicks.                       
        self.kicks = []
        self._generate_wake_kicks(slices)

        
    def _generate_wake_kicks(self, slices):

        # Dipole wake kick x.
        if any(self.Yokoya_X1):
            wake_function = self._function_total(self._function_transverse, self.Yokoya_X1)
            self.kicks.append(DipoleWakeKickX(wake_function, slices))

        # Quadrupole wake kick x.
        if any(self.Yokoya_X2):
            wake_function = self._function_total(self._function_transverse, self.Yokoya_X2)
            self.kicks.append(QuadrupoleWakeKickX(wake_function, slices))

        # Dipole wake kick y.
        if any(self.Yokoya_Y1):
            wake_function = self._function_total(self._function_transverse, self.Yokoya_Y1)
            self.kicks.append(DipoleWakeKickY(wake_function, slices))

        # Quadrupole wake kick y.
        if any(self.Yokoya_Y2):
            wake_function = self._function_total(self._function_transverse, self.Yokoya_Y2)
            self.kicks.append(QuadrupoleWakeKickY(wake_function, slices))

        # Constant wake kick z.
        if any(self.Yokoya_Z):
            wake_function = self._function_total(self._function_longitudinal, self.Yokoya_Z)
            self.kicks.append(ConstantWakeKickZ(wake_function, slices))


    def _function_transverse(self, R_shunt, frequency, Q, Yokoya_factor):

        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega**2 - alpha**2))

        # Taken from definition in HEADTAIL
        def wake(beta, z):

            t = z.clip(max=0) / (beta*c)
            if Q > 0.5:
                y =  Yokoya_factor * R_shunt * omega**2 / (Q*omegabar) * np.exp(alpha*t) * sin(omegabar*t)
            elif Q == 0.5:
                y =  Yokoya_factor * R_shunt * omega**2 / Q * np.exp(alpha*t) * t
            else:
                y =  Yokoya_factor * R_shunt * omega**2 / (Q*omegabar) * np.exp(alpha*t) * np.sinh(omegabar*t)
            return y

        return wake
    

    def _function_longitudinal(self, R_shunt, frequency, Q, Yokoya_factor):

        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega ** 2 - alpha ** 2))

        def wake(beta, z):

            t = z.clip(max=0) / (beta*c)
            if Q > 0.5:
                y =  - Yokoya_factor * (np.sign(z)-1) * R_shunt * alpha * np.exp(alpha*t) * (cos(omegabar*t) \
                                                                            + alpha/omegabar * sin(omegabar*t))
            elif Q == 0.5:
                y =  - Yokoya_factor * (np.sign(z)-1) * R_shunt * alpha * np.exp(alpha*t) * (1. + alpha*t)
            elif Q < 0.5:
                y =  - Yokoya_factor * (np.sign(z)-1) * R_shunt * alpha * np.exp(alpha*t) * (np.cosh(omegabar*t) \
                                                                            + alpha/omegabar * np.sinh(omegabar*t))
            return y

        return wake


    def _function_total(self, function_single, Yokoya_factor):
        return reduce(lambda x, y: x + y,
                      [function_single(self.R_shunt[i], self.frequency[i], self.Q[i], Yokoya_factor[i])
                       for i in np.arange(len(self.Q)) if Yokoya_factor[i]!=0])

    
class ResistiveWall(object):

    def __init__(self, pipe_radius, resistive_wall_length, conductivity, dz_min,
                 Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, slices):

        self.pipe_radius = np.array([pipe_radius]).flatten()
        self.resistive_wall_length = resistive_wall_length
        self.conductivity = conductivity
        self.dz_min = dz_min

        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y2 = Yokoya_Y2

        # Generate wake kicks and append to list of kicks.                       
        self.kicks = []
        self._generate_wake_kicks(slices)


    def _generate_wake_kicks(self, slices):

        if self.Yokoya_X1:
            wake_function = self._function_transverse(self.Yokoya_X1)
            self.kicks.append(DipoleWakeKickX(wake_function, slices))

        if self.Yokoya_X2:
            wake_function = self._function_transverse(self.Yokoya_X2)
            self.kicks.append(QuadrupoleWakeKickX(wake_function, slices))

        if self.Yokoya_Y1:
            wake_function = self._function_transverse(self.Yokoya_Y1)
            self.kicks.append(DipoleWakeKickY(wake_function, slices))

        if self.Yokoya_Y2:
            wake_function = self._function_transverse(self.Yokoya_Y2)
            self.kicks.append(QuadrupoleWakeKickY(wake_function, slices))


    def _function_transverse(self, Yokoya_factor):

        Z0 = physical_constants['characteristic impedance of vacuum'][0]
        lambda_s = 1. / (Z0*self.conductivity)
        mu_r = 1

        def wake(beta, z):
            y = Yokoya_factor * (np.sign(z + np.abs(self.dz_min)) - 1) / 2 * beta * c \
                * Z0 * self.resistive_wall_length / np.pi / self.pipe_radius ** 3 \
                * np.sqrt(-lambda_s * mu_r / np.pi / z.clip(max=-abs(self.dz_min)))
            
            return y

        return wake
