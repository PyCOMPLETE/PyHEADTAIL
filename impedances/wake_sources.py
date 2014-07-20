'''
@class Wakefields
@author Hannes Bartosik & Kevin Li & Giovanni Rumolo & Michael Schenk
@date March 2014
@Class for creation and management of wakefields from impedance sources
@copyright CERN
'''
from __future__ import division


from functools import partial
import numpy as np
from scipy.constants import c, e
from scipy.constants import physical_constants


sin = np.sin
cos = np.cos

class WakeSources(object): pass


def BB_Resonator_Circular(R_shunt, frequency, Q, slices=None):
    return BB_Resonator_transverse(R_shunt, frequency, Q, Yokoya_X1=1, Yokoya_Y1=1, Yokoya_X2=0, Yokoya_Y2=0, slices=slices)


def BB_Resonator_ParallelPlates(R_shunt, frequency, Q, slices=None):
    return BB_Resonator_transverse(R_shunt, frequency, Q, Yokoya_X1=np.pi**2/24, Yokoya_Y1=np.pi**2/12, Yokoya_X2=-np.pi**2/24, Yokoya_Y2=np.pi**2/24, slices=slices)


def Resistive_wall_Circular(pipe_radius, length_resistive_wall, conductivity=5.4e17, dz_min=1e-4, slices=None):
    return Resistive_wall_transverse(pipe_radius, length_resistive_wall, conductivity, dz_min, Yokoya_X1=1, Yokoya_Y1=1, Yokoya_X2=0, Yokoya_Y2=0, slices=slices)


def Resistive_wall_ParallelPlates(pipe_radius, length_resistive_wall, conductivity=5.4e17, dz_min=1e-4, slices=None):
    return BB_Resonator_transverse(pipe_radius, length_resistive_wall, conductivity, dz_min, Yokoya_X1=np.pi**2/24, Yokoya_Y1=np.pi**2/12, Yokoya_X2=-np.pi**2/24, Yokoya_Y2=np.pi**2/24, slices=slices)


class BB_Resonator_transverse(WakeSources):

    def __init__(self, R_shunt, frequency, Q, Yokoya_X1=1, Yokoya_Y1=1, Yokoya_X2=0, Yokoya_Y2=0, slices):

        # wake_function()

        Wakefields.__init__(self, slices)
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q
        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y2 = Yokoya_Y2

    def memo(self, fn):
        cache = {}
        def call(*args):
            if args not in cache:
                cache[args] = fn(*args)

            return cache[args]

        return call

    def function_transverse(self, bunch, z):
        Rs = self.R_shunt
        frequency = self.frequency
        Q = self.Q
        beta_r = bunch.beta

        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega ** 2 - alpha ** 2))

        # Taken from definition in HEADTAIL
        if self.Q > 0.5:
            wake =  Rs * omega ** 2 / Q / omegabar * np.exp(alpha * z.clip(max=0) / c / beta_r) * \
                    sin(omegabar * z.clip(max=0) / c / beta_r)
        elif self.Q == 0.5:
            wake =  Rs * omega ** 2 / Q * z.clip(max=0) / c / beta_r * \
                    np.exp(alpha * z.clip(max=0) / c / beta_r)
        else:
            wake =  Rs * omega ** 2 / Q / omegabar * np.exp(alpha * z.clip(max=0) / c / beta_r) * \
                    np.sinh(omegabar * z.clip(max=0) / c / beta_r)
        return wake

    def dipole_wake_x(self, bunch, z):
        return self.Yokoya_X1 * self.wake_transverse(bunch, z)

    def dipole_wake_x_memo(self, bunch, z):
        wake_partial = partial(self.wake_transverse, bunch)
        wake_transverse = self.memo(wake_partial)
        z_shape = z.shape
        W = np.array(map(wake_transverse, z.flatten())).reshape(z_shape)
        return self.Yokoya_X1 * W#wake_transverse(bunch, z)

    def dipole_wake_y(self, bunch, z):
        return self.Yokoya_Y1 * self.wake_transverse(bunch, z)

    def quadrupole_wake_x(self, bunch, z):
        if self.Yokoya_X2: return self.Yokoya_X2 * self.wake_transverse(bunch, z)
        return 0

    def quadrupole_wake_y(self, bunch, z):
        if self.Yokoya_Y2: return self.Yokoya_Y2 * self.wake_transverse(bunch, z)
        return 0


class Resistive_wall_transverse(Wakefields):

    def __init__(self, pipe_radius, length_resistive_wall, conductivity=5.4e17, dz_min= 1e-4, Yokoya_X1=1, Yokoya_Y1=1, Yokoya_X2=0, Yokoya_Y2=0, slices=None):

        Wakefields.__init__(self, slices)
        self.pipe_radius = np.array([pipe_radius]).flatten()
        self.length_resistive_wall = length_resistive_wall
        self.conductivity = conductivity
        self.dz_min = dz_min
        self.Yokoya_X1 = Yokoya_X1
        self.Yokoya_Y1 = Yokoya_Y1
        self.Yokoya_X2 = Yokoya_X2
        self.Yokoya_Y2 = Yokoya_Y2

    def wake_transverse(self, bunch, z):
        Z0 = physical_constants['characteristic impedance of vacuum'][0]
        lambda_s = 1. / (Z0 * self.conductivity)
        mu_r = 1
        wake = (np.sign(z + np.abs(self.dz_min)) - 1) / 2 * bunch.beta * c * Z0 * self.length_resistive_wall / np.pi / self.pipe_radius ** 3 * np.sqrt(-lambda_s * mu_r / np.pi / z.clip(max=-abs(self.dz_min)))
        return wake

    def dipole_wake_x(self, bunch, z):
        return self.Yokoya_X1 * self.wake_transverse(bunch, z)

    def dipole_wake_y(self, bunch, z):
        return self.Yokoya_Y1 * self.wake_transverse(bunch, z)

    def quadrupole_wake_x(self, bunch, z):
        if self.Yokoya_X2: return self.Yokoya_X2 * self.wake_transverse(bunch, z)
        return 0

    def quadrupole_wake_y(self, bunch, z):
        if self.Yokoya_Y2: return self.Yokoya_Y2 * self.wake_transverse(bunch, z)
        return 0


class Wake_table(Wakefields):

    def __init__(self, wake_file, keys, slices):
        Wakefields.__init__(self, slices)
        table = np.loadtxt(wake_file, delimiter="\t")
        self.wake_table = dict(zip(keys, np.array(zip(*table))))
        self.unit_conversion()

    def unit_conversion(self):
        transverse_wakefield_keys   = ['dipolar_x', 'dipolar_y', 'quadrupolar_x', 'quadrupolar_y']
        longitudinal_wakefield_keys = ['longitudinal']
        self.wake_field_keys = []
        print 'Converting wake table to correct units ... '
        self.wake_table['time'] *= 1e-9 # unit convention [ns]
        print '\t converted time from [ns] to [s]'
        for wake in transverse_wakefield_keys:
            try:
                self.wake_table[wake] *= - 1.e15 # unit convention [V/pC/mm] and sign convention !!
                print '\t converted "' + wake + '" wake from [V/pC/mm] to [V/C/m] and inverted sign'
                self.wake_field_keys += [wake]
            except:
                print '\t "' + wake + '" wake not provided'
        for wake in longitudinal_wakefield_keys:
            try:
                self.wake_table[wake] *= - 1.e12 # unit convention [V/pC] and sign convention !!
                print '\t converted "' + wake + '" wake from [V/pC] to [V/C]'
                self.wake_field_keys += [wake]
            except:
                print '\t "' + wake + '" wake not provided'

    #~ @profile
    def wake_transverse(self, key, bunch, z):
        time = np.array(self.wake_table['time'])
        wake = np.array(self.wake_table[key])
        # insert zeros at origin if wake functions at (or below) zero not provided
        if time[0] > 0:
            time = np.append(0, time)
            wake = np.append(0, wake)
        # insert zero value of wake field if provided wake begins with a finite value
        if wake[0] != 0:
            wake = np.append(0, wake)
            time = np.append(time[0] - np.diff(time[1], time[0]), time)
        return np.interp(- z / c / bunch.beta, time, wake, left=0, right=0)


    def dipole_wake_x(self, bunch, z):
        if 'dipolar_x' in self.wake_field_keys: return self.wake_transverse('dipolar_x', bunch, z)
        return 0

    def dipole_wake_y(self, bunch, z):
        if 'dipolar_y' in self.wake_field_keys: return self.wake_transverse('dipolar_y', bunch, z)
        return 0

    def quadrupole_wake_x(self, bunch, z):
        if 'quadrupolar_x' in self.wake_field_keys: return self.wake_transverse('quadrupolar_x', bunch, z)
        return 0

    def quadrupole_wake_y(self, bunch, z):
        if 'quadrupolar_y' in self.wake_field_keys: return self.wake_transverse('quadrupolar_y', bunch, z)
        return 0

    def wake_longitudinal(self, bunch, z):
        time = np.array(self.wake_table['time'])
        wake = np.array(self.wake_table['longitudinal'])
        wake_interpolated = np.interp(- z / c / bunch.beta, time, wake, left=0, right=0)
        if time[0] < 0:
            return wake_interpolated
        elif time[0] == 0:
            # beam loading theorem: half value of wake at z=0;
            return (np.sign(-z) + 1) / 2 * wake_interpolated

    def track(self, bunch):
        if not self.slices:
            self.slices = bunch.slices

        # bunch.compute_statistics()
        self.slices.update_slices(bunch)
        self.slices.compute_statistics(bunch)

        if ('dipolar_x' or 'quadrupolar_x') in self.wake_field_keys:
            wakefield_kicks_x = self.transverse_wakefield_kicks('x')
            wakefield_kicks_x(bunch)
        if ('dipolar_y' or 'quadrupolar_y') in self.wake_field_keys:
            wakefield_kicks_y = self.transverse_wakefield_kicks('y')
            wakefield_kicks_y(bunch)
        if 'longitudinal' in self.wake_field_keys:
            self.longitudinal_wakefield_kicks(bunch)


class BB_Resonator_longitudinal(Wakefields):
    '''
    classdocs
    '''
    def __init__(self, R_shunt, frequency, Q, slices=None):
        '''
        Constructor
        '''
        Wakefields.__init__(self, slices)
        self.R_shunt = np.array([R_shunt]).flatten()
        self.frequency = np.array([frequency]).flatten()
        self.Q = np.array([Q]).flatten()
        assert(len(self.R_shunt) == len(self.frequency) == len(self.Q))

    def wake_longitudinal(self, bunch, z):
        return reduce(lambda x,y: x+y, [self.wake_BB_resonator(self.R_shunt[i], self.frequency[i], self.Q[i], bunch, z) for i in np.arange(len(self.Q))])

    def wake_BB_resonator(self, R_shunt, frequency, Q, bunch, z):
        # Taken from Alex Chao's resonator model (2.82)
        omega = 2 * np.pi * frequency
        alpha = omega / (2 * Q)
        omegabar = np.sqrt(np.abs(omega ** 2 - alpha ** 2))

        if Q > 0.5:
            wake =  - (np.sign(z) - 1) * R_shunt * alpha * np.exp(alpha * z.clip(max=0) / c / bunch.beta) * \
                    (cos(omegabar * z.clip(max=0) / c / bunch.beta) + alpha / omegabar * sin(omegabar * z.clip(max=0) / c / bunch.beta))
        elif Q == 0.5:
            wake =  - (np.sign(z) - 1) * R_shunt * alpha * np.exp(alpha * z.clip(max=0) / c / bunch.beta) * \
                    (1. + alpha * z.clip(max=0) / c / bunch.beta)
        elif Q < 0.5:
            wake =  - (np.sign(z) - 1) * R_shunt * alpha * np.exp(alpha * z.clip(max=0) / c / bunch.beta) * \
                    (np.cosh(omegabar * z.clip(max=0) / c / bunch.beta) + alpha / omegabar * np.sinh(omegabar * z.clip(max=0) / c / bunch.beta))
        return wake

    def track(self, bunch):
        # if not self.slices:
            # self.slices = bunch.slices
        self.slices.compute_statistics(bunch)

        self.longitudinal_wakefield_kicks(bunch)
