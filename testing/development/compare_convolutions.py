from __future__ import division

import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

import time, copy
import numpy as np
from mpi4py import MPI
import scipy
from scipy.constants import c, e, m_p
import matplotlib.pyplot as plt
from collections import deque

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField
from PyHEADTAIL.machines.synchrotron import Synchrotron


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class WakeConvolution(object):

    def __init__(self,t,x, simulation_parameters):

        convert_to_V_per_Cm = -1e15
        self._t = t*1e-9
        self._x = x*convert_to_V_per_Cm
        self._n_turns = simulation_parameters['n_turns_wake']
        self._z_values = None
        self._previous_kicks = deque(maxlen=simulation_parameters['n_turns_wake'])

        self._simulation_parameters = simulation_parameters

    def _wake_factor(self):
        """Universal scaling factor for the strength of a wake field
        kick. Copied from PyHEADTAIL.
        """
        beta = self._simulation_parameters['beta']
        charge = self._simulation_parameters['charge']
        mass = self._simulation_parameters['mass']
        gamma = self._simulation_parameters['gamma']

        wake_factor = (-(charge)**2 / (mass * gamma * (beta * c)**2))
        return wake_factor

    def _init(self, z):
        """Calculates wake functions for different turns
        """

        self._turn_by_turn_wake_functions = []
        turn_length = (self._simulation_parameters['circumference'])/c
        normalized_z = (z - z[0])/c

        for i in xrange(self._n_turns):
            self._previous_kicks.append(np.zeros(len(normalized_z)))
            z_values = normalized_z + float(i)*turn_length

            temp_values = np.interp(z_values, self._t, self._x)
#            if i == 0:
#                temp_values[0] = 0.

            self._turn_by_turn_wake_functions.append(temp_values)

    def operate(self, z, x, xp, intensity):
        if not hasattr(self, '_turn_by_turn_wake_functions'):
            self._init(z)

        source = np.copy(x*intensity)
        source = source[::-1]

        # calculates turn by turn circular convolutions for wake kicks
        for i, wake in enumerate(self._turn_by_turn_wake_functions):
            kick = np.real(np.fft.ifft(np.fft.fft(source) * np.fft.fft(wake)))

            # the total wake kick is accumulated by adding the calculated values to
            # the values from previous tracking turns. The index i+1 comes from the property
            # of a deque object, i.e. the last kick is added to the end of the deque
            # which pops out the first value.
            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += kick
            else:
                self._previous_kicks.append(kick)

        # the kick is applied to the xp values
        return xp + self._wake_factor()*self._previous_kicks[0][::-1]


def calculate_pure_convolution(bunches, wakes, simulation_parameters):

    # loads numerical wake function values from the PyHEADTAIL object
    wf = wakes.function_transverse(1)
    T0 = simulation_parameters['circumference']/(simulation_parameters['beta'] * scipy.constants.c)
    tt = np.linspace(0, ((simulation_parameters['n_turns_wake']+1)*T0), 10000) * -1
    data_t = tt
    data_x = wf(tt) / -1e15

    # creates an object, which calculates turn by turn wake kicks
    kick_calculator = WakeConvolution(data_t*-1e9, data_x, simulation_parameters)

    # data objects for the point-like-bunch beam
    bunch_list = bunches.split()
    z = np.zeros(len(bunch_list))
    x = np.zeros(len(bunch_list))
    xp = np.zeros(len(bunch_list))
    intensity = np.zeros(len(bunch_list))

    # initial values are copied from the PyHEADTAIL bunches
    for i, bunch in enumerate(bunch_list):
        z[i] = i*simulation_parameters['circumference']/float(simulation_parameters['h_RF'])
        x[i] = bunch.mean_x()
        xp[i] = bunch.mean_xp()
        intensity[i] = bunch.intensity

    # tracked data, which is compared to the PyHEADTAIL simulations
    x_data = np.zeros((simulation_parameters['n_turns'], len(bunch_list)))

    for i in xrange(simulation_parameters['n_turns']):

        angle = 2.*np.pi*simulation_parameters['accQ_x']
        s = np.sin(angle)
        c = np.cos(angle)
        beta_x = simulation_parameters['beta_x']

        # beam is rotated in x-xp plane
        new_x = c * x + beta_x * s * xp
        new_xp = (-1. / beta_x) * s * x + c * xp
        np.copyto(x, new_x)
        np.copyto(xp, new_xp)

        # wake kick is applied
        np.copyto(xp, kick_calculator.operate(z, x, xp, intensity))

        # tracking data is copied
        np.copyto(x_data[i,:], x)

    return x_data


def generate_machine_and_bunches():

    # BEAM AND MACHNINE PARAMETERS
    # ============================

    n_macroparticles = 200
    intensity = 1e11

    charge = e
    mass = m_p
    alpha = 53.86**-2

    accQ_x = 62.31
    accQ_y = 60.32
    circumference = 1000.
    s = None
    alpha_x = None
    alpha_y = None
    beta_x = circumference / (2.*np.pi*accQ_x)
    beta_y = circumference / (2.*np.pi*accQ_y)
    D_x = 0
    D_y = 0
    optics_mode = 'smooth'
    name = None
    n_segments = 1

    # detunings
    Qp_x = 0
    Qp_y = 0

    app_x = 0
    app_y = 0
    app_xy = 0

    longitudinal_mode = 'linear'

    h_RF = 100
    dphi_RF = 0
    p_increment = 0
    p0 = 26e9 * e / c
    V_RF = 1e1
    wrap_z = False

    machine = Synchrotron(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), V_RF=np.atleast_1d(V_RF),
            dphi_RF=np.atleast_1d(dphi_RF), p0=p0, p_increment=p_increment,
            charge=charge, mass=mass, wrap_z=wrap_z)

    machine.one_turn_map = machine.one_turn_map[1:]

    # every bucker is filled
    filling_scheme = sorted([i for i in range(h_RF)])

    # BEAM
    # ====
    epsn_x = 2.e-6
    epsn_y = 2.e-6
    sigma_z = 0.081
    allbunches = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme, matched=False)

    # WAKE PARAMETERS
    # ============================
    n_turns_wake = 1
    wakes = CircularResonator(1e10, 40e5, 6, n_turns_wake=n_turns_wake)

    # SIMULATION PARAMETERS
    # ============================
    n_turns = 7

    simulation_parameters = {
            'circumference': circumference,
            'accQ_x': accQ_x,
            'accQ_y': accQ_y,
            'beta_x': beta_x,
            'beta_y': beta_y,
            'h_RF': h_RF,
            'charge': charge,
            'mass': mass,
            'beta': machine.beta,
            'gamma': machine.gamma,
            'n_turns_wake': n_turns_wake,
            'n_turns': n_turns
            }

    # a bunch on the middle is kicked
    bunch_list = allbunches.split()

    for i in xrange(len(bunch_list)):
        bunch_list[i].x[:] = bunch_list[i].x[:] * 0
        bunch_list[i].xp[:] = bunch_list[i].xp[:] * 0.

    idx = int(len(bunch_list)/2)
    bunch_list[idx].x[:] = bunch_list[idx].x[:] + 1e-3

    return machine, sum(bunch_list), wakes, simulation_parameters


def track_n_turns(machine, bunches, wakes, mpi_settings,simulation_parameters):

    n_turns = simulation_parameters['n_turns']

    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields = UniformBinSlicer(1, z_cuts=(-4., 4.))

    # CREATE WAKES
    # ============
    if mpi_settings is not None:

        wake_field = WakeField(slicer_for_wakefields, wakes,
                               circumference=machine.circumference, h_bunch=simulation_parameters['h_RF'], mpi=mpi_settings)

        machine.one_turn_map.append(wake_field)

    bunch_list = bunches.split()

    x_data = np.zeros((n_turns, len(bunch_list)))

    for i in range(n_turns):
        t0 = time.clock()

        machine.track(bunches)

        if i == 0:
            print 'mpi_settings = ' + str(mpi_settings) + ', n_bunches = ' + str(len(bunch_list))
        t1 = time.clock()
        bunch_list = bunches.split()
        for j, bunch in enumerate(bunch_list):
            x_data[i,j] = bunch.mean_x()

        print('Turn {:d}, {:g} ms, {:s}'.format(i, (t1-t0)*1e3, time.strftime(
            "%d/%m/%Y %H:%M:%S", time.localtime())))

    return x_data


mpi_settings_for_testing = [
#    None,
    'dummy',
    'memory_optimized',
    'mpi_full_ring_fft',
#    'loop_minimized',
#    True,

    ]

mpi_setting_labels = [
#    'without wake objects',
    'without wakes',
    'memory_optimized',
    'mpi_full_ring_fft',
#    'loop_minimized',
#    'original',
    'pure_convolution',
    ]

data_x = []

# gathers data from different wake implementations in PyHEADTAIL
data_x.append([])
ref_machine, ref_bunches, wakes, simulation_parameters = generate_machine_and_bunches()
n_turns = simulation_parameters['n_turns']
for j, mpi_settings in enumerate(mpi_settings_for_testing):
    machine = copy.deepcopy(ref_machine)
    bunches = copy.deepcopy(ref_bunches)
    x_data = track_n_turns(machine, bunches, wakes, mpi_settings, simulation_parameters)
    data_x[-1].append(x_data)

# gathers data from pure convolution calculations
x_data = calculate_pure_convolution(ref_bunches, wakes, simulation_parameters)
data_x[-1].append(x_data)

# plots bunch position data from the last turn
fig, (ax11) = plt.subplots(1, figsize=(16, 9))
for i in xrange(len(data_x[0])):
    ax11.plot(data_x[0][i][-1,:], label = mpi_setting_labels[i])
ax11.legend()
ax11.set_xlabel('Bunch index')
ax11.set_ylabel('mean_x')


# plots turn by turn data
ref_values = np.zeros(simulation_parameters['n_turns'])
for i in xrange(simulation_parameters['n_turns']):
    ref_values[i] = np.sum(np.abs(data_x[0][1][i,:]))

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16, 9), sharex=True, tight_layout=False)
for i in xrange(len(data_x[0])):
    ax1.plot(data_x[0][i][:,0], label = mpi_setting_labels[i])
    ax2.plot(data_x[0][i][:,-1])
    if i > 1:
        values = np.zeros(simulation_parameters['n_turns'])
        for j in xrange(simulation_parameters['n_turns']):
            values[j] = np.sum(np.abs(data_x[0][i][j,:]))

        ref_diff_1 = (values-ref_values)/ref_values
        ax3.plot(ref_diff_1, label = mpi_setting_labels[i])

ax1.legend()
ax3.legend()
ax1.set_ylabel('First bunch, mean_x')
ax2.set_ylabel('Last bunch, mean_x')
ax3.set_ylabel('Average relative error \n of mean_x values')
ax3.set_xlabel('Turn')
plt.show()

# -*- coding: utf-8 -*-

