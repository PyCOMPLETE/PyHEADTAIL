#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import time as time
from copy import deepcopy
from scipy.constants import m_p, c, e

from PyHEADTAIL.machines.synchrotron import *
from PyHEADTAIL.impedances.wakes import WakeField, WakeSource, WakeTable
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.general.printers import SilentPrinter

def compute_wake_potential_with_convolution(n_slices_wake, method):
    n_macroparticles = n_macroparticles_per_slice * n_slices_wake
    slicer_uniform = UniformBinSlicer(
                        n_slices_wake, z_cuts=(-z_cut, z_cut))
    ww1 = WakeTable(wakefile1, wakefile_columns, n_turns_wake=n_turns_wake, 
                    method=method, beta=machine.beta, slicer=slicer_uniform)
    wake_field = WakeField(slicer_uniform, ww1)

    bunch = machine.generate_6D_Gaussian_bunch(n_macroparticles=n_macroparticles,
                                               intensity=intensity, epsn_x=epsn_x,
                                               epsn_y=epsn_y, sigma_z=sigma_z)
    bunch.x += 1e-3
    bunch.y += 1e-3

    slice_set = bunch.get_slices(slicer_uniform, statistics=['mean_xp', 'mean_yp'])
    xp_before_kick = slice_set.mean_xp.copy()   

    wake_field.track(bunch)

    slice_set2 = bunch.get_slices(slicer_uniform, statistics=['mean_xp', 'mean_yp'])
    kick_xp = slice_set2.mean_xp - xp_before_kick

    window = slice_set2.z_centers.size//10
    z_convolution = slice_set2.z_centers[window//2:-window//2+1]
    wake_potential_convolution = np.convolve(kick_xp, np.ones(window), "valid")/window
    
    return z_convolution, wake_potential_convolution



R = 100 # PS radius
C = 2*R*np.pi

# CREATE SYNCHROTRON (TRANSVERSE MAP + LONGITUDINAL MAP)
# =====================
n_segments = 1

Q_x = 6.225
Q_y = 6.24
Q_s = 0.00126 # Used when longitudinal_mode = 'linear'

Qp_x           = 0.
Qp_y           = 0.

s              = np.arange(0, n_segments + 1) * C / n_segments
alpha_x        = 0 * np.ones(n_segments + 1)
beta_x         = 16  * np.ones(n_segments)
D_x            = 0     * np.ones(n_segments)
alpha_y        = 0 * np.ones(n_segments)
beta_y         = 16  * np.ones(n_segments)
D_y            = 0     * np.ones(n_segments)

charge           = e
mass             = m_p
alpha            = 0.02687
h                = [7]
V                = [50e3]
dphi             = 0.

gamma = 3.13
p0 = np.sqrt(gamma**2 - 1) * m_p * c
p_increment = 0.

# Machine with PS parameters
machine = Synchrotron(optics_mode = 'smooth', circumference = C,
                      n_segments = n_segments, alpha_x = alpha_x,
                      beta_x = beta_x, D_x = D_x, alpha_y = alpha_y,
                      beta_y = beta_y, D_y = D_y, accQ_x = Q_x,
                      accQ_y = Q_y, Qp_x = Qp_x, Qp_y = Qp_y,
                      alpha_mom_compaction = alpha,
                      longitudinal_mode = 'linear', Q_s = Q_s, h_RF = h,
                      dphi_RF = dphi, charge = e, mass = m_p, p0 = p0,
                      p_increment = p_increment)

machine.gamma = gamma

###############################################################################3
# CREATE BEAM
# ===========
intensity = 1e11

epsn_x = 3.25e-6 # in [m rad]
epsn_y = 3.25e-6 # in [m rad]
sigma_z = 12.36 # in [m]
z_cut = 3 * sigma_z


# CREATE BEAM SLICERS
# ===================
n_macroparticles_per_slice = 700

# CREATE WAKES
# ============

# Wake Table
wakefile1 = ('/wakes/PS_wall_impedance_Ekin_2.0GeV.wake')
n_turns_wake = 1

wakefile_columns = ['time', 'dipole_x', 'nodipole_y', 'quadrupole_x', 'noquadrupole_y']
ww1 = WakeTable(wakefile1, wakefile_columns, n_turns_wake=n_turns_wake, 
                method='interpolated', beta=machine.beta
               )


print('Plot wake table content')

plt.figure(figsize=(8,6))
plt.plot(ww1.wake_table['time'], ww1.wake_table['dipole_x'], label='dipole_x')
plt.plot(ww1.wake_table['time'], ww1.wake_table['quadrupole_x'], label='quadrupole_x')
plt.xlabel('Time [ns]')
plt.ylabel('W [V/pC/mm]')
plt.legend()
plt.xlim(-1, 2)
plt.show()


print('Methods comparison /n')
print('Integrated method might take up some time to integrate the wake file '
      'as it contains many points.')
n_slices_integrated = 600
n_slices_interpolated = 5000
plt.figure()
plt.plot(*compute_wake_potential_with_convolution(n_slices_integrated, 'integrated'), 
         label=f'Integrated method, {n_slices_integrated:.0f} slices')
plt.plot(*compute_wake_potential_with_convolution(n_slices_interpolated, 'interpolated'), 
         ls='--', label=f'Integrated method, {n_slices_interpolated:.0f} slices')
plt.xlabel('z [m]')
plt.ylabel('Kick [a.u.]')
plt.grid()
plt.legend(bbox_to_anchor=(1,1))
plt.show()


print('Results are exactly matching when both methods have converged. '
      'Using the integrated method results in a factor ~5 reduction of '
      'number of slices while giving the wake potential.')
