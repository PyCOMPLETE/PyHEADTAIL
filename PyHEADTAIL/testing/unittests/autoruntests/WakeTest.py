
# coding: utf-8

# In[1]:

import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)



# In[2]:

import numpy as np
from scipy.constants import m_p, c, e

from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.impedances.wakes import (
    WakeField, WakeTable, Resonator, CircularResonator)
from PyHEADTAIL.impedances.wakes import (
    ParallelHorizontalPlatesResonator as ParallelPlatesResonator)
from PyHEADTAIL.impedances.wakes import (
    ResistiveWall, CircularResistiveWall, ParallelPlatesResistiveWall)
from PyHEADTAIL.impedances.wakes import (
    ParallelHorizontalPlatesResistiveWall as ParallelPlatesResistiveWall)
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.particles.particles import Particles
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer
from PyHEADTAIL.general.printers import SilentPrinter


# In[3]:

# HELPERS
def run():

    def track_n_save(bunch, map_):
        mean_x = np.empty(n_turns)
        mean_y = np.empty(n_turns)
        sigma_z = np.empty(n_turns)

        for i in xrange(n_turns):
            mean_x[i] = bunch.mean_x()
            mean_y[i] = bunch.mean_y()
            sigma_z[i] = bunch.sigma_z()

            for m_ in map_:
                m_.track(bunch)

        return mean_x, mean_y, sigma_z

    def my_fft(data):
        t = np.arange(len(data))
        fft = np.fft.rfft(data)
        fft_freq = np.fft.rfftfreq(t.shape[-1])

        return fft_freq, np.abs(fft.real)

    def generate_bunch(n_macroparticles, alpha_x, alpha_y, beta_x, beta_y, linear_map):

        intensity = 1.05e11
        sigma_z = 0.059958
        gamma = 3730.26
        p0 = np.sqrt(gamma**2 - 1) * m_p * c

        beta_z = (linear_map.eta(dp=0, gamma=gamma) * linear_map.circumference /
                  (2 * np.pi * linear_map.Q_s))

        epsn_x = 3.75e-6 # [m rad]
        epsn_y = 3.75e-6 # [m rad]
        epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)

        bunch = generators.generate_Gaussian6DTwiss(
            macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
            gamma=gamma, mass=m_p, circumference=C,
            alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
            alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
            beta_z=beta_z, epsn_z=epsn_z)
        # print ('bunch sigma_z=' + bunch.sigma_z())

        return bunch

    def track_n_show(bunch, slicer, map_woWakes, wake_field):
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(16,16))

        xp_diff  = np.zeros(n_macroparticles)

        for i in xrange(n_turns):
            for m_ in map_woWakes:
                m_.track(bunch)

            # Dipole X kick.
            if i == (n_turns - 1):
                xp_old = bunch.xp.copy()
            wake_field.track(bunch)

            if i == (n_turns - 1):
                xp_diff[:] = bunch.xp[:] - xp_old[:]

        # Plot bunch.z vs. slice index of particle. Mark particles within
        # z cuts in green.
        nsigmaz_lbl = ' (nsigmaz =' + str(n_sigma_z) + ')'

        slice_set = bunch.get_slices(slicer)
        pidx = slice_set.particles_within_cuts
        slidx = slice_set.slice_index_of_particle

        z_cut_tail, z_cut_head = slice_set.z_cut_tail, slice_set.z_cut_head






    # In[4]:
        # Basic parameters.
    n_turns = 10
    n_segments = 1
    n_macroparticles = 50

    Q_x = 64.28
    Q_y = 59.31
    Q_s = 0.0020443

    C = 26658.883
    R = C / (2.*np.pi)

    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj = 66.0064
    beta_y_inj = 71.5376
    alpha_0 = [0.0003225]

    #waketable filename
    fn = os.path.join(os.path.dirname(__file__), 'wake_table.dat')
    # In[5]:

    # Parameters for transverse map.
    s = np.arange(0, n_segments + 1) * C / n_segments

    alpha_x = alpha_x_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)

    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_y = np.zeros(n_segments)


    # In[6]:



    # In[7]:



    # In[8]:

    # CASE TEST SETUP
    trans_map = TransverseMap(
        s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    long_map = LinearMap(alpha_0, C, Q_s)

    bunch = generate_bunch(
        n_macroparticles, alpha_x_inj, alpha_y_inj, beta_x_inj, beta_y_inj,
        long_map)


    # In[9]:

    # CASE I
    # Transverse and long. tracking (linear), and wakes from WakeTable source.
    # DIPOLE X, UniformBinSlicer

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    wake_file_columns = ['time', 'dipole_x', 'no_dipole_y', 'no_quadrupole_x', 'no_quadrupole_y',
                         'no_dipole_xy', 'no_dipole_yx']
    table = WakeTable(fn, wake_file_columns,
                      printer=SilentPrinter(),
                      warningprinter=SilentPrinter())
    wake_field = WakeField(uniform_bin_slicer, table)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)

    # In[10]:

    # CASE II
    # Transverse and long. tracking (linear), and wakes from WakeTable source.
    # DIPOLE X, UniformChargeSlicer

    n_sigma_z = 2
    n_slices = 15
    uniform_charge_slicer = UniformChargeSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    wake_file_columns = ['time', 'dipole_x', 'no_dipole_y', 'no_quadrupole_x', 'no_quadrupole_y',
                         'no_dipole_xy', 'no_dipole_yx']
    table = WakeTable(fn, wake_file_columns,
                      printer=SilentPrinter(),
                      warningprinter=SilentPrinter())
    wake_field = WakeField(uniform_charge_slicer, table)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[11]:

    # CASE III
    # Transverse and long. tracking (linear), and wakes from WakeTable source.
    # Quadrupole X, UniformChargeSlicer

    n_sigma_z = 2
    n_slices = 15
    uniform_charge_slicer = UniformChargeSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    wake_file_columns = ['time', 'no_dipole_x', 'no_dipole_y', 'quadrupole_x', 'no_quadrupole_y',
                         'no_dipole_xy', 'no_dipole_yx']
    table = WakeTable(fn, wake_file_columns,
                      printer=SilentPrinter(),
                      warningprinter=SilentPrinter())
    wake_field = WakeField(uniform_charge_slicer, table)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[12]:

    # CASE IV
    # Transverse and long. tracking (linear), and wakes from WakeTable source.
    # Quadrupole X, UniformBinSlicer

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    wake_file_columns = ['time', 'no_dipole_x', 'no_dipole_y', 'quadrupole_x', 'no_quadrupole_y',
                         'no_dipole_xy', 'no_dipole_yx']
    table = WakeTable(fn, wake_file_columns,
                      printer=SilentPrinter(),
                      warningprinter=SilentPrinter())
    wake_field = WakeField(uniform_bin_slicer, table)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[15]:

    # CASE V
    # Transverse and long. tracking (linear),
    # Resonator circular

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    reson_circ = CircularResonator(R_shunt=1e6, frequency=1e8, Q=1)
    wake_field = WakeField(uniform_bin_slicer, reson_circ)

    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[16]:

    # CASE V b.
    # Transverse and long. tracking (linear),
    # Several Resonators circular

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    reson_circ = CircularResonator(R_shunt=1e6, frequency=1e8, Q=1)
    reson_circ2 = CircularResonator(R_shunt=1e6, frequency=1e9, Q=0.8)
    reson_circ3 = CircularResonator(R_shunt=5e6, frequency=1e6, Q=0.2)

    wake_field = WakeField(uniform_bin_slicer, reson_circ, reson_circ2, reson_circ3)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[17]:

    # CASE V c.
    # Transverse and long. tracking (linear),
    # Resonator parallel_plates

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    reson_para = ParallelPlatesResonator(R_shunt=1e6, frequency=1e8, Q=1)
    wake_field = WakeField(uniform_bin_slicer, reson_para)

    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[18]:

    # CASE V d.
    # Transverse and long. tracking (linear),
    # Resonator w. longitudinal wake

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    reson = Resonator(R_shunt=1e6, frequency=1e8, Q=1, Yokoya_X1=1, Yokoya_X2=1, Yokoya_Y1=1, Yokoya_Y2=1, switch_Z=True)
    wake_field = WakeField(uniform_bin_slicer, reson)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[19]:

    # CASE VI
    # Transverse and long. tracking (linear),
    # ResistiveWall circular

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    resis_circ = CircularResistiveWall(pipe_radius=5e-2, resistive_wall_length=C,
                                        conductivity=1e6, dt_min=1e-3)
    wake_field = WakeField(uniform_bin_slicer, resis_circ)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[20]:

    # CASE VI b.
    # Transverse and long. tracking (linear),
    # ResistiveWall parallel_plates

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    resis_para = ParallelPlatesResistiveWall(pipe_radius=5e-2, resistive_wall_length=C,
                                        conductivity=1e6, dt_min=1e-3)
    wake_field = WakeField(uniform_bin_slicer, resis_para)


    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


    # In[21]:

    # CASE VII.
    # Transverse and long. tracking (linear),
    # Pass mixture of WakeSources to define WakeField.

    n_sigma_z = 2
    n_slices = 15
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    # Definition of WakeField as a composition of different sources.
    resis_circ = CircularResistiveWall(pipe_radius=5e-2, resistive_wall_length=C,
                                        conductivity=1e6, dt_min=1e-3)
    reson_para = ParallelPlatesResonator(R_shunt=1e6, frequency=1e8, Q=1)
    wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y',
                         'dipole_xy', 'dipole_yx']
    table = WakeTable(fn, wake_file_columns,
                      printer=SilentPrinter(),
                      warningprinter=SilentPrinter())

    wake_field = WakeField(uniform_bin_slicer, resis_circ, reson_para, table)

    trans_map = [ m for m in trans_map ]
    map_woWakes = trans_map + [long_map]

    track_n_save(bunch,map_woWakes)


if __name__ == '__main__':
    run()

# In[19]:
