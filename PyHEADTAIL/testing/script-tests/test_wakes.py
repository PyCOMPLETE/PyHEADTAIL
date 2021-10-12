

from collections import OrderedDict

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import m_p, c, e

import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.general.printers import SilentPrinter
from PyHEADTAIL.impedances.wakes import CircularResistiveWall, ParallelHorizontalPlatesResistiveWall
from PyHEADTAIL.impedances.wakes import WakeField, WakeTable, Resonator, CircularResonator, \
    ParallelHorizontalPlatesResonator
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap

# Set up simulation
# =================
n_turns = 500
n_slices = 15
n_sigma_z = 2
n_segments = 1
n_macroparticles = 500


def main():
    # Set up machine (LHC-type)
    # =========================
    p0 = 3.5e12
    C = 26658.883
    R = C / (2. * np.pi)
    mass = m_p * c ** 2 / e
    gamma = np.sqrt(1 + (p0 / mass) ** 2)

    Q_x = 64.28
    Q_y = 59.31
    Q_s = 0.0020443

    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj = 66.0064
    beta_y_inj = 71.5376
    alpha_0 = [0.0003225]

    s = np.arange(0, n_segments + 1) * C / n_segments

    alpha_x = alpha_x_inj * np.ones(n_segments)
    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)
    D_y = np.zeros(n_segments)

    trans_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    long_map = LinearMap(alpha_0, C, Q_s)
    beta_z = long_map.beta_z(gamma)

    # Set up beam
    # ===========
    np.random.seed(42)

    intensity = 1.05e11
    sigma_z = 0.059958

    epsn_x = 3.75e-6  # [m rad]
    epsn_y = 3.75e-6  # [m rad]
    epsn_z = 4 * np.pi * sigma_z ** 2 * p0 / c / beta_z

    bunch = generators.generate_Gaussian6DTwiss(
        macroparticlenumber=n_macroparticles, intensity=intensity,
        charge=e, mass=m_p, gamma=gamma, circumference=C,
        alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
        alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z)

    # Create wake fields
    # ==================
    uniform_charge_slicer = UniformChargeSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)
    uniform_bin_slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z, circumference=0, h_bunch=10)

    # Error case (I) for WakeTable. Number of wake_file_columns does not correspond to that of the wake_file.
    try:
        wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y', 'dipole_xy', 'dipole_yx',
                             'nonsense']
        table = WakeTable('wake_table.dat', wake_file_columns)
        print('test NOT passed. No error raised!')
    except ValueError as exc:
        print('test passed: the expected ValueError due to mismatched column contents ' +
              'vs column description occured.\n')
        print('Error message:\n' + str(exc))

    # Error case (II) for WakeTable. No wake_file_column 'time' defined.
    try:
        wake_file_columns = ['nonsense', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                             'dipole_yx']
        table = WakeTable('wake_table.dat', wake_file_columns)
        print('test NOT passed. No error raised!')
    except ValueError as exc:
        print('test passed: the expected ValueError due to missing \'time\' column occured.\n ')
        print('Error message:\n' + str(exc))

    # Test different wakes
    # ====================
    # (I) waketable only dipole - uniform bin
    wake_file_columns = [
        'time', 'dipole_x', 'no_dipole_y', 'no_quadrupole_x', 'no_quadrupole_y', 'no_dipole_xy', 'no_dipole_yx']
    table = WakeTable('wake_table.dat', wake_file_columns, warningprinter=SilentPrinter())
    wake_dipole = WakeField(uniform_bin_slicer, table)

    # (II) waketable only dipole - uniform charge
    wake_dipole_unicharge = WakeField(uniform_charge_slicer, table)

    # (III) waketable only quadrupole - uniform bin
    wake_file_columns = [
        'time', 'no_dipole_x', 'no_dipole_y', 'quadrupole_x', 'no_quadrupole_y', 'no_dipole_xy', 'no_dipole_yx']
    table = WakeTable('wake_table.dat', wake_file_columns, warningprinter=SilentPrinter())
    wake_quads = WakeField(uniform_bin_slicer, table)

    # (IV) waketable quads - uniform charge
    wake_quads_unicharge = WakeField(uniform_charge_slicer, table)

    # (V) - circular resonator wake
    reson_circ = CircularResonator(R_shunt=1e6, frequency=1e8, Q=1)
    wake_resonator = WakeField(uniform_bin_slicer, reson_circ)

    # (VI) - many resonators
    reson_circ1 = CircularResonator(R_shunt=1e6, frequency=1e8, Q=1)
    reson_circ2 = CircularResonator(R_shunt=1e6, frequency=1e9, Q=0.8)
    reson_circ3 = CircularResonator(R_shunt=5e6, frequency=1e6, Q=0.2)
    wake_resonators = WakeField(uniform_bin_slicer, [reson_circ1, reson_circ2, reson_circ3])

    # (VII) - parallel plates
    reson_para = ParallelHorizontalPlatesResonator(R_shunt=1e6, frequency=1e8, Q=1)
    wake_resonator_parallel = WakeField(uniform_bin_slicer, reson_para)

    # (VIII) - resonator with long. wake
    reson = Resonator(R_shunt=1e6, frequency=1e8, Q=1, Yokoya_X1=1, Yokoya_X2=1, Yokoya_Y1=1, Yokoya_Y2=1,
                      switch_Z=True)
    wake_longitudinal = WakeField(uniform_bin_slicer, reson)

    # (IX) - transverse and longitudinal resistive wall
    resis_circ = CircularResistiveWall(pipe_radius=5e-2, resistive_wall_length=C, conductivity=1e6, dt_min=1e-3 / c)
    wake_rewall = WakeField(uniform_bin_slicer, resis_circ)

    # (X) - rewall parallel plates
    resis_para = ParallelHorizontalPlatesResistiveWall(pipe_radius=5e-2, resistive_wall_length=C, conductivity=1e6,
                                                       dt_min=1e-3 / c)
    wake_rewall_parallel = WakeField(uniform_bin_slicer, resis_para)

    # (XI) - mixed
    resis_circ = CircularResistiveWall(pipe_radius=5e-2, resistive_wall_length=C, conductivity=1e6, dt_min=1e-3 / c)
    wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y', 'dipole_xy', 'dipole_yx']
    reson_para = ParallelHorizontalPlatesResonator(R_shunt=1e6, frequency=1e8, Q=1)
    table = WakeTable('wake_table.dat', wake_file_columns, warningprinter=SilentPrinter())
    wake_mixed = WakeField(uniform_bin_slicer, [resis_circ, reson_para, table])

    # Python 2 way
    wakes_collection = OrderedDict([
        ('wake_dipole',
         [wake_dipole, uniform_bin_slicer, "(I) dipole wake", "dipole_x"]),
        # ('wake_dipole_unicharge',
        #  [wake_dipole_unicharge, uniform_charge_slicer, "(II) dipole wake uniform charge", "dipole_x"]),
        # ('wake_quads_unicharge',
        #  [wake_quads_unicharge, uniform_charge_slicer, "(III) quad wake uniform charge", "quadrupole_x"]),
        # ('wake_quads',
        #  [wake_quads, uniform_bin_slicer, "(IV) quad wake", "quadrupole_x"]),
        # ('wake_resonator',
        #  [wake_resonator, uniform_bin_slicer, "(V) resonator wake", "dipole_x"]),
        # ('wake_resonators',
        #  [wake_resonators, uniform_bin_slicer, "(VI) resonators list wake", "dipole_x"]),
        # ('wake_resonator_parallel',
        #  [wake_resonator_parallel, uniform_bin_slicer, "(VII) parallel plates resonator wake", "dipole_x"]),
        # ('wake_longitudinal',
        #  [wake_longitudinal, uniform_bin_slicer, "(VIII) longitudinal resonator wake", "longitudinal"]),
        # ('wake_rewall',
        #  [wake_rewall, uniform_bin_slicer, "(IX) resistive wall wake", "dipole_x"]),
        # ('wake_rewall_parallel',
        #  [wake_rewall_parallel, uniform_bin_slicer, "(X) parallel plates resistive wall wakes", "dipole_x"]),
        # ('wake_mixed',
        #  [wake_mixed, uniform_bin_slicer, "(XI) several mixed wakes", "dipole_x"])
    ])

    # Python 3 way
    # wakes_collection = OrderedDict(
    #     wake_dipole=[wake_dipole, uniform_bin_slicer, "(I) dipole wake", "dipole_x"],
    #     wake_dipole_unicharge=[wake_dipole_unicharge, uniform_charge_slicer, "(II) dipole wake uniform charge",
    #                            "dipole_x"],
    #     wake_quads=[wake_quads, uniform_bin_slicer, "(III) quad wake", "quadrupole_x"],
    #     wake_quads_unicharge=[wake_quads_unicharge, uniform_charge_slicer, "(IV) quad wake uniform charge",
    #                           "quadrupole_x"],
    #     wake_resonator=[wake_resonator, uniform_bin_slicer, "(V) resonator wake", "dipole_x"],
    #     wake_resonator_parallel=[wake_resonator_parallel, uniform_bin_slicer, "(VI) parallel plates resonator wake",
    #                              "dipole_x"],
    #     wake_resonators=[wake_resonators, uniform_bin_slicer, "(VII) resonators list wake", "dipole_x"],
    #     wake_longitudinal=[wake_longitudinal, uniform_bin_slicer, "(VIII) longitudinal resonator wake", "longitudinal"],
    #     wake_rewall=[wake_rewall, uniform_bin_slicer, "(IX) resistive wall wake", "dipole_x"],
    #     wake_rewall_parallel=[wake_rewall_parallel, uniform_bin_slicer, "(X) parallel plates resistive wall wakes",
    #                           "dipole_x"],
    #     wake_mixed=[wake_mixed, uniform_bin_slicer, "(XI) several mixed wakes", "dipole_x"]
    # )

    trans_map = [m for m in trans_map]
    one_turn_map = trans_map + [long_map]

    # Tracking loop
    # =============
    for i, wakes_entry in enumerate(wakes_collection.values()):
        wake, slicer, label, component = wakes_entry
        test_wake_kick(bunch, slicer, one_turn_map, wake, case=label)
        # evaluate_wake_kick(bunch, slicer, one_turn_map, wake, case=label)
        # if i < 4:
        #     print(i, label)
        #     show_sampled_wake(bunch, uniform_bin_slicer, table, wake_component=component, case=label)


# Helper functions
# ================
# def plot_data(sigma_z, mean, Q, Qs):
#     fig = plt.figure(figsize=(16, 16))
#     ax1 = fig.add_subplot(311)
#     ax2 = fig.add_subplot(312)
#     ax3 = fig.add_subplot(313)
#
#     ax1.plot(mean, '-', c='b')
#     # ax1.plot(mean_y, '-', c='r')
#     ax1.set_xlabel('turns')
#     ax1.set_ylabel('mean [m]')
#
#     ax2.plot(sigma_z, '-', c='b')
#     ax2.set_xlabel('turns')
#     ax2.set_ylabel('sigma_z [m]')
#
#     fr_x, ax_x = my_fft(mean)
#     markerline, stemlines, baseline = ax3.stem(fr_x, ax_x, label=r'bunch spectrum')
#     plt.setp(baseline, 'color', 'b', 'linewidth', 2)
#     ax3.axvline(Q % 1, color='r', label='transverse main tune')
#     ax3.axvline(Q % 1 - Qs, color='g', linestyle='dashed', label=r'1st synchrotron sidebands')
#     ax3.axvline(Q % 1 + Qs, color='g', linestyle='dashed')
#     handles, labels = ax3.get_legend_handles_labels()
#     ax3.legend(handles, labels, loc='upper left')
#     ax3.set_xlabel('tune')
#     ax3.set_ylabel('amplitude')
#     ax3.set_xlim((0.25, 0.32))
#
#     # plt.show()


# def track_n_save(bunch, map_):
#     mean_x = np.empty(n_turns)
#     mean_y = np.empty(n_turns)
#     sigma_z = np.empty(n_turns)
#
#     for i in xrange(n_turns):
#         mean_x[i] = bunch.mean_x()
#         mean_y[i] = bunch.mean_y()
#         sigma_z[i] = bunch.sigma_z()
#
#         for m_ in map_:
#             m_.track(bunch)
#
#     return mean_x, mean_y, sigma_z


# def my_fft(data):
#     t = np.arange(len(data))
#     fft = np.fft.rfft(data)
#     fft_freq = np.fft.rfftfreq(t.shape[-1])
#
#     return fft_freq, np.abs(fft.real)


def evaluate_wake_kick(bunch, slicer, one_turn_map, wake_field, case, show=True, savefig=False, test=True):
    fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(10, 10))

    print('\nCase %s\n' % case)

    xp_diff = np.zeros(n_macroparticles)

    for i in range(n_turns):
        for m in one_turn_map:
            m.track(bunch)

        # Dipole X kick.
        if i == (n_turns - 1):
            xp_old = bunch.xp.copy()
        wake_field.track(bunch)

        if i == (n_turns - 1):
            xp_diff[:] = bunch.xp[:] - xp_old[:]

    # Plot bunch.z vs. slice index of particle. Mark particles within z cuts in green.
    # nsigmaz_lbl = ' (nsigmaz =' + str(n_sigma_z) + ')'

    slice_set = bunch.get_slices(slicer)
    pidx = slice_set.particles_within_cuts
    slidx = slice_set.slice_index_of_particle

    z_cut_tail, z_cut_head = slice_set.z_cut_tail, slice_set.z_cut_head

    ax1.plot(slidx, bunch.z, '.r', ms=10, label='all particles')
    ax1.plot(slidx.take(pidx), bunch.z.take(pidx), '.g', label='particles within z cuts')
    ax1.axhline(z_cut_tail, color='m', linestyle='dashed', label='slicer boundaries')
    ax1.axhline(z_cut_head, color='m', linestyle='dashed')
    ax1.axvline(0, color='b', linestyle='dashed', label='first and last slices')
    ax1.axvline(slice_set.n_slices - 1, color='b', linestyle='dashed')
    [ax1.axhline(z, color='m', linestyle='dashed') for z in slice_set.z_bins]
    ax1.legend(loc='upper left')

    # Show dipole and qudrupole kicks applied for each particle for the last turn.
    # ax13.plot(slidx, xp_diff_quad, '.g', ms=10, label='quad x kicks')
    ax2.plot(slidx, xp_diff, '.r', label='x kicks')
    ax2.axvline(0, color='b', linestyle='dashed', label='first and last slices')
    ax2.axvline(n_slices - 1, color='b', linestyle='dashed')
    ax2.axhline(0, color='black', ls='dashed')
    # ax13.axvline(0, color='b', linestyle='dashed', label='first and last slices' + nsigmaz_lbl)
    # ax13.axvline(n_slices-1, color='b', linestyle='dashed')
    ax2.legend(loc='lower right')
    # ax13.legend(loc='lower right')

    xmax = max(slidx)
    xmax += 2
    xmin = min(slidx)
    xmin -= 2

    ymax = max(xp_diff)
    ymax += ymax * 0.2
    ymin = min(xp_diff)
    ymin += ymin * 0.2

    ax1.set_xlim((xmin, xmax))
    # ax2.set_xlim((xmin, xmax))
    # ax13.set_xlim((xmin, xmax))
    fig.subplots_adjust(bottom=0.05, top=0.93, hspace=0.16)
    fig.suptitle(case)

    if test:
        ix = case.split('(')[1].split(')')[0]
        xp_diff_ref = np.load("reference_data/wake_kicks/case_{:s}_kick_data.npy".format(ix))
        xx = xp_diff / abs(xp_diff).max()
        xx_ref = xp_diff_ref / abs(xp_diff_ref).max()
        ax2.plot(slidx, xx, 'd', color='cyan', label='x kicks')
        ax2.plot(slidx, xx_ref, 'o', color='magenta', label='x kicks')
        print(np.sum((xx - xx_ref) ** 2))
    if savefig:
        fig.savefig('output/Case_%s.png' % (case))
    if show:
        plt.show()


def test_wake_kick(bunch, slicer, one_turn_map, wake_field, case):
    xp_diff = np.zeros(n_macroparticles)

    for i in range(n_turns):
        for m in one_turn_map:
            m.track(bunch)

        # Dipole X kick.
        if i == (n_turns - 1):
            xp_old = bunch.xp.copy()
        wake_field.track(bunch)

        if i == (n_turns - 1):
            xp_diff[:] = bunch.xp[:] - xp_old[:]

    ix = case.split('(')[1].split(')')[0]
    xp_diff_ref = np.load("reference_data/wake_kicks/case_{:s}_kick_data.npy".format(ix))
    xx = xp_diff / abs(xp_diff).max()
    xx_ref = xp_diff_ref / abs(xp_diff_ref).max()
    # ax2.plot(slidx, xx, 'd', color='cyan', label='x kicks')
    # ax2.plot(slidx, xx_ref, 'o', color='magenta', label='x kicks')

    chi_square = np.sum((xx - xx_ref) ** 2)
    # print(chi_square)
    assert chi_square < 1e-9


def show_sampled_wake(bunch, slicer, wake_table, wake_component, case):
    slice_set = bunch.get_slices(slicer)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot wake table and show where wake is sampled (values obtained from interp1d)
    # for dipole and quadrupole X kicks.
    ax1.plot(1e-9 * wake_table.wake_table['time'][:-1], abs(1e15 * wake_table.wake_table[wake_component][:-1]),
             color='b')
    # ax21.plot(wake_table.wake_table['time'][:-1], abs(wake_table.wake_table['quadrupole_x'][:-1]),
    #          color='r', label='quadrupole x')

    sampled_wake = wake_table.function_transverse(wake_component)
    dt = np.concatenate(
        (slice_set.convert_to_time(slice_set.z_centers) - slice_set.convert_to_time(slice_set.z_centers[-1]),
         (slice_set.convert_to_time(slice_set.z_centers) - slice_set.convert_to_time(slice_set.z_centers[0]))[1:]))

    ax1.plot(abs(dt), abs(sampled_wake(dt)), '.g', ms=15, label='sampled and interpolated wake')

    slice_width = (slice_set.z_cut_head - slice_set.z_cut_tail) / slice_set.n_slices
    dzz = np.arange(0, n_slices * slice_width, slice_width)
    [ax1.axvline(z / (bunch.beta * c), color='black', ls='dashed') for z in dzz[1:]]
    ax1.axvline(dzz[0] / (bunch.beta * c), color='black', ls='dashed', label='slice widths')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('abs. wake strength [V/C/m]')
    ax1.legend(loc='upper right')

    ax1.set_xlim((1e-9 * wake_table.wake_table['time'][0], 1e-9 * wake_table.wake_table['time'][-2]))

    plt.show()
    # fig.savefig('output/Case_%s_show_sampled_wake.png' % (case))


if __name__ == '__main__':
    main()
