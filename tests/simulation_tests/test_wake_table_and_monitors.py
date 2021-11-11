import os
import pathlib

import h5py
import numpy as np
from scipy.constants import c, e
from scipy.stats import linregress
from scipy.signal import hilbert

from LHC import LHC
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor


def test_monitors():

    n_attempts = 5

    outputpath = './'  # outputpath relative to this file

    n_macroparticles = 10000
    n_turns = 3000

    # Beam and machine parameters
    intensity = 1.8e11
    epsn_x = 1.8e-6  # normalised horizontal emittance
    epsn_y = 1.8e-6  # normalised vertical emittance

    Qp_x = -10.0
    Qp_y = -10.0
    i_oct = 5
    dampingrate = 0

    wake_folder = pathlib.Path(__file__).parent.joinpath(
        '../../examples/impedances').absolute()

    # Injection
    machine_configuration = 'Injection'
    p0 = 450e9*e/c
    wakefile = wake_folder.joinpath(
        'wakes/wakeforhdtl_PyZbase_Allthemachine_450GeV'
        '_B1_LHC_inj_450GeV_B1.dat')

    # Detuners
    app_x = 2 * p0 * 27380.10941 * i_oct / 100.
    app_y = 2 * p0 * 28875.03442 * i_oct / 100.
    app_xy = 2 * p0 * -21766.48714 * i_oct / 100.
    Qpp_x = 4889.00298 * i_oct / 100.
    Qpp_y = -2323.147896 * i_oct / 100.

    # Create machine
    machine = LHC(n_segments=1,
                  machine_configuration=machine_configuration,
                  **{'app_x': app_x, 'app_y': app_y, 'app_xy': app_xy,
                     'Qp_x': [Qp_x, Qpp_x], 'Qp_y': [Qp_y, Qpp_y]})

    sigma_z = 1.2e-9 * machine.beta*c/4.  # RMS bunch length in meters

    # Wakes
    slicer_for_wakefields = UniformBinSlicer(
        n_slices=500, z_cuts=(-3*sigma_z, 3*sigma_z))
    wake_table1 = WakeTable(wakefile,
                            ['time', 'dipole_x', 'dipole_y',
                             'noquadrupole_x', 'noquadrupole_y',
                             'dipole_xy', 'dipole_yx',
                             ])
    wake_field = WakeField(slicer_for_wakefields, wake_table1)

    # Damper
    damper = TransverseDamper(dampingrate, dampingrate)

    machine.one_turn_map.append(wake_field)
    machine.one_turn_map.append(damper)

    # Loop over attempts
    i_attempt = 0
    while i_attempt < n_attempts:

        print(f"Attempt {i_attempt+1}:")

        # Create beam
        bunch = machine.generate_6D_Gaussian_bunch_matched(
            n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

        print("\n--> Bunch length and emittance: {:g} m, {:g} eVs.".format(
            bunch.sigma_z(), bunch.epsn_z()))

        # Monitors
        try:
            bucket = machine.longitudinal_map.get_bucket(bunch)
        except AttributeError:
            bucket = machine.rfbucket

        simulation_parameters_dict = {
            'gamma': machine.gamma,
            'beta': machine.beta,
            'intensity': intensity,
            'Qx': machine.Q_x,
            'Qy': machine.Q_y,
            'Qs': bucket.Q_s,
            'beta_x': bunch.beta_Twiss_x(),
            'beta_y': bunch.beta_Twiss_y(),
            'beta_z': bucket.beta_z,
            'epsn_x': bunch.epsn_x(),
            'epsn_y': bunch.epsn_y(),
            'sigma_z': bunch.sigma_z(),
        }
        bunchmonitor = BunchMonitor(
            filename=outputpath+'/bunchmonitor', n_steps=n_turns,
            parameters_dict=simulation_parameters_dict,
            write_buffer_to_file_every=512, buffer_size=4096)

        slicer_for_slicemonitor = UniformBinSlicer(
            n_slices=50, z_cuts=(-3*sigma_z, 3*sigma_z))

        bunch_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp']

        slicemonitor = SliceMonitor(
            filename=outputpath+'/slicemonitor', n_steps=n_turns,
            slicer=slicer_for_slicemonitor,
            parameters_dict=simulation_parameters_dict,
            write_buffer_every=64, buffer_size=256,
            **{'bunch_stats_to_store': bunch_stats_to_store},
        )

        # Save for plotting
        x = np.zeros(n_turns, dtype=float)

        # Tracking loop
        for i in range(n_turns):

            # track the beam around the machine for one turn:
            machine.track(bunch)

            x[i] = bunch.mean_x()

            # monitor the bunch and slice statistics (once per turn):
            bunchmonitor.dump(bunch)
            slicemonitor.dump(bunch)

        # Get data from monitor files
        bunch_file = h5py.File(outputpath+'/bunchmonitor.h5')
        slice_file = h5py.File(outputpath+'/slicemonitor.h5')

        x_from_bunch_file = bunch_file['Bunch']['mean_x'][:]
        x_from_slice_file = slice_file['Bunch']['mean_x'][:]

        n_per_slice = slice_file['Slices']['n_macroparticles_per_slice'][:]
        x_per_slice = slice_file['Slices']['mean_x'][:]
        x_from_slices = np.average(x_per_slice, axis=0, weights=n_per_slice)

        sx = np.sqrt(epsn_x * bunch_file.attrs['beta_x']
                     / bunch_file.attrs['gamma'] / bunch_file.attrs['beta'])

        # Check results
        turns = np.arange(n_turns)

        iMin = 500
        iMax = n_turns - 500
        ampl = np.abs(hilbert(x))
        b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
        print(f"Growth rate {b*1e4:.2f} [10^-4/turn]")

        ampl = np.abs(hilbert(x_from_slices))
        b_from_slices, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
        print(f"Growth rate from slices {b_from_slices*1e4:.2f} [10^-4/turn]")

        h5py.File.close(bunch_file)
        h5py.File.close(slice_file)

        check_bunch = np.allclose(x_from_bunch_file/sx, x/sx)
        check_slice = np.allclose(x_from_slice_file/sx, x/sx)
        check_growth_rate = np.allclose(b_from_slices, b, rtol=3e-2)

        # assert np.allclose(x_from_bunch_file/sx, x/sx), \
        #     "x from bunch file doesn't match"
        # assert np.allclose(x_from_slice_file/sx, x/sx), \
        #     "x from slice file doesn't match"
        # assert np.allclose(b_from_slices, b, rtol=3e-2), \
        #     "Growth rate from slices doesn't match"

        assert check_bunch or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts x from bunch file doesn't match"
        assert check_slice or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts x from slice file doesn't match"
        assert check_growth_rate or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts Growth rate from slices doesn't match"

        os.remove(outputpath+'/bunchmonitor.h5')
        os.remove(outputpath+'/slicemonitor.h5')

        if check_bunch and check_slice and check_growth_rate:
            print(f"Passed on {i_attempt + 1}. attempt.")
            break
        i_attempt += 1
