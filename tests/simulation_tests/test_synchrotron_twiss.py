import pathlib

import numpy as np
import pickle
from scipy.constants import c as c_light
from scipy.signal import find_peaks_cwt

from LHC import LHC


def test_synchrotron_twiss():

    macroparticlenumber_track = 5000
    macroparticlenumber_optics = 100000

    n_turns = 5000

    epsn_x  = 2.5e-6
    epsn_y  = 3.5e-6
    sigma_z = 0.6e-9 / 4.0 * c_light

    intensity = 1e11

    # Create machine using twiss parameters from optics pickle

    optics_folder = pathlib.Path(__file__).parent.joinpath(
        '../../examples/synchrotron').absolute()
    optics_file = optics_folder.joinpath('lhc_2015_80cm_optics.pkl')
    with open(optics_file, 'rb') as fid:
        optics = pickle.load(fid, encoding='latin1')
    optics.pop('circumference')
    optics.pop('part')
    optics.pop('L_interaction')

    Qx_expected = optics['accQ_x'][-1]
    Qy_expected = optics['accQ_y'][-1]
    Qs_expected = 0.00168

    machine = LHC(machine_configuration='6.5_TeV_collision',
                  optics_mode = 'non-smooth', V_RF=10e6,
                  **optics)
    print(f'Q_x = {machine.Q_x}')
    print(f'Q_y = {machine.Q_y}')
    print(f'Q_s = {machine.Q_s}')

    assert np.isclose(machine.Q_x, Qx_expected, rtol=0, atol=1e-2)
    assert np.isclose(machine.Q_y, Qy_expected, rtol=0, atol=1e-2)
    assert np.isclose(machine.Q_s, Qs_expected, rtol=0, atol=1e-5)

    # Create bunch for optics test
    print('Create bunch for optics...')
    bunch_optics = machine.generate_6D_Gaussian_bunch_matched(
        macroparticlenumber_optics,
        intensity, epsn_x, epsn_y,
        sigma_z=sigma_z)

    print('Done.')

    # Kick bunch
    bunch_optics.x += 10.
    bunch_optics.y += 20.
    bunch_optics.z += .020

    # Temporarily remove longitudinal map
    ix = machine.one_turn_map.index(machine.longitudinal_map)
    machine.one_turn_map.remove(machine.longitudinal_map)

    # Lists for saving
    beam_alpha_x = []
    beam_beta_x = []
    beam_alpha_y = []
    beam_beta_y = []

    # Track through optics elements
    print('Track through optics elements')
    for i_ele, m in enumerate(machine.one_turn_map):
        beam_alpha_x.append(bunch_optics.alpha_Twiss_x())
        beam_beta_x.append(bunch_optics.beta_Twiss_x())
        beam_alpha_y.append(bunch_optics.alpha_Twiss_y())
        beam_beta_y.append(bunch_optics.beta_Twiss_y())
        m.track(bunch_optics)

    # Check optics
    assert np.allclose(optics['alpha_x'], machine.transverse_map.alpha_x,
                       rtol=0., atol=0.)
    assert np.allclose(optics['alpha_y'], machine.transverse_map.alpha_y,
                       rtol=0., atol=0.)
    assert np.allclose(optics['beta_x'], machine.transverse_map.beta_x,
                       rtol=0., atol=0.)
    assert np.allclose(optics['beta_y'], machine.transverse_map.beta_y,
                       rtol=0., atol=0.)

    # assert np.allclose(beam_alpha_x, optics['alpha_x'][:-1],
    #                    rtol=5e-3, atol=5e-3)
    # assert np.allclose(beam_alpha_y, optics['alpha_y'][:-1],
    #                    rtol=5e-3, atol=5e-2)
    assert np.allclose(beam_beta_x, optics['beta_x'][:-1], rtol=2e-2)
    assert np.allclose(beam_beta_y, optics['beta_y'][:-1], rtol=2e-2)

    machine.one_turn_map.insert(ix, machine.longitudinal_map)

    # Create bunch for tracking
    print('Create bunch for tracking...')
    bunch = machine.generate_6D_Gaussian_bunch_matched(
        macroparticlenumber_track, intensity, epsn_x, epsn_y, sigma_z=sigma_z)
    print('Done.')

    # Lists for saving
    beam_x = []
    beam_y = []
    beam_z = []

    # Tracking loop
    print(f'Track for {n_turns} turns')
    for i_turn in range(n_turns):

        machine.track(bunch)

        beam_x.append(bunch.mean_x())
        beam_y.append(bunch.mean_y())
        beam_z.append(bunch.mean_z())

    # Find tunes
    freq_x = np.fft.rfftfreq(len(beam_x), d=1.)
    ampl_x = np.abs(np.fft.rfft(beam_x))
    ind_peaks_x = find_peaks_cwt(ampl_x, 5)
    ind_max_peak_x = np.argmax(ampl_x[ind_peaks_x])
    f_peak_x = freq_x[ind_peaks_x[ind_max_peak_x]]
    print(f'Q_x found at {f_peak_x:.2f}')

    freq_y = np.fft.rfftfreq(len(beam_y), d=1.)
    ampl_y = np.abs(np.fft.rfft(beam_y))
    ind_peaks_y = find_peaks_cwt(ampl_y, 5)
    ind_max_peak_y = np.argmax(ampl_y[ind_peaks_y])
    f_peak_y = freq_y[ind_peaks_y[ind_max_peak_y]]
    print(f'Q_y found at {f_peak_y:.2f}')

    freq_z = np.fft.rfftfreq(len(beam_z), d=1.)
    ampl_z = np.abs(np.fft.rfft(beam_z))
    ind_peaks_z = find_peaks_cwt(ampl_z, 5)
    ind_max_peak_z = np.argmax(ampl_z[ind_peaks_z])
    f_peak_z = freq_z[ind_peaks_z[ind_max_peak_z]]
    print(f'Q_s found at {f_peak_z:.4f}')

    assert np.isclose(f_peak_x, np.modf(machine.Q_x)[0], rtol=0, atol=1e-2)
    assert np.isclose(f_peak_y, np.modf(machine.Q_y)[0], rtol=0, atol=1e-2)
    assert np.isclose(f_peak_z, np.modf(machine.Q_s)[0], rtol=0, atol=1.7e-4)

