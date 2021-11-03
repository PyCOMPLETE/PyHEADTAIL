import pathlib

import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.stats import linregress
from scipy.signal import hilbert

from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformBinSlicer


def test_waketable():

    n_attempts = 5

    expected_growth_rate = 12.5e-4
    rel_tolerance = 5e-2

    n_turns = 5000
    n_macroparticles = int(1e4)

    # Beam and machine parameters
    bunch_intensity = 2e11
    epsn = 1.8e-6
    sigma_z = 1.2e-9 / 4.0 * c_light

    energy = 7e12  # [eV]
    rest_energy = m_p * c_light**2 / qe  # [eV]
    gamma = energy / rest_energy
    betar = np.sqrt(1 - 1 / gamma ** 2)
    p0 = m_p * betar * gamma * c_light

    beta_x = 68.9
    beta_y = 70.34

    Q_x = 64.31
    Q_y = 59.32

    alpha_mom = 3.483575072011584e-04

    eta = alpha_mom - 1.0 / gamma**2
    voltage = 12e6
    h = 35640
    Q_s = np.sqrt(qe * voltage * eta * h / (2 * np.pi * betar * c_light * p0))

    circumference = 26658.883199999

    # Detuners
    Qp_x = -20.0
    Qp_y = 0.0
    i_oct = 15.
    detx_x = 1.4e5 * i_oct / 550.0
    detx_y = -1.0e5 * i_oct / 550.0

    # Create machine
    machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                          n_segments=1,
                          alpha_x=0.0, beta_x=beta_x, D_x=0.0,
                          alpha_y=0.0, beta_y=beta_y, D_y=0.0,
                          accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y,
                          app_x=detx_x * p0, app_y=detx_x * p0,
                          app_xy=detx_y * p0,
                          alpha_mom_compaction=alpha_mom,
                          longitudinal_mode='linear', Q_s=Q_s,
                          dphi_RF=0.0, p_increment=0.0, p0=p0,
                          charge=qe, mass=m_p, RF_at='end_of_transverse')

    # Wake field
    n_slices_wakes = 200
    limit_z = 3 * sigma_z
    slicer_for_wakefields = UniformBinSlicer(n_slices_wakes,
                                             z_cuts=(-limit_z, limit_z))
    wake_folder = pathlib.Path(__file__).parent.joinpath(
        '../../examples/impedances').absolute()
    wakefile = wake_folder.joinpath(
        'wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV'
        '_B1_2021_TeleIndex1_wake.dat')
    waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y',
                                     'quadrupole_x', 'quadrupole_y'])
    wake_field = WakeField(slicer_for_wakefields, waketable)

    # Damper
    damping_time = 7000  # [turns]
    damper = TransverseDamper(dampingrate_x=damping_time,
                              dampingrate_y=damping_time)

    machine.one_turn_map.append(wake_field)
    machine.one_turn_map.append(damper)

    # Loop over attempts
    i_attempt = 0
    while i_attempt < n_attempts:

        print(f"Attempt {i_attempt+1}:")

        particles = machine.generate_6D_Gaussian_bunch_matched(
            n_macroparticles,
            intensity=bunch_intensity,
            epsn_x=epsn,
            epsn_y=epsn,
            sigma_z=sigma_z,
        )

        x = np.zeros(n_turns, dtype=float)

        # Tracking loop
        for turn in range(n_turns):

            machine.track(particles)
            x[turn] = particles.mean_x()

        # Check results
        turns = np.arange(n_turns)
        iMin = 1000
        iMax = n_turns - 1000

        ampl = np.abs(hilbert(x))
        b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
        print(f"Growth rate {b*1e4:.2f} [10^-4/turn]")

        # assert np.isclose(b, expected_growth_rate, rtol=rel_tolerance), \
        #     "Horizontal growth rate does not match expectation."

        check = np.isclose(b, expected_growth_rate, rtol=rel_tolerance)

        assert check or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts horizontal growth rate does not \
            match expectation."
        if check:
            print(f"Passed on {i_attempt + 1}. attempt.")
            break
        i_attempt += 1

