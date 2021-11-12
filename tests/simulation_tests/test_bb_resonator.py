import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.impedances import wakes
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformBinSlicer


def test_bbresonator():

    n_attempts = 5

    n_turns = 3000
    macroparticlenumber = int(1e5)

    expected_growth_rate_x = 59e-4
    expected_growth_rate_y = 25e-4
    rel_tolerance = 5e-2

    # Beam and machine parameters
    intensity = 2.5e11
    epsn_x = 2e-6   # normalised horizontal emittance
    epsn_y = 2e-6   # normalised vertical emittance
    sigma_z = 0.23   # RMS bunch length in meters

    circumference = 6911.5038378975451

    p0_eVperc = 26e9
    p0 = p0_eVperc * qe / c_light

    beta_x = 54.644808743169399
    beta_y = 54.509415262636274

    Q_x = 20.13
    Q_y = 20.18

    alpha_mom = 0.0030864197530864196

    h_RF = [4620, 4*4620]
    V_RF = [4.5e6, 0.45e6]
    dphi_RF = [0., np.pi]
    p_increment = 0.

    # Create machine
    machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                          n_segments=1, beta_x=beta_x, beta_y=beta_y,
                          D_x=0.0, D_y=0.0,
                          accQ_x=Q_x, accQ_y=Q_y, Qp_x=0.0, Qp_y=0.0,
                          alpha_mom_compaction=alpha_mom,
                          longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                          dphi_RF=dphi_RF, p_increment=p_increment,
                          p0=p0, charge=qe, mass=m_p)

    # Create BB resonator wake
    # Resonator parameters
    R_shunt = 10.2e6   # Shunt impedance [Ohm/m]
    frequency = 1.3e9   # Resonance frequency [Hz]
    Q = 1   # Quality factor

    slices = 200
    slicer_for_wakes = UniformBinSlicer(slices, n_sigma_z=6)

    # Wake
    wake = wakes.CircularResonator(R_shunt, frequency, Q)
    wake_field = wakes.WakeField(slicer_for_wakes, wake)

    machine.one_turn_map.append(wake_field)

    # Loop over attempts
    i_attempt = 0
    while i_attempt < n_attempts:

        print(f"Attempt {i_attempt+1}:")

        # Create beam
        bunch = machine.generate_6D_Gaussian_bunch_matched(
            n_macroparticles=macroparticlenumber,
            intensity=intensity,
            epsn_x=epsn_x,
            epsn_y=epsn_y,
            sigma_z=sigma_z,
        )

        # Create arrays for saving
        x = np.zeros(n_turns, dtype=float)
        y = np.zeros(n_turns, dtype=float)

        # Tracking loop
        for i in range(n_turns):

            for m in machine.one_turn_map:
                m.track(bunch)

            x[i], y[i] = bunch.mean_x(), bunch.mean_y()

        # Check results
        turns = np.arange(n_turns)

        iMin = 1500
        iMax = n_turns

        ampl_x = np.abs(hilbert(x))
        b, a, r, p, stde = linregress(turns[iMin:iMax], np.log(ampl_x[iMin:iMax]))
        print(f"Growth rate x {b*1e4:.2f} [10^-4/turn]")

        # assert np.isclose(b, expected_growth_rate_x, rtol=rel_tolerance), \
        #     "Horizontal growth rate does not match expectation."
        check_x = np.isclose(b, expected_growth_rate_x, rtol=rel_tolerance)

        ampl_y = np.abs(hilbert(y))
        b, a, r, p, stde = linregress(turns[iMin:iMax], np.log(ampl_y[iMin:iMax]))
        print(f"Growth rate y {b*1e4:.2f} [10^-4/turn]")

        # assert np.isclose(b, expected_growth_rate_y, rtol=rel_tolerance), \
        #     "Vertical growth rate does not match expectation."
        check_y = np.isclose(b, expected_growth_rate_y, rtol=rel_tolerance)

        assert check_x or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts horizontal growth rate " \
            "doesn't match expectation."
        assert check_y or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts vertical growth rate " \
            "doesn't match expectation."
        if check_x and check_y:
            print(f"Passed on {i_attempt + 1}. attempt")
            break
        i_attempt += 1

