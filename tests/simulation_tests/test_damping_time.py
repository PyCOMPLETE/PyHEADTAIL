import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.machines.synchrotron import Synchrotron


def test_damping_time():

    n_attempts = 5

    n_turns = 800
    macroparticlenumber = int(1e5)

    # Beam and machine parameters
    intensity = 2e11
    epsn_x = 2e-6   # normalised horizontal emittance
    epsn_y = 2e-6   # normalised vertical emittance

    p0_eVperc = 6.5e12
    p0 = p0_eVperc * qe / c_light

    beta_x = 92.7
    beta_y = 93.2

    Q_x = 64.31
    Q_y = 59.32

    alpha_momentum = 3.225e-4
    h_RF = 35640
    V_RF = 12.0e6
    circumference = 26658.883199999

    # Create machine
    machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                          n_segments=1, beta_x=beta_x, beta_y=beta_y,
                          D_x=0.0, D_y=0.0, accQ_x=Q_x, accQ_y=Q_y,
                          alpha_mom_compaction=alpha_momentum,
                          longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                          dphi_RF=0, p_increment=0.0,
                          p0=p0, charge=qe, mass=m_p)

    sigma_z = 1e-9 * machine.beta * c_light / 4.   # RMS bunch length in meters

    # Damper
    damping_time = 200  # [turns]
    damper = TransverseDamper(dampingrate_x=damping_time,
                              dampingrate_y=damping_time)

    machine.one_turn_map.append(damper)

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

        # Kick beam
        kick_in_sigmas = 0.75
        bunch.x += kick_in_sigmas * bunch.sigma_x()
        bunch.x += kick_in_sigmas * bunch.sigma_y()

        # Create arrays for saving
        x = np.zeros(n_turns, dtype=float)
        y = np.zeros(n_turns, dtype=float)

        # Tracking loop
        for i in range(n_turns):

            for m in machine.one_turn_map:
                m.track(bunch)

            x[i], y[i] = bunch.mean_x(), bunch.mean_y()

        # Check results
        turns = range(n_turns)

        iMin = 5
        iMax = n_turns - 5

        ampl_x = np.abs(hilbert(x))
        b_x, a, r, p, stderr = linregress(turns[iMin:iMax],
                                          np.log(ampl_x[iMin:iMax]))
        print(f"Damping time x {-1/b_x:.0F} [turns]")

        ampl_y = np.abs(hilbert(y))
        b_y, a, r, p, stderr = linregress(turns[iMin:iMax],
                                          np.log(ampl_y[iMin:iMax]))
        print(f"Damping time y {-1/b_y:.0F} [turns]")

        # assert np.isclose(-1/b_x, damping_time, rtol=2e-2), \
        #     "Horizontal damping time doesn't match"
        # assert np.isclose(-1/b_y, damping_time, rtol=2e-2), \
        #     "Vertical damping time doesn't match"

        check_x = np.isclose(-1/b_x, damping_time, rtol=2e-2)
        check_y = np.isclose(-1/b_y, damping_time, rtol=2e-2)

        assert check_x or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts horizontal damping time doesn't match"
        assert check_y or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts vertical damping time doesn't match"
        if check_x and check_y:
            print(f"Passed on {i_attempt + 1}. attempt.")
            break
        i_attempt += 1


def test_damping_time_xy():

    n_attempts = 5
    n_turns = 800
    macroparticlenumber = int(1e5)

    # Beam and machine parameters
    intensity = 2e11
    epsn_x = 2e-6   # normalised horizontal emittance
    epsn_y = 2e-6   # normalised vertical emittance

    p0_eVperc = 6.5e12
    p0 = p0_eVperc * qe / c_light

    beta_x = 92.7
    beta_y = 93.2

    Q_x = 64.31
    Q_y = 59.32

    alpha_momentum = 3.225e-4
    h_RF = 35640
    V_RF = 12.0e6
    circumference = 26658.883199999

    # Create machine
    machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                          n_segments=1, beta_x=beta_x, beta_y=beta_y,
                          D_x=0.0, D_y=0.0, accQ_x=Q_x, accQ_y=Q_y,
                          alpha_mom_compaction=alpha_momentum,
                          longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                          dphi_RF=0, p_increment=0.0,
                          p0=p0, charge=qe, mass=m_p)

    sigma_z = 1e-9 * machine.beta * c_light / 4.   # RMS bunch length in meters

    # Damper
    damping_time = 200  # [turns]
    damper_x = TransverseDamper(dampingrate_x=damping_time,
                                dampingrate_y=0, phase=87,
                                local_beta_function=93.)
    damper_y = TransverseDamper(dampingrate_x=0,
                                dampingrate_y=damping_time, phase=93,
                                local_beta_function=93.)

    machine.one_turn_map.append(damper_x)
    machine.one_turn_map.append(damper_y)

    # Loop over attempts
    i_attempt = 0
    while i_attempt < n_attempts:

        print(f"Attempt {i_attempt+1}")

        # Create beam
        bunch = machine.generate_6D_Gaussian_bunch_matched(
            n_macroparticles=macroparticlenumber,
            intensity=intensity,
            epsn_x=epsn_x,
            epsn_y=epsn_y,
            sigma_z=sigma_z,
        )

        # Kick beam
        kick_in_sigmas = 0.75
        bunch.x += kick_in_sigmas * bunch.sigma_x()
        bunch.x += kick_in_sigmas * bunch.sigma_y()

        # Create arrays for saving
        x = np.zeros(n_turns, dtype=float)
        y = np.zeros(n_turns, dtype=float)

        # Tracking loop
        for i in range(n_turns):

            for m in machine.one_turn_map:
                m.track(bunch)

            x[i], y[i] = bunch.mean_x(), bunch.mean_y()

        # Check results
        turns = range(n_turns)

        iMin = 5
        iMax = damping_time - 5

        ampl_x = np.abs(hilbert(x))
        b_x, a, r, p, stderr = linregress(turns[iMin:iMax],
                                          np.log(ampl_x[iMin:iMax]))
        print(f"Damping time x {-1/b_x:.0F} [turns]")

        ampl_y = np.abs(hilbert(y))
        b_y, a, r, p, stderr = linregress(turns[iMin:iMax],
                                          np.log(ampl_y[iMin:iMax]))
        print(f"Damping time y {-1/b_y:.0F} [turns]")

        # assert np.isclose(-1/b_x, damping_time, rtol=2e-2), \
        #     "Horizontal damping time doesn't match"
        # assert np.isclose(-1/b_y, damping_time, rtol=2e-2), \
        #     "Vertical damping time doesn't match"

        check_x = np.isclose(-1/b_x, damping_time, rtol=2e-2)
        check_y = np.isclose(-1/b_y, damping_time, rtol=2e-2)

        assert check_x or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts horizontal damping time doesn't match"
        assert check_y or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts vertical damping time doesn't match"
        if check_x and check_y:
            print(f"Passed on {i_attempt + 1}. attempt.")
            break
        i_attempt += 1

