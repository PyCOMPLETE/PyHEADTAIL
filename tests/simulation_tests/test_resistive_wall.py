import numpy as np
from scipy.constants import c as c_light, e as qe, m_p
from scipy.signal import hilbert
from scipy.stats import linregress

from PyHEADTAIL.impedances import wakes
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.particles.slicing import UniformChargeSlicer


def test_reswall():

    n_attempts = 5

    expected_growth_rate = 70e-4
    rel_tolerance = 5e-2

    n_turns = 1000
    macroparticlenumber = int(1e5)

    # Beam and machine parameters
    intensity = 1.6e+12
    epsn_x = 2e-6   # Normalised horizontal emittance [m]
    epsn_y = 2e-6   # Normalised vertical emittance [m]
    sigma_z = 12.4  # [m]

    E_kin = 1.4e9  # [eV]
    E_rest = m_p * c_light**2 / qe  # [eV]
    gamma = 1 + E_kin/E_rest
    betagamma = np.sqrt(gamma**2 - 1)
    p0 = betagamma * m_p * c_light

    circumference = 2*np.pi*100

    Q_x = 6.22
    Q_y = 6.25

    xi_x = -0.1
    xi_y = -0.1

    Qp_x = Q_x * xi_x
    Qp_y = Q_y * xi_y

    beta_x = 16.  # circumference/(2*np.pi*Q_x)
    beta_y = 16.  # circumference/(2*np.pi*Q_y)

    alpha_mom = 0.027

    # Non-linear map
    h_RF = 7
    V_RF = 20e3
    dphi_RF = np.pi
    p_increment = 0.0

    # Create machine
    machine = Synchrotron(optics_mode='smooth', circumference=circumference,
                          n_segments=1, beta_x=beta_x, beta_y=beta_y,
                          D_x=0.0, D_y=0.0,
                          accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y,
                          alpha_mom_compaction=alpha_mom,
                          longitudinal_mode='non-linear', h_RF=h_RF, V_RF=V_RF,
                          dphi_RF=dphi_RF, p_increment=p_increment,
                          p0=p0, charge=qe, mass=m_p)

    # Create wakes
    slices_for_wake = 500
    slicer_for_wake = UniformChargeSlicer(slices_for_wake, n_sigma_z=3)

    # Resistive wall wake
    conductivity = 1e6
    pipe_sigma_y = 0.0356   # sigma_y * 8.
    wake = wakes.ParallelHorizontalPlatesResistiveWall(
        conductivity=conductivity, pipe_radius=pipe_sigma_y,
        resistive_wall_length=10000, dt_min=1e-12)
    wake_field = wakes.WakeField(slicer_for_wake, wake)

    machine.one_turn_map.append(wake_field)

    # Loop over attempts
    i_attempt = 0
    while i_attempt < n_attempts:

        print(f"Attempt {i_attempt+1}:")

        # Create particles
        bunch = machine.generate_6D_Gaussian_bunch_matched(
            n_macroparticles=macroparticlenumber,
            intensity=intensity,
            epsn_x=epsn_x,
            epsn_y=epsn_y,
            sigma_z=sigma_z,
        )

        # Array for saving
        y = np.zeros(n_turns, dtype=float)

        # Tracking loop
        for i in range(n_turns):

            for m in machine.one_turn_map:
                m.track(bunch)

            y[i] = bunch.mean_y()

        # Check results
        turns = np.arange(n_turns)

        iMin = 100
        iMax = n_turns
        ampl = np.abs(hilbert(y))
        b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
        print(f"Growth rate {b*1e4:.2f} [10^-4/turn]")

        check = np.isclose(b, expected_growth_rate, rtol=rel_tolerance) 
        
        assert check or i_attempt < n_attempts-1, \
            f"After {n_attempts} attempts vertical growth rate \
            doesn't match expectation."
        if check:
            print(f"Passed on {i_attempt + 1}. attempt.")
            break
        i_attempt += 1

