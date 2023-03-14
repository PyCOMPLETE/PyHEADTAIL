import numpy as np

class Wakefield:

    def __init__(self,
                source_moments,
                kick,
                scale_kick,
                function,
                z_slice_range=None, # These are [a, b] in the paper
                num_slices=None, # Per bunch, this is N_1 in the paper
                slicer=None, # alternatively, a slicer can be used
                z_period=None, # This is P in the paper
                i_period_range=None, # This is [A, B] in the paper
                num_periods=None,
                num_turns=1,
                circumference=None):

        if i_period_range is not None:
            raise NotImplementedError('i_period_range is not implemented yet')

        self.dz = (z_slice_range[1] - z_slice_range[0]) / num_slices # h in the paper
        self._a = z_slice_range[0]

        self._N_1 = num_slices # N_1 in the
        self._P = z_period # P in the paper
        self._N_S = num_periods # N_S in the paper
        self.num_turns = num_turns

        self._N_aux = self._N_1 + self._N_2 # N_aux in the paper

        # Compute M_aux
        self._M_aux = (self._N_S + self._N_T - 1) * self._N_aux # M_aux in the paper

        self.moments_names = source_moments
        self.moments_data = np.zeros(
            (len(source_moments), self.num_turns, self._M_aux), dtype=np.float64)

        self._G_hat = 'TODO'
        self.scale_kick = scale_kick
        self.function = function

    @property
    def num_slices(self):
        return self._N_1

    @property
    def num_periods(self):
        return self._N_S

    @property
    def z_period(self):
        return self._P

    @property
    def _N_2(self):
        return self._N_1 # For the wakefiled, N_1 = N_2

    @property
    def _N_T(self):
        return self._N_S # For the wakefiled, N_S = N_T

    def set_moments(self, i_source, i_turn, moments):

        """
        Set the moments for a given source and turn.

        Parameters
        ----------
        i_source : int
            The source index, 0 <= i_source < self.num_periods
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns
        moments : dict
            A dictionary of the form {moment_name: moment_value}

        """

        assert np.isscalar(i_source)
        assert np.isscalar(i_turn)

        assert i_source < self._N_S
        assert i_source >= 0

        assert i_turn < self.num_turns
        assert i_turn >= 0

        for nn, vv in moments.items():
            assert nn in self.moments_names, (
                f'Moment {nn} not in defined moments_names')
            assert len(vv) == self._N_1, (
                f'Length of moment {nn} is not equal to num_slices')
            i_moment = self.moments_names.index(nn)
            i_start_in_moments_data = self._M_aux - (i_source + 1) * self._N_aux
            i_end_in_moments_data = i_start_in_moments_data + self._N_1

            self.moments_data[i_moment, i_turn,
                              i_start_in_moments_data:i_end_in_moments_data,
                             ] = vv

    def get_moment_profile(self, moment_name, i_turn):

        '''
        Get the moment profile for a given turn.

        Parameters
        ----------
        moment_name : str
            The name of the moment to get
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns

        Returns
        -------
        z_out : np.ndarray
            The z positions within the moment profile
        moment_out : np.ndarray
            The moment profile
        '''

        z_out = np.zeros(self._N_S * self._N_1)
        moment_out = np.zeros(self._N_S * self._N_1)
        i_moment = self.moments_names.index(moment_name)
        for i_source in range(self._N_S):
            i_start_out = (self._N_S - (i_source + 1)) * self._N_1
            i_end_out = i_start_out + self._N_1
            z_out[i_start_out:i_end_out] = (
                self._a + self.dz / 2
                - i_source * self._P + self.dz * np.arange(self._N_1))

            i_start_in_moments_data = self._M_aux - (i_source + 1) * self._N_aux
            i_end_in_moments_data = i_start_in_moments_data + self._N_1
            moment_out[i_start_out:i_end_out] = (
                self.moments_data[i_moment, i_turn,
                                  i_start_in_moments_data:i_end_in_moments_data,
                                  ])

        return z_out, moment_out

    def track(self, particles):

        self._next_turn() # Trash the oldest turn and make space for new one

        i_bin_particles = self._update_moments(particles) # associate to each particle

        self._mpi_sync()

        conv_res = self._convolve_with_G_hat() # Conv res has a similar structure to moments_data (M_aux, n_turns)

        conv_res_tot = np.sum(conv_res, axis=1) # Sum over n_turns

        conv_res_particles = np.take(conv_res_tot, i_bin_particles, axis=0) # Reorder the convolved result to match the particles

        kick_per_particle = conv_res_particles

        for nn in self.scale_kick:
            if isinstance(nn, str):
                kick_per_particle *= getattr(particles, nn)
            else:
                kick_per_particle *= nn

        getattr(particles, self.kick)[:] += kick_per_particle

class ResonatorFunction:
    def __init__(self, Rs, Qs, f0):
        self.Rs = Rs
        self.Qs = Qs
        self.f0 = f0

    def __call__(self, t):
        res = 0
        return res

circumference = 26658.8832
rf_bucket_length = 1

# Quadrupolar wakefield
wf = Wakefield(
    source_moments=['num_particles', 'x'],
    kick='x',
    scale_kick=[3.4, 'x'], # The kick is scaled by position of the particle for quadrupolar, would be None for dipolar
    function=ResonatorFunction(Rs=3, Qs=4, f0=5),
    z_slice_range=[-rf_bucket_length/2, rf_bucket_length/2], # These are [a, b] in the paper
    slicer=None, # alternatively, a slicer can be used
    num_slices=5, # Per bunch, this is N_1 in the paper
    z_period=10, # This is P in the paper
    num_periods=4, # This is N_S
    num_turns=3,
    circumference=circumference,
)

charge_test_0 = np.linspace(-1, 1, 5)
charge_test_1 = np.linspace(-2, 2, 5)
charge_test_2 = np.linspace(-3, 3, 5)

wf.set_moments(
    i_source=0,
    i_turn=1,
    moments={'num_particles': charge_test_0,
             'x': 2 * charge_test_0}
)

wf.set_moments(
    i_source=1,
    i_turn=1,
    moments={'num_particles': charge_test_1,
                'x': 2 * charge_test_1,}
)

wf.set_moments(
    i_source=2,
    i_turn=1,
    moments={'num_particles': charge_test_2,
                'x': 2 * charge_test_2}
)


z_profile, num_particles_profile = wf.get_moment_profile('num_particles', 1)

assert np.allclose(z_profile,
    [-30.4, -30.2, -30. , -29.8, -29.6,
     -20.4, -20.2, -20. , -19.8, -19.6,
     -10.4, -10.2, -10. ,  -9.8,  -9.6,
       -0.4, -0.2,   0. ,   0.2,   0.4],
    rtol=0, atol=1e-12)

assert np.allclose(num_particles_profile,
    [0. ,  0. ,  0. ,  0. ,  0. ,
    -3. , -1.5,  0. ,  1.5,  3. ,
    -2. , -1. ,  0. ,  1. ,  2. ,
    -1. , -0.5,  0. ,  0.5,  1. ],
    rtol=0, atol=1e-12)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(z_profile, num_particles_profile, '.')

plt.show()



