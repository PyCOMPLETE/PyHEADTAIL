import numpy as np

class DeltaFunction:

    def __init__(self, tol_z):
        self.delta_z = tol_z

    def __call__(self, z):
        return np.exp(z) * (z < 0)
        #return np.float64(np.abs(z) < self.delta_z)




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

        self.function = function

        self.dz = (z_slice_range[1] - z_slice_range[0]) / num_slices # h in the paper
        self._z_a = z_slice_range[0]
        self._z_b = z_slice_range[1]
        self.circumference = circumference

        self._N_1 = num_slices # N_1 in the
        self._z_P = z_period # P in the paper
        self._N_S = num_periods # N_S in the paper
        self.num_turns = num_turns

        self._BB = 1 # B in the paper
                     # (for now we assume that B=0 is the first bunch in time
                     # and the last one in zeta)
        self._AA = self._BB - self._N_S

        self._N_aux = self._N_1 + self._N_2 # N_aux in the paper

        # Compute M_aux
        self._M_aux = (self._N_S + self._N_T - 1) * self._N_aux # M_aux in the paper

        self.moments_names = source_moments
        self.moments_data = np.zeros(
            (len(source_moments), self.num_turns, self._M_aux), dtype=np.float64)

        # Build wake matrix
        z_c = self._z_a # For wakefield, z_c = z_a
        z_d = self._z_b # For wakefield, z_d = z_b
        self.z_wake = np.zeros((self.num_turns, self._M_aux))
        for tt in range(self.num_turns):
            z_a_turn = self._z_a + tt * self.circumference
            z_b_turn = self._z_b + tt * self.circumference
            temp_z = np.arange(z_c - z_b_turn, z_d - z_a_turn, self.dz)
            for ii, ll in enumerate(range(
                                self._CC - self._BB + 1, self._DD - self._AA)):
                self.z_wake[tt, ii*self._N_aux:(ii+1)*self._N_aux] = (
                                                    temp_z + ll * self._z_P)

        self.G_aux = self.function(self.z_wake)

        self._G_hat = 'TODO'

    @property
    def num_slices(self):
        return self._N_1

    @property
    def num_periods(self):
        return self._N_S

    @property
    def z_period(self):
        return self._z_P

    @property
    def _N_2(self):
        return self._N_1 # For the wakefield, N_1 = N_2

    @property
    def _N_T(self):
        return self._N_S # For the wakefield, N_S = N_T

    @property
    def _CC(self):
        return self._AA # For the wakefield, A = C

    @property
    def _DD(self):
        return self._BB # For the wakefield, B = D


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
                self._z_a + self.dz / 2
                - i_source * self._z_P + self.dz * np.arange(self._N_1))

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



rf_bucket_length = 1

# Quadrupolar wakefield
wf = Wakefield(
    source_moments=['num_particles', 'x'],
    kick='x',
    scale_kick=[3.4, 'x'], # The kick is scaled by position of the particle for quadrupolar, would be None for dipolar
    function=DeltaFunction(tol_z=1e-12),
    z_slice_range=[-rf_bucket_length/2, rf_bucket_length/2], # These are [a, b] in the paper
    slicer=None, # alternatively, a slicer can be used
    num_slices=5, # Per bunch, this is N_1 in the paper
    z_period=10, # This is P in the paper
    num_periods=4, # This is N_S
    num_turns=3,
    circumference=100.,
)

assert wf.moments_data.shape == (2, 3, 70)
assert wf._M_aux == 70
assert wf._N_aux == 10
assert wf._N_1 == 5
assert wf._N_S == 4
assert wf._z_P == 10

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



