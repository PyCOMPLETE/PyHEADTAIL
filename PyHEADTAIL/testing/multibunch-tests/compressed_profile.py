import numpy as np

class CompressedProfile:

    def __init__(self,
                moments,
                z_slice_range=None, # These are [a, b] in the paper
                num_slices=None, # Per bunch, this is N_1 in the paper
                slicer=None, # alternatively, a slicer can be used
                z_period=None, # This is P in the paper
                i_period_range=None, # This is [A, B] in the paper
                num_periods=None,
                num_turns=1,
                num_targets=None,
                num_slices_target=None,
                circumference=None,
                ):

        if i_period_range is not None:
            raise NotImplementedError('i_period_range is not implemented yet')

        if num_turns > 1:
            assert circumference is not None, (
                'circumference must be specified if num_turns > 1')

        self.circumference = circumference

        assert slicer is None, 'slicer is not implemented yet'

        self.dz = (z_slice_range[1] - z_slice_range[0]) / num_slices # h in the paper
        self._z_a = z_slice_range[0]
        self._z_b = z_slice_range[1]

        self._N_1 = num_slices # N_1 in the
        self._z_P = z_period # P in the paper
        self._N_S = num_periods # N_S in the paper

        if num_slices_target is not None:
            self._N_2 = num_slices_target
        else:
            self._N_2 = self._N_1

        if num_targets is not None:
            self._N_T = num_targets
        else:
            self._N_T = self._N_S

        self.num_turns = num_turns

        self._N_aux = self._N_1 + self._N_2 # N_aux in the paper

        # Compute M_aux
        self._M_aux = (self._N_S + self._N_T - 1) * self._N_aux # M_aux in the paper

        self.moments_names = moments
        self.data = np.zeros(
            (len(moments), self.num_turns, self._M_aux), dtype=np.float64)

    def __getitem__(self, key):
        assert isinstance(key, str), ('other modes not supported yet')
        assert key in self.moments_names, (
            f'Moment {key} not in defined moments_names')
        i_moment = self.moments_names.index(key)
        return self.data[i_moment]

    def __setitem__(self, key, value):
        self[key][:] = value

    @property
    def num_slices(self):
        return self._N_1

    @property
    def num_periods(self):
        return self._N_S

    @property
    def z_period(self):
        return self._z_P

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

            self.data[i_moment, i_turn,
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
                self.data[i_moment, i_turn,
                                  i_start_in_moments_data:i_end_in_moments_data,
                                  ])

        return z_out, moment_out

