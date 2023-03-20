import numpy as np
from compressed_profile import CompressedProfile

from scipy.constants import c as clight
from scipy.signal import fftconvolve

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
                circumference=None,
                _flatten=False):

        self._flatten = _flatten

        if i_period_range is not None:
            raise NotImplementedError('i_period_range is not implemented yet')

        self.function = function

        self.moments_data = CompressedProfile(
                moments=source_moments + ['result'],
                z_slice_range=z_slice_range,
                num_slices=num_slices,
                slicer=slicer,
                z_period=z_period,
                i_period_range=i_period_range,
                num_periods=num_periods,
                num_turns=num_turns,
                circumference=circumference)

        if not _flatten:

            self._N_aux = self.moments_data._N_aux
            self._M_aux = self.moments_data._M_aux
            self._N_S = self.moments_data._N_S
            self._N_T = self._N_S
            self._BB = 1 # B in the paper
                        # (for now we assume that B=0 is the first bunch in time
                        # and the last one in zeta)
            self._AA = self._BB - self._N_S
            self._CC = self._AA
            self._DD = self._BB

            # Build wake matrix
            self.z_wake = _build_z_wake(self._z_a, self._z_b, self.num_turns,
                        self._N_aux, self._M_aux,
                        self.circumference, self.dz, self._AA, self._BB, self._CC,
                        self._DD, self._z_P)

            self.G_aux = self.function(self.z_wake)

            phase_term = np.exp(1j * 2 * np.pi * np.arange(self._M_aux)
                            * ((self._N_S - 1)* self._N_aux + self._N_1)
                            / self._M_aux)

            self._G_hat_dephased = phase_term * np.fft.fft(self.G_aux, axis=1)

        else:
            self._N_S_flatten = self.moments_data._N_S * self.num_turns
            self._N_T_flatten = self.moments_data._N_S
            self._N_aux = self.moments_data._N_aux
            self._M_aux_flatten = ((self._N_S_flatten + self._N_T_flatten - 1)
                                   * self._N_aux)
            self._BB_flatten = 1 # B in the paper
            # (for now we assume that B=0 is the first bunch in time
            # and the last one in zeta)
            self._AA_flatten = self._BB_flatten - self._N_S_flatten
            self._CC_flatten = self._AA_flatten
            self._DD_flatten = self._AA_flatten + self.moments_data._N_S

            # Build wake matrix
            self.z_wake = _build_z_wake(self._z_a, self._z_b,
                        1, # num_turns
                        self._N_aux, self._M_aux_flatten,
                        0, # circumference
                        self.dz,
                        self._AA_flatten, self._BB_flatten,
                        self._CC_flatten, self._DD_flatten, self._z_P)
            self.G_aux = self.function(self.z_wake)

            phase_term = np.exp(1j * 2 * np.pi
                            * np.arange(self._M_aux_flatten//2 + 1)
                            * ((self._N_S_flatten - 1) * self._N_aux + self._N_1)
                            / self._M_aux_flatten)

            self._G_hat_dephased = phase_term * np.fft.rfft(self.G_aux, axis=1)
            self._G_aux_shifted = np.fft.irfft(self._G_hat_dephased, axis=1)

    def _compute_convolution(self, moment_names):

        if isinstance(moment_names, str):
            moment_names = [moment_names]

        rho_aux = np.ones(shape=self.moments_data['result'].shape,
                        dtype=np.float64)

        for nn in moment_names:
            rho_aux *= self.moments_data[nn]

        if not self._flatten:
            print('not flatten')
            rho_hat = np.fft.fft(rho_aux, axis=1)
            res = np.fft.ifft(rho_hat * self._G_hat_dephased, axis=1)
        else:
            rho_flatten = np.zeros((1, self._M_aux_flatten), dtype=np.float64)
            _N_aux_turn = self.moments_data._N_S * self._N_aux
            for tt in range(self.num_turns):
                #tt_flatten = self.num_turns - tt - 1
                rho_flatten[
                    0, tt * _N_aux_turn: (tt + 1) * _N_aux_turn] = \
                        rho_aux[tt, :_N_aux_turn]

            # rho_hat_flatten = np.fft.rfft(rho_flatten, axis=1)
            # res_flatten = np.fft.irfft(
            #     rho_hat_flatten * self._G_hat_dephased, axis=1).real
            # self._res_flatten_fft = res_flatten # for debugging

            res = rho_aux * 0

            res_flatten = fftconvolve(rho_flatten, self._G_aux_shifted, mode='full')
            self._res_flatten_full = res_flatten # for debugging
            res_flatten = res_flatten[:, -len(rho_flatten[0, :])+1:]

            res[0, :_N_aux_turn] = res_flatten[0, :_N_aux_turn]

            # for tt in range(self.num_turns):
            #     #tt_flatten = self.num_turns - tt - 1
            #     res[tt, :_N_aux_turn] = res_flatten[0,
            #         tt * _N_aux_turn: (tt + 1) * _N_aux_turn]
            self._res_flatten = res_flatten # for debugging
            self._rho_flatten = rho_flatten # for debugging

        self.moments_data['result'] = res.real


    # Parameters from CompressedProfile
    @property
    def _N_1(self):
        return self.moments_data._N_1

    @property
    def _N_2(self):
        return self.moments_data._N_2


    @property
    def _z_a(self):
        return self.moments_data._z_a

    @property
    def _z_b(self):
        return self.moments_data._z_b

    @property
    def z_period(self):
        return self.moments_data.z_period

    @property
    def _z_P(self):
        return self.moments_data._z_P

    @property
    def circumference(self):
        return self.moments_data.circumference

    @property
    def dz(self):
        return self.moments_data.dz

    @property
    def num_slices(self):
        return self.moments_data.num_slices

    @property
    def num_periods(self):
        return self.moments_data.num_periods

    @property
    def num_turns(self):
        return self.moments_data.num_turns


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

        self.moments_data.set_moments(i_source, i_turn, moments)

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

        z_out, moment_out = self.moments_data.get_moment_profile(
                moment_name, i_turn)

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


class TempResonatorFunction:
    def __init__(self, R_shunt, frequency, Q):
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q

    def __call__(self, z):
        R_s = self.R_shunt
        Q = self.Q
        f_r = self.frequency
        omega_r = 2 * np.pi * f_r
        alpha_t = omega_r / (2 * Q)
        omega_bar = np.sqrt(omega_r**2 - alpha_t**2)

        res = (z < 0) * (R_s * omega_r**2 / (Q * omega_bar)
               * np.exp(alpha_t * z / clight)
                * np.sin(omega_bar * z / clight))# Wake definition
        return res

def _build_z_wake(z_a, z_b, num_turns, N_aux, M_aux, circumference, dz,
                 AA, BB, CC, DD, z_P):
    z_c = z_a # For wakefield, z_c = z_a
    z_d = z_b # For wakefield, z_d = z_b
    z_wake = np.zeros((num_turns, M_aux))
    for tt in range(num_turns):
        z_a_turn = z_a + tt * circumference
        z_b_turn = z_b + tt * circumference
        temp_z = np.arange(
            z_c - z_b_turn, z_d - z_a_turn + dz/10, dz)[:-1]
        for ii, ll in enumerate(range(
                            CC - BB + 1, DD - AA)):
            z_wake[tt, ii*N_aux:(ii+1)*N_aux] = temp_z + ll * z_P
    return z_wake