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
                num_periods=1, # This is N_S
                num_turns=1,
                circumference=None):

        self._N_1 = num_slices # N_1 in the
        self._P = z_period # P in the paper
        self._N_S = num_periods # N_S in the paper
        self.num_turns = num_turns

        self._N_aux = self._N_1 + self._N_2 # N_aux in the paper

        # Compute M_aux
        self._M_aux = (self._N_S + self._N_T - 1) * self._N_aux # M_aux in the paper

        self.moments_names = source_moments
        self.moments_data = np.zeros(
            (self._M_aux, self.num_turns, len(source_moments)), dtype=np.float64)

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

    def set_moments(self, i_bunch, i_turn, moments):
        pass

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
rf_bucket_length = 5

# Quadrupolar wakefield
wf = Wakefield(
    source_moments=['charge'], # would be ['charge', 'x'] for dipolar
    kick='x',
    scale_kick=[3.4, 'x'], # The kick is scaled by position of the particle for quadrupolar, would be None for dipolar
    function=ResonatorFunction(Rs=3, Qs=4, f0=5),
    z_slice_range=[-rf_bucket_length/2, rf_bucket_length/2], # These are [a, b] in the paper
    slicer=None, # alternatively, a slicer can be used
    num_slices=100, # Per bunch, this is N_1 in the paper
    z_period=rf_bucket_length, # This is P in the paper
    num_periods=1, # This is N_S
    num_turns=3,
    circumference=circumference,
)

wf.set_moments(
    i_bunch=3,
    i_turn=2,
    moments={'num_particles': [1,2,3],
             'x': [4,5,6],
             'y': [4,5,6]}
)


