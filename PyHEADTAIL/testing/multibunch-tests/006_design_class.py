import numpy as np

class Wakefield:

    def __init__(self,
                source_moments,
                kick,
                scale_kick,
                function,
                z_slice_range=None, # These are [a, b] in the paper
                n_slices=None, # Per bunch, this is N_1 in the paper
                slicer=None, # alternatively, a slicer can be used
                z_period=None, # This is P in the paper
                n_periods=1, # This is N_S
                n_turns=1,
                circumference=None):

        N_1 = n_slices
        P = z_period
        N_S = n_periods

        # Compute M_aux
        M_aux = 999 # TODO

        self.moments_names = source_moments
        self.moments_data = np.zeros(M_aux, n_turns, len(source_moments))

        self._G_hat = 'TODO'
        self.scale_kick = scale_kick

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



resonator_function = ResonatorFunction(Rs, Qs, f0)

# Can be used to plot the function
t = np.linspace(-10, 10, 1000)
plt.plot(t, resonator_function.evaluate(t))

rf_bucket_length = 3.
circumference = 26658.8832

# Quadrupolar wakefield
wf = Wakefield(
    source_moment=['charge'], # would be ['charge', 'x'] for dipolar
    kick='x',
    scale_kick=[3.4, 'x'], # The kick is scaled by position of the particle for quadrupolar, would be None for dipolar
    function=ResonatorFunction(Rs=3, Qs=4, f0=5),
    z_slice_range=[-rf_bucket_length/2, rf_bucket_length/2], # These are [a, b] in the paper
    slicer=None, # alternatively, a slicer can be used
    n_slices=100, # Per bunch, this is N_1 in the paper
    z_period=rf_bucket_length, # This is P in the paper
    n_periods=1, # This is N_S
    n_turns=3,
    circumference=circumference,
)


# You can always access the slicer
wf.slicer # e.g. to use it in a monitor

# You can also access the function
wf.function # e.g. to plot the wake

wf.track(particles) # This is the tracking function


class WakefieldResonator(Wakefield):

    def __init__(self, Rs, Qs, f0, z_slice_range, n_slices):
        self.Rs = Rs
        self.Qs = Qs
        self.f0 = f0
        self.rf_bucket_length = rf_bucket_length
        self.circumference = circumference
        self.function = ResonatorFunction(Rs, Qs, f0)


