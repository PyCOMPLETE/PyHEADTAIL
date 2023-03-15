import numpy as np

from wakefield import Wakefield

circumference=40.

class DeltaFunction:

    def __init__(self, tol_z, delay=0.):
        self.delta_z = tol_z
        self.delay = delay

    def __call__(self, z):
        #return np.exp(z) * (z < 0)
        return np.float64(np.abs(z+self.delay) < self.delta_z)



rf_bucket_length = 1

# Quadrupolar wakefield
wf = Wakefield(
    source_moments=['num_particles', 'x', 'y'],
    kick='x',
    scale_kick=[3.4, 'x'], # The kick is scaled by position of the particle for quadrupolar, would be None for dipolar
    function=DeltaFunction(tol_z=1e-12, delay=circumference),
    z_slice_range=[-rf_bucket_length/2, rf_bucket_length/2], # These are [a, b] in the paper
    slicer=None, # alternatively, a slicer can be used
    num_slices=5, # Per bunch, this is N_1 in the paper
    z_period=10, # This is P in the paper
    num_periods=4, # This is N_S
    num_turns=3,
    circumference=circumference,
    _flatten=True
)

assert wf.moments_data.data.shape == (4, 3, 70)
assert wf._M_aux == 70
assert wf._N_aux == 10
assert wf._N_1 == 5
assert wf._N_S == 4
assert wf._z_P == 10



for i_source in range(3):
    wf.set_moments(
        i_source=i_source,
        i_turn=1,
        moments={'num_particles': np.linspace(-1, 1, 5) * (i_source + 1)})


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

for i_source in range(3):
    wf.set_moments(
        i_source=i_source,
        i_turn=1,
        moments={'num_particles': np.linspace(-1, 1, 5) * (i_source + 1)})
wf._compute_convolution(moment_names=['num_particles'])
plt.plot(*wf.get_moment_profile('result', 0), 'x')

plt.show()


