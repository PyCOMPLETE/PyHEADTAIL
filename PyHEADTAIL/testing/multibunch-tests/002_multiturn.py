import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c, e, m_p
from scipy.signal import fftconvolve
import time

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.particles import Particles
# from PyHEADTAIL.impedances.wakes import CircularResistiveWall, WakeField
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
from PyHEADTAIL.monitors.monitors import BunchMonitor
from PyHEADTAIL.machines.synchrotron import Synchrotron


# Machine settings
n_turns = 300

n_macroparticles = 100000 # per bunch
intensity = 2.3e11

alpha = 53.86**-2

p0 = 7000e9 * e / c

accQ_x = 0.31
accQ_y = 0.32
Q_s = 2.1e-3
chroma = 0

circumference = 26658.883 / 35640 * 20

beta_x = 100 #circumference / (2.*np.pi*accQ_x)
beta_y = 100 #circumference / (2.*np.pi*accQ_y)

h_RF = 20
h_bunch = 20

epsn_x = 2e-6
epsn_y = 2e-6
sigma_z = 0.09

machine = Synchrotron(
        optics_mode='smooth', circumference=circumference,
        n_segments=1, s=None, name=None,
        alpha_x=None, beta_x=beta_x, D_x=0,
        alpha_y=None, beta_y=beta_y, D_y=0,
        accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=chroma, Qp_y=chroma,
        app_x=0, app_y=0, app_xy=0,
        alpha_mom_compaction=alpha, longitudinal_mode='linear',
        h_RF=np.atleast_1d(h_RF), p0=p0,
        charge=e, mass=m_p, wrap_z=False, Q_s=Q_s)

transverse_map = machine.transverse_map.segment_maps[0]

# Filling scheme

bunch_spacing_buckets = 5
n_bunches = 3
filling_scheme = [i*bunch_spacing_buckets for i in range(n_bunches)]

# Initialise beam
allbunches = machine.generate_6D_Gaussian_bunch(n_macroparticles, intensity,
                                                epsn_x, epsn_y, sigma_z=sigma_z,
                                                filling_scheme=filling_scheme,
                                                matched=False)

bucket_id_set = list(set(allbunches.bucket_id))

bucket_length = machine.circumference / h_RF
z_all = -allbunches.bucket_id * bucket_length + allbunches.z

amplitude = 1e-3
wavelength = 2
allbunches.x = amplitude * np.sin(2 * np.pi * z_all / wavelength)
allbunches.xp *= 0

# allbunches.x[allbunches.z < 0] = 0

for b_id in bucket_id_set:
    mask = allbunches.bucket_id == b_id
    z_centroid = np.mean(allbunches.z[mask])
    z_std = np.std(allbunches.z[mask])
    mask_tails = mask & (np.abs(allbunches.z - z_centroid) > z_std)
    allbunches.x[mask_tails] = 0
    # if b_id != 0:
    #     allbunches.x[mask] = 0

beam = Particles(macroparticlenumber=allbunches.macroparticlenumber,
                 particlenumber_per_mp=allbunches.particlenumber_per_mp,
                 charge=allbunches.charge, mass=allbunches.mass,
                 circumference=allbunches.circumference, gamma=allbunches.gamma,
                 coords_n_momenta_dict=dict(x=allbunches.x.copy(),
                                            y=allbunches.y.copy(),
                                            xp=allbunches.xp.copy(),
                                            yp=allbunches.yp.copy(),
                                            z=z_all.copy(),
                                            dp=allbunches.dp.copy(),
                 ))

# Initialise wakes

n_slices = 100
slicer = UniformBinSlicer(n_slices, z_cuts=(-0.5*bucket_length, 0.5*bucket_length),
                          circumference=machine.circumference, h_bunch=h_bunch)

n_turns_wake = 3

# pipe radius [m]
b = 13.2e-3
# length of the pipe [m]
L = 100000.
# conductivity of the pipe 1/[Ohm m]
sigma = 1. / 7.88e-10

# wakes = CircularResistiveWall(b, L, sigma, b/c, beta_beam=machine.beta)
wakes = CircularResonator(R_shunt=135e6, frequency=1.97e9*0.6, Q=31000/100, n_turns_wake=n_turns_wake)

# mpi_settings = 'circular_mpi_full_ring_fft'
# wake_field = WakeField(slicer, wakes, mpi=mpi_settings, Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

mpi_settings = False
# mpi_settings = 'memory_optimized'
wake_field = WakeField(slicer, wakes, mpi=mpi_settings)

# Wake full beam

n_buckets_slicer = max(filling_scheme) + 2
n_buckets_slicer = max(filling_scheme) + 1

slicer_full_beam = UniformBinSlicer(n_buckets_slicer * slicer.n_slices,
                                    z_cuts=((0.5 - n_buckets_slicer)*bucket_length, 0.5*bucket_length),
                                    circumference=machine.circumference, h_bunch=h_bunch)
slicer_full_beam.force_absolute = True

wakes_full_beam = CircularResonator(R_shunt=wakes.R_shunt, frequency=wakes.frequency, Q=wakes.Q, n_turns_wake=wakes.n_turns_wake)
wake_field_full_beam = WakeField(slicer_full_beam, wakes_full_beam, mpi=False)


# import pdb
# pdb.set_trace()

plt.close('all')

fig0, (ax00, ax01) = plt.subplots(2, 1)
ax01.sharex(ax00)

skip = 10
ax00.plot(z_all[::skip], allbunches.x[::skip], '.')
ax00.plot(beam.z[::skip], beam.x[::skip], '.')

x_at_wake_allbunches = []
xp_before_wake_allbunches = []
xp_after_wake_allbunches = []
slice_set_before_wake_allbunches = []
slice_set_after_wake_allbunches = []

x_at_wake_beam = []
xp_before_wake_beam = []
xp_after_wake_beam = []
slice_set_before_wake_beam = []
slice_set_after_wake_beam = []


n_turns = 3
color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i_turn in range(n_turns):

    allbunches.clean_slices()
    slice_set_before_wake_allbunches.append(allbunches.get_slices(slicer,
                        statistics=['mean_x', 'mean_xp', 'mean_y', 'mean_yp']))
    allbunches.clean_slices()

    beam.clean_slices()
    slice_set_before_wake_beam.append(beam.get_slices(slicer_full_beam,
                        statistics=['mean_x', 'mean_xp', 'mean_y', 'mean_yp']))
    beam.clean_slices()

    x_at_wake_allbunches.append(allbunches.x.copy())
    xp_before_wake_allbunches.append(allbunches.xp.copy())

    x_at_wake_beam.append(beam.x.copy())
    xp_before_wake_beam.append(beam.xp.copy())

    wake_field.track(allbunches)
    wake_field_full_beam.track(beam)

    allbunches.clean_slices()
    slice_set_after_wake_allbunches.append(allbunches.get_slices(slicer,
                        statistics=['mean_x', 'mean_xp', 'mean_y', 'mean_yp']))
    allbunches.clean_slices()

    beam.clean_slices()
    slice_set_after_wake_beam.append(beam.get_slices(slicer_full_beam,
                        statistics=['mean_x', 'mean_xp', 'mean_y', 'mean_yp']))
    beam.clean_slices()

    xp_after_wake_allbunches.append(allbunches.xp.copy())
    xp_after_wake_beam.append(beam.xp.copy())

    transverse_map.track(allbunches)
    transverse_map.track(beam)


    ax00.plot(slice_set_after_wake_beam[-1].z_centers,
              slice_set_after_wake_beam[-1].mean_x, 'x', color=color_list[i_turn])

    ax01.plot(z_all[::skip],
              xp_after_wake_allbunches[-1][::skip] - xp_before_wake_allbunches[-1][::skip],
              '.', label=f'MB turn {i_turn}', color=color_list[i_turn])
    ax01.plot(beam.z[::skip], xp_after_wake_beam[-1][::skip] - xp_before_wake_beam[-1][::skip],
              '.', label=f'SB turn {i_turn}', color=color_list[i_turn])

    ax01.plot(
        slice_set_after_wake_beam[-1].z_centers,
        slice_set_after_wake_beam[-1].mean_xp - slice_set_before_wake_beam[-1].mean_xp,
        'x', color=color_list[i_turn])

ax01.legend()

plt.show()


z_centers = slice_set_before_wake_beam[0].z_centers
dz = z_centers[1] - z_centers[0]

dxp_ref = slice_set_after_wake_beam[0].mean_xp - slice_set_before_wake_beam[0].mean_xp
z_ref = z_centers

z_centers_time_sorted = z_centers[::-1]

##############
# Build wake #
##############

# Wake formula
p0_SI = machine.p0
mean_x_slice = slice_set_before_wake_beam[0].mean_x
num_charges_slice = slice_set_before_wake_beam[0].charge_per_slice/e

n_wake = len(z_centers) + 100
z_wake = np.arange(0, -(n_wake)*dz, -dz)[::-1] # HEADTAIL order (time reversed)
assert np.max(np.abs(z_wake)) < machine.circumference
z_wake_mt = np.array(
    [z_wake - i_turn * machine.circumference for i_turn in range(n_turns)][::-1])



R_s = wakes.R_shunt
Q = wakes.Q
f_r = wakes.frequency
omega_r = 2 * np.pi * f_r
alpha_t = omega_r / (2 * Q)
omega_bar = np.sqrt(omega_r**2 - alpha_t**2)
Wmt_r = (R_s * omega_r**2 / (Q * omega_bar) * np.exp(alpha_t * z_wake_mt / c)
      * np.sin(omega_bar * z_wake_mt / c))# Wake definition
Wmt_scaled = -e**2 / (p0_SI * c) * Wmt_r # Put all constants in front of the wake

# Plot wakes
fig10 = plt.figure(10)
ax10 = fig10.add_subplot(111)

ax10.plot(z_wake_mt.T, Wmt_scaled.T, label='Wake')
ax10.set_xlabel('z [m]')
ax10.set_ylabel('W(z)')

num_charges_slice = np.zeros(shape=(n_turns_wake, len(z_centers)), dtype=float)
mean_x_slice = np.zeros(shape=(n_turns_wake, len(z_centers)), dtype=float)

for i_turn in range(n_turns_wake):
    num_charges_slice[i_turn] = slice_set_before_wake_beam[i_turn].charge_per_slice/e
    mean_x_slice[i_turn] = slice_set_before_wake_beam[i_turn].mean_x

dip_moment_slice = num_charges_slice * mean_x_slice


Wmt_scaled_time_sorted = Wmt_scaled[:, ::-1]
dip_moment_slice_time_sorted = dip_moment_slice[:, ::-1]

# Convolution

from numpy.fft import fft, ifft

len_fft = len(Wmt_scaled_time_sorted[0, :])+len(dip_moment_slice_time_sorted[0, :])-1
W_transf = fft(Wmt_scaled_time_sorted, n=len_fft, axis=1)

dxp_fft_time_sorted = ifft(
    W_transf *
    fft(dip_moment_slice_time_sorted, n=len_fft, axis = 1),
    axis=1).real
# Keep only the first n_centers points
dxp_fft_time_sorted = dxp_fft_time_sorted[:,:len(z_centers)]

# Back to HEADTAIL order
dxp_fft = dxp_fft_time_sorted[:, ::-1]

plt.show()

prrrrr

#######################
# Chopped and compressed convolution #
#######################

z_wake_mt_time_sorted = z_wake_mt[:, ::-1]

K_period = n_slices * bunch_spacing_buckets
L_preserve = n_slices

n_periods = len(Wmt_scaled_time_sorted[0,:]) // K_period

WW_compressed = []
WW = Wmt_scaled_time_sorted
dip_moments_compressed_time_sorted = []
z_centers_compressed_time_sorted = []
z_wake_compressed_time_sorted = []
for ii in range(n_periods+1):
    start_preserve = ii*K_period - L_preserve + 1
    if start_preserve < 0:
        start_preserve = 0
    end_preserve = ii*K_period + L_preserve
    if end_preserve > len(WW[0,:]):
        end_preserve = len(WW[0,:])
    part_preserve = slice(start_preserve, end_preserve)

    WW_compressed.append(WW[:, part_preserve])
    z_wake_compressed_time_sorted.append(z_wake_mt_time_sorted[:, part_preserve])

    dip_moments_compressed_time_sorted.append(
        dip_moment_slice_time_sorted[:, part_preserve])
    z_centers_compressed_time_sorted.append(
        z_centers_time_sorted[part_preserve])

WW_compressed = np.concatenate(WW_compressed, axis=1)
z_wake_compressed_time_sorted = np.concatenate(
    z_wake_compressed_time_sorted, axis=1)
dip_moments_compressed_time_sorted = np.concatenate(
    dip_moments_compressed_time_sorted, axis=1)
z_centers_compressed_time_sorted = np.concatenate(
    z_centers_compressed_time_sorted)

# FFT size
len_fft = len(WW_compressed[0, :])+len(dip_moment_slice_time_sorted[0, :])-1

# FFT convolution
WW_fft = np.fft.rfft(WW_compressed, n=len_fft, axis=1)
dip_moments_fft = np.fft.rfft(dip_moments_compressed_time_sorted, n=len_fft, axis=1)

dxp_compressed_time_sorted = np.fft.irfft(WW_fft * dip_moments_fft, axis=1)

# Keep only the first n_centers_compressed points
dxp_compressed_time_sorted = dxp_compressed_time_sorted[
                :, :len(z_centers_compressed_time_sorted)]

# Plot
plt.figure(100)
plt.plot(z_centers_compressed_time_sorted,
         np.sum(dxp_compressed_time_sorted, axis=0), '.')

plt.show()
