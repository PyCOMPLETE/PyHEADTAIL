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

accQ_x = 62.31
accQ_y = 60.32
Q_s = 2.1e-3
chroma = 0

circumference = 26658.883

beta_x = circumference / (2.*np.pi*accQ_x)
beta_y = circumference / (2.*np.pi*accQ_y)

h_RF = 35640
h_bunch = 35640

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
bucket_id_set.sort()

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
    #      allbunches.x[mask] = 0

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
wakes = CircularResonator(R_shunt=135e6, frequency=1.97e9*0.6, Q=31000/1000, n_turns_wake=n_turns_wake)

# mpi_settings = 'circular_mpi_full_ring_fft'
# wake_field = WakeField(slicer, wakes, mpi=mpi_settings, Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

mpi_settings = False
# mpi_settings = 'memory_optimized'
wake_field = WakeField(slicer, wakes, mpi=mpi_settings)

# Wake full beam

n_buckets_slicer = max(filling_scheme) + 2
n_buckets_slicer = max(filling_scheme) + 1

slicer_full_beam = UniformBinSlicer(n_buckets_slicer * slicer.n_slices,
                                    z_cuts=((0.5 - n_buckets_slicer)*bucket_length, 0.5*bucket_length))
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


n_turns = 1
for i_turn in range(n_turns):

    wake_field.track(allbunches)
    wake_field_full_beam.track(beam)

    ax00.plot(beam._slice_sets[slicer_full_beam].z_centers, beam._slice_sets[slicer_full_beam].mean_x, 'o')

    ax01.plot(z_all[::skip], allbunches.xp[::skip], '.', label=f'MB turn {i_turn}')
    ax01.plot(beam.z[::skip], beam.xp[::skip], '.', label=f'SB turn {i_turn}')
    ax01.plot(beam._slice_sets[slicer_full_beam].z_centers, wake_field_full_beam.wake_kicks[0]._last_dipole_kick[0], 'x')

ax01.legend()

wake_function_x = wake_field_full_beam.wake_kicks[0].wake_function

z_centers = beam._slice_sets[slicer_full_beam].z_centers
dz = z_centers[1] - z_centers[0]

dxp_ref = wake_field_full_beam.wake_kicks[0]._last_dipole_kick[0]
z_ref = z_centers

z_centers_time_sorted = z_centers[::-1]

##############
# Build wake #
##############

# Wake formula
p0_SI = machine.p0
mean_x_slice = beam._slice_sets[slicer_full_beam].mean_x
num_charges_slice = beam._slice_sets[slicer_full_beam].charge_per_slice/e

n_wake = len(z_centers) + 100
z_wake = np.arange(0, -(n_wake)*dz, -dz)[::-1] # HEADTAIL order (time reversed)
assert len(z_wake) == n_wake

z_wake_time_sorted = z_wake[::-1]


R_s = wakes.R_shunt
Q = wakes.Q
f_r = wakes.frequency
omega_r = 2 * np.pi * f_r
alpha_t = omega_r / (2 * Q)
omega_bar = np.sqrt(omega_r**2 - alpha_t**2)
W_r = (R_s * omega_r**2 / (Q * omega_bar) * np.exp(alpha_t * z_wake / c)
      * np.sin(omega_bar * z_wake / c))# Wake definition
W_scaled = -e**2 / (p0_SI * c) * W_r # Put all constants in front of the wake

# Plot wakes
fig10 = plt.figure(10)
ax10 = fig10.add_subplot(111)

ax10.plot(z_wake, W_scaled, label='Wake')
ax10.set_xlabel('z [m]')
ax10.set_ylabel('W(z)')

# Compute dipole moments
dip_moment_slice = num_charges_slice * mean_x_slice

###################################
# Convolution with HEADTAIL order #
###################################
dxp = fftconvolve(dip_moment_slice, W_scaled, mode='full')
# Keep only the last n_centers points
dxp = dxp[-len(z_centers):]

######################################
# Convolution with time sorted order #
######################################

W_scaled_time_sorted = W_scaled[::-1]
dip_moment_slice_time_sorted = dip_moment_slice[::-1]

dxp_time_sorted = np.convolve(
    dip_moment_slice_time_sorted, W_scaled_time_sorted, mode='full')
# Keep only the first n_centers points
dxp_time_sorted = dxp_time_sorted[:len(z_centers)]

# Back to HEADTAIL order
dxp_time_sorted = dxp_time_sorted[::-1]

########################
# Convolution with FFT #
########################

from numpy.fft import fft, ifft

len_fft = len(W_scaled)+len(dip_moment_slice)-1

dxp_fft_time_sorted = ifft(
    fft(W_scaled_time_sorted, n=len_fft)
    * fft(dip_moment_slice_time_sorted, n=len_fft)).real
# Keep only the first n_centers points
dxp_fft_time_sorted = dxp_fft_time_sorted[:len(z_centers)]

# Back to HEADTAIL order
dxp_fft = dxp_fft_time_sorted[::-1]

###############
# As in latex #
###############

z_source_matrix = np.zeros((n_bunches, n_slices))
dipole_moment_matrix = np.zeros((n_bunches, n_slices))

for i_bunch in range(n_bunches):
    i_bucket = bucket_id_set[::-1][i_bunch]
    print(f"{i_bunch=} {i_bucket=}")
    i_z_bunch_center = np.argmin(np.abs(z_centers - (-i_bucket * bucket_length)))

    i_z_a_bunch = i_z_bunch_center - n_slices//2

    dipole_moment_matrix[i_bunch, :] = dip_moment_slice[
                                        i_z_a_bunch : i_z_a_bunch+n_slices]

    z_source_matrix[i_bunch, :] = z_centers[
                                    i_z_a_bunch : i_z_a_bunch+n_slices]

z_target_matrix = z_source_matrix.copy()


z_a = z_source_matrix[0, 0]
z_b = z_source_matrix[0, -1] + dz
z_c = z_a
z_d = z_b

PP = 5 * bucket_length
AA = -2
BB = 1
CC = AA
DD = BB

N_S = BB - AA
N_T = DD - CC

N_1 = z_source_matrix.shape[1]
N_2 = z_target_matrix.shape[1]

N_aux = N_1 + N_2

M_aux = N_aux * (N_S + N_T - 1)

z_wake_abcd_matrix = np.zeros((N_S + N_T - 1, N_aux))
for ii, ll in enumerate(range(CC-BB+1, DD-AA)):
    z_wake_abcd_matrix[ii, :] = np.arange(
        z_c - z_b, z_d - z_a, dz) + ll * PP
W_r_abcd_matrix = (
      R_s * omega_r**2 / (Q * omega_bar) * np.exp(alpha_t * z_wake_abcd_matrix / c)
      * np.sin(omega_bar * z_wake_abcd_matrix / c))# Wake definition
W_r_abcd_matrix[z_wake_abcd_matrix > 0] = 0
W_scaled_abcd_matrix = -e**2 / (p0_SI * c) * W_r_abcd_matrix # Put all constants in front of the wake

G_segm = W_scaled_abcd_matrix

G_aux = np.zeros(M_aux)
rho_aux = np.zeros(M_aux)
z_aux = np.zeros(M_aux)

for ii in range(N_S + N_T - 1):
    G_aux[ii*N_aux:(ii+1)*N_aux] = G_segm[ii, :]

for ii in range(N_S):
    rho_aux[ii*N_aux:(ii)*N_aux+N_1] = dipole_moment_matrix[ii, :]
    z_aux[ii*N_aux:(ii)*N_aux+N_1] = z_source_matrix[ii, :]

G_hat = fft(G_aux)
rho_hat = fft(rho_aux)

# phase_term = 1
# phase_term = (np.exp(-1j * 2 * np.pi * np.arange(M_aux) * N_S * N_aux / M_aux)
#                 * np.exp(-1j * 2 * np.pi * np.arange(M_aux) * (N_1) / M_aux))
phase_term = np.exp(1j * 2 * np.pi * np.arange(M_aux)
                    * ((N_S - 1)* N_aux + N_1) / ((N_S + N_T - 1)* N_aux))
phi_hat = G_hat * rho_hat * phase_term

phi_aux = ifft(phi_hat).real

phi_target_matrix = np.zeros((N_T, N_2))
for ii in range(N_T):
    phi_target_matrix[ii, :] = phi_aux[ii*N_aux:ii*N_aux+N_2]


# Plot results

fig2 = plt.figure(2, figsize=(6.4*1.4, 4.8*1.4))
ax21 = fig2.add_subplot(311)
ax22 = fig2.add_subplot(312, sharex=ax21)
ax23 = fig2.add_subplot(313, sharex=ax21)

ax21.plot(z_centers, num_charges_slice, label='num. charges')
ax21.set_ylabel('Number of charges per slice')
ax22.plot(z_centers, amplitude * np.sin(2 * np.pi * z_centers / wavelength), '--')
ax22.plot(z_centers, mean_x_slice, label='mean x')

ax22.set_ylabel('Mean x per slice')

ax23.plot(z_ref, dxp_ref, label='ref.')
ax23.plot(z_centers, dxp, '--', label='conv.')
ax23.plot(z_centers, dxp_time_sorted, '--', label='conv. t-sorted')
ax23.plot(z_centers, dxp_fft, '--', label='conv. fft')
# ax23.plot(z_aux, phi_aux, '.', label='conv. latex')
ax23.plot(z_target_matrix.T, phi_target_matrix.T, '-r',lw=3, label='conv. latex')
ax23.set_ylabel('Dipole kick per slice')
ax23.set_xlabel('z [m]')


# Put legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.subplots_adjust(right=0.75, hspace=0.3)

plt.show()