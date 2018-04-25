from __future__ import division

import sys, time
import numpy as np
from mpi4py import MPI
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
from PyHEADTAIL.monitors.monitors import BunchMonitor
from PyHEADTAIL.machines.synchrotron import Synchrotron

import matplotlib.pyplot as plt


def run(frequency, settings_idx):
    
    R_shunt = 0.5e10
    Q=0.65
    
#    intensities = np.logspace(np.log10(1e13),np.log10(1e15),11)
    intensity = 1e12
    
    n_turns = 15000
    
    chroma = 0

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('I am rank ' + str(rank) + ' out of ' + str(size))
    print('  ')
    np.random.seed(0+rank)

       # BEAM AND MACHNINE PARAMETERS
    # ============================

    n_macroparticles = 10000

    charge = e
    mass = m_p
    alpha = 53.86**-2

    p0 = 7000e9 * e / c

    accQ_x = 62.31
    accQ_y = 60.32
    Q_s = 2.1e-3
    circumference = 2665.8883
    s = None
    alpha_x = None
    alpha_y = None
    beta_x = circumference / (2.*np.pi*accQ_x)
    beta_y = circumference / (2.*np.pi*accQ_y)
    D_x = 0
    D_y = 0
    optics_mode = 'smooth'
    name = None
    n_segments = 1

    # detunings
    Qp_x = chroma
    Qp_y = chroma


    app_x = 0
    app_y = 0
    app_xy = 0

    longitudinal_mode = 'linear'

#    h_RF = 3564
    h_RF = 123
    wrap_z = False

    machine = Synchrotron(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), p0=p0,
            charge=charge, mass=mass, wrap_z=wrap_z, Q_s=Q_s)
    
#    removes longitudinal tracking
#    machine.one_turn_map = machine.one_turn_map[1:]
#    print(machine.one_turn_map)

    # every bucker is filled
    filling_scheme = sorted([i for i in range(h_RF)])

    # BEAM
    # ====
    epsn_x = 2e-6
    epsn_y = 2e-6
    sigma_z = 0.09
    allbunches = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme, matched=False)

    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields = UniformBinSlicer(1, z_cuts=(-3.*sigma_z, 3.*sigma_z),
                               circumference=circumference, h_bunch=h_RF)
    print('Delta z: ' + str(3.*sigma_z+3.*sigma_z))
    print('Delta t: ' + str((3.*sigma_z+3.*sigma_z)/c))

    # WAKE PARAMETERS
    # ============================
    n_turns_wake = 5
    if settings_idx==0:
        mpi_settings = 'linear_mpi_full_ring_fft'
    elif settings_idx==1:
        mpi_settings = 'circular_mpi_full_ring_fft'
    else:
        raise ValueError('Unknown wake type!')
    wakes = CircularResonator(R_shunt, frequency, Q, n_turns_wake=n_turns_wake)

    wake_field = WakeField(slicer_for_wakefields, wakes, mpi=mpi_settings,
                           Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

    machine.one_turn_map.append(wake_field)

    # SIMULATION PARAMETERS
    # ============================
    write_every = 100

    # TRACKING LOOP
    # =============
    print('\n--> Begin tracking...\n')
    
    import PyHEADTAIL.mpi.mpi_data as mpiTB    
    n_total_bunches = h_RF

    beam_data = np.zeros(n_turns)
    n_data_points = 0 
    for i in range(n_turns):
        t0 = time.clock()

        machine.track(allbunches)
        bunch_list = allbunches.split_to_views()
        
        my_abs_mean_x = 0.
        my_abs_mean_y = 0.
        
        for b in bunch_list:
            my_abs_mean_x += np.abs(b.mean_x())/n_total_bunches
            my_abs_mean_y += np.abs(b.mean_y())/n_total_bunches
        
        total_abs_mean_x = np.sum(mpiTB.share_numbers(my_abs_mean_x))
        total_abs_mean_y = np.sum(mpiTB.share_numbers(my_abs_mean_y))
        
        if (total_abs_mean_x > 1e-2) or (total_abs_mean_y > 1e-2):
            break

        
        if rank == 0:
            beam_data[i] = total_abs_mean_y
            n_data_points += 1
            if i%write_every is not 0:
                continue
            bunch = bunch_list[0]
            
            print('Bunch: {:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.sigma_z(), bunch.sigma_dp(), str(time.clock() - t0)))
            print('Beam : {:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, allbunches.mean_x(), allbunches.mean_y(), allbunches.mean_z(), allbunches.epsn_x(), allbunches.epsn_y(), allbunches.epsn_z(), allbunches.sigma_z(), allbunches.sigma_dp(), str(time.clock() - t0)))

    return beam_data[:n_data_points]

if __name__=="__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    f_rs = np.logspace(np.log10(0.3e5),np.log10(1e6),10)
#    f_rs = np.logspace(np.log10(1e5),np.log10(1e6),2)
    if rank == 0:
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        growth_rates = np.zeros(len(f_rs))
        growth_rates2 = np.zeros(len(f_rs))
    
    for i, f in enumerate(f_rs):
        values = run(f, 0)
        if rank == 0:
            turns = np.linspace(1,len(values),len(values))
            n_valid_points = int(len(values)/3)
            coeffs = np.polyfit(turns[-n_valid_points:], np.log(values[-n_valid_points:]),1)
            fit_line = np.exp(turns*coeffs[0]+coeffs[1])
            
            growth_rates[i] = coeffs[0]
            ax1.semilogy(turns, values)
            ax1.semilogy(turns, fit_line, 'r--')
            ax1.set_xlabel('Turn')
            ax1.set_ylabel('Avg. beam displacement')
    
    for i, f in enumerate(f_rs):
        values = run(f, 1)
        if rank == 0:
            turns = np.linspace(1,len(values),len(values))
            n_valid_points = int(len(values)/5)
            coeffs = np.polyfit(turns[-n_valid_points:], np.log(values[-n_valid_points:]),1)
            fit_line = np.exp(turns*coeffs[0]+coeffs[1])
            
            growth_rates2[i] = coeffs[0]
            ax2.semilogy(turns, values)
            ax2.semilogy(turns, fit_line, 'r--')
            ax2.set_xlabel('Turn')
            ax2.set_ylabel('Avg. beam displacement')
            
    if rank == 0:

        ax3.semilogx(f_rs,growth_rates,'o',label='Linear convolution')
        ax3.semilogx(f_rs,growth_rates2,'x',label='Circular convolution')
        ax3.set_xlabel('Resonator frequency [Hz]')
        ax3.set_ylabel('Growth rate [1/turn]')
        ax3.legend()
        plt.tight_layout()
        plt.show()

    