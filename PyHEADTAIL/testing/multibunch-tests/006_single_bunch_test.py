from __future__ import division

import sys, os,time
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)

import numpy as np
from mpi4py import MPI
from scipy.constants import c, e, m_p

# from PyHEADTAIL.mpi.MPI import MPI
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
from PyHEADTAIL.monitors.monitors import BunchMonitor
from PyHEADTAIL.machines.synchrotron import Synchrotron

import matplotlib.pyplot as plt


def run(settings):
    
    R_shunt = 0.5e10
    Q = 0.5
    frequency = 1e8
    intensity = 3e11
    
    n_turns = 1000
    
    chroma = 0

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('I am rank ' + str(rank) + ' out of ' + str(size))
    print('  ')
    np.random.seed(0+rank)

    # BEAM AND MACHNINE PARAMETERS
    # ============================

    n_macroparticles = 100000

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
    h_RF = 13
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
    
    if settings == False:
        filling_scheme = [0]
    else:
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
    slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-3.*sigma_z, 3.*sigma_z),
                               circumference=circumference, h_bunch=h_RF)
    print('Delta z: ' + str(3.*sigma_z+3.*sigma_z))
    print('Delta t: ' + str((3.*sigma_z+3.*sigma_z)/c))

    # WAKE PARAMETERS
    # ============================
    n_turns_wake = 5
    wakes = CircularResonator(R_shunt, frequency, Q, n_turns_wake=n_turns_wake)

    wake_field = WakeField(slicer_for_wakefields, wakes, mpi=settings,
                           Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

    machine.one_turn_map.append(wake_field)

    # SIMULATION PARAMETERS
    # ============================
    
    write_every = 100

    # TRACKING LOOP
    # =============
    print('')
    print('mpi = ' + str(settings))
    print('\n--> Begin tracking...\n')
    
    import PyHEADTAIL.mpi.mpi_data as mpiTB    

    turns = np.linspace(1,n_turns,n_turns)
    bunch_data = np.zeros(n_turns)
    n_data_points = 0 
    for i in range(n_turns):
        t0 = time.time()

        machine.track(allbunches)
        bunch_list = allbunches.split_to_views()
        
        if rank == 0:
            bunch_data[i] = bunch_list[0].mean_x()
            n_data_points += 1
            if i%write_every is not 0:
                continue
            bunch = bunch_list[0]
            
            print('Bunch: {:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.sigma_z(), bunch.sigma_dp(), str(time.time() - t0)))
    slices = bunch_list[0].get_slices(slicer_for_wakefields,
                                     statistics=['mean_x', 'mean_y'])
    zbins = slices.z_bins

    return turns[:n_data_points], bunch_data[:n_data_points], zbins[:-1], slices.mean_x*slices.n_macroparticles_per_slice
#    return turns[:n_data_points], bunch_data[:n_data_points], zbins[:-1], slices.mean_x

if __name__== "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    mpi_settings = [
            False,
            True,
            'memory_optimized',
            'linear_mpi_full_ring_fft',
            'circular_mpi_full_ring_fft',
            ]
    
    labels = [
            "False",
            "True",
            "'memory_optimized'",
            "'linear_mpi_full_ring_fft'",
            "'circular_mpi_full_ring_fft'",
            ]
    
    if rank == 0:
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
    for i, settings in enumerate(mpi_settings):
        turns, mean_x, slice_z, slice_x =  run(settings)
        if rank == 0:
            ax1.plot(turns,mean_x, label = labels[i])
            ax2.plot(np.linspace(1,len(slice_z),len(slice_z)),slice_x, label = labels[i])
    
    ax1.set_xlim(800,1000)
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('BPM signal')
    ax1.legend()
    ax2.set_xlabel('Slice #')
    ax2.set_ylabel('BPM signal')
    ax2.legend()        

    plt.tight_layout()
    plt.show()

    
