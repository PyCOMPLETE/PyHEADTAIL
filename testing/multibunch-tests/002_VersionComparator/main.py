from __future__ import division

import sys, os
import sys, time
import numpy as np
from mpi4py import MPI
from scipy.constants import c, e, m_p


import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)


def run(argv):
    case = int(argv[0])
        
    # This is the main PyHEADTAIL script, which can be used     
    
    BIN = os.path.expanduser("../")
    sys.path.append(BIN)
    
    if case == 0:
        print('Case 0: ')
        outputpath = 'data_case_0'
#        from PyHEADTAIL_feedback.feedback import OneboxFeedback
#        from PyHEADTAIL_feedback.processors.multiplication import ChargeWeighter
#        from PyHEADTAIL_feedback.processors.convolution import Gaussian
        
    elif case == 1:
        print ('Case 1:')
        outputpath = 'data_case_1'
#        from PyHEADTAIL.feedback.feedback import OneboxFeedback
#        from PyHEADTAIL.feedback.processors.multiplication import ChargeWeighter
#        from PyHEADTAIL.feedback.processors.convolution import Gaussian  


    from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer
    from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField
    from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor
    from PyHEADTAIL.machines.synchrotron import Synchrotron
    
    job_id = 0
    chroma = 0

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Random number seed must be defined for the bunch-by-bunch trace comparison
    # but different seed must be used in each rank in order to avoid weird coherent
    # oscillations at the beginning.

    np.random.seed(0+rank)
#    np.random.seed(0)
    
    print('I am rank ' + str(rank) + ' out of ' + str(size))
    print('  ')

    # BEAM AND MACHNINE PARAMETERS
    # ============================

    n_macroparticles = 1500
    intensity = 1e12

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

# Number of bunches simulated
#    h_RF = 2748
#    h_RF = 274
    h_RF = 156


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

# Removes the longitudinal map from the one turn map
#    machine.one_turn_map = machine.one_turn_map[1:]

# Only the longitudinal map
#    machine.one_turn_map = [machine.one_turn_map[0]]
    print(machine.one_turn_map)

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
    
#    if rank == 1:
#    bunch_list = allbunches.split_to_views()
#    bunch_list[-1].x += 1.

    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-3.*sigma_z, 3.*sigma_z),
                               circumference=machine.circumference, h_bunch=h_RF)
    slicer1_for_slicemonitor = UniformBinSlicer(50, z_cuts=(-2 * sigma_z, 2 * sigma_z))
    slicer2_for_slicemonitor = UniformBinSlicer(50, z_cuts=(-2 * sigma_z, 2 * sigma_z))

    # WAKE PARAMETERS
    # ============================
    if case == 0:
        mpi_settings = 'linear_mpi_full_ring_fft'     
    elif case == 1:
        mpi_settings = True
        
    n_turns_wake = 8  
    f_rs = np.logspace(np.log10(3e4),np.log10(1e6),20)
    R_shunt = 1e10
    frequency = f_rs[12]
    Q=0.65
    wakes = CircularResonator(R_shunt, frequency, Q, n_turns_wake=n_turns_wake)

    wake_field = WakeField(slicer_for_wakefields, wakes, mpi=mpi_settings,
                           Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)


#    if case == 1:
    machine.one_turn_map.append(wake_field)

    # FEEDBACK
    # ========
    
#    feedback_gain = 2./50.
#    f_c = 1e6
#    f_harmonic = 1./(circumference/c/float(h_RF))
#    
#    processors_bunch_x = [
#        ChargeWeighter(normalization = 'segment_average'),
#        Gaussian(f_c, normalization=('bunch_by_bunch', f_harmonic)),
#    ]
#    processors_bunch_y = [
#        ChargeWeighter(normalization = 'segment_average'),
#        Gaussian(f_c, normalization=('bunch_by_bunch', f_harmonic)),
#    ]
#    
#    feedback_map = OneboxFeedback(feedback_gain, slicer_for_wakefields,
#                                  processors_bunch_x,processors_bunch_y,
#                                  mpi=True)
#
#    machine.one_turn_map.append(feedback_map)


    # SIMULATION PARAMETERS
    # ============================
    n_turns = 300


    # CREATE MONITORS
    # ===============
    # bucket = machine.longitudinal_map.get_bucket(bunch)
    simulation_parameters_dict = {'gamma'           : machine.gamma,\
                                  'intensity'       : intensity,\
                                  'Qx'              : accQ_x,\
                                  'Qy'              : accQ_y,\
                                  'Qs'              : Q_s,\
                                  'beta_x'          : beta_x,\
                                  'beta_y'          : beta_y,\
    #                               'beta_z'          : bucket.beta_z,\
                                  'epsn_x'          : epsn_x,\
                                  'epsn_y'          : epsn_y,\
                                  'sigma_z'         : sigma_z,\
                                 }
    bunchmonitor = BunchMonitor(
        outputpath+'/bunchmonitor_{:04d}_chroma={:g}'.format(job_id, chroma), n_turns,
        simulation_parameters_dict, write_buffer_to_file_every=n_turns, buffer_size=n_turns, mpi=True, filling_scheme=filling_scheme)
    
#    if rank == 0:
#        slicemonitor1 = SliceMonitor(
#            outputpath + '/slicemonitor_bunch_0_{:04d}_chroma={:g}'.format(job_id, chroma),
#            n_turns, slicer1_for_slicemonitor,
#            simulation_parameters_dict, write_buffer_to_file_every=n_turns, buffer_size=n_turns)
#        
#        slicemonitor2 = SliceMonitor(
#            outputpath + '/slicemonitor_bunch_255_{:04d}_chroma={:g}'.format(job_id, chroma),
#            n_turns, slicer2_for_slicemonitor,
#            simulation_parameters_dict, write_buffer_to_file_every=n_turns, buffer_size=n_turns)


    # TRACKING LOOP
    # =============
    print '\n--> Begin tracking...\n'

    for i in range(n_turns):
        t0 = time.clock()

        machine.track(allbunches)

        if rank == 0:
            bunch_list = allbunches.split_to_views()
            bunch = bunch_list[0]
            print 'Bunch: {:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.sigma_z(), bunch.sigma_dp(), str(time.clock() - t0))
        bunchmonitor.dump(allbunches)

        if rank == 0:
#            print 'I am dumping'
#            slicemonitor1.dump(bunch)
#            slicemonitor2.dump(bunch_list[-1])

            if i%1 is not 0:
                continue
            
            print 'Beam : {:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, allbunches.mean_x(), allbunches.mean_y(), allbunches.mean_z(), allbunches.epsn_x(), allbunches.epsn_y(), allbunches.epsn_z(), allbunches.sigma_z(), allbunches.sigma_dp(), str(time.clock() - t0))
            

    print '\n*** Successfully completed!'


if __name__=="__main__":
	run(sys.argv[1:])
