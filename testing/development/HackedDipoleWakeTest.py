from __future__ import division
import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)
import numpy as np
from PyHEADTAIL.particles.slicing import UniformBinSlicer
#from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor
from PyHEADTAIL.impedances.hacked_dipole_wakes import HackedDipoleWake
#CellMonitor


import time
from scipy.constants import c, e, m_p


try:
    print outputpath
except NameError:
    outputpath = '.'


n_macroparticles = 1000
n_turns          = 10240
wakefile1        = '../interactive-tests/wake_table.dat'
n_bunches = 80
it = 0

def run(intensity, chroma, dampingtime = 1e6, bandwidth = 100.):


    # BEAM AND MACHNINE PARAMETERS
    # ============================
    from HLLHC import HLLHC
    machine = HLLHC(mass=m_p, charge=e, n_segments=1,
                    machine_configuration='7_TeV_collision_tunes',
                    longitudinal_focusing='non-linear',
                    Qp_x=chroma, Qp_y=chroma, wrap_z=True)



    h = np.min(machine.longitudinal_map.harmonics) * 1.
    print 'np.min(machine.longitudinal_map.harmonics) * 1. -> ' + str(h)
    h_counter = -1
    filling_scheme = []

    for i in xrange(2):
        for j in xrange(78):
            h_counter += 1
            filling_scheme.append(10*h_counter)
        for j in xrange(8):
            h_counter += 1

    h_rf = 133650
    circumference = 100200.
    bunch_spacing = 10. * circumference/float(h_rf)/c
    bunch_length = bunch_spacing/5.


    # BEAM
    # ====
    epsn_x  = 2.2e-6
    epsn_y  = 2.2e-6
    sigma_z = 0.08
    bunch   = machine.generate_6D_Gaussian_bunch_matched(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme)
    sigma_z = bunch.sigma_z()


    # CREATE BEAM SLICERS
    # ===================

    slicer_for_wakefields   = UniformBinSlicer(10, z_cuts=(-0.75* bunch_spacing/10.*c, 0.75*bunch_spacing/10.*c))
    # slicer_for_wakefields = UniformBinSlicer(10, z_cuts=(-1 * sigma_z, 1 * sigma_z))


    # CREATE WAKES
    # ============
#     wake_table1          = WakeTable(wakefile1,
#                                      ['time', 'Nolong', 'dipole_x', 'dipole_y',
#                                       'quadrupole_x', 'quadrupole_y'], n_turns_wake=10,
#                                       circumference=machine.circumference)
#     wake_field           = WakeField(slicer_for_wakefields, wake_table1,
#                                       circumference=machine.circumference, mpi=True)

    wakedata = np.loadtxt(wakefile1)
    print 'wakedata.shape' + str(wakedata.shape)
    data_t = wakedata[:,0]
    data_x = wakedata[:,2]
    data_y = wakedata[:,3]
    wake_field = HackedDipoleWake(data_t, data_x, data_y, slicer_for_wakefields, n_turns_wakes=5, circumference=machine.circumference, mpi=True)

    # CREATE MONITORS
    # ===============
    bucket = machine.longitudinal_map.get_bucket(bunch)
    simulation_parameters_dict = {'gamma'           : machine.gamma,\
                                  'intensity'       : intensity,\
                                  'Qx'              : machine.Q_x,\
                                  'Qy'              : machine.Q_y,\
                                  'Qs'              : bucket.Qs,\
                                  'beta_x'          : bunch.beta_Twiss_x(),\
                                  'beta_y'          : bunch.beta_Twiss_y(),\
                                  'beta_z'          : bucket.beta_z,\
                                  'epsn_x'          : bunch.epsn_x(),\
                                  'epsn_y'          : bunch.epsn_y(),\
                                  'sigma_z'         : bunch.sigma_z(),\
                                 }
#    bunchmonitor = BunchMonitor(
#        outputpath+'/bunchmonitor_{:04d}_chroma={:g}'.format(it, chroma), n_turns,
#        simulation_parameters_dict, write_buffer_to_file_every=1024, buffer_size=1024, mpi=True, filling_scheme=filling_scheme)

    # TRACKING LOOP
    # =============
    machine.one_turn_map.append(wake_field)


    print '\n--> Begin tracking...\n'



    for i in range(n_turns):
        t0 = time.clock()

        machine.track(bunch)
#        bunchmonitor.dump(bunch)
        print '{:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3e} \t {:3f} \t {:3f} \t {:3f} \t {:3s}'.format(i, bunch.mean_x(), bunch.mean_y(), bunch.mean_z(), bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.sigma_z(), bunch.sigma_dp(), str(time.clock() - t0))

    print '\n*** Successfully completed!'


run(1e11, 0, 50, 40)
