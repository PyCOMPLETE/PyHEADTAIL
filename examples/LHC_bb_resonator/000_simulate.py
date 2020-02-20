from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import m_p

from PyHEADTAIL.machines.synchrotron import Synchrotron

machine = Synchrotron(
         optics_mode='smooth',
         charge=qe,
         mass=m_p,
         p0=6.5e12*qe/clight,
         circumference=26659.,
         n_segments=1,
         beta_x=100.,
         beta_y=100.,
         D_x=0.,
         D_y=0.,
         accQ_x=62.31,
         accQ_y=60.32,
         Qp_x=0,
         Qp_y=0,
         app_x=0,
         app_y=0,
         app_xy=0,
         longitudinal_mode='linear',
         Q_s=1.909e-3,
         alpha_mom_compaction=3.48e-4,
        )

bunch = machine.generate_6D_Gaussian_bunch(
        n_macroparticles=100000,
        intensity=3e11,
        epsn_x=2.5e-6,
        epsn_y=2.5e-6,
        sigma_z=1e-9/4*clight)

bunch.x += 1e-4

from PyHEADTAIL.particles.slicing import UniformBinSlicer
slicer_for_wakefields = UniformBinSlicer(
    1000, z_cuts=(-1.1*bunch.sigma_z(), 1.1*bunch.sigma_z()))


import PyHEADTAIL.impedances.wakes as wakes
wake = wakes.CircularResonator(R_shunt=25e6,
        frequency=2e9, Q=1)
wake_element = wakes.WakeField(slicer_for_wakefields, wake)

machine.one_turn_map.append(wake_element)

N_turns = 10000
x_list = []
for ii in range(N_turns):
    if ii % 100 == 0:
        print(f'{ii}/{N_turns}, pos={bunch.mean_x():.2e}')
    x_list.append(bunch.mean_x())
    machine.track(bunch)

