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
