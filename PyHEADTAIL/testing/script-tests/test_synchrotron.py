import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)

import numpy as np

from scipy.constants import c, e, m_p
from PyHEADTAIL.machines.synchrotron import BasicSynchrotron

circumference = 26658.8832
n_segments = 15
charge = e
mass = m_p


beta_x         = 92.7 
D_x            = 0
beta_y         = 93.2 
D_y            = 0 

Q_x            = 64.28
Q_y            = 59.31

Qp_x           = 10.
Qp_y           = 15.

app_x          = 0.0000e-9
app_y          = 0.0000e-9
app_xy         = 0

alpha             = 3.225e-04

h1, h2       = 35640, 35640*2
V1, V2       = 6e6, 0.
dphi1, dphi2 = 0, np.pi


longitudinal_mode = 'non-linear'

p0 = 450e9 * e / c

p_increment = 0



machine = BasicSynchrotron(optics_mode='smooth', circumference=circumference, n_segments=n_segments, 
             beta_x=beta_x, D_x=D_x, beta_y=beta_y, D_y=D_y,
             accQ_x=Q_x, accQ_y=Q_y, Qp_x=Qp_x, Qp_y=Qp_y, app_x=app_x, app_y=app_y, app_xy=app_xy,
             alpha_mom_compaction=alpha, longitudinal_mode='non-linear',
             h_RF=[h1,h2], V_RF=[V1,V2], dphi_RF=[dphi1,dphi2], p0=p0, p_increment=p_increment,
             charge=charge, mass=mass)


