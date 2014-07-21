import numpy as np
import pylab as pl



import ecloud.PyECLOUD_for_PyHEADTAIL as pyecl
from particles.particles import *
from scipy.constants import e, m_e
import numpy as np
from particles.slicer import *


C = 6911.
R = C / (2 * np.pi)
gamma_tr = 18.
gamma = 27.7
eta = 1. / gamma_tr ** 2 - 1 / gamma ** 2
Qx = 20.13
Qy = 20.18
Qs = 0.017
beta_x = 54.6
beta_y = 54.6
beta_z = np.abs(eta) * R / Qs
epsn_x = 2.5
epsn_y = 2.5
epsn_z = 0.5*(0.2/0.23)**2

n_turns = 1

n_segments = 1

L_ecloud = C/n_segments



# Beam
bunch = Particles.as_gaussian(100000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)

#ecloud
beamslicer = Slicer(64, nsigmaz=3)
ecloud = pyecl.Ecloud(L_ecloud, beamslicer, Dt_ref = 25e-12, pyecl_input_folder='drift_for_benchmark')

# Betatron

s = np.arange(n_segments + 1) * C / n_segments
ltm = TransverseTracker.from_copy(s,
    np.zeros(n_segments), np.ones(n_segments) * beta_x, np.zeros(n_segments),
    np.zeros(n_segments), np.ones(n_segments) * beta_y, np.zeros(n_segments),
    Qx, 0, 0, 0, Qy, 0, 0, 0)
