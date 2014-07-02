from __future__ import division
import cProfile, itertools, sys, time, timeit

from scipy.constants import c, e, m_p

from particles.particles import *
from particles.slicer import *
from cobra_functions import stats, random
from monitors.monitors import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *

import pylab as plt


# plt.ion()


# Parameters
# ==========

x_min = 1.
x_max = 2.
y_min = 3.
y_max = 4.




# Object definitions
# ==================
# Bunch - TODO: fix initialisation of longitudinal phase space
bunch = Particles.as_uniformXY(10000, e, 1., 1.15e11, m_p, x_min, x_max, y_min, y_max)
# ecloud = Cloud.from_parameters(100000, 10e11, plt.std(bunch.x) * 20, plt.std(bunch.y) * 20, C)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
ax1.plot(bunch.x, bunch.y, 'g.')

ax2.plot(bunch.x, bunch.xp, 'g.')

ax3.plot(bunch.y, bunch.yp, 'g.')

plt.suptitle('%e'%bunch.intensity)
plt.show()
exit(-1)

