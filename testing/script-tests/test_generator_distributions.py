'''
@author Kevin Li, Michael Schenk, Adrian Oeftiger, Stefan Hegglin
@date 30.03.2015
@brief module for generating & matching particle distributions
'''
from __future__ import division

import numpy as np
import seaborn as sns
from seaborn import plt

from PyHEADTAIL.particles.generators import waterbag2D, waterbag4D, KV2D, KV4D, kv2D, kv4D

sns.set(context='notebook', style='whitegrid')


xx, xp, yy, yp = 1, 2, 3, 4

af = waterbag2D(xx, xp)
bf = waterbag4D(xx, xp, yy, yp)
cf = KV2D(xx, xp)
df = KV4D(xx, xp, yy, yp)
ef = kv2D(xx, xp)
gf = kv4D(xx, xp, yy, yp)
names = ['Waterbag 2D', 'Waterbag 4D',
         'K-V 2d', 'K-V 4D', 'kv2D', 'kv4D']

funcs = [af, bf, cf, df, ef, gf]

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for i, f in enumerate(funcs):
    try:
        a, _, b, _ = f(100000)
    except ValueError:
        a, b = f(100000)

    axes[i//2, i%2].plot(a, b, '.')
    # axes[i//2, i%2].set_aspect('equal')
    axes[i//2, i%2].set_title(names[i])

plt.show()
