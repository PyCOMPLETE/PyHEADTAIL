from pylab import *

x = arange(0.,10.,1)

y = 0*x
y[:] = x#[:]

y[5] = -1.
