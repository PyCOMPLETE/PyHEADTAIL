from pylab import *
import h5py

# Open hdf5 file containing bunch information for every turn and get handle 'hf'.
hf = h5py.File('bunch.h5', 'r')
A = hf['Bunch']
y, yp = A['mean_y'], A['mean_yp']
dz, dp = A['mean_dz'], A['mean_dp']

fig, ( ax1, ax2, ax3 ) = subplots(3)
fig.subplots_adjust(hspace = 0.3)

r = sqrt(y[:] ** 2 + (54.5054 * yp[:]) ** 2)
ax1.plot(r, c='purple')
ax1.plot(-r, c='purple')
ax1.plot(y)
ax1.set_xlabel('#turns')
ax1.set_ylabel('y pos.')
#ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax2.plot(dz)
ax2.set_xlabel('#turns')
ax2.set_ylabel('dz')
#ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax3.plot(dp)
ax3.set_xlabel('#turns')
ax3.set_ylabel('dp')
ax3.ticklabel_format(style='sci', axis='dp', scilimits=(0,0))

#ax3.scatter(, , marker='.')
show()
