
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py, os


def get_data(datafile, filling_scheme):
    h5f = h5py.File(datafile,'r')

    data_mean_x = None
    data_mean_z = None
    data_mean_dp = None
    data_epsn_x = None

    for i, bunch_id in enumerate(filling_scheme):
        t_mean_x = h5f['Bunches'][str(bunch_id)]['mean_x'][:]
        t_mean_y = h5f['Bunches'][str(bunch_id)]['mean_y'][:]
        t_mean_z = h5f['Bunches'][str(bunch_id)]['mean_z'][:]
        t_mean_dp = h5f['Bunches'][str(bunch_id)]['mean_dp'][:]
        t_epsn_x = h5f['Bunches'][str(bunch_id)]['epsn_x'][:]

        if bunch_id % 100 == 0:
            print bunch_id/10
        if data_mean_x is None:
            data_mean_x = np.zeros((len(t_mean_x),len(filling_scheme)))
            data_mean_y = np.zeros((len(t_mean_y),len(filling_scheme)))
            data_mean_z = np.zeros((len(t_mean_x),len(filling_scheme)))
            data_mean_dp = np.zeros((len(t_mean_x),len(filling_scheme)))
            data_epsn_x = np.zeros((len(t_mean_x),len(filling_scheme)))
        np.copyto(data_mean_x[:,i],t_mean_x)
        np.copyto(data_mean_y[:,i],t_mean_y)
        np.copyto(data_mean_z[:,i],t_mean_z)
        np.copyto(data_mean_dp[:,i],t_mean_dp)
        np.copyto(data_epsn_x[:,i],t_epsn_x)
        
    return data_mean_x, data_mean_y, data_mean_z,  data_epsn_x, data_mean_dp

def get_slice_data(datafile):
    h5f = h5py.File(datafile,'r')

    data_mean_x = h5f['Slices']['mean_x'][:]
    data_mean_z = h5f['Slices']['mean_z'][:]
    data_mean_dp = h5f['Slices']['mean_dp'][:]
    data_epsn_x = h5f['Slices']['epsn_x'][:]

    return data_mean_x, data_mean_z,  data_epsn_x, data_mean_dp


## Launches the simulations

import subprocess
print "start"
# The first argument is a number of processors given to the mpirun and the second one is the case index
subprocess.call("./run_local_job.sh 4 0", shell=True)
subprocess.call("./run_local_job.sh 4 1", shell=True)
print "end"


## Filling scheme
# **Must be exactly same as given in the PyHEADTAIL script!!**

h_RF = 274
h_RF = 156
# h_RF = 2748
filling_scheme = sorted([i for i in range(h_RF)])

n_bunches = len(filling_scheme)


## Reads data from the h5 files

data_set_0 = './data_case_0/'
data_set_1 = './data_case_1/'
filename = 'bunchmonitor_0000_chroma=0.h5'

datafile_0 = data_set_0 + filename
datafile_1 = data_set_1 + filename

# slice_filename = 'slicemonitor_bunch_255_0000_chroma=0.h5'

# slicefile_new = data_set_new + slice_filename
# slicefile_org = data_set_org + slice_filename

case_0_mean_x, case_0_mean_y, case_0_mean_z,  case_0_epsn_x, case_0_mean_dp = get_data(datafile_0, filling_scheme)
case_1_mean_x, case_1_mean_y, case_1_mean_z,  case_1_epsn_x, case_1_mean_dp = get_data(datafile_1, filling_scheme)

# slice_new_mean_x, slice_new_mean_z, slice_new_epsn_x, slice_new_mean_dp = get_slice_data(slicefile_new)
# slice_org_mean_x, slice_org_mean_z, slice_org_epsn_x, slice_org_mean_dp = get_slice_data(slicefile_org)

# os.remove(datafile_0)
# os.remove(datafile_1)


## Mean_x comparison

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Case 0")
ax1.set_xlabel('Turn')
ax1.set_ylabel('Bunch mean_x')

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Case 1")
ax2.set_ylabel('Bunch mean_x')

ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlabel('Turn')
ax3.set_ylabel('Position difference')

ax1.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])
ax2.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])
ax3.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])

for i in xrange(n_bunches):
    ax1.plot(case_0_mean_x[:,i]*1e6)
    ax2.plot(case_1_mean_x[:,i]*1e6)
    ax3.plot((case_0_mean_x[:,i]-case_1_mean_x[:,i])*1e6)
    
plt.tight_layout()
plt.show()


## Mean_y comparison

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Case 0")
ax1.set_xlabel('Turn')
ax1.set_ylabel('Bunch mean_y')

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Case 1")
ax2.set_ylabel('Bunch mean_y')

ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlabel('Turn')
ax3.set_ylabel('Position difference')

ax1.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])
ax2.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])
ax3.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])

for i in xrange(n_bunches):
    ax1.plot(case_0_mean_y[:,i]*1e6)
    ax2.plot(case_1_mean_y[:,i]*1e6)
    ax3.plot((case_0_mean_y[:,i]-case_1_mean_y[:,i])*1e6)
    
plt.tight_layout()
plt.show()


## Mean_dp comparison

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Case 0")
ax1.set_xlabel('Turn')
ax1.set_ylabel('Bunch mean_dp')

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Case 1")
ax2.set_ylabel('Bunch mean_dp')

ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlabel('Turn')
ax3.set_ylabel('Position difference')

ax1.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])
ax2.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])
ax3.set_color_cycle([plt.cm.viridis(i) for i in np.linspace(0, 1, n_bunches)])

for i in xrange(n_bunches):
    ax1.plot(case_0_mean_dp[:,i]*1e6)
    ax2.plot(case_1_mean_dp[:,i]*1e6)
    ax3.plot((case_0_mean_dp[:,i]-case_1_mean_dp[:,i])*1e6)
    
plt.tight_layout()
plt.show()

