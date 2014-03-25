'''
@author: Kevin Li
@date: 11.02.2014
'''


import h5py as hp
import numpy as np


from abc import ABCMeta, abstractmethod


class Monitor(object):

    @abstractmethod
    def dump(bunch):
        pass

class BunchMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        self.i_steps = 0

        self.h5file.create_group('Bunch')
        self.h5file.create_group('LSlice')
        self.h5file.create_group('RSlice')
        self.h5file.create_group('Slices')

    # def __del__(self):

    #     self.h5file.close()
    #     print "Closed!"
        
    def dump(self, bunch):

        if not self.i_steps:
            n_steps = self.n_steps
            n_slices = bunch.slices.n_slices

            self.create_data(self.h5file['Bunch'], (n_steps,))
            self.create_data(self.h5file['LSlice'], (n_steps,))
            self.create_data(self.h5file['RSlice'], (n_steps,))
            self.create_data(self.h5file['Slices'], (n_slices, n_steps))

            self.write_data(bunch, np.s_[-2], self.h5file['Bunch'], self.i_steps)
            self.write_data(bunch, np.s_[0], self.h5file['LSlice'], self.i_steps)
            self.write_data(bunch, np.s_[-3], self.h5file['RSlice'], self.i_steps)
            self.write_data(bunch, np.s_[1:-3], self.h5file['Slices'], self.i_steps, rank=2)
        else:
            self.write_data(bunch, np.s_[-2], self.h5file['Bunch'], self.i_steps)
            self.write_data(bunch, np.s_[0], self.h5file['LSlice'], self.i_steps)
            self.write_data(bunch, np.s_[-3], self.h5file['RSlice'], self.i_steps)
            self.write_data(bunch, np.s_[1:-3], self.h5file['Slices'], self.i_steps, rank=2)

        self.i_steps += 1

    def create_data(self, h5group, dims):

        h5group.create_dataset("mean_x", dims)
        h5group.create_dataset("mean_xp", dims)
        h5group.create_dataset("mean_y", dims)
        h5group.create_dataset("mean_yp", dims)
        h5group.create_dataset("mean_dz", dims)
        h5group.create_dataset("mean_dp", dims)
        h5group.create_dataset("sigma_x", dims)
        h5group.create_dataset("sigma_y", dims)
        h5group.create_dataset("sigma_dz", dims)
        h5group.create_dataset("sigma_dp", dims)
        h5group.create_dataset("epsn_x", dims)
        h5group.create_dataset("epsn_y", dims)
        h5group.create_dataset("epsn_z", dims)
        h5group.create_dataset("charge", dims)

    def write_data(self, bunch, indices, h5group, i_steps, rank=1):

        if rank == 1:
            h5group["mean_x"][i_steps] = bunch.slices.mean_x[indices]
            h5group["mean_xp"][i_steps] = bunch.slices.mean_xp[indices]
            h5group["mean_y"][i_steps] = bunch.slices.mean_y[indices]
            h5group["mean_yp"][i_steps] = bunch.slices.mean_yp[indices]
            h5group["mean_dz"][i_steps] = bunch.slices.mean_dz[indices]
            h5group["mean_dp"][i_steps] = bunch.slices.mean_dp[indices]
            h5group["sigma_x"][i_steps] = bunch.slices.sigma_x[indices]
            h5group["sigma_y"][i_steps] = bunch.slices.sigma_y[indices]
            h5group["sigma_dz"][i_steps] = bunch.slices.sigma_dz[indices]
            h5group["sigma_dp"][i_steps] = bunch.slices.sigma_dp[indices]
            h5group["epsn_x"][i_steps] = bunch.slices.epsn_x[indices]
            h5group["epsn_y"][i_steps] = bunch.slices.epsn_y[indices]
            h5group["epsn_z"][i_steps] = bunch.slices.epsn_z[indices]
            h5group["charge"][i_steps] = bunch.slices.charge[indices]
        elif rank == 2:
            h5group["mean_x"][:,i_steps] = bunch.slices.mean_x[indices]
            h5group["mean_xp"][:,i_steps] = bunch.slices.mean_xp[indices]
            h5group["mean_y"][:,i_steps] = bunch.slices.mean_y[indices]
            h5group["mean_yp"][:,i_steps] = bunch.slices.mean_yp[indices]
            h5group["mean_dz"][:,i_steps] = bunch.slices.mean_dz[indices]
            h5group["mean_dp"][:,i_steps] = bunch.slices.mean_dp[indices]
            h5group["sigma_x"][:,i_steps] = bunch.slices.sigma_x[indices]
            h5group["sigma_y"][:,i_steps] = bunch.slices.sigma_y[indices]
            h5group["sigma_dz"][:,i_steps] = bunch.slices.sigma_dz[indices]
            h5group["sigma_dp"][:,i_steps] = bunch.slices.sigma_dp[indices]
            h5group["epsn_x"][:,i_steps] = bunch.slices.epsn_x[indices]
            h5group["epsn_y"][:,i_steps] = bunch.slices.epsn_y[indices]
            h5group["epsn_z"][:,i_steps] = bunch.slices.epsn_z[indices]
            h5group["charge"][:,i_steps] = bunch.slices.charge[indices]
        else:
            raise ValueError("Rank > 2 not supported!")

class ParticleMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.h5file = hp.File(filename + '.h5part', 'w')
        self.n_steps = n_steps
        self.i_steps = 0

    def dump(self, bunch):

        if not self.i_steps:
            self.z0 = np.copy(bunch.dz)

        n_particles = len(bunch.x)
        h5group = self.h5file.create_group("Step#" + str(self.i_steps))
        self.create_data(h5group, (n_particles,))
        self.write_data(bunch, h5group)

        self.i_steps += 1

    def create_data(self, h5group, dims):

        h5group.create_dataset("x", dims)
        h5group.create_dataset("xp", dims)
        h5group.create_dataset("y", dims)
        h5group.create_dataset("yp", dims)
        h5group.create_dataset("dz", dims)
        h5group.create_dataset("dp", dims)
        h5group.create_dataset("identity", dims)

        h5group.create_dataset("c", dims)

    def write_data(self, bunch, h5group):

        h5group["x"][:] = bunch.x
        h5group["xp"][:] = bunch.xp
        h5group["y"][:] = bunch.y
        h5group["yp"][:] = bunch.yp
        h5group["dz"][:] = bunch.dz
        h5group["dp"][:] = bunch.dp
        h5group["identity"][:] = bunch.identity
        
        h5group["c"][:] = self.z0
