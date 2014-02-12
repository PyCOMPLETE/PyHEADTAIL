'''
@author: Kevin Li
@date: 11.02.2014
'''


import h5py as hp
import numpy as np


from abc import ABCMeta, abstractmethod


class Monitor():

    @abstractmethod
    def dump(bunch):
        pass

class BunchMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        self.i_steps = 0

    def dump(self, bunch, i_bunch):

        bunchname = 'Bunch_{:04d}'.format(i_bunch)
        try:
            self.h5file[bunchname]
        except KeyError:
            self.h5file.create_group(bunchname)
            self.create_data(i_bunch)
        finally:
            self.mean_x[self.i_steps] = bunch.slices.mean_x[-1]
            self.mean_xp[self.i_steps] = bunch.slices.mean_xp[-1]
            self.mean_y[self.i_steps] = bunch.slices.mean_y[-1]
            self.mean_yp[self.i_steps] = bunch.slices.mean_yp[-1]
            self.mean_dz[self.i_steps] = bunch.slices.mean_dz[-1]
            self.mean_dp[self.i_steps] = bunch.slices.mean_dp[-1]
            self.sigma_x[self.i_steps] = bunch.slices.sigma_x[-1]
            self.sigma_y[self.i_steps] = bunch.slices.sigma_y[-1]
            self.sigma_dz[self.i_steps] = bunch.slices.sigma_dz[-1]
            self.sigma_dp[self.i_steps] = bunch.slices.sigma_dp[-1]
            self.epsn_x[self.i_steps] = bunch.slices.epsn_x[-1]
            self.epsn_y[self.i_steps] = bunch.slices.epsn_y[-1]
            self.epsn_z[self.i_steps] = bunch.slices.epsn_z[-1]

        self.i_steps += 1

    def create_data(self, i_bunch):

        bunchname = "Bunch_{:04d}".format(i_bunch)
        self.mean_x = self.h5file[bunchname].create_dataset("mean_x", (self.n_steps,))
        self.mean_xp = self.h5file[bunchname].create_dataset("mean_xp", (self.n_steps,))
        self.mean_y = self.h5file[bunchname].create_dataset("mean_y", (self.n_steps,))
        self.mean_yp = self.h5file[bunchname].create_dataset("mean_yp", (self.n_steps,))
        self.mean_dz = self.h5file[bunchname].create_dataset("mean_dz", (self.n_steps,))
        self.mean_dp = self.h5file[bunchname].create_dataset("mean_dp", (self.n_steps,))
        self.sigma_x = self.h5file[bunchname].create_dataset("sigma_x", (self.n_steps,))
        self.sigma_y = self.h5file[bunchname].create_dataset("sigma_y", (self.n_steps,))
        self.sigma_dz = self.h5file[bunchname].create_dataset("sigma_dz", (self.n_steps,))
        self.sigma_dp = self.h5file[bunchname].create_dataset("sigma_dp", (self.n_steps,))
        self.epsn_x = self.h5file[bunchname].create_dataset("epsn_x", (self.n_steps,))
        self.epsn_y = self.h5file[bunchname].create_dataset("epsn_y", (self.n_steps,))
        self.epsn_z = self.h5file[bunchname].create_dataset("epsn_z", (self.n_steps,))
        

class SliceMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        
    def dump(self, bunch):
        pass

class ParticleMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.h5file = hp.File(filename + '.h5part', 'w')
        self.n_steps = n_steps

    def dump(self, bunch):
        pass
