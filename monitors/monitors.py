'''
@author: Kevin Li, Michael Schenk
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

    def __init__(self, filename, n_steps, dictionary=None):
        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        self.i_steps = 0

        if dictionary:
            for key in dictionary:
                self.h5file.attrs[key] = dictionary[key]

        self.h5file.create_group('Bunch')

    def dump(self, bunch):
        # This method may be called several times in different places of the code. Ok. for now.
        bunch.compute_statistics()
        
        if not self.i_steps:
            n_steps = self.n_steps
            self.create_data(self.h5file['Bunch'], (n_steps,))
            self.write_data(bunch, self.h5file['Bunch'], self.i_steps)
        else:
            self.write_data(bunch, self.h5file['Bunch'], self.i_steps)

        self.i_steps += 1

    def create_data(self, h5group, dims):
        h5group.create_dataset("mean_x",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_xp",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_y",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_yp",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_z",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_dp",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_x",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_y",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_z",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_dp", dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_x",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_y",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_z",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("n_macroparticles", dims, compression="gzip", compression_opts=9)

    def write_data(self, bunch, h5group, i_steps):
        h5group["mean_x"][i_steps]   = bunch.mean_x
        h5group["mean_xp"][i_steps]  = bunch.mean_xp
        h5group["mean_y"][i_steps]   = bunch.mean_y
        h5group["mean_yp"][i_steps]  = bunch.mean_yp
        h5group["mean_z"][i_steps]   = bunch.mean_z
        h5group["mean_dp"][i_steps]  = bunch.mean_dp
        h5group["sigma_x"][i_steps]  = bunch.sigma_x
        h5group["sigma_y"][i_steps]  = bunch.sigma_y
        h5group["sigma_z"][i_steps]  = bunch.sigma_z
        h5group["sigma_dp"][i_steps] = bunch.sigma_dp
        h5group["epsn_x"][i_steps]   = bunch.epsn_x
        h5group["epsn_y"][i_steps]   = bunch.epsn_y
        h5group["epsn_z"][i_steps]   = bunch.epsn_z
        h5group["n_macroparticles"][i_steps] = bunch.n_macroparticles

    def close(self):
        self.h5file.close()

        
class SliceMonitor(Monitor):

    def __init__(self, filename, n_steps, slices=None, dictionary=None):
        self.h5file  = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        self.slices  = slices
        self.i_steps = 0

        if dictionary:
            for key in dictionary:
                self.h5file.attrs[key] = dictionary[key]
        
        self.h5file.create_group('Bunch')
        self.h5file.create_group('Slices')

    def dump(self, bunch):
        if not self.slices:
            self.slices = bunch.slices

        # These methods may be called several times in different places of the code. Ok. for now.
        bunch.compute_statistics()
        self.slices.update_slices(bunch)
        self.slices.compute_statistics(bunch)
        
        if not self.i_steps:
            n_steps = self.n_steps
            n_slices = self.slices.n_slices

            self.create_data(self.h5file['Bunch'],  (n_steps,))
            self.create_data(self.h5file['Slices'], (n_slices, n_steps))

            self.write_data(bunch, self.h5file['Bunch'], self.i_steps)
            self.write_data(self.slices, self.h5file['Slices'], self.i_steps, rank=2)
        else:
            self.write_data(bunch, self.h5file['Bunch'], self.i_steps)
            self.write_data(self.slices, self.h5file['Slices'], self.i_steps, rank=2)

        self.i_steps += 1

    def create_data(self, h5group, dims):
        h5group.create_dataset("mean_x",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_xp",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_y",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_yp",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_z",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_dp",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_x",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_y",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_z",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_dp", dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_x",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_y",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_z",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("n_macroparticles", dims, compression="gzip", compression_opts=9)

    def write_data(self, data, h5group, i_steps, rank=1):
        if rank == 1:
            h5group["mean_x"][i_steps]   = data.mean_x
            h5group["mean_xp"][i_steps]  = data.mean_xp
            h5group["mean_y"][i_steps]   = data.mean_y
            h5group["mean_yp"][i_steps]  = data.mean_yp
            h5group["mean_z"][i_steps]   = data.mean_z
            h5group["mean_dp"][i_steps]  = data.mean_dp
            h5group["sigma_x"][i_steps]  = data.sigma_x
            h5group["sigma_y"][i_steps]  = data.sigma_y
            h5group["sigma_z"][i_steps]  = data.sigma_z
            h5group["sigma_dp"][i_steps] = data.sigma_dp
            h5group["epsn_x"][i_steps]   = data.epsn_x
            h5group["epsn_y"][i_steps]   = data.epsn_y
            h5group["epsn_z"][i_steps]   = data.epsn_z
            h5group["n_macroparticles"][i_steps] = data.n_macroparticles
        elif rank == 2:
            h5group["mean_x"][:,i_steps]   = data.mean_x
            h5group["mean_xp"][:,i_steps]  = data.mean_xp
            h5group["mean_y"][:,i_steps]   = data.mean_y
            h5group["mean_yp"][:,i_steps]  = data.mean_yp
            h5group["mean_z"][:,i_steps]   = data.mean_z
            h5group["mean_dp"][:,i_steps]  = data.mean_dp
            h5group["sigma_x"][:,i_steps]  = data.sigma_x
            h5group["sigma_y"][:,i_steps]  = data.sigma_y
            h5group["sigma_z"][:,i_steps]  = data.sigma_z
            h5group["sigma_dp"][:,i_steps] = data.sigma_dp
            h5group["epsn_x"][:,i_steps]   = data.epsn_x
            h5group["epsn_y"][:,i_steps]   = data.epsn_y
            h5group["epsn_z"][:,i_steps]   = data.epsn_z
            h5group["n_macroparticles"][:,i_steps] = data.n_macroparticles
        else:
            raise ValueError("Rank > 2 not supported!")

    def close(self):
        self.h5file.close()
        
        
class ParticleMonitor(Monitor):

    def __init__(self, filename, stride=1, dictionary=None):

        self.h5file = hp.File(filename + '.h5part', 'w')
        if dictionary:
            for key in dictionary:
                self.h5file.attrs[key] = dictionary[key]

        self.stride = stride
        # self.n_steps = n_steps
        self.i_steps = 0

    def dump(self, bunch):

        if not self.i_steps:
            resorting_indices = np.argsort(bunch.id)[::self.stride]
            self.z0 = np.copy(bunch.z[resorting_indices])

        h5group = self.h5file.create_group("Step#" + str(self.i_steps))
        self.create_data(h5group, (bunch.n_macroparticles // self.stride,))
        self.write_data(bunch, h5group)

        self.i_steps += 1

    def create_data(self, h5group, dims):

        h5group.create_dataset("x",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("xp", dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("y",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("yp", dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("z",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("dp", dims, compression="gzip", compression_opts=9)

        # Do we need/want this here?
        h5group.create_dataset("id", dims, dtype=np.int)

        h5group.create_dataset("c", dims)

    def write_data(self, bunch, h5group):

        resorting_indices = np.argsort(bunch.id)[::self.stride]

        h5group["x"][:]  = bunch.x[resorting_indices]
        h5group["xp"][:] = bunch.xp[resorting_indices]
        h5group["y"][:]  = bunch.y[resorting_indices]
        h5group["yp"][:] = bunch.yp[resorting_indices]
        h5group["z"][:]  = bunch.z[resorting_indices]
        h5group["dp"][:] = bunch.dp[resorting_indices]

        # Do we need/want this here?
        h5group["id"][:] = bunch.id[resorting_indices]

        h5group["c"][:] = self.z0

    def close(self):
        self.h5file.close()
