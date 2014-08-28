'''
@author: Kevin Li, Michael Schenk
@date: 11.02.2014
'''

import h5py as hp
import numpy as np
import sys

from abc import ABCMeta, abstractmethod


class Monitor(object):

    @abstractmethod
    def dump(bunch):
        pass


class BunchMonitor(Monitor):

    def __init__(self, filename, n_steps, dictionary=None):

        self.stats_quantities = [ 'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
                                  'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y', 'epsn_z',
                                  'n_macroparticles' ]
        self.n_steps  = n_steps
        self.i_steps  = 0
        self.filename = filename

        self.buffer_size = 2048
        self.write_buffer_to_file_every = 1024
        self.buffer = np.zeros((len(self.stats_quantities), self.buffer_size))

        self._create_file_structure(dictionary)


    def dump(self, bunch):

        self._write_data_to_buffer(bunch)
        if (self.i_steps + 1) % self.write_buffer_to_file_every == 0 or (self.i_steps + 1) == self.n_steps:
            self._write_buffer_to_file()

        self.i_steps += 1


    def _create_file_structure(self, dictionary):

        try:
            h5file = hp.File(self.filename + '.h5', 'w')

            if dictionary:
                for key in dictionary:
                    h5file.attrs[key] = dictionary[key]

            h5file.create_group('Bunch')
            h5group = h5file['Bunch']
            for stats in sorted(self.stats_quantities):
                h5group.create_dataset(stats, shape=(self.n_steps,), compression='gzip', compression_opts=9)

            h5file.close()
        except:
            print 'Creation of bunch monitor file failed.'
            sys.exit(-1)
            

    def _write_data_to_buffer(self, bunch):

        for k, stats in enumerate(sorted(self.stats_quantities)):
            evaluate_stats = getattr(bunch, stats)
            try:
                self.buffer[k,0] = evaluate_stats()
            except TypeError:
                self.buffer[k,0] = evaluate_stats

        self.buffer = np.roll(self.buffer, shift=-1, axis=1)
        

    def _write_buffer_to_file(self):

        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer   = self.buffer_size - n_entries_in_buffer
        low_pos_in_file     = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file      = self.i_steps + 1

        try:       
            h5file  = hp.File(self.filename + '.h5', 'a')
            h5group = h5file['Bunch']
            for k, stats in enumerate(sorted(self.stats_quantities)):
                h5group[stats][low_pos_in_file:up_pos_in_file] = self.buffer[k, low_pos_in_buffer:]
            h5file.close()
            
        except:
            print 'Bunch monitor file is not accessible.'

            
class SliceMonitor(Monitor):

    def __init__(self, filename, n_steps, dictionary=None, slices=None):
        
        self.stats_quantities = [ 'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
                                  'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y', 'epsn_z',
                                  'n_macroparticles' ]
        self.n_steps  = n_steps
        self.i_steps  = 0
        self.filename = filename
        self.slices   = slices

        self.buffer_size = 2048
        self.write_buffer_to_file_every = 1024
        self.buffer_bunch = np.zeros((len(self.stats_quantities), self.buffer_size))
        self.buffer_slice = np.zeros((len(self.stats_quantities), self.slices.n_slices, self.buffer_size))

        self._create_file_structure(dictionary)


    def dump(self, bunch):

        self._write_data_to_buffer(bunch)
        if (self.i_steps + 1) % self.write_buffer_to_file_every == 0 or (self.i_steps + 1) == self.n_steps:
            self._write_buffer_to_file()

        self.i_steps += 1


    def _create_file_structure(self, dictionary):
                
        try:
            h5file = hp.File(self.filename + '.h5', 'w')

            if dictionary:
                for key in dictionary:
                    h5file.attrs[key] = dictionary[key]

            h5file.create_group('Bunch')
            h5file.create_group('Slices')        
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']
            for stats in sorted(self.stats_quantities):
                h5group_bunch.create_dataset(stats, shape=(self.n_steps,), compression='gzip', compression_opts=9)
                h5group_slice.create_dataset(stats, shape=(self.slices.n_slices, self.n_steps), compression='gzip', compression_opts=9)
            h5file.close()
            
        except:
            print 'Creation of slice monitor file failed.'
            sys.exit(-1)


    def _write_data_to_buffer(self, bunch):

        for k, stats in enumerate(sorted(self.stats_quantities)):
            evaluate_stats_bunch = getattr(bunch, stats)
            evaluate_stats_slice = getattr(self.slices, stats)
            try:
                self.buffer_bunch[k,0] = evaluate_stats_bunch()
                self.buffer_slice[k,:,0] = evaluate_stats_slice(bunch)
            except TypeError:
                self.buffer_bunch[k,0] = evaluate_stats_bunch
                self.buffer_slice[k,:,0] = evaluate_stats_slice
                
        self.buffer_bunch = np.roll(self.buffer_bunch, shift=-1, axis=1)
        self.buffer_slice = np.roll(self.buffer_slice, shift=-1, axis=2)
        

    def _write_buffer_to_file(self):

        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer   = self.buffer_size - n_entries_in_buffer
        low_pos_in_file     = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file      = self.i_steps + 1

        try:       
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']
            for k, stats in enumerate(sorted(self.stats_quantities)):
                h5group_bunch[stats][low_pos_in_file:up_pos_in_file]   = self.buffer_bunch[k, low_pos_in_buffer:]
                h5group_slice[stats][:,low_pos_in_file:up_pos_in_file] = self.buffer_slice[k,:,low_pos_in_buffer:]
            h5file.close()
        except:
            print 'Slices monitor file is not accessible.'


class ParticleMonitor(Monitor):

    def __init__(self, filename, stride=1, dictionary=None, slices=None):

        self.slices = slices
        self.stride = stride
        self.i_steps = 0

        self.h5file = hp.File(filename + '.h5part', 'w')
        if dictionary:
            for key in dictionary:
                self.h5file.attrs[key] = dictionary[key]

        self.h5file.flush()


    def dump(self, bunch):

        if not self.i_steps:
            resorting_indices = np.argsort(bunch.id)[::self.stride]
            self.z0 = np.copy(bunch.z[resorting_indices])

        h5group = self.h5file.create_group("Step#" + str(self.i_steps))
        self._create_data(h5group, (bunch.n_macroparticles // self.stride,))
        self._write_data(h5group, bunch)

        self.i_steps += 1
        self.h5file.flush()


    def _create_data(self, h5group, dims):

        h5group.create_dataset("x",           dims, compression="gzip", compression_opts=9, dtype=np.float64)
        h5group.create_dataset("xp",          dims, compression="gzip", compression_opts=9, dtype=np.float64)
        h5group.create_dataset("y",           dims, compression="gzip", compression_opts=9, dtype=np.float64)
        h5group.create_dataset("yp",          dims, compression="gzip", compression_opts=9, dtype=np.float64)
        h5group.create_dataset("z",           dims, compression="gzip", compression_opts=9, dtype=np.float64)
        h5group.create_dataset("dp",          dims, compression="gzip", compression_opts=9, dtype=np.float64)
        h5group.create_dataset("slice_index", dims, compression="gzip", compression_opts=9, dtype=np.float64)

        # Do we need/want this here?
        h5group.create_dataset("id", dims, dtype=np.int)
        h5group.create_dataset("c",  dims)


    def _write_data(self, h5group, bunch):

        resorting_indices = np.argsort(bunch.id)[::self.stride]
    
        h5group["x"][:]           = bunch.x[resorting_indices]
        h5group["xp"][:]          = bunch.xp[resorting_indices]
        h5group["y"][:]           = bunch.y[resorting_indices]
        h5group["yp"][:]          = bunch.yp[resorting_indices]
        h5group["z"][:]           = bunch.z[resorting_indices]
        h5group["dp"][:]          = bunch.dp[resorting_indices]

        particle_id = bunch.id[resorting_indices]
        if self.slices:
            h5group["slice_index"][:] = self.slices.slice_index_of_particle[particle_id]

        # Do we need/want this here?
        h5group["id"][:] = particle_id
        h5group["c"][:] = self.z0
