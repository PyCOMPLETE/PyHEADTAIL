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

        self.stats_to_store = [ 'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
                                'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y', 'epsn_z',
                                'n_macroparticles' ]
        self.filename = filename
        self.n_steps  = n_steps
        self.i_steps  = 0

        self.buffer_size = 4096
        self.write_buffer_to_file_every = 512
        self.buffer = {}
        for stats in self.stats_to_store:
            self.buffer[stats] = np.zeros(self.buffer_size)

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
            for stats in sorted(self.stats_to_store):
                h5group.create_dataset(stats, shape=(self.n_steps,), compression='gzip', compression_opts=9)

            h5file.close()
        except:
            print 'Creation of bunch monitor file failed.'
            sys.exit(-1)


    def _write_data_to_buffer(self, bunch):

        for stats in self.stats_to_store:
            evaluate_stats = getattr(bunch, stats)
       
            try:
                self.buffer[stats][0] = evaluate_stats()
            except TypeError:
                self.buffer[stats][0] = evaluate_stats

            self.buffer[stats] = np.roll(self.buffer[stats], shift=-1, axis=0)


    def _write_buffer_to_file(self):

        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer   = self.buffer_size - n_entries_in_buffer
        low_pos_in_file     = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file      = self.i_steps + 1

        try:       
            h5file  = hp.File(self.filename + '.h5', 'a')
            h5group = h5file['Bunch']
            for stats in self.stats_to_store:
                h5group[stats][low_pos_in_file:up_pos_in_file] = self.buffer[stats][low_pos_in_buffer:]
            h5file.close()
            
        except:
            print 'Bunch monitor file is not accessible.'

            
class SliceMonitor(Monitor):

    def __init__(self, filename, n_steps, dictionary=None, slices=None):
        
        self.stats_to_store = [ 'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
                                'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y', 'epsn_z',
                                'n_macroparticles' ]
        self.filename = filename
        self.n_steps  = n_steps
        self.i_steps  = 0
        self.slices   = slices

        self.buffer_size = 4096
        self.write_buffer_to_file_every = 512
        self.buffer_bunch = {}
        self.buffer_slice = {}
        for stats in self.stats_to_store:
            self.buffer_bunch[stats] = np.zeros(self.buffer_size)
            self.buffer_slice[stats] = np.zeros((self.slices.n_slices, self.buffer_size))
        
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
            for stats in sorted(self.stats_to_store):
                h5group_bunch.create_dataset(stats, shape=(self.n_steps,), compression='gzip', compression_opts=9)
                h5group_slice.create_dataset(stats, shape=(self.slices.n_slices, self.n_steps), compression='gzip', compression_opts=9)
            h5file.close()
            
        except:
            print 'Creation of slice monitor file failed.'
            sys.exit(-1)


    def _write_data_to_buffer(self, bunch):

        for stats in self.stats_to_store:
            evaluate_stats_bunch = getattr(bunch, stats)
            evaluate_stats_slice = getattr(self.slices, stats)

            try:
                self.buffer_bunch[stats][0]   = evaluate_stats_bunch()
            except TypeError:
                self.buffer_bunch[stats][0]   = evaluate_stats_bunch

            try:
                self.buffer_slice[stats][:,0] = evaluate_stats_slice(bunch)
            except TypeError:
                self.buffer_slice[stats][:,0] = evaluate_stats_slice

            self.buffer_bunch[stats] = np.roll(self.buffer_bunch[stats], shift=-1, axis=0)
            self.buffer_slice[stats] = np.roll(self.buffer_slice[stats], shift=-1, axis=1)


    def _write_buffer_to_file(self):

        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer   = self.buffer_size - n_entries_in_buffer
        low_pos_in_file     = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file      = self.i_steps + 1

        try:       
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']
            for stats in self.stats_to_store:
                h5group_bunch[stats][low_pos_in_file:up_pos_in_file]   = self.buffer_bunch[stats][low_pos_in_buffer:]
                h5group_slice[stats][:,low_pos_in_file:up_pos_in_file] = self.buffer_slice[stats][:,low_pos_in_buffer:]
            h5file.close()
        except:
            print 'Slices monitor file is not accessible.'


class ParticleMonitor(Monitor):

    def __init__(self, filename, stride=1, dictionary=None): #, slices=None):

        self.quantities_to_store = [ 'x', 'xp', 'y', 'yp', 'z', 'dp', 'id' ]
        self.filename = filename
        self.stride   = stride
        self.i_steps  = 0
        # self.slices = slices
        self._create_file_structure(dictionary)


    def dump(self, bunch):

        self._write_data_to_file(bunch)
        self.i_steps += 1


    def _create_file_structure(self, dictionary):

        h5file = hp.File(self.filename + '.h5part', 'w')
        if dictionary:
            for key in dictionary:
                h5file.attrs[key] = dictionary[key]
        h5file.close()


    def _write_data_to_file(self, bunch):

        h5file  = hp.File(self.filename + '.h5part', 'a')
        h5group = h5file.create_group('Step#' + str(self.i_steps))
        dims    = (bunch.n_macroparticles // self.stride,)
        
        resorting_indices = np.argsort(bunch.id)[::self.stride]
        for quant in self.quantities_to_store:
            quant_values = getattr(bunch, quant)
            h5group.create_dataset(quant, shape=dims, compression='gzip', compression_opts=9, dtype=quant_values.dtype)
            h5group[quant][:] = quant_values[resorting_indices]

        h5file.close()
            
        # h5group.create_dataset("slice_index", dims, compression="gzip", compression_opts=9, dtype=np.float64)
        # h5group.create_dataset("c",  dims)

        # if self.slices:
            # h5group["slice_index"][:] = self.slices.slice_index_of_particle[particle_id]

        # Do we need/want this here?
        # h5group["id"][:] = particle_id
        # h5group["c"][:] = self.z0
