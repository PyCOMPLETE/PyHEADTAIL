"""
@author Kevin Li, Michael Schenk, Stefan Hegglin
@date 11. February 2014
@brief Implementation of monitors to store bunch-, slice- or particle-
       specific data to a HDF5 file.
@copyright CERN
"""
from __future__ import division

import h5py as hp
import numpy as np
import sys

from abc import ABCMeta, abstractmethod
from . import Printing
from ..general import pmath as pm
from ..gpu import gpu_utils as gpu_utils
from ..general import decorators as decorators

from ..cobra_functions import stats as cp

# from .. import cobra_functions.stats.calc_cell_stats as calc_cell_stats


class Monitor(Printing):
    """ Abstract base class for monitors. A monitor can request
    statistics data such as mean value and standard deviation and store
    the results in an HDF5 file. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def dump(bunch):
        """ Write particle data given by bunch (instance of Particles
        class) to buffer and/or file at the specific time the method is
        called. Data can e.g. be bunch-specific, slice_set-specific or
        particle-specific. """
        pass


class BunchMonitor(Monitor):
    """ Class to store bunch-specific data to a HDF5 file. This monitor
    uses a buffer (a shift register) to reduce the number of writing
    operations to file. This also helps to avoid IO errors and loss of
    data when writing to a file that may become temporarily unavailable
    (e.g. if file is located on network) during the simulation. """
    def __init__(self, filename, n_steps, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096,
                 *args, **kwargs):
        """ Create an instance of a BunchMonitor class. Apart from
        initializing the HDF5 file, a self.buffer dictionary is
        prepared to buffer the data before writing them to file.

          filename:           Path and name of HDF5 file. Without file
                              extension.
          n_steps:            Number of entries to be reserved for each
                              of the quantities in self.stats_to_store.
          parameters_dict:    Metadata for HDF5 file containing main
                              simulation parameters.
          write_buffer_every: Number of steps after which buffer
                              contents are actually written to file.
          buffer_size:        Number of steps to be buffered.

          Optionally pass a list called stats_to_store which specifies
          which members/methods of the bunch will be called/stored.
          """
        stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'macroparticlenumber' ]
        self.stats_to_store = kwargs.pop('stats_to_store', stats_to_store)
        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0

        self._create_file_structure(parameters_dict)

        # Prepare buffer.
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every
        self.buffer = None

    def _init_buffer(self, bunch):
        '''
        Init the correct buffer type (np.zeros, gpuarrays.zeros)
        '''
        self.buffer = pm.init_bunch_buffer(bunch, self.stats_to_store,
                                           self.buffer_size)

    def dump(self, bunch):
        """ Evaluate the statistics like mean and standard deviation for
        the given bunch and write the data to the HDF5 file. Make use of
        a buffer to reduce the number of writing operations to file.
        This helps to avoid IO errors and loss of data when writing data
        to a file that may become temporarily unavailable (e.g. if file
        is on network). during the simulation. Buffer contents are
        written to file only every self.write_buffer_every steps.
        The buffer gets initialized in the first dump() call. This allows
        for a dynamic creation of the buffer memory on either CPU or GPU"""
        if self.buffer is None:
            self._init_buffer(bunch)
        self._write_data_to_buffer(bunch)
        if ((self.i_steps + 1) % self.write_buffer_every == 0 or
                (self.i_steps + 1) == self.n_steps):
            self._write_buffer_to_file()

        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file and create its basic structure (groups
        and datasets). One group is created for bunch-specific data.
        One dataset for each of the quantities defined in
        self.stats_to_store is generated.
        If specified by the user, write the contents of the
        parameters_dict as metadata (attributes) to the file.
        Maximum file compression is activated. """
        try:
            h5file = hp.File(self.filename + '.h5', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]

            h5file.create_group('Bunch')
            h5group = h5file['Bunch']
            for stats in sorted(self.stats_to_store):
                h5group.create_dataset(stats, shape=(self.n_steps,),
                                       compression='gzip', compression_opts=9)
            h5file.close()
        except Exception as err:
            self.warns('Problem occurred during Bunch monitor creation.')
            self.warns(err.message)
            raise

    @decorators.synchronize_gpu_streams_after
    def _write_data_to_buffer(self, bunch):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. """
        val_buf = {}
        p_write = {}
        for stats in self.stats_to_store:
            evaluate_stats = getattr(bunch, stats)

            # Handle the different statistics quantities, which can
            # either be methods (like mean(), ...) or simply attributes
            # (macroparticlenumber) of the bunch.
            write_pos = self.i_steps % self.buffer_size
            try:
                if pm.device is 'is_.2slowerwiththis':#'GPU':
                    #val_bf[stat]
                    st = next(gpu_utils.stream_pool)
                    val_buf[stats] = evaluate_stats(stream=st)
                    p_write[stats] = int(self.buffer[stats].gpudata) + write_pos*self.buffer[stats].strides[0]
                    sze = 8#val.nbytes
                    gpu_utils.driver.memcpy_dtod_async(dest=p_write[stats], src=val_buf[stats].gpudata, size=sze, stream=st)
                else:
                    self.buffer[stats][write_pos] = evaluate_stats()
            except TypeError:
                self.buffer[stats][write_pos] = evaluate_stats

    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened and
        closed each time the buffer is written to file to prevent from
        loss of data in case of a crash.
        buffer_tmp is an extra buffer which is always on the CPU. If
        self.buffer is on the GPU, copy the data to buffer_tmp and write
        the result to the file."""

        buffer_tmp = {} # always on CPU
        shift = - (self.i_steps + 1 % self.buffer_size)
        for stats in self.stats_to_store:
            try:
                buffer_tmp[stats] = np.roll(self.buffer[stats].get(),
                        shift=shift, axis=0)
            except:
                buffer_tmp[stats] = np.roll(self.buffer[stats].copy(),
                        shift=shift, axis=0)
        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer = self.buffer_size - n_entries_in_buffer
        low_pos_in_file = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file = self.i_steps + 1

        # Try to write data to file. If file is not available, skip this
        # step and repeat it again after self.write_buffer_every. As
        # long as self.buffer_size is not exceeded, no data are lost.
        try:
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group = h5file['Bunch']
            for stats in self.stats_to_store:
                h5group[stats][low_pos_in_file:up_pos_in_file] = \
                    buffer_tmp[stats][low_pos_in_buffer:]
            h5file.close()
        except IOError:
            self.warns('Bunch monitor file is temporarily unavailable. \n')


class SliceMonitor(Monitor):
    """ Class to store bunch- and slice_set-specific data to a HDF5
    file. This monitor uses two buffers (shift registers) to reduce the
    number of writing operations to file. This also helps to avoid IO
    errors and loss of data when writing to a file that may become
    temporarily unavailable (e.g. if file is located on network) during
    the simulation. """

    def __init__(self, filename, n_steps, slicer, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096,
                 *args, **kwargs):
        """ Create an instance of a SliceMonitor class. Apart from
        initializing the HDF5 file, two buffers self.buffer_bunch and
        self.buffer_slice are prepared to buffer the bunch-specific and
        slice_set-specific data before writing them to file.

          filename:           Path and name of HDF5 file. Without file
                              extension.
          n_steps:            Number of entries to be reserved for each
                              of the quantities in self.stats_to_store.
          slicer:             Instance of the Slicer class containing
                              the configuration defining a slice_set.
          parameters_dict:    Metadata for HDF5 file containing main
                              simulation parameters.
          write_buffer_every: Number of steps after which buffer
                              contents are actually written to file.
          buffer_size:        Number of steps to be buffered.

          optionally pass a list called bunch_stats_to_store or
          slice_stats_to_store which specifie
          which members/methods of the bunch will be called/stored.
        """
        bunch_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'macroparticlenumber' ]
        slice_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'n_macroparticles_per_slice' ]

        self.bunch_stats_to_store = kwargs.pop('bunch_stats_to_store',
                bunch_stats_to_store)
        self.slice_stats_to_store = kwargs.pop('slice_stats_to_store',
                slice_stats_to_store)

        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0
        self.slicer = slicer

        # Prepare buffers.
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every
        self.buffer_bunch = None
        self.buffer_slice = None
        self._create_file_structure(parameters_dict)

    def _init_buffer(self, bunch, slice_set):
        self.buffer_bunch = pm.init_bunch_buffer(bunch,
            self.bunch_stats_to_store, self.buffer_size)
        self.buffer_slice = pm.init_slice_buffer(slice_set,
            self.slice_stats_to_store, self.buffer_size)

    def dump(self, bunch):
        """ Evaluate the statistics like mean and standard deviation for
        the given slice_set and the bunch and write the data to the
        buffers and/or to the HDF5 file. The buffers are used to reduce
        the number of writing operations to file. This helps to avoid IO
        errors and loss of data when writing data to a file that may
        become temporarily unavailable (e.g. if file is on network)
        during the simulation. Buffer contents are written to file only
        every self.write_buffer_every steps. """
        self._write_data_to_buffer(bunch)
        if ((self.i_steps + 1) % self.write_buffer_every == 0 or
                (self.i_steps + 1) == self.n_steps):
            self._write_buffer_to_file()

        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file and create its basic structure (groups
        and datasets). Two groups are created, one for slice_set-
        specific and one for bunch-specific data. One dataset for each
        of the quantities defined in self.bunch_stats_to_store and
        self.slice_stats_to_store resp. is generated. If specified by
        the user, write the contents of the parameters_dict as metadata
        (attributes) to the file. Maximum file compression is
        activated. """
        try:
            h5file = hp.File(self.filename + '.h5', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]

            h5file.create_group('Bunch')
            h5file.create_group('Slices')
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']

            for stats in self.bunch_stats_to_store:
                h5group_bunch.create_dataset(stats, shape=(self.n_steps,),
                    compression='gzip', compression_opts=9)
            for stats in self.slice_stats_to_store:
                h5group_slice.create_dataset(stats, shape=(self.slicer.n_slices,
                    self.n_steps), compression='gzip', compression_opts=9)
            h5file.close()
        except Exception as err:
            self.warns('Problem occurred during Slice monitor creation.')
            self.warns(err.message)
            raise

    def _write_data_to_buffer(self, bunch):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. To
        find the slice_set-specific data, a slice_set, defined by the
        slicing configuration self.slicer must be requested from the
        bunch (instance of the Particles class), including all the
        statistics that are to be saved. """
        slice_set = bunch.get_slices(self.slicer, statistics=True)
        if self.buffer_bunch is None:
            self._init_buffer(bunch, slice_set)
        # Handle the different statistics quantities, which can
        # either be methods (like mean(), ...) or simply attributes
        # (macroparticlenumber or n_macroparticles_per_slice) of the bunch
        # or slice_set resp.

        # bunch-specific data.
        write_pos = self.i_steps % self.buffer_size
        for stats in self.bunch_stats_to_store:
            evaluate_stats_bunch = getattr(bunch, stats)
            try:
                self.buffer_bunch[stats][write_pos] = evaluate_stats_bunch()
            except TypeError:
                self.buffer_bunch[stats][write_pos] = evaluate_stats_bunch

        # slice_set-specific data.
        for stats in self.slice_stats_to_store:
            self.buffer_slice[stats][:, write_pos] = getattr(slice_set, stats)

    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened
        and closed each time the buffer is written to file to prevent
        from loss of data in case of a crash. """
        buffer_tmp_bunch = {} # always on CPU
        buffer_tmp_slice = {}
        shift = - (self.i_steps + 1 % self.buffer_size)
        for stats in self.bunch_stats_to_store:
            try:
                buffer_tmp_bunch[stats] = np.roll(self.buffer_bunch[stats].get(),
                        shift=shift, axis=0)
            except:
                buffer_tmp_bunch[stats] = np.roll(self.buffer_bunch[stats].copy(),
                        shift=shift, axis=0)
        for stats in self.slice_stats_to_store:
            try:
                buffer_tmp_slice[stats] = np.roll(self.buffer_slice[stats].get(),
                        shift=shift, axis=1)
            except:
                buffer_tmp_slice[stats] = np.roll(self.buffer_slice[stats].copy(),
                        shift=shift, axis=1)

        # Keep track of where to read from buffers and where to store
        # data in file.
        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer = self.buffer_size - n_entries_in_buffer
        low_pos_in_file = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file = self.i_steps + 1

        # Try to write data to file. If file is not available, skip this
        # step and repeat it again after self.write_buffer_every. As
        # long as self.buffer_size is not exceeded, no data are lost.
        try:
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']
            for stats in self.bunch_stats_to_store:
                h5group_bunch[stats][low_pos_in_file:up_pos_in_file] = \
                    buffer_tmp_bunch[stats][low_pos_in_buffer:]
                    #self.buffer_bunch[stats][low_pos_in_buffer:]
            for stats in self.slice_stats_to_store:
                h5group_slice[stats][:,low_pos_in_file:up_pos_in_file] = \
                    buffer_tmp_slice[stats][:,low_pos_in_buffer:]
                    #self.buffer_slice[stats][:,low_pos_in_buffer:]
            h5file.close()
        except IOError:
            self.warns('Slice monitor file is temporarily unavailable. \n')


class ParticleMonitor(Monitor):
    """Class to store particle-specific data to a HDF5 file, i.e. the
    coordinates and conjugate momenta as well as the id of individual
    macroparticles of a bunch. """

    def __init__(self, filename, stride=1, parameters_dict=None,
                 *args, **kwargs):
        """ Create an instance of a ParticleMonitor class. The HDF5 file
        is initialized, and if specified, the parameters_dict is written
        to file.

          filename:        Path and name of HDF5 file. Without file
                           extension.
          stride:          Only store data of macroparticles for which
                           id % stride == 0.
          parameters_dict: Metadata for HDF5 file containing main
                           simulation parameters.

          Optionally pass a list called quantities_to_store which
          specifies which members of the bunch will be called/stored.
        """
        quantities_to_store = [ 'x', 'xp', 'y', 'yp', 'z', 'dp', 'id' ]
        self.quantities_to_store = kwargs.pop('quantities_to_store',
                                              quantities_to_store)
        self.filename = filename
        self.stride = stride
        self.i_steps = 0

        self._create_file_structure(parameters_dict)

    def dump(self, bunch, arrays_dict=None):
        """ Write particle data to file. See docstring of method
        self._write_data_to_file . """
        self._write_data_to_file(bunch, arrays_dict)
        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file. If specified by the user, write the
        contents of the parameters_dict as metadata (attributes)
        to the file. Maximum file compression is activated. """
        try:
            h5file = hp.File(self.filename + '.h5part', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]
            h5file.close()
        except Exception as err:
            self.warns('Problem occurred during Particle monitor creation.')
            self.warns(err.message)
            raise

    def _write_data_to_file(self, bunch, arrays_dict):
        """ Write macroparticle data (x, xp, y, yp, z, dp, id) of a
        selection of particles to the HDF5 file. Optionally, data in
        additional_quantities can also be added if provided in the
        constructor. The file is opened and closed every time to prevent
        from loss of data in case of a crash.
        For each simulation step, a new group with name 'Step#..' is
        created. It contains one dataset for each of the quantities
        given in self.quantities_to_store. """
        h5file = hp.File(self.filename + '.h5part', 'a')
        h5group = h5file.create_group('Step#' + str(self.i_steps))
        dims = (bunch.macroparticlenumber // self.stride,)
        dims = bunch.get_coords_n_momenta_dict().values()[0][::self.stride].shape # more robust implementation

        # resorting_indices = np.argsort(bunch.id)[::self.stride]
        all_quantities = {}
        for quant in self.quantities_to_store:
            quant_values = getattr(bunch, quant)
            if pm.device is 'GPU':
                stream = next(gpu_utils.stream_pool)
                quant_values = quant_values.get_async(stream=stream)
            all_quantities[quant] = quant_values
        if pm.device is 'GPU':
            for stream in gpu_utils.streams:
                stream.synchronize()

        if arrays_dict is not None:
            all_quantities.update(arrays_dict)

        for quant in all_quantities.keys():
            quant_values = all_quantities[quant]
            h5group.create_dataset(quant, shape=dims, compression='gzip',
                compression_opts=9, dtype=quant_values.dtype)
            h5group[quant][:] = quant_values[::self.stride]

        h5file.close()


class CellMonitor(Monitor):
    """ Class to store cell (z, dp) specific data (for the moment only
    mean_x, mean_y, mean_z, mean_dp and n_particles_in_cell) to a HDF5
    file. This monitor uses a buffer (shift register) to reduce the
    number of writing operations to file. This also helps to avoid IO
    errors and loss of data when writing to a file that may become
    temporarily unavailable (e.g. if file is located on network) during
    the simulation. """

    def __init__(self, filename, n_steps, n_azimuthal_slices, n_radial_slices,
                 radial_cut, beta_z, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096, *args, **kwargs):
        """ Create an instance of a CellMonitor class. Apart from
        initializing the HDF5 file, a buffer self.buffer_cell is
        prepared to buffer the cell-specific data before writing them
        to file.

          filename:           Path and name of HDF5 file. Without file
                              extension.
          n_steps:            Number of entries to be reserved for each
                              of the quantities in self.stats_to_store.
          n_azimuthal_slices: Number of pizza slices (azimuthal slicing).
          n_radial_slices:    Number of rings (radial slicing).
          radial_cut:         'Radius' of the outermost ring in
                              longitudinal phase space (using beta_z*dp)
          parameters_dict:    Metadata for HDF5 file containing main
                              simulation parameters.
          write_buffer_every: Number of steps after which buffer
                              contents are actually written to file.
          buffer_size:        Number of steps to be buffered. """
        stats_to_store = [
            'mean_x', 'mean_xp',
            'mean_y', 'mean_yp',
            'mean_z', 'mean_dp',
            'macroparticlenumber']
        self.stats_to_store = kwargs.pop('stats_to_store', stats_to_store)

        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0
        self.n_azimuthal_slices = n_azimuthal_slices
        self.n_radial_slices = n_radial_slices
        self.radial_cut = radial_cut
        self.beta_z = beta_z

        # Prepare buffer.
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every
        self.buffer_cell = {}

        for stats in self.stats_to_store:
            self.buffer_cell[stats] = (
                np.zeros((self.n_azimuthal_slices, self.n_radial_slices, self.buffer_size)))
        self._create_file_structure(parameters_dict)

    def dump(self, bunch):
        """ Evaluate the statistics for the given cells and write the
        data to the buffer and/or to the HDF5 file. The buffer is used
        to reduce the number of writing operations to file. This helps
        to avoid IO errors and loss of data when writing data to a file
        that may become temporarily unavailable (e.g. if file is on
        network) during the simulation. Buffer contents are written to
        file only every self.write_buffer_every steps. """
        self._write_data_to_buffer(bunch)
        if ((self.i_steps + 1) % self.write_buffer_every == 0 or
                (self.i_steps + 1) == self.n_steps):
            self._write_buffer_to_file()
        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file and create its basic structure (groups
        and datasets). One dataset for each of the quantities defined
        in self.stats_to_store is generated. If specified by
        the user, write the contents of the parameters_dict as metadata
        (attributes) to the file. Maximum file compression is
        activated. """
        try:
            h5file = hp.File(self.filename + '.h5', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]

            h5file.create_group('Cells')
            h5group_cells = h5file['Cells']
            for stats in self.stats_to_store:
                h5group_cells.create_dataset(
                    stats, compression='gzip', compression_opts=9,
                    shape=(self.n_azimuthal_slices, self.n_radial_slices,
                           self.n_steps))
            h5file.close()
        except Exception as err:
            self.warns('Problem occurred during Cell monitor creation.')
            self.warns(err.message)
            raise

    def _write_data_to_buffer(self, bunch):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. The
        cell-specific data are computed by a cython function. """

        ps_coords = {'x': None, 'xp': None, 'y': None,
                     'yp': None, 'z': None, 'dp': None}
        for coord in ps_coords:
            ps_coords[coord] = getattr(bunch, coord)
            if pm.device is 'GPU':
                stream = next(gpu_utils.stream_pool)
                ps_coords[coord] = ps_coords[coord].get_async(stream=stream)
        if pm.device is 'GPU':
            for stream in gpu_utils.streams:
                stream.synchronize()

        # TODO: calculate these cell stats on the GPU instead of the CPU!!!
        n_cl, x_cl, xp_cl, y_cl, yp_cl, z_cl, dp_cl = cp.calc_cell_stats(
            ps_coords['x'], ps_coords['xp'], ps_coords['y'],
            ps_coords['yp'], ps_coords['z'], ps_coords['dp'],
            self.beta_z, self.radial_cut, self.n_radial_slices,
            self.n_azimuthal_slices)

        self.buffer_cell['mean_x'][:,:,0] = x_cl[:,:]
        self.buffer_cell['mean_xp'][:,:,0] = xp_cl[:,:]
        self.buffer_cell['mean_y'][:,:,0] = y_cl[:,:]
        self.buffer_cell['mean_yp'][:,:,0] = yp_cl[:,:]
        self.buffer_cell['mean_z'][:,:,0] = z_cl[:,:]
        self.buffer_cell['mean_dp'][:,:,0] = dp_cl[:,:]
        self.buffer_cell['macroparticlenumber'][:,:,0] = n_cl[:,:]

        for stats in self.stats_to_store:
            self.buffer_cell[stats] = np.roll(
                self.buffer_cell[stats], shift=-1, axis=2)

    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened
        and closed each time the buffer is written to file to prevent
        from loss of data in case of a crash. """
        # Keep track of where to read from buffers and where to store
        # data in file.
        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer = self.buffer_size - n_entries_in_buffer
        low_pos_in_file = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file = self.i_steps + 1

        # Try to write data to file. If file is not available, skip this
        # step and repeat it again after self.write_buffer_every. As
        # long as self.buffer_size is not exceeded, no data are lost.
        try:
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group_cells = h5file['Cells']

            for stats in self.stats_to_store:
                h5group_cells[stats][:,:,low_pos_in_file:up_pos_in_file] = \
                    self.buffer_cell[stats][:,:,low_pos_in_buffer:]
            h5file.close()
        except Exception as err:
            self.warns(err.message)
