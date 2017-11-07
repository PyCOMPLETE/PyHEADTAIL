import numpy as np
from scipy.constants import m_p, c, e, pi

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib as mpl
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt

from PyHEADTAIL.machines.synchrotron import Synchrotron

import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import LinearMap
from PyHEADTAIL.particles.slicing import UniformBinSlicer
# plt.xkcd()
class Machine(Synchrotron):
    """ Note that this is a toy machine for testing, which pameters are tuned for testing
        purposes.
    """

    def __init__(self, n_segments = 1, Q_x=64.28, Q_y=59.31):

        optics_mode='smooth'
#        longitudinal_mode='non-linear'
        longitudinal_mode='linear'

        charge = e
        mass = m_p
        alpha = 53.86**-2
        self.h_RF = 35640
        self.h_bunch = 3564

        p0 = 450e9 * e / c
        p_increment = 0.
        self.accQ_x = Q_x
        self.accQ_y = Q_y
        V_RF = 16e6
        dphi_RF = 0

        self.circumference = 26658.883
        s = None
        alpha_x = None
        alpha_y = None
        self.beta_x = self.circumference / (2.*np.pi*self.accQ_x)
        self.beta_y = self.circumference / (2.*np.pi*self.accQ_y)
        D_x = 0
        D_y = 0

        # detunings
        Qp_x = 0
        Qp_y = 0

        app_x = 0
        app_y = 0
        app_xy = 0
        
#        wrap_z=True
        wrap_z=False

        super(Machine, self).__init__(
            optics_mode=optics_mode, circumference=self.circumference,
            n_segments=n_segments, s=s, name=None,
            alpha_x=alpha_x, beta_x=self.beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=self.beta_y, D_y=D_y,
            accQ_x=self.accQ_x, accQ_y=self.accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(self.h_RF), V_RF=np.atleast_1d(V_RF),
            dphi_RF=np.atleast_1d(dphi_RF), p0=p0, p_increment=p_increment,
            charge=charge, mass=mass, wrap_z=wrap_z)


def generate_objects(machine,n_macroparticles, n_slices, n_sigma_z,
                     intensity=1e11, sigma_z=0.1124,
                     epsn_x=3.5e-6, epsn_y=3.5e-6,
                     filling_scheme = [0], matched=False):

    bunch = machine.generate_6D_Gaussian_bunch(
    n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
    filling_scheme=filling_scheme, matched=matched)

    slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    return bunch, slicer, machine.transverse_map, machine.longitudinal_map


def track(n_turns, bunch, total_map, bunch_dump):
    for i in range(n_turns):
        bunch_dump.dump()

        for m_ in total_map:
            m_.track(bunch)


class BunchTracker(object):
    def __init__(self,bunch):
        self.counter = 0

        self.bunch = bunch
        self.turn = np.array([])

        self.mean_x = np.array([])
        self.mean_y = np.array([])
        self.mean_z = np.array([])

        self.mean_xp = np.array([])
        self.mean_yp = np.array([])
        self.mean_dp = np.array([])

        self.sigma_x = np.array([])
        self.sigma_y = np.array([])
        self.sigma_z = np.array([])

        self.sigma_xp = np.array([])
        self.sigma_yp = np.array([])
        self.sigma_dp = np.array([])

        self.epsn_x = np.array([])
        self.epsn_y = np.array([])
        self.epsn_z = np.array([])

    def dump(self):
        self.turn=np.append(self.turn,[self.counter])
        self.counter += 1

        self.mean_x=np.append(self.mean_x,[self.bunch.mean_x()])
        self.mean_y=np.append(self.mean_y,[self.bunch.mean_y()])
        self.mean_z=np.append(self.mean_z,[self.bunch.mean_z()])

        self.mean_xp=np.append(self.mean_xp,[self.bunch.mean_xp()])
        self.mean_yp=np.append(self.mean_yp,[self.bunch.mean_yp()])
        self.mean_dp=np.append(self.mean_dp,[self.bunch.mean_dp()])

        self.sigma_x=np.append(self.sigma_x,[self.bunch.sigma_x()])
        self.sigma_y=np.append(self.sigma_y,[self.bunch.sigma_y()])
        self.sigma_z=np.append(self.sigma_z,[self.bunch.sigma_z()])

        self.sigma_xp=np.append(self.sigma_xp,[self.bunch.sigma_xp()])
        self.sigma_yp=np.append(self.sigma_yp,[self.bunch.sigma_yp()])
        self.sigma_dp=np.append(self.sigma_dp,[self.bunch.sigma_dp()])

        self.epsn_x = np.append(self.epsn_x,[self.bunch.epsn_x()])
        self.epsn_y = np.append(self.epsn_y,[self.bunch.epsn_y()])
        self.epsn_z = np.append(self.epsn_z,[self.bunch.epsn_z()])

class BeamTracker(object):
    def __init__(self,beam):
        self.bunch_ids = None
        self.counter = 0

        self.beam = beam
        self.turn = np.array([])

        self.mean_x = np.array([])
        self.mean_y = np.array([])
        self.mean_z = np.array([])

        self.mean_xp = np.array([])
        self.mean_yp = np.array([])
        self.mean_dp = np.array([])

        self.sigma_x = np.array([])
        self.sigma_y = np.array([])
        self.sigma_z = np.array([])

        self.sigma_xp = np.array([])
        self.sigma_yp = np.array([])
        self.sigma_dp = np.array([])

        self.epsn_x = np.array([])
        self.epsn_y = np.array([])
        self.epsn_z = np.array([])

    def dump(self):
        self.turn=np.append(self.turn,[self.counter])
        
        bunch_list = self.beam.split_to_views()
        
        temp_mean_x = np.zeros(len(bunch_list))
        temp_mean_y = np.zeros(len(bunch_list))
        temp_mean_z = np.zeros(len(bunch_list))

        temp_mean_xp = np.zeros(len(bunch_list))
        temp_mean_yp = np.zeros(len(bunch_list))
        temp_mean_dp = np.zeros(len(bunch_list))

        temp_sigma_x = np.zeros(len(bunch_list))
        temp_sigma_y = np.zeros(len(bunch_list))
        temp_sigma_z = np.zeros(len(bunch_list))

        temp_sigma_xp = np.zeros(len(bunch_list))
        temp_sigma_yp = np.zeros(len(bunch_list))
        temp_sigma_dp = np.zeros(len(bunch_list))

        temp_epsn_x = np.zeros(len(bunch_list))
        temp_epsn_y = np.zeros(len(bunch_list))
        temp_epsn_z = np.zeros(len(bunch_list))
        
        if self.bunch_ids is None:
            get_ids = True
            self.bunch_ids = []
        else:
            get_ids = False
        
        for i, bunch in enumerate(bunch_list):
            if get_ids:
                self.bunch_ids.append(bunch.bucket_id[0])
        
            temp_mean_x[i] = bunch.mean_x()
            temp_mean_y[i] = bunch.mean_y()
            temp_mean_z[i] = bunch.mean_z()
    
            temp_mean_xp[i] = bunch.mean_xp()
            temp_mean_yp[i] = bunch.mean_yp()
            temp_mean_dp[i] = bunch.mean_dp()
    
            temp_sigma_x[i] = bunch.sigma_x()
            temp_sigma_y[i] = bunch.sigma_y()
            temp_sigma_z[i] = bunch.sigma_z()
    
            temp_sigma_xp[i] = bunch.sigma_xp()
            temp_sigma_yp[i] = bunch.sigma_yp()
            temp_sigma_dp[i] = bunch.sigma_dp()
    
            temp_epsn_x[i] = bunch.epsn_x()
            temp_epsn_y[i] = bunch.epsn_y()
            temp_epsn_z[i] = bunch.epsn_z()
            
        if self.counter == 0:
            self.mean_x = [temp_mean_x]
            self.mean_y = [temp_mean_y]
            self.mean_z = [temp_mean_z]
    
            self.mean_xp = [temp_mean_xp]
            self.mean_yp = [temp_mean_yp]
            self.mean_dp = [temp_mean_dp]
    
            self.sigma_x = [temp_sigma_x]
            self.sigma_y = [temp_sigma_y]
            self.sigma_z = [temp_sigma_z]
    
            self.sigma_xp = [temp_sigma_xp]
            self.sigma_yp = [temp_sigma_yp]
            self.sigma_dp = [temp_sigma_dp]
    
            self.epsn_x = [temp_epsn_x]
            self.epsn_y = [temp_epsn_y]
            self.epsn_z = [temp_epsn_z]   
            
        else:
            self.mean_x = np.concatenate((self.mean_x,[temp_mean_x]), axis=0)
            self.mean_y = np.concatenate((self.mean_y,[temp_mean_y]), axis=0)
            self.mean_z = np.concatenate((self.mean_z,[temp_mean_z]), axis=0)
    
            self.mean_xp = np.concatenate((self.mean_xp,[temp_mean_xp]), axis=0)
            self.mean_yp = np.concatenate((self.mean_yp,[temp_mean_yp]), axis=0)
            self.mean_dp = np.concatenate((self.mean_dp,[temp_mean_dp]), axis=0)
    
            self.sigma_x = np.concatenate((self.sigma_x,[temp_sigma_x]), axis=0)
            self.sigma_y = np.concatenate((self.sigma_y,[temp_sigma_y]), axis=0)
            self.sigma_z = np.concatenate((self.sigma_z,[temp_sigma_z]), axis=0)
    
            self.sigma_xp = np.concatenate((self.sigma_xp,[temp_sigma_xp]), axis=0)
            self.sigma_yp = np.concatenate((self.sigma_yp,[temp_sigma_yp]), axis=0)
            self.sigma_dp = np.concatenate((self.sigma_dp,[temp_sigma_dp]), axis=0)
    
            self.epsn_x = np.concatenate((self.epsn_x,[temp_epsn_x]), axis=0)
            self.epsn_y = np.concatenate((self.epsn_y,[temp_epsn_y]), axis=0)
            self.epsn_z = np.concatenate((self.epsn_z,[temp_epsn_z]), axis=0)
        
        self.counter += 1

def compare_traces(trackers, labels, bunch_idx=None):
    fig = plt.figure(figsize=(16, 8))
    ax_x_mean = fig.add_subplot(231)
    ax_x_sigma = fig.add_subplot(232)
    ax_x_epsn = fig.add_subplot(233)

    ax_y_mean = fig.add_subplot(234)
    ax_y_sigma = fig.add_subplot(235)
    ax_y_epsn = fig.add_subplot(236)

    if len(trackers[0].mean_x.shape) > 1:

        if bunch_idx is None:
            b_f = 0
            b_t = len(trackers[0].mean_x[0,:])
        else:
            b_f = bunch_idx
            b_t = bunch_idx + 1

        for i, tracker in enumerate(trackers):
            ax_x_mean.plot(tracker.turn, np.mean(tracker.mean_x[:,b_f:b_t], axis=1) * 1000, label=labels[i])
            ax_x_sigma.plot(tracker.turn, np.mean(tracker.sigma_x[:,b_f:b_t], axis=1) * 1000, label=labels[i])
            ax_x_epsn.plot(tracker.turn, np.mean(tracker.epsn_x[:,b_f:b_t], axis=1) * 1e6, label=labels[i])
            ax_y_mean.plot(tracker.turn, np.mean(tracker.mean_y[:,b_f:b_t], axis=1) * 1000, label=labels[i])
            ax_y_sigma.plot(tracker.turn, np.mean(tracker.sigma_y[:,b_f:b_t], axis=1) * 1000, label=labels[i])
            ax_y_epsn.plot(tracker.turn, np.mean(tracker.epsn_y[:,b_f:b_t], axis=1) * 1e6, label=labels[i])
            
    else:
        for i, tracker in enumerate(trackers):
            ax_x_mean.plot(tracker.turn, tracker.mean_x * 1000, label=labels[i])
            ax_x_sigma.plot(tracker.turn, tracker.sigma_x * 1000, label=labels[i])
            ax_x_epsn.plot(tracker.turn, tracker.epsn_x * 1e6, label=labels[i])
            ax_y_mean.plot(tracker.turn, tracker.mean_y * 1000, label=labels[i])
            ax_y_sigma.plot(tracker.turn, tracker.sigma_y * 1000, label=labels[i])
            ax_y_epsn.plot(tracker.turn, tracker.epsn_y * 1e6, label=labels[i])
        
    ax_x_mean.legend(loc='upper right')
    ax_x_mean.set_ylabel('<x> [mm]')
    ax_x_mean.ticklabel_format(useOffset=False)

    ax_x_sigma.set_ylabel('sigma_x [mm]')
    ax_x_sigma.ticklabel_format(useOffset=False)

    ax_x_epsn.set_ylabel('epsn_x [mm mrad]')
    ax_x_epsn.set_xlabel('Turn')
    ax_x_epsn.ticklabel_format(useOffset=False)

    ax_y_mean.legend(loc='upper right')
    ax_y_mean.set_ylabel('<y> [mm]')
    ax_y_mean.ticklabel_format(useOffset=False)

    ax_y_sigma.set_ylabel('sigma_y [mm]')
    ax_y_sigma.ticklabel_format(useOffset=False)

    ax_y_epsn.set_ylabel('epsn_y  [mm mrad]')
    ax_y_epsn.set_xlabel('Turn')
    ax_y_epsn.ticklabel_format(useOffset=False)

    plt.show()


def compare_projections(bunches, labels, n_particles = 300):
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('z-x and z-y projections of bunches', fontsize=14, fontweight='bold')
    ax_z_x = fig.add_subplot(121)
    ax_z_y = fig.add_subplot(122)
    plot_every_n = None

    for i, bunch in enumerate(bunches):
        if plot_every_n is None:
            plot_every_n = int(np.ceil(len(bunch.z)/float(n_particles)))
        ax_z_x.plot(bunch.z[::plot_every_n], bunch.x[::plot_every_n] * 1000, '.', label=labels[i])
    ax_z_x.legend(loc='upper right')
    ax_z_x.set_xlabel('z [m]')
    ax_z_x.set_ylabel('x [mm]')
    for i, bunch in enumerate(bunches):
        ax_z_y.plot(bunch.z[::plot_every_n], bunch.y[::plot_every_n] * 1000, '.', label=labels[i])
    ax_z_y.legend(loc='upper right')
    ax_z_y.set_xlabel('z [m]')
    ax_z_y.set_ylabel('y [mm]')
    plt.tight_layout()
    plt.show()

def compare_beam_projections(beams, labels):
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('z-x and z-y projections of bunches', fontsize=14, fontweight='bold')
    ax_z_x = fig.add_subplot(121)
    ax_z_y = fig.add_subplot(122)

    for j, beam in enumerate(beams):
        
        bunch_list = beam.split()
        z_data = np.zeros(len(bunch_list))
        x_data = np.zeros(len(bunch_list))
        y_data = np.zeros(len(bunch_list))
        
        for i, bunch in enumerate(bunch_list):
            z_data[i] = bunch.mean_z()
            x_data[i] = bunch.mean_x()
            y_data[i] = bunch.mean_y()
            
        ax_z_x.plot(z_data, x_data * 1000, '.', label=labels[j])
        ax_z_y.plot(z_data, y_data * 1000, '.', label=labels[j])
    ax_z_x.legend(loc='upper right')
    ax_z_x.set_xlabel('z [m]')
    ax_z_x.set_ylabel('x [mm]')
    ax_z_y.legend(loc='upper right')
    ax_z_y.set_xlabel('z [m]')
    ax_z_y.set_ylabel('y [mm]')
    plt.tight_layout()
    plt.show()

def particle_position_difference(ref_bunch,bunch):
    diff_x = np.sum((ref_bunch.x - bunch.x)**2)/float(len(ref_bunch.x))
    sum_x = np.sum((bunch.x)**2)/float(len(ref_bunch.x))
    diff_y = np.sum((ref_bunch.y - bunch.y)**2)/float(len(ref_bunch.y))
    sum_y = np.sum((bunch.y)**2)/float(len(ref_bunch.y))
    
    print('An average relative particle position difference in x-axis: ' + str(diff_x/sum_x))
    print('An average relative particle position difference in y-axis: ' + str(diff_y/sum_y))


def trace_difference(ref_tracker, tracker):
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('z-x and z-y projections of bunches', fontsize=14, fontweight='bold')
    ax_x = fig.add_subplot(121)
    ax_y = fig.add_subplot(122)
    
    ax_x.plot(tracker.turn, (tracker.mean_x - ref_tracker.mean_x) * 1000)    
    ax_y.plot(tracker.turn, (tracker.mean_y - ref_tracker.mean_y) * 1000)
    
    ax_x.set_xlabel('Turn')
    ax_x.set_ylabel('Difference [mm]')

    ax_y.set_xlabel('Turn')
    ax_y.set_ylabel('Difference [mm]')
    plt.show()
    



def plot_debug_data(processors, source = 'input'):


    def pick_signals(processor, source = 'input'):
        """
        A function which helps to visualize the signals passing the signal processors.
        :param processor: a reference to the signal processor
        :param source: source of the signal, i.e, 'input' or 'output' signal of the processor
        :return: (t, z, bins, signal), where 't' and 'z' are time or position values for the signal values (which can be used
            as x values for plotting), 'bins' are data for visualizing sampling and 'signal' is the actual signal.
        """

        if source == 'input':
            bin_edges = processor.input_parameters['bin_edges']
            raw_signal = processor.input_signal
        elif source == 'output':
            bin_edges = processor.output_parameters['bin_edges']
            raw_signal = processor.output_signal
        else:
            raise ValueError('Unknown value for the data source')
        t = np.zeros(len(raw_signal)*4)
        bins = np.zeros(len(raw_signal)*4)
        signal = np.zeros(len(raw_signal)*4)
        value = 1.

        for i, edges in enumerate(bin_edges):
            t[4*i] = edges[0]
            t[4*i+1] = edges[0]
            t[4*i+2] = edges[1]
            t[4*i+3] = edges[1]
            bins[4*i] = 0.
            bins[4*i+1] = value
            bins[4*i+2] = value
            bins[4*i+3] = 0.
            signal[4*i] = 0.
            signal[4*i+1] = raw_signal[i]
            signal[4*i+2] = raw_signal[i]
            signal[4*i+3] = 0.
            value *= -1

        z = t * c
        return (t, z, bins, signal)

    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(211)
    ax11 = ax1.twiny()
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax22 = ax2.twiny()

    coeff = 1.
    
    min_t = None
    max_t = None
    
    min_z = None
    max_z = None

    for i, processor in enumerate(processors):
        if source == 'input':
            if hasattr(processor, 'input_signal'):
                if processor.debug:
                    t, z, bins, signal = pick_signals(processor,'input')
                    label=processor.label
                    ax1.plot(t*1e9,0.33*bins + coeff, label=label)
                    ax11.plot(z, np.zeros(len(z)))
                    ax11.cla()
                    coeff += 1.0
                    ax2.plot(t*1e9,signal*1e3)
                    ax22.plot(z, np.zeros(len(z)))
                    ax22.cla()
                    
                    if (min_t is None) or (np.min(t*1e9)<min_t):
                        min_t = np.min(t*1e9)
                    if (max_t is None) or (np.max(t*1e9)>max_t):
                        max_t = np.max(t*1e9)
                        
                    if (min_z is None) or (np.min(z)<min_z):
                        min_z = np.min(t)
                    if (max_z is None) or (np.max(z)>max_z):
                        max_z = np.max(z)
                    
        elif source == 'output':
            if hasattr(processor, 'output_signal'):
                if processor.debug:
                    t, z, bins, signal = pick_signals(processor,'output')
                    label=processor.label
                    ax1.plot(t*1e9,0.33*bins + coeff, label=label)
                    ax11.plot(z, np.zeros(len(z)))
                    ax11.cla()
                    coeff += 1.0
                    ax2.plot(t*1e9,signal*1e3)
                    ax22.plot(z, np.zeros(len(z)))
                    ax22.cla()
                    
                    if (min_t is None) or (np.min(t*1e9)<min_t):
                        min_t = np.min(t*1e9)
                    if (max_t is None) or (np.max(t*1e9)>max_t):
                        max_t = np.max(t*1e9)
                        
                    if (min_z is None) or (np.min(z)<min_z):
                        min_z = np.min(t)
                    if (max_z is None) or (np.max(z)>max_z):
                        max_z = np.max(z)

    ax1.set_ylim(coeff,0)
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    ax1.set_xticklabels(())
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Signal processor #')
    ax1.set_xlabel('Time [ns]')
    ax1.set_xlim(min_t,max_t)
    ax11.set_xlim(min_z,max_z)
    ax11.set_xlabel('Distance [m]')

    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Signal [mm]')
    ax2.set_xlim(min_t,max_t)
    ax22.set_xlim(min_z,max_z)
    ax22.set_xticklabels(())
#    ax1.set_xticklabels(())

    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2


def plot3Dtraces(tracker, labels, show_holes=True, bunches=None,
                 first_turn=0):
    

    def make_holes(data_z,data_x):
        bunch_spacing = data_z[0]-data_z[1]
        plot_x = None
        plot_z = None
        
        i_from = 0
        for i in range(len(data_z)-1):
            if (-data_z[i+1]+data_z[i]) > 1.5 * bunch_spacing:
                i_to = i+1
                if plot_x is None:
                    plot_x = data_x[i_from:i_to]
                    plot_z = data_z[i_from:i_to]
                else:
                    plot_x = np.append(plot_x,data_x[i_from:i_to])
                    plot_z = np.append(plot_z,data_z[i_from:i_to])
                plot_x = np.append(plot_x,[0])
                plot_x = np.append(plot_x,[0])
                plot_z = np.append(plot_z,[data_z[i]])
                plot_z = np.append(plot_z,[data_z[i+1]])
                    
                i_from = i+1
                
        i_to = None
        if plot_x is None:
            plot_x = data_x[i_from:i_to]
            plot_z = data_z[i_from:i_to]
        else:
            plot_x = np.append(plot_x,data_x[i_from:i_to])
            plot_z = np.append(plot_z,data_z[i_from:i_to])
        plot_x = np.append(plot_x,[0])
        plot_z = np.append(plot_z,[data_z[-1]])
        
        return plot_z, plot_x
                    
    def pick_max_values(turns, data, first_turn):
        
        from scipy.optimize import curve_fit
        def func(x, a, b, c, d):
            return a*np.exp(-b*x)
        
        n_values_per_window = 4.
        
        t_segments = np.array_split(turns[first_turn:],int(np.ceil(len(turns[first_turn:])/n_values_per_window)))
        d_segments = np.array_split(data[first_turn:],int(np.ceil(len(np.abs(data[first_turn:]))/n_values_per_window)))
       
        max_turns = np.zeros(len(t_segments))
        max_data = np.zeros(len(d_segments))
        
        for i, (t,d) in enumerate(zip(t_segments,d_segments)):
            if len(d) > n_values_per_window-1:
                idx = np.argmax(d)
                max_turns[i] = t[idx]
                max_data[i] = d[idx]
         
        max_turns =  max_turns[max_data > 0.]
        max_data =  max_data[max_data > 0.]
        
            
        popt = np.polyfit(max_turns, np.log(max_data),1)
    
        return turns, np.exp(popt[1])*np.exp(popt[0]*turns), -1./popt[0], max_turns, max_data
    
    data_x = tracker.mean_x
#    data_z = tracker.mean_z
    data_z = np.array(tracker.bunch_ids)
    turns = np.linspace(1,len(data_x[:,0]),len(data_x[:,0]))
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax_damping = fig.add_subplot(122)
#    ax_raw = fig.add_subplot(133)
#    ax = fig.gca(projection='3d')
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

#    xs = np.arange(5, 10, 0.4)
    verts = []
    for i in range(len(turns)):
        
        xs, ys = make_holes(data_z,np.abs(data_x[i,:]))
        
#        ys = np.append(np.abs(data_x[i,:]),[0])
##        ys = np.append(data_x[i,:],[0])
#        xs = np.append(data_z[i,:],[data_z[i,-1]])
        verts.append(list(zip(xs, ys)))
    
    poly = PolyCollection(verts, facecolors = [cc('w')]*len(turns), linewidths=1, edgecolor="black")
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=turns, zdir='y')
    
    ax.set_xlabel('Bucket #')
    ax.set_xlim3d(np.min(data_z),np.max(data_z))
    ax.set_ylabel('Turn')
    ax.set_ylim3d(np.max(turns)+1,0)
    ax.set_zlabel('|mean_x|')
    ax.set_zlim3d(0, 10e-3)
    
    if bunches is not None:
        colormap = mpl.cm.Set1.colors
        if type(bunches) is int:
            bunches = [bunches]
            
        for j,i in enumerate(bunches):
            t, d, t_d, rt, rd = pick_max_values(turns, data_x[:,i], first_turn)
            ax_damping.plot(turns,data_x[:,i], color= colormap[j], label = r'Bucket #{:03d}, $\tau_d=${:.2f} turns'.format(tracker.bunch_ids[i],t_d))
            ax_damping.plot(t,d,'--', color= colormap[j])
            ax.plot(data_z[i]*np.ones(len(t)), t, d,'--', lw=1.5, color= colormap[j])
#        else:
#            t, d, t_d, rt, rd = pick_max_values(turns, data_x[:,bunches], first_turn)
#            print 'Dampint time: ' + str(t_d) + ' turns'
#            ax_damping.plot(turns,data_x[:,bunches], color= colormap[0], label = 'bucket #' + str(bunches))
#            ax_damping.plot(t,d,'--', color= colormap[0], label = 'bucket #' + str(bunches))
#            ax_raw.semilogy(rt, rd, label = 'bucket #' + str(bunches))
#            ax_raw.semilogy(t, d, label = 'bucket #' + str(bunches))
                
        ax_damping.set_xlabel('Turn')
        ax_damping.set_ylabel('mean_x')
        ax_damping.legend(loc='upper right')
        
    
    plt.tight_layout()
    plt.show()
    
