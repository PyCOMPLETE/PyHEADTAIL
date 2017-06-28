import numpy as np
from scipy.constants import m_p, c, e, pi
import matplotlib.pyplot as plt

from PyHEADTAIL.machines.synchrotron import Synchrotron



import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import LinearMap
from PyHEADTAIL.particles.slicing import UniformBinSlicer


class Machine(Synchrotron):

    def __init__(self, n_segments = 1, Q_x=64.28, Q_y=59.31, Q_s=0.0020443):

        optics_mode='smooth'
        longitudinal_mode='non-linear'

        charge = e
        mass = m_p
        alpha = 53.86**-2
        self.h_RF = 35640

        p0 = 7000e9 * e / c
        p_increment = 0.
        self.accQ_x = Q_x
        self.accQ_y = Q_y
        V_RF = 16e6
        dphi_RF = 0

#        n_segments = kwargs['n_segments']
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
        wrap_z=True

#        name = None
        self.sigma_z = 0.081
        self.epsn_x = 3.75e-6  # [m rad]
        self.epsn_y = 3.75e-6  # [m rad]
        self.intensity = 1.05e11

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


def generate_objects(machine,n_macroparticles, n_slices,n_sigma_z,
    filling_scheme = None, matched=True):

    bunch = machine.generate_6D_Gaussian_bunch(
    n_macroparticles, machine.intensity, machine.epsn_x, machine.epsn_y, sigma_z=machine.sigma_z,
    filling_scheme=filling_scheme, matched=matched)

    slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)

    return bunch, slicer, machine.transverse_map, machine.longitudinal_map


def track(n_turns, bunch, total_map, bunch_dump):
    for i in xrange(n_turns):
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


def compare_traces(trackers, labels):
    fig = plt.figure(figsize=(16, 8))
    ax_x_mean = fig.add_subplot(231)
    ax_x_sigma = fig.add_subplot(232)
    ax_x_epsn = fig.add_subplot(233)

    for i, tracker in enumerate(trackers):
        ax_x_mean.plot(tracker.turn, tracker.mean_x * 1000, label=labels[i])
    ax_x_mean.legend(loc='upper right')
    ax_x_mean.set_ylabel('<x> [mm]')
    ax_x_mean.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_x_sigma.plot(tracker.turn, tracker.sigma_x * 1000, label=labels[i])
    ax_x_sigma.set_ylabel('sigma_x [mm]')
    ax_x_sigma.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_x_epsn.plot(tracker.turn, tracker.epsn_x * 1e6, label=labels[i])
    ax_x_epsn.set_ylabel('epsn_x [mm mrad]')
    ax_x_epsn.set_xlabel('Turn')
    ax_x_epsn.ticklabel_format(useOffset=False)

    ax_y_mean = fig.add_subplot(234)
    ax_y_sigma = fig.add_subplot(235)
    ax_y_epsn = fig.add_subplot(236)

    for i, tracker in enumerate(trackers):
        ax_y_mean.plot(tracker.turn, tracker.mean_y * 1000, label=labels[i])
    ax_y_mean.legend(loc='upper right')
    ax_y_mean.set_ylabel('<y> [mm]')
    ax_y_mean.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_y_sigma.plot(tracker.turn, tracker.sigma_y * 1000, label=labels[i])
    ax_y_sigma.set_ylabel('sigma_y [mm]')
    ax_y_sigma.ticklabel_format(useOffset=False)

    for i, tracker in enumerate(trackers):
        ax_y_epsn.plot(tracker.turn, tracker.epsn_y * 1e6, label=labels[i])
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
            print 'plot_every_n: '
            print plot_every_n
        ax_z_x.plot(bunch.z[::plot_every_n], bunch.x[::plot_every_n] * 1000, '.', label=labels[i])
    ax_z_x.legend(loc='upper right')
    ax_z_x.set_xlabel('z [m]')
    ax_z_x.set_ylabel('x [mm]')
    for i, bunch in enumerate(bunches):
        ax_z_y.plot(bunch.z[::plot_every_n], bunch.y[::plot_every_n] * 1000, '.', label=labels[i])
    ax_z_y.legend(loc='upper right')
    ax_z_y.set_xlabel('z [m]')
    ax_z_y.set_ylabel('y [mm]')


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
        z = np.zeros(len(raw_signal)*4)
        bins = np.zeros(len(raw_signal)*4)
        signal = np.zeros(len(raw_signal)*4)
        value = 1.

        for i, edges in enumerate(bin_edges):
            z[4*i] = edges[0]
            z[4*i+1] = edges[0]
            z[4*i+2] = edges[1]
            z[4*i+3] = edges[1]
            bins[4*i] = 0.
            bins[4*i+1] = value
            bins[4*i+2] = value
            bins[4*i+3] = 0.
            signal[4*i] = 0.
            signal[4*i+1] = raw_signal[i]
            signal[4*i+2] = raw_signal[i]
            signal[4*i+3] = 0.
            value *= -1

        t = z/c
        return (t, z, bins, signal)

    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(211)
    ax11 = ax1.twiny()
    ax2 = fig.add_subplot(212)
    ax22 = ax2.twiny()

    coeff = 1.


    for i, processor in enumerate(processors):
        if source == 'input':
            if hasattr(processor, 'input_signal'):
                if processor.debug:
                    t, z, bins, signal = pick_signals(processor,'input')
                    ax1.plot(t,bins*coeff)
                    ax11.plot(z, np.zeros(len(z)))
                    ax11.cla()
                    coeff *= 0.9
                    ax2.plot(t,signal)
                    ax22.plot(z, np.zeros(len(z)))
                    ax22.cla()
        elif source == 'output':
            if hasattr(processor, 'output_signal'):
                if processor.debug:
                    t, z, bins, signal = pick_signals(processor,'output')
                    ax1.plot(t,bins*coeff)
                    ax11.plot(z, np.zeros(len(z)))
                    ax11.cla()
                    coeff *= 0.9
                    ax2.plot(t,signal)
                    ax22.plot(z, np.zeros(len(z)))
                    ax22.cla()

    plt.show()
    return fig, ax1, ax2
