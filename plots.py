import h5py
import pylab as plt


def plot_slices(bunch):
    pdf, bins, patches = plt.hist(bunch.dz, 64)
    [plt.axvline(vl, c='r') for vl in bunch.slices.dz_bins]
    plt.stem(bunch.slices.dz_centers, bunch.slices.charge[:-1],
             linefmt='-g', markerfmt='og')
    print sum(bunch.slices.charge), sum(pdf)
    print bunch.slices.charge[-12:-1]
    print  pdf[-10:]


def plot_phasespace(bunch, r):
    plt.clf()

    # normalization = np.max(bunch.dz) / np.max(bunch.dp)
    # r = bunch.dz ** 2 + (normalization * bunch.dp) ** 2

    ax = plt.gca()
    ax.scatter(bunch.dz, bunch.dp, c=r, marker='.', lw=0)
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-0.5e-2, 0.5e-2)
    plt.draw()

def plot_bunch(filename):

    filename = filename + '.h5'

    bunch = h5py.File(filename, 'r')['Bunch']

    beta_x = plt.amax(bunch['mean_x'][:]) / plt.amax(bunch['mean_xp'][:])
    r = plt.sqrt(bunch['mean_x'][:] ** 2 + (beta_x * bunch['mean_xp'][:]) ** 2)

    ax1 = plt.gca()
    ax2 = plt.twinx(ax1)
    ax1.plot(bunch['mean_x'], 'r')
    ax1.plot(+r, c='brown', lw=2)
    ax1.plot(-r, c='brown', lw=2)
    plt.show()
    
    
def plot_emittance(filename):

    filename = filename + '.h5'

    #~ bunch = h5py.File(filename, 'r')['Bunch']
#~ 
    #~ ax1 = plt.gca()
    #~ ax2 = plt.twinx(ax1)
    #~ ax1.plot(bunch['mean_x'], 'r')
    #~ ax1.plot(+r, c='brown', lw=2)
    #~ ax1.plot(-r, c='brown', lw=2)
    #~ plt.show()
