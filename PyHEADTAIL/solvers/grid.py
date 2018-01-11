from __future__ import division


import numpy as np

from grid_functions import gather_from, scatter_to

from . import Element

class UniformGrid(Element):

    def __init__(self, extension_x, extension_y, nx, ny):

        self.nx, self.ny, self.n_points = nx, ny, nx * ny

        self.ex = np.zeros((ny, nx))
        self.ey = np.zeros((ny, nx))
        self.ez = np.zeros((ny, nx))
        self.rho = np.zeros((ny, nx))
        self.phi = np.zeros((ny, nx))

        self.dx = 2 * extension_x / (nx - 1)
        self.dy = 2 * extension_y / (ny - 1)
        mx = np.arange(-extension_x, extension_x + self.dx, self.dx)
        my = np.arange(-extension_y, extension_y + self.dy, self.dy)
        self.x, self.y = np.meshgrid(mx, my)

        from types import MethodType
        UniformGrid.gather_from = MethodType(gather_from, None, UniformGrid)
        UniformGrid.scatter_to = MethodType(scatter_to, None, UniformGrid)

    def gather(self, bunch, i_slice):
        '''
        Cell
        3 ------------ 4
        |     |        |
        |     |        |
        |     |        |
        |     |        |
        |-----x--------|
        |     |        |
        1 ------------ 2
        '''

        # Initialise
        self.rho[:] = 0

        # H, xedges, yedges = np.histogram2d(ix, iy, bins=self.rho.shape)
        x0, x1 = self.x[0,0] - self.dx / 2, self.x[0,-1] + self.dx / 2
        y0, y1 = self.y[0,0] - self.dy / 2, self.y[-1,0] + self.dy / 2
        xedges = np.linspace(x0, x1, self.nx + 1)
        yedges = np.linspace(y0, y1, self.ny + 1)
        H, xedges, yedges = np.histogram2d(bunch.x, bunch.y, bins=(xedges, yedges))

        self.rho += H.T

    # @profile
    def track(self, bunch):

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

        for i in xrange(bunch.slices.n_slices):
            self.gather(bunch, i)
        print self.x.shape, self.y.shape, self.rho.shape
        ax1.contourf(self.x, self.y, self.rho.T.T, 100)
        ax1.scatter(bunch.x, bunch.y, c='y', marker='.', alpha=0.1, lw=0)
        ax1.set_aspect('equal')

        for i in xrange(bunch.slices.n_slices):
            self.fastgather(bunch.x, bunch.y)
        ax2.contourf(self.x, self.y, self.rho, 100)
        ax2.scatter(bunch.x, bunch.y, c='y', marker='.', alpha=0.1, lw=0)
        ax2.set_aspect('equal')

        plt.show()
        exit(-1)


class AdaptiveGrid(object):
    '''
    classdocs
    '''

    def __init__(self, extension_x, extension_y, nx, ny):
        '''
        l = sum_{k=1}^n dx_k
        dx_k = m * k
        => l = m * sum_{k=1}^n k = m * (n * (n + 1) / 2); m = 2 * l / (n * (n + 1))
        '''
        nx, ny = nx // 2, ny // 2
        mx = 2 * extension_x / (nx * (nx + 1))
        my = 2 * extension_y / (ny * (ny + 1))
        # dxk = np.arange(1, nx + 1) * mx
        # dyk = np.arange(1, ny + 1) * my
        lx, ly = np.zeros(nx), np.zeros(ny)
        lx[0], ly[0] = mx, my
        for i in xrange(1, nx):
            lx[i] = lx[i - 1] + mx * (i + 1)
        for i in xrange(1, ny):
            ly[i] = ly[i - 1] + my * (i + 1)

        self.mx = np.hstack((-lx[::-1], lx))
        self.my = np.hstack((-ly[::-1], ly))
