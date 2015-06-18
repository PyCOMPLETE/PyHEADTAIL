from __future__ import division
'''
Created on 08.01.2014

@author: Kevin Li
'''


import numpy as np


import copy
from grid import UniformGrid
from compute_potential_fgreenm2m import compute_potential_fgreenm2m, compute_potential_fgreenp2m


class PoissonFFT(UniformGrid):
    '''
    FFT Poisson solver operates on a grid
    '''

    # @profile
    def __init__(self, *args, **kwargs):
        '''
        Constructor
        '''
        super(PoissonFFT, self).__init__(*args, **kwargs)

        self.tmprho = np.zeros((2 * self.ny, 2 * self.nx))
        self.fgreen = np.zeros((2 * self.ny, 2 * self.nx))

        mx = -self.dx / 2 + np.arange(self.nx + 1) * self.dx
        my = -self.dy / 2 + np.arange(self.ny + 1) * self.dy
        x, y = np.meshgrid(mx, my)
        r2 = x ** 2 + y ** 2
        # Antiderivative
        tmpfgreen = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
                   + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy

        # Integration and circular Green's function
        self.fgreen[:self.ny, :self.nx] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
        self.fgreen[self.ny:, :self.nx] = self.fgreen[self.ny:0:-1, :self.nx]
        self.fgreen[:self.ny, self.nx:] = self.fgreen[:self.ny, self.nx:0:-1]
        self.fgreen[self.ny:, self.nx:] = self.fgreen[self.ny:0:-1, self.nx:0:-1]
        # # Would expect to be fully symmetric
        # self.fgreen[self.nx:, :self.ny] = self.fgreen[self.nx - 1::-1, :self.ny]
        # self.fgreen[:self.nx, self.ny:] = self.fgreen[:self.nx, self.ny - 1::-1]
        # self.fgreen[self.nx:, self.ny:] = self.fgreen[self.nx - 1::-1, self.ny - 1::-1]

        from types import MethodType
        PoissonFFT.compute_potential_fgreenm2m = MethodType(compute_potential_fgreenm2m, None, PoissonFFT)
        PoissonFFT.compute_potential_fgreenp2m = MethodType(compute_potential_fgreenp2m, None, PoissonFFT)

        try:
            import pyfftw

            # Arrays
            self.fftw_fgreen = pyfftw.n_byte_align_empty((2 * self.ny, 2 * self.nx), 16, 'complex128')
            self.fftw_rho = pyfftw.n_byte_align_empty((2 * self.ny, 2 * self.nx), 16, 'complex128')
            self.fftw_phi = pyfftw.n_byte_align_empty((2 * self.ny, 2 * self.nx), 16, 'complex128')
            self.ifftw_fgreen = pyfftw.n_byte_align_empty((2 * self.ny, 2 * self.nx), 16, 'complex128')
            self.ifftw_rho = pyfftw.n_byte_align_empty((2 * self.ny, 2 * self.nx), 16, 'complex128')
            self.ifftw_phi = pyfftw.n_byte_align_empty((2 * self.ny, 2 * self.nx), 16, 'complex128')

            # Plans
            self.pfftw_fgreen = pyfftw.FFTW(self.fftw_fgreen, self.ifftw_fgreen, axes=(0,1))#, flags=('FFTW_EXHAUSTIVE',), threads=1)
            self.pfftw_rho = pyfftw.FFTW(self.fftw_rho, self.ifftw_rho, axes=(0,1))#, flags=('FFTW_EXHAUSTIVE',), threads=1)
            self.pfftw_phi = pyfftw.FFTW(self.ifftw_phi, self.fftw_phi, axes=(0,1), direction='FFTW_BACKWARD')#, flags=('FFTW_EXHAUSTIVE',), threads=1)

            self.compute_potential = self.compute_potential_fftw
        except ImportError:
            print '*** WARNING: pyfftw not available. Falling back to NumPy FFT.'
            self.compute_potential = self.compute_potential_numpy

    # @profile
    def compute_potential_numpy(self, rho, phi):

        self.tmprho[:self.ny, :self.nx] = rho

        fftphi = np.fft.fft2(self.tmprho) * np.fft.fft2(self.fgreen)

        tmpphi = np.fft.ifft2(fftphi)
        phi[:] = np.abs(tmpphi[:self.ny, :self.nx])

    # @profile
    def compute_potential_fftw(self, rho, phi):

        # Fill arrays
        self.fftw_fgreen[:] = self.fgreen
        self.fftw_rho[:self.ny, :self.nx] = rho

        # Solve
        tmpfgreen = self.pfftw_fgreen()
        tmprho = self.pfftw_rho()

        self.ifftw_phi[:] = np.asarray(tmprho) * np.asarray(tmpfgreen)
        # self.ifftw_phi[:] = np.dot(tmprho, tmpfgreen)
        tmpphi = self.pfftw_phi()
        phi[:] = np.abs(tmpphi[:self.ny, :self.nx])

    def compute_fields(self, phi, ex, ey):

        ey[:], ex[:] = np.gradient(phi, self.dy, self.dx)
        ex[:] *= -1
        ey[:] *= -1

    # @profile
    def py_green_m2m(self):

        self.phi = np.zeros((self.ny, self.nx))

        da = self.dx * self.dy
        x, y = self.x.flatten(), self.y.flatten()
        phi, rho = self.phi.flatten(), self.rho.flatten()

        for i in xrange(self.n_points):
            xi, yi = x[i], y[i]
            for j in xrange(self.n_points):
                xj, yj = x[j], y[j]
                r2 = (xi - xj) ** 2 + (yi - yj) ** 2 + 1e-6
                phi[i] += (-rho[j] * da * 1 / 2 * np.log(r2)) # * self.dx * self.dy / (2 * np.pi))

        self.phi = np.reshape(phi, (self.ny, self.nx))
