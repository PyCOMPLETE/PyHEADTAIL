from __future__ import division


import numpy as np


class UniformGrid(object):
    '''
    classdocs
    '''
    
    def __init__(self, extension_x, extension_y, nx, ny):
        
        dx = 2 * extension_x / (nx - 1)
        dy = 2 * extension_y / (ny - 1)

        self.mx = np.arange(-extension_x, extension_x + dx, dx)
        self.my = np.arange(-extension_y, extension_y + dy, dy)


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
