from __future__ import division

from ..trackers import wrapper as def_wrapper

from pycuda.elementwise import ElementwiseKernel

class LongWrapperGPU(def_wrapper.LongWrapper):
    '''Wrap particles that go out of the z range covering the circumference.
    GPU version.'''
    def __init__(self, *args, **kwargs):
        super(LongWrapperGPU, self).__init__(*args, **kwargs)
        self._wrap = ElementwiseKernel(
            'double *z',
            'z[i] -= floor((z[i] - {z_min:}) / {circ}) * {circ}'.format(
                circ=self.circumference,
                z_min=self.z_min
            ),
            'wrap_z'
        )

    def track(self, beam):
        self._wrap(beam.z)
