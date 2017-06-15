'''Translates PyPIC/GPU calls to the corresponding GPU or CPU
class depending on the current context (in pmath).

PyPIC can be found under https://github.com/PyCOMPLETE/PyPIC .

@authors: Adrian Oeftiger
@date:    13.06.2017
'''

from ..general import pmath as pm
from ..general.contextmanager import UnknownContextManagerError

from PyPIC.GPU import pypic

def make_PyPIC(*args, **kwargs):
    '''Factory method for PyPIC.GPU.pypic.PyPIC(_GPU) classes.
    Return PyPIC_GPU instance launched with args and kwargs if current
    context is on the GPU, otherwise return PyPIC instance.
    '''
    if pm.device is 'CPU':
        return pypic.PyPIC(*args, **kwargs)
    elif pm.device is 'GPU':
        from pycuda.autoinit import context
        if 'context' not in kwargs:
            kwargs.update(context=context)
        return pypic.PyPIC_GPU(*args, **kwargs)
    else:
        raise UnknownContextManagerError()
