'''
@authors: Adrian Oeftiger
@date:    02/10/2014

Provide useful decorators for PyHEADTAIL.
'''

import warnings
from functools import wraps
from ..gpu import gpu_utils


def deprecated(message):
    '''Deprecation warning as described in warnings documentation.
    '''
    def deprecated_decorator(func):
        @wraps(func)
        def deprecated_wrapper(*args, **kwargs):
            if func.__name__ == "__init__":
                name = args[0].__name__
            else:
                name = func.__name__
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn('\n\n*** DEPRECATED: "{:s}" will be replaced in a future '
                          'PyHEADTAIL release!'.format(name),
                          category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            print message
            return func(*args, **kwargs)
        return deprecated_wrapper
    return deprecated_decorator


def memoize(function):
    '''Memoizes the output of a function for given arguments (no keyword arguments)
    and returns the correspondingly saved value after the first evaluation.
    '''
    store = {}

    @wraps(function)
    def evaluate(*args):
        signature = (args)
        if not store.has_key(signature):
            store[signature] = function(*args)
        return store[signature]
    return evaluate

def synchronize_gpu_streams_before(func):
    '''
    Use this decorator if you need the results of all the streams
    synchronized before this function is called
    '''
    def sync_before_wrap(*args, **kwargs):
        for stream in gpu_utils.streams:
            stream.synchronize()
        return func(*args, **kwargs)
    return sync_before_wrap

def synchronize_gpu_streams_after(func):
    '''
    Use this decorator if you need the results of all the streams
    synchronized after this function is called
    '''
    def sync_after_wrap(*args, **kwargs):
        res = func(*args, **kwargs)
        for stream in gpu_utils.streams:
            stream.synchronize()
        return res
    return sync_after_wrap
