'''
@authors: Adrian Oeftiger
@date:    02/10/2014

Provide useful decorators for PyHEADTAIL.
'''

import warnings
from functools import wraps


def deprecated(message):
    '''Deprecation warning as described in warnings documentaion.
    '''
    def deprecated_decorator(func):
        @wraps(func)
        def deprecated_wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn('\n\n*** DEPRECATED function: ' +
                          '"{:s}" '.format(func.__name__) +
                          'will be replaced in one of the future ' +
                          'PyHEADTAIL releases!',
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
