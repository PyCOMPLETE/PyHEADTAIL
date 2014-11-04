
from functools import wraps

def memoize(function):
    '''Memoizes the output of a function for given arguments
    (no keyword arguments) and returns the correspondingly saved value
    after the first evaluation.
    '''
    store = {}
    @wraps(function)
    def evaluate(*args):
        signature = (args)
        if not store.has_key(signature):
            store[signature] = function(*args)
        return store[signature]
    return evaluate
