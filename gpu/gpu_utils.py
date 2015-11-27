'''
GPU Utils
Memory pool, ...
This could also be the place to store the context, device, streams, etc...
The module is automatically a singleton
@author Stefan Hegglin
'''
try:
    import pycuda.tools
    import pycuda.driver as drv
    has_pycuda = True
    try:
        pycuda.driver.mem_get_info()
    except pycuda._driver.LogicError: #the error pycuda throws if no context initialized
        print ('No context initialized. Please import pycuda.autoinit at the '
               'beginning of your script if you want to use GPU functionality')
        has_pycuda=False
except ImportError:
    has_pycuda = False
import atexit
################################################################################

if has_pycuda:
    memory_pool = pycuda.tools.DeviceMemoryPool()

    import skcuda.misc #s
    skcuda.misc.init(allocator=memory_pool.allocate)
    atexit.register(skcuda.misc.shutdown)

    n_streams = 3
    streams = [drv.Stream() for i in xrange(n_streams)]
