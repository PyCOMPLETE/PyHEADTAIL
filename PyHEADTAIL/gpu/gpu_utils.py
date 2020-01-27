'''
GPU Utils
Memory pool, ...
This could also be the place to store the context, device, streams, etc...
The module is automatically a singleton
@author Stefan Hegglin
'''

import atexit
from itertools import cycle

try:
    import pycuda.tools
    import pycuda.driver as drv
    import pycuda.elementwise
    has_pycuda = True
    try:
        pycuda.driver.mem_get_info()
    except pycuda._driver.LogicError: #the error pycuda throws if no context initialized
        # print ('No context initialized. Please import pycuda.autoinit at the '
        #        'beginning of your script if you want to use GPU functionality')
        has_pycuda = False
except ImportError:
    has_pycuda = False

################################################################################

if has_pycuda:

    device = pycuda.autoinit.device
    context = pycuda.autoinit.context
    driver = drv

    memory_pool = pycuda.tools.DeviceMemoryPool()

    import skcuda.misc #s
    skcuda.misc.init(allocator=memory_pool.allocate)
    atexit.register(skcuda.misc.shutdown)

    n_streams = 4
    streams = [drv.Stream() for i in range(n_streams)]
    stream_pool = cycle(streams)

    stream_emittance = [drv.Stream() for i in range(6)]


    def dummy_1(gpuarr, stream=None):
        __dummy1(gpuarr, stream=stream)
        return gpuarr

    def dummy_2(gpuarr, stream=None):
        __dummy2(gpuarr, stream=stream)
        return gpuarr


else:
    streams = [] # this way nothing bad happens if 'for stream in streams: sync'

################################################################################
