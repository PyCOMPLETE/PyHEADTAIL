'''
GPU Utils
Memory pool, ...
This could also be the place to store the context, device, streams, etc...
The module is automatically a singleton
@author Stefan Hegglin
'''
try:
    import pycuda.tools
    has_pycuda = True
    try:
        pycuda.driver.mem_get_info()
    except pycuda._driver.LogicError: #the error pycuda throws if no context initialized
        print ('No context initialized. Please import pycuda.autoinit at the '
               'beginning of your script if you want to use GPU functionality')
        has_pycuda=False
except ImportError:
    has_pycuda = False

################################################################################

if has_pycuda:
    memory_pool = pycuda.tools.DeviceMemoryPool()
