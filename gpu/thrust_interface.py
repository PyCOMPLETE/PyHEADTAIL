from cgen import FunctionBody, \
        FunctionDeclaration, Typedef, POD, Value, \
        Pointer, Module, Block, Initializer, Assign, \
        Include, Statement, If
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule

import codepy.jit, codepy.toolchain

# INFO: the following code goes much along the tutorial to be found at
# http://wiki.tiker.net/PyCuda/Examples/ThrustInterop
# many thanks to Bryan Catanzaro!

#Make a host_module, compiled for CPU
host_mod = BoostPythonModule()

#Make a device module, compiled with NVCC
nvcc_mod = CudaModule(host_mod)

#Describe device module code
#NVCC includes
nvcc_includes = [
    'thrust/sort.h',
    'thrust/binary_search.h',
    'thrust/device_vector.h',
    'cuda.h',
    'thrust/copy.h',
    'thrust/device_ptr.h',
    'thrust/sequence.h',
    #'helper.h'
    ]
#Add includes to module
nvcc_mod.add_to_preamble([Include(x) for x in nvcc_includes])

device_structs = [
'namespace helper {'

	'struct between'
	'{'
		'const int minimum;'
		'const int maximum;'
		'__host__ __device__ between(int min, int max): minimum(min), maximum(max) {};'
	    '__host__ __device__ bool operator()(const int x) {return !(x < minimum) && !(maximum < x);};'
	'};'
'}'
]
nvcc_mod.add_to_preamble([Statement(x) for x in device_structs])

#NVCC function
nvcc_functions = [
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_sort_double'),
                            [Value('double*', 'input_ptr'),
                             Value('int', 'length')]),
        Block([Statement('thrust::device_ptr<double> thrust_ptr(input_ptr)'),
               Statement('thrust::sort(thrust_ptr, thrust_ptr + length)')])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_sort_by_key_double'),
                            [Value('double*', 'key_ptr'),
                             Value('int', 'length'),
                             Value('double*', 'val_ptr')]),
        Block([Statement('thrust::device_ptr<double> thrust_key_ptr(key_ptr)'),
               Statement('thrust::device_ptr<double> thrust_val_ptr(val_ptr)'),
               Statement('thrust::sort_by_key(thrust_key_ptr, thrust_key_ptr + length, thrust_val_ptr)')])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_get_sort_perm_double'),
                            [Value('double*', 'input_ptr'),
                             Value('int', 'length'),
                             Value('int*', 'perm_ptr')]),
        Block([Statement('thrust::device_ptr<double> thrust_ptr(input_ptr)'),
               Statement('thrust::device_ptr<int> indices(perm_ptr)'),
               Statement('thrust::sequence(indices, indices + length)'),
               Statement('thrust::sort_by_key(thrust_ptr, thrust_ptr + length, indices)'),
              ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_get_sort_perm_int'),
                            [Value('int*', 'input_ptr'),
                             Value('int', 'length'),
                             Value('int*', 'perm_ptr')]),
        Block([Statement('thrust::device_ptr<int> thrust_ptr(input_ptr)'),
               Statement('thrust::device_ptr<int> indices(perm_ptr)'),
               Statement('thrust::sequence(indices, indices + length)'),
               Statement('thrust::sort_by_key(thrust_ptr, thrust_ptr + length, indices)'),
              ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_apply_sort_perm_double'),
                            [Value('double*', 'input_ptr'),
                             Value('int', 'length'),
                             Value('double*', 'output_ptr'),
                             Value('int*', 'perm_ptr')]),
        Block([Statement('thrust::device_ptr<double> thrust_input_ptr(input_ptr)'),
               Statement('thrust::device_ptr<double> thrust_output_ptr(output_ptr)'),
               Statement('thrust::device_ptr<int> indices(perm_ptr)'),
               Statement('thrust::gather(indices, indices + length, thrust_input_ptr, thrust_output_ptr)'),
              ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_apply_sort_perm_int'),
                            [Value('int*', 'input_ptr'),
                             Value('int', 'length'),
                             Value('int*', 'output_ptr'),
                             Value('int*', 'perm_ptr')]),
        Block([Statement('thrust::device_ptr<int> thrust_input_ptr(input_ptr)'),
               Statement('thrust::device_ptr<int> thrust_output_ptr(output_ptr)'),
               Statement('thrust::device_ptr<int> indices(perm_ptr)'),
               Statement('thrust::gather(indices, indices + length, thrust_input_ptr, thrust_output_ptr)'),
              ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_lower_bound_int'),
                            [Value('int*', 'sorted_ptr'),
                             Value('int', 'sorted_length'),
                             Value('int*', 'bounds_ptr'),
                             Value('int', 'bounds_length'),
                             Value('int*', 'output_ptr')]),
        Block([Statement('thrust::device_ptr<int> thrust_sorted_ptr(sorted_ptr)'),
               Statement('thrust::device_ptr<int> thrust_bounds_ptr(bounds_ptr)'),
               Statement('thrust::device_ptr<int> thrust_output_ptr(output_ptr)'),
               Statement('thrust::lower_bound('
                             'thrust_sorted_ptr, '
                             'thrust_sorted_ptr + sorted_length, '
                             'thrust_bounds_ptr, '
                             'thrust_bounds_ptr + bounds_length, '
                             'thrust_output_ptr)'),
              ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'thrust_upper_bound_int'),
                            [Value('int*', 'sorted_ptr'),
                             Value('int', 'sorted_length'),
                             Value('int*', 'bounds_ptr'),
                             Value('int', 'bounds_length'),
                             Value('int*', 'output_ptr')]),
        Block([Statement('thrust::device_ptr<int> thrust_sorted_ptr(sorted_ptr)'),
               Statement('thrust::device_ptr<int> thrust_bounds_ptr(bounds_ptr)'),
               Statement('thrust::device_ptr<int> thrust_output_ptr(output_ptr)'),
               Statement('thrust::upper_bound('
                             'thrust_sorted_ptr, '
                             'thrust_sorted_ptr + sorted_length, '
                             'thrust_bounds_ptr, '
                             'thrust_bounds_ptr + bounds_length, '
                             'thrust_output_ptr)'),
              ])),
    FunctionBody(
        FunctionDeclaration(Value('int', 'thrust_copy_if_min_max'),
                            [Value('int*', 'input_ptr'),
                             Value('int', 'input_length'),
                             Value('int*', 'output_ptr'),
                             Value('int', 'mini'),
                             Value('int', 'maxi')]),
        Block([Statement(x) for x in
            [
                # get device ptr
                'thrust::device_ptr<int> d_input_ptr(input_ptr)',
                'thrust::device_ptr<int> d_output_ptr(output_ptr)',
                'thrust::device_ptr<int> d_end_input_ptr(d_input_ptr+input_length)',

                # define the predicate

                'thrust::device_ptr<int> output_end = thrust::copy_if(d_input_ptr, d_end_input_ptr, d_output_ptr, helper::between(mini, maxi))',
                'int output_length = output_end - d_output_ptr',
                # 'int N = 10',
                # 'thrust::device_vector<int> v1(N)',
                # 'thrust::sequence(v1.begin(), v1.end())',
                # 'thrust::device_vector<int> result(N)',
                # 'thrust::device_ptr<int> d_in = v1.data()',
                # 'thrust::device_ptr<int> d_out = result.data()',
                # 'thrust::copy_if(d_in, d_in + N, d_out, helper::between(0,2))',



                'return output_length',
            ]
              ])),
]


#Add declaration to nvcc_mod
#Adds declaration to host_mod as well
for fct in nvcc_functions:
    nvcc_mod.add_function(fct)

host_includes = [
    'boost/python/extract.hpp',
    ]
#Add host includes to module
host_mod.add_to_preamble([Include(x) for x in host_includes])

host_namespaces = [
    'namespace p = boost::python',
    ]

#Add BPL using statement
host_mod.add_to_preamble([Statement(x) for x in host_namespaces])

host_functions = [
    FunctionBody(
        FunctionDeclaration(Value('p::object', 'sort_double'),
                            [Value('p::object', 'gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(gpu_array.attr("shape"))',
                'int length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr ptr = p::extract<CUdeviceptr>(gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_sort_double((double*) ptr, length)',
                #Return result
                'return gpu_array',
            ]
        ])),
    FunctionBody( # IMPORTANT INFO: thrust::sort_by_key modifies also the key_gpu_array!
        FunctionDeclaration(Value('p::object', 'sort_by_key_double'),
                            [Value('p::object', 'key_gpu_array'),
                             Value('p::object', 'val_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(key_gpu_array.attr("shape"))',
                'int length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr key_ptr = p::extract<CUdeviceptr>(key_gpu_array.attr("ptr"))',
                'CUdeviceptr val_ptr = p::extract<CUdeviceptr>(val_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_sort_by_key_double((double*) key_ptr, length, (double*) val_ptr)',
                #Return result
                'return val_gpu_array',
            ]
        ])),
    FunctionBody( # IMPORTANT INFO: thrust::sort_by_key modifies also the gpu_array!
        FunctionDeclaration(Value('void', 'get_sort_perm_double'),
                            [Value('p::object', 'gpu_array'),
                             Value('p::object', 'perm_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(gpu_array.attr("shape"))',
                'int length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr ptr = p::extract<CUdeviceptr>(gpu_array.attr("ptr"))',
                'CUdeviceptr perm_ptr = p::extract<CUdeviceptr>(perm_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_get_sort_perm_double((double*) ptr, length, (int*) perm_ptr)',
            ]
        ])),
    FunctionBody( # IMPORTANT INFO: thrust::sort_by_key modifies also the gpu_array!
        FunctionDeclaration(Value('void', 'get_sort_perm_int'),
                            [Value('p::object', 'gpu_array'),
                             Value('p::object', 'perm_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(gpu_array.attr("shape"))',
                'int length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr ptr = p::extract<CUdeviceptr>(gpu_array.attr("ptr"))',
                'CUdeviceptr perm_ptr = p::extract<CUdeviceptr>(perm_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_get_sort_perm_int((int*) ptr, length, (int*) perm_ptr)',
            ]
        ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'apply_sort_perm_double'),
                            [Value('p::object', 'input_gpu_array'),
                             Value('p::object', 'output_gpu_array'),
                             Value('p::object', 'perm_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(input_gpu_array.attr("shape"))',
                'int length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr input_ptr = p::extract<CUdeviceptr>(input_gpu_array.attr("ptr"))',
                'CUdeviceptr output_ptr = p::extract<CUdeviceptr>(output_gpu_array.attr("ptr"))',
                'CUdeviceptr perm_ptr = p::extract<CUdeviceptr>(perm_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_apply_sort_perm_double((double*) input_ptr, length, '
                                              '(double*) output_ptr, (int*) perm_ptr)',
            ]
        ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'apply_sort_perm_int'),
                            [Value('p::object', 'input_gpu_array'),
                             Value('p::object', 'output_gpu_array'),
                             Value('p::object', 'perm_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(input_gpu_array.attr("shape"))',
                'int length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr input_ptr = p::extract<CUdeviceptr>(input_gpu_array.attr("ptr"))',
                'CUdeviceptr output_ptr = p::extract<CUdeviceptr>(output_gpu_array.attr("ptr"))',
                'CUdeviceptr perm_ptr = p::extract<CUdeviceptr>(perm_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_apply_sort_perm_int((int*) input_ptr, length, '
                                           '(int*) output_ptr, (int*) perm_ptr)',
            ]
        ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'lower_bound_int'),
                            [Value('p::object', 'sorted_gpu_array'),
                             Value('p::object', 'bounds_gpu_array'),
                             Value('p::object', 'output_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple sorted_shape = p::extract<p::tuple>(sorted_gpu_array.attr("shape"))',
                'int sorted_length = p::extract<int>(sorted_shape[0])',
                'p::tuple bounds_shape = p::extract<p::tuple>(bounds_gpu_array.attr("shape"))',
                'int bounds_length = p::extract<int>(bounds_shape[0])',
                #Get data pointer
                'CUdeviceptr sorted_ptr = p::extract<CUdeviceptr>(sorted_gpu_array.attr("ptr"))',
                'CUdeviceptr bounds_ptr = p::extract<CUdeviceptr>(bounds_gpu_array.attr("ptr"))',
                'CUdeviceptr output_ptr = p::extract<CUdeviceptr>(output_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_lower_bound_int((int*) sorted_ptr, sorted_length, '
                                       '(int*) bounds_ptr, bounds_length, (int*) output_ptr)',
            ]
        ])),
    FunctionBody(
        FunctionDeclaration(Value('void', 'upper_bound_int'),
                            [Value('p::object', 'sorted_gpu_array'),
                             Value('p::object', 'bounds_gpu_array'),
                             Value('p::object', 'output_gpu_array')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple sorted_shape = p::extract<p::tuple>(sorted_gpu_array.attr("shape"))',
                'int sorted_length = p::extract<int>(sorted_shape[0])',
                'p::tuple bounds_shape = p::extract<p::tuple>(bounds_gpu_array.attr("shape"))',
                'int bounds_length = p::extract<int>(bounds_shape[0])',
                #Get data pointer
                'CUdeviceptr sorted_ptr = p::extract<CUdeviceptr>(sorted_gpu_array.attr("ptr"))',
                'CUdeviceptr bounds_ptr = p::extract<CUdeviceptr>(bounds_gpu_array.attr("ptr"))',
                'CUdeviceptr output_ptr = p::extract<CUdeviceptr>(output_gpu_array.attr("ptr"))',
                #Call Thrust routine, compiled into the CudaModule
                'thrust_upper_bound_int((int*) sorted_ptr, sorted_length, '
                                       '(int*) bounds_ptr, bounds_length, (int*) output_ptr)',
            ]
        ])),

    FunctionBody(
        FunctionDeclaration(Value('int', 'copy_if_min_max'),
                            [Value('p::object', 'input_gpuarray'),
                             Value('p::object', 'output_gpuarray'),
                             Value('int', 'minimum'),
                             Value('int', 'maximum')]),
        Block([Statement(x) for x in
            [
                #Extract information from PyCUDA GPUArray
                #Get length
                'p::tuple shape = p::extract<p::tuple>(input_gpuarray.attr("shape"))',
                'int input_length = p::extract<int>(shape[0])',
                #Get data pointer
                'CUdeviceptr input_ptr = p::extract<CUdeviceptr>(input_gpuarray.attr("ptr"))',
                'CUdeviceptr output_ptr = p::extract<CUdeviceptr>(output_gpuarray.attr("ptr"))',

                'return thrust_copy_if_min_max((int*)input_ptr, input_length,'
                                              '(int*)output_ptr, minimum, maximum)',
            ]
        ])),
]

for fct in host_functions:
    host_mod.add_function(fct)

gcc_toolchain = codepy.toolchain.guess_toolchain()
nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()

# COMPILED CODE:
'''Compiled thrust functionality, use this module to access thrust
functions.
'''
compiled_module = nvcc_mod.compile(gcc_toolchain, nvcc_toolchain, debug=False)
