#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
//#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
// namespace helper {struct between{const int minimum;const int maximum;__host__ __device__ between(int min, int max): minimum(min), maximum(max) {};__host__ __device__ bool operator()(const int x) {return !(x < minimum) && !(maximum < x);};};};

extern "C" { //required to avoid name mangling in PyCUDA: https://devtalk.nvidia.com/default/topic/471412/pycuda-thrust-example-in-case-someone-is-curious/

void thrust_sort_double(double* input_ptr, int length)
{
  thrust::device_ptr<double> thrust_ptr(input_ptr);
  thrust::sort(thrust_ptr, thrust_ptr + length);
}
void thrust_sort_by_key_double(double* key_ptr, int length, double* val_ptr)
{
  thrust::device_ptr<double> thrust_key_ptr(key_ptr);
  thrust::device_ptr<double> thrust_val_ptr(val_ptr);
  thrust::sort_by_key(thrust_key_ptr, thrust_key_ptr + length, thrust_val_ptr);
}
void thrust_get_sort_perm_double(double* input_ptr, int length, int* perm_ptr)
{
  thrust::device_ptr<double> thrust_ptr(input_ptr);
  thrust::device_ptr<int> indices(perm_ptr);
  thrust::sequence(indices, indices + length);
  thrust::sort_by_key(thrust_ptr, thrust_ptr + length, indices);
}
void thrust_get_sort_perm_int(int* input_ptr, int length, int* perm_ptr)
{
  thrust::device_ptr<int> thrust_ptr(input_ptr);
  thrust::device_ptr<int> indices(perm_ptr);
  thrust::sequence(indices, indices + length);
  thrust::sort_by_key(thrust_ptr, thrust_ptr + length, indices);
}
void thrust_apply_sort_perm_double(double* input_ptr, int length, double* output_ptr, int* perm_ptr)
{
  thrust::device_ptr<double> thrust_input_ptr(input_ptr);
  thrust::device_ptr<double> thrust_output_ptr(output_ptr);
  thrust::device_ptr<int> indices(perm_ptr);
  thrust::gather(indices, indices + length, thrust_input_ptr, thrust_output_ptr);
}
void thrust_apply_sort_perm_int(int* input_ptr, int length, int* output_ptr, int* perm_ptr)
{
  thrust::device_ptr<int> thrust_input_ptr(input_ptr);
  thrust::device_ptr<int> thrust_output_ptr(output_ptr);
  thrust::device_ptr<int> indices(perm_ptr);
  thrust::gather(indices, indices + length, thrust_input_ptr, thrust_output_ptr);
}
void thrust_lower_bound_int(int* sorted_ptr, int sorted_length, int* bounds_ptr, int bounds_length, int* output_ptr)
{
  thrust::device_ptr<int> thrust_sorted_ptr(sorted_ptr);
  thrust::device_ptr<int> thrust_bounds_ptr(bounds_ptr);
  thrust::device_ptr<int> thrust_output_ptr(output_ptr);
  thrust::lower_bound(thrust_sorted_ptr, thrust_sorted_ptr + sorted_length, thrust_bounds_ptr, thrust_bounds_ptr + bounds_length, thrust_output_ptr);
}
void thrust_upper_bound_int(int* sorted_ptr, int sorted_length, int* bounds_ptr, int bounds_length, int* output_ptr)
{
  thrust::device_ptr<int> thrust_sorted_ptr(sorted_ptr);
  thrust::device_ptr<int> thrust_bounds_ptr(bounds_ptr);
  thrust::device_ptr<int> thrust_output_ptr(output_ptr);
  thrust::upper_bound(thrust_sorted_ptr, thrust_sorted_ptr + sorted_length, thrust_bounds_ptr, thrust_bounds_ptr + bounds_length, thrust_output_ptr);
}
void thrust_cumsum_double(double* data_ptr, int length, double* sum_ptr)
{
  thrust::device_ptr<double> thrust_data_ptr(data_ptr);
  thrust::device_ptr<double> thrust_sum_ptr(sum_ptr);
  thrust::inclusive_scan(thrust_data_ptr, thrust_data_ptr + length, thrust_sum_ptr);
}
void thrust_cumsum_int(int* data_ptr, int length, int* sum_ptr)
{
  thrust::device_ptr<int> thrust_data_ptr(data_ptr);
  thrust::device_ptr<int> thrust_sum_ptr(sum_ptr);
  thrust::inclusive_scan(thrust_data_ptr, thrust_data_ptr + length, thrust_sum_ptr);
}
} // end extern "C"
