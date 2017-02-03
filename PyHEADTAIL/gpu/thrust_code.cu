#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
//#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cmath>
#include "math.h"

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

// ---------------- slice statistics using thrust ---------------- //

// StackOverflow inspired:
// http://stackoverflow.com/questions/12380966/standard-deviation-using-cuda

typedef double T;

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// structure used to accumulate the moments and other
// statistical properties encountered so far.
struct summary_stats_data
{
    T n;
    T mean;
    T M2;

    // initialize to the identity element
    void initialize()
    {
      n = mean = M2 = 0;
    }

    __host__ __device__
    T variance()   { return M2 / (n - 1); }
    __host__ __device__
    T variance_n() { return M2 / n; }
};

// stats_unary_op is a functor that takes in a value x and
// returns a summary_stats_data whose mean value is initialized to x.
struct summary_stats_unary_op
{
    __host__ __device__
    summary_stats_data operator()(const T& x) const
    {
         summary_stats_data result;
         result.n    = 1;
         result.mean = x;
         result.M2   = 0;

         return result;
    }
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for
// all values that have been agregated so far
struct summary_stats_binary_op
    : public thrust::binary_function<const summary_stats_data&,
                                     const summary_stats_data&,
                                           summary_stats_data >
{
    __host__ __device__
    summary_stats_data operator()(const summary_stats_data& x,
                                  const summary_stats_data& y) const
    {
        summary_stats_data result;

        // precompute some common subexpressions
        T n  = x.n + y.n;

        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;

        //Basic number of samples (n)
        result.n   = n;

        result.mean = x.mean + delta * y.n / n;

        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};

struct extract_mean
{
    __host__ __device__
    T operator()(summary_stats_data& x)
    {
        return x.mean;
    }
};
struct extract_std
{
    __host__ __device__
    T operator()(summary_stats_data& x)
    {
        if (x.n < 1.001) {
            return 0.;
        } else {
        return std::sqrt(x.variance());
        }
    }
};

void thrust_stats_per_slice(
    int* particle_slice_id_ptr, double* u, const int n_mp,
//    int* n_macroparticles_per_slice, const int n_slices, // inputs end
    int* slice_id_ptr, double* slice_mean_ptr,
    double* slice_std_ptr, int* n_relevant_entries // outputs end
) {
    // set up arguments
    summary_stats_unary_op  unary_op;
    summary_stats_binary_op binary_op;
    thrust::equal_to<int> binary_pred;
    extract_mean unary_extract_mean;
    extract_std unary_extract_std;

    // input pointers
    thrust::device_ptr<double> thrust_u(u);
    thrust::device_ptr<int> thrust_p_sid(particle_slice_id_ptr);

    // output pointers
    thrust::device_ptr<int> thrust_sid(slice_id_ptr);
    thrust::device_ptr<double> thrust_mean(slice_mean_ptr);
    thrust::device_ptr<double> thrust_std(slice_std_ptr);

    // intermediate summary_stats_data type arrays
    thrust::device_vector<summary_stats_data> stats_vec(n_mp);
    thrust::device_vector<summary_stats_data> stats_vec_out(n_mp);

    // convert array to summary_stats_data type
    thrust::transform(thrust_u, thrust_u + n_mp, stats_vec.begin(), unary_op);

    // pointers to end of relevant reduced entries
    typedef thrust::device_ptr<int> Iterator1;
    typedef thrust::device_vector<summary_stats_data>::iterator Iterator2;
    thrust::pair<Iterator1, Iterator2> new_end;

    // compute statistics for each slice
    new_end = thrust::reduce_by_key(
      thrust_p_sid,
      thrust_p_sid + n_mp,
      stats_vec.begin(),
      thrust_sid,
      stats_vec_out.begin(),
      binary_pred,
      binary_op
    );

    // how many relevant reduced entries in the output arrays:
    *n_relevant_entries = new_end.first - thrust_sid;

    // extract results and write to output arrays
    thrust::transform(stats_vec_out.begin(), new_end.second,
                      thrust_mean, unary_extract_mean);
    thrust::transform(stats_vec_out.begin(), new_end.second,
                      thrust_std, unary_extract_std);
}

} // end extern "C"
