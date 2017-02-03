__global__ void sorted_mean_per_slice(unsigned int* lower_bounds,
                                      unsigned int* upper_bounds,
                                      double* u,                    // array of particle quantity sorted by slice
                                      unsigned int n_slices,
                                      double* mean_u)               // output array of length n_slices with mean values for each slice
/**
    Iterate once through all the particles within the
    slicing region and calculate simultaneously the mean
    value of quantity u for each slice separately.

    Assumes the particle array u to be sorted by slices.

    The index arrays lower_bounds and upper_bounds
    indicate the start and end indices
    within the sorted particle arrays for each slice. The respective
    slice id is identical to the index within lower_bounds and
    upper_bounds.
*/
{
    double sum_u;
    unsigned int n_macroparticles; // in current slice
    for (int sid = blockIdx.x * blockDim.x + threadIdx.x;
         sid < n_slices;
         sid += blockDim.x * gridDim.x)
    {
        sum_u = 0;
        n_macroparticles = upper_bounds[sid] - lower_bounds[sid];
        if (n_macroparticles == 0) {
            mean_u[sid] = 0;
        } else {
            for (int pid = lower_bounds[sid]; pid < upper_bounds[sid]; pid++)
            {
                sum_u += u[pid];
            }
            mean_u[sid] = sum_u / n_macroparticles;
        }
    }
}

__global__ void sorted_std_per_slice(unsigned int* lower_bounds,
                                     unsigned int* upper_bounds,
                                     double* u,                     // array of particle quantity sorted by slice
                                     unsigned int n_slices,
                                     double* cov_u)                 // output array of length n_slices with mean values for each slice
/**
    Iterate once through all the particles within the
    slicing region and calculate simultaneously the
    standard deviation of quantity u for each slice separately.

    Assumes the particle array u to be sorted by slices.

    The index arrays lower_bounds and upper_bounds
    indicate the start and end indices
    within the sorted particle arrays for each slice. The respective
    slice id is identical to the index within lower_bounds and
    upper_bounds.
*/
{
    double sum_u, mean_u, l_cov_u, du;
    unsigned int n_macroparticles; // in current slice
    for (int sid = blockIdx.x * blockDim.x + threadIdx.x;
         sid < n_slices;
         sid += blockDim.x * gridDim.x)
    {
        sum_u = 0;
        n_macroparticles = upper_bounds[sid] - lower_bounds[sid];
        if (n_macroparticles <= 1) {
            cov_u[sid] = 0;
        } else {
            for (int pid = lower_bounds[sid]; pid < upper_bounds[sid]; pid++)
            {
                sum_u += u[pid];
            }
            mean_u = sum_u / n_macroparticles;

            l_cov_u = 0;
            for (int pid = lower_bounds[sid]; pid < upper_bounds[sid]; pid++)
            {
                du = u[pid] - mean_u;
                l_cov_u += du * du;
            }
            cov_u[sid] = sqrt(l_cov_u / (n_macroparticles - 1));
        }
    }
}


__global__ void sorted_cov_per_slice(unsigned int* lower_bounds,
                                     unsigned int* upper_bounds,
                                     double* u,                     // array of particle quantity sorted by slice
                                     double* v,                     // 2nd array of particles
                                     unsigned int n_slices,
                                     double* cov_uv)                 // output array of length n_slices with mean values for each slice
/**
    Iterate once through all the particles within the
    slicing region and calculate simultaneously the
    covariance of the quantities u,v for each slice separately.

    Assumes the particle array u to be sorted by slices.

    The index arrays lower_bounds and upper_bounds
    indicate the start and end indices
    within the sorted particle arrays for each slice. The respective
    slice id is identical to the index within lower_bounds and
    upper_bounds.
*/
{
    for (int sid = blockIdx.x * blockDim.x + threadIdx.x;
         sid < n_slices;
         sid += blockDim.x * gridDim.x)
    {
        double sum_u = 0.;
        double sum_v = 0.;
        unsigned int n_macroparticles = upper_bounds[sid] - lower_bounds[sid];
        if (n_macroparticles <= 1) {
            cov_uv[sid] = 0;
        } else {
            for (int pid = lower_bounds[sid]; pid < upper_bounds[sid]; pid++)
            {
                sum_u += u[pid];
                sum_v += v[pid];
            }
            double mean_u = sum_u / n_macroparticles;
            double mean_v = sum_v / n_macroparticles;

            double l_cov_u = 0;
            for (int pid = lower_bounds[sid]; pid < upper_bounds[sid]; pid++)
            {
                l_cov_u += (u[pid] - mean_u) * (v[pid] - mean_v);
            }
            cov_uv[sid] = l_cov_u / (n_macroparticles - 1);
        }
    }
}
