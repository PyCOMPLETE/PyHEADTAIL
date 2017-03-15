#define RADIUS      2
#define BLOCK_SIZE  32 // needs to be equal to blockDim.x

__constant__ double WEIGHT = 1. / (2.*RADIUS + 1);

__global__ void smoothing_stencil_1d(double *in, double *out, int length)
/**
   Smoothing stencil in one dimension using average over the radius
   defined by RADIUS .

   Inspired by http://www.orangeowlsolutions.com/archives/1119 .
*/
{
   __shared__ double temp[BLOCK_SIZE + 2 * RADIUS];

   int gindex = threadIdx.x + blockIdx.x * blockDim.x;

   int lindex = threadIdx.x + RADIUS;

   // Read input elements into shared memory

   temp[lindex] = in[gindex];

   if (threadIdx.x < RADIUS) {
      if (gindex < RADIUS) {
         temp[lindex - RADIUS] = in[0];
      } else {
         temp[lindex - RADIUS] = in[gindex - RADIUS];
      }
      if (gindex >= length - BLOCK_SIZE) {
         temp[lindex + BLOCK_SIZE] = in[length-1];
      } else {
         temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
      }
   }

   __syncthreads();

   // Apply the stencil

   double result = 0;

   for (int offset = -RADIUS; offset <= RADIUS; offset++)

      result += temp[lindex + offset];

   // Store the result

   out[gindex] = result * WEIGHT;
}


#define W4 0.0001338306246147
#define W3 0.0044318616200313
#define W2 0.0539911274207044
#define W1 0.2419714456566007
#define W0 0.3989434693560978

// __device__ __constant__ double GAUSSIAN_SMOOTHING_WEIGHT[] =
//   {
//     0.0001338306246147,
//     0.0044318616200313,
//     0.0539911274207044,
//     0.2419714456566007,
//     0.3989434693560978,
//     0.2419714456566007,
//     0.0539911274207044,
//     0.0044318616200313,
//     0.0001338306246147,
//   };

#define GAUSSIAN_FILTER_RADIUS 4
#define GAUSSIAN_BLOCK_SIZE       32

// template<int GAUSSIAN_FILTER_RADIUS, int GAUSSIAN_BLOCK_SIZE>
__global__ void gaussian_smoothing_1d(double *in, double *out, int length)
{
  // needs length GAUSSIAN_BLOCK_SIZE + 2*GAUSSIAN_FILTER_RADIUS
  __shared__ double temp[GAUSSIAN_BLOCK_SIZE + 2*GAUSSIAN_FILTER_RADIUS];
  int lindex, offset;

  for (int gindex = blockIdx.x * blockDim.x + threadIdx.x;
       gindex < length;
       gindex += blockDim.x * gridDim.x)
  {
    lindex = threadIdx.x + GAUSSIAN_FILTER_RADIUS;

    temp[lindex] = in[gindex];

    if (threadIdx.x < GAUSSIAN_FILTER_RADIUS) {
      // local edge handling

      // left
      if (gindex < GAUSSIAN_FILTER_RADIUS) {
        // local edge is global edge
        temp[lindex - GAUSSIAN_FILTER_RADIUS] = in[0];
      } else {
        temp[lindex - GAUSSIAN_FILTER_RADIUS] =
            in[gindex - GAUSSIAN_FILTER_RADIUS];
      }

      //right
      if (gindex >= length - GAUSSIAN_BLOCK_SIZE) {
        // local edge is global edge
        temp[lindex + GAUSSIAN_BLOCK_SIZE] = in[length - 1];
      } else {
        temp[lindex + GAUSSIAN_BLOCK_SIZE] = in[gindex + GAUSSIAN_BLOCK_SIZE];
      }
    }

    __syncthreads();

    out[gindex] =   temp[lindex - 4] * W4
                  + temp[lindex - 3] * W3
                  + temp[lindex - 2] * W2
                  + temp[lindex - 1] * W1
                  + temp[lindex]     * W0
                  + temp[lindex + 1] * W1
                  + temp[lindex + 2] * W2
                  + temp[lindex + 3] * W3
                  + temp[lindex + 4] * W4;


    // for (offset = lindex - GAUSSIAN_FILTER_RADIUS;
    //      offset <= lindex + GAUSSIAN_FILTER_RADIUS;
    //      offset++)
    // {
    //   out[gindex] += temp[offset] *
    //                  GAUSSIAN_SMOOTHING_WEIGHT[offset + GAUSSIAN_FILTER_RADIUS - lindex];
    // }
  }
}
