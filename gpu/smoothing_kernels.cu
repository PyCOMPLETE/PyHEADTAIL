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

__constant__ double w4 = 0.0001338306246147;
__constant__ double w3 = 0.0044318616200313;
__constant__ double w2 = 0.0539911274207044;
__constant__ double w1 = 0.2419714456566007;
__constant__ double w0 = 0.3989434693560978;

__global__ void gauss_1sig_smoothing(double *in, double *out, int length)
{
   __shared__ double temp[BLOCK_SIZE + 8];

   int gindex = threadIdx.x + blockIdx.x * blockDim.x;
   int lindex = threadIdx.x + 4;

   temp[lindex] = in[gindex];

   if (threadIdx.x < 4) {
      // local edge handling

      // left
      if (gindex < 4) {
         // local edge is global edge
         temp[lindex - 4] = in[0];
      } else {
         temp[lindex - 4] = in[gindex - 4];
      }

      //right
      if (gindex >= length - BLOCK_SIZE) {
         // local edge is global edge
         temp[lindex + BLOCK_SIZE] = in[length - 1];
      } else {
         temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
      }
   }

   __syncthreads();

   out[gindex] =   temp[lindex - 4] * w4
                 + temp[lindex - 3] * w3
                 + temp[lindex - 2] * w2
                 + temp[lindex - 1] * w1
                 + temp[lindex]     * w0
                 + temp[lindex + 1] * w1
                 + temp[lindex + 2] * w2
                 + temp[lindex + 3] * w3
                 + temp[lindex + 4] * w4;

}
