#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

template <class Real_t>
__global__ void chebPoly(Real_t* in, Real_t* out, int d, int n);

template <class Real_t>
__global__ void chebPoly_sharedMem(Real_t* in, Real_t* out, int d, int n);

void chebPoly_helper(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock);

void chebPoly_helper_shared(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock);
