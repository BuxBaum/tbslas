#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

template <class Real_t>
__global__ void chebPoly(Real_t* in, Real_t* out, int d, int n);

template <class Real_t>
__global__ void chebPoly_sharedMem(Real_t* in, Real_t* out, int d, int n);

template <class Real_t>
__global__ void vec_eval_kernel (int n, int d0, int d, int dof, Real_t* px, Real_t* py, Real_t* pz, Real_t* coeff, Real_t* tmp_out);

void reorderResult_helper(double* in, double* out, int n, int dof, int numBlocks, int threadsPerBlock);

void chebPoly_helper(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock);

void chebPoly_helper_stream(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock, cudaStream_t stream);

void vec_eval_kernel_helper_dof3(int n, int d0, int d, int dof, double* px, double* py, double* pz, double* coeff, double* tmp_out, int numBlocks, int threadsPerBlock);

void vec_eval_kernel_helper_dof1_new_complete(int n, double* px, double* py, double* pz, double* coeff, double* tmp_out, int* pointnode, int numBlocks, int threadsPerBlock);

void vec_eval_kernel_helper_dof3_new_complete(int n, double* px, double* py, double* pz, double* coeff, double* tmp_out, int* pointnode, int numBlocks, int threadsPerBlock);
