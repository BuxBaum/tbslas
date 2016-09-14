#include "chebEval.h"
#include "preprocessordefines.h"
#include "setChebyshevDegree.h"

// Computes the Chebyshev polynomials recursively
template <class Real_t>
__global__ void chebPoly_unrolled(Real_t* in, Real_t* out, int d, int n)
{
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i<n)
	{
		
			Real_t x = in[i];
			Real_t y0 = 1.0;		
			
			out[i] = y0;
			out[i+n] = x;
			
			Real_t y1 = x;
			
			
			Real_t* y2 = &out[2*n+i];
			#pragma unroll
			for (int j=2; j<=D; j++) {
				*y2 = 2*x*y1-y0;
				y0=y1;
				y1=*y2;
				y2=&y2[n];
			}
	}
	
}

// just for reference, doesn't work anymore due to NodeFieldFunctor changes
// fastest triple for loop version
// coeffs in shared, 1 point per thread (3 dof)
// for d > 16 shared memory needs to be allocated externally
template <class Real_t>
__global__ void vec_eval_kernel_dof3_shared_new (int n, int d0, int d, Real_t* px, Real_t* py, Real_t* pz, Real_t* coeff, Real_t* tmp_out)
{
	// compute thread id
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int id_shared = threadIdx.x;
	
	__shared__ Real_t c1[1024];
	__shared__ Real_t c2[1024];
	__shared__ Real_t c3[1024];
	
	int coeffsize = d*(d+1)*(d+2)/6;
	
	if (id_shared < coeffsize)
	{
		c1[id_shared] = coeff[id_shared];
		c2[id_shared] = coeff[coeffsize + id_shared];
		c3[id_shared] = coeff[2*coeffsize + id_shared];
	}
	
	if (id<n) 
	{
		
		// registers
		Real_t u0 = 0.0;
		Real_t v0 = 0.0;
		Real_t w0 = 0.0;
		Real_t u1 = 0.0;
		Real_t v1 = 0.0;
		Real_t w1 = 0.0;
		Real_t u2 = 0.0;
		Real_t v2 = 0.0;
		Real_t w2 = 0.0;
		
		Real_t tmp;
		
		int index = 0;
		
		
		for (int i = 0; i < d; i++)
		{
			v0 = 0.0;
			v1 = 0.0;
			v2 = 0.0;
			
			
			for (int j = 0; j < (d - i); j++)
			{
				w0 = 0.0;
				w1 = 0.0;
				w2 = 0.0;
				
				
				for (int k = 0; k < (d - i - j); k++)
				{
					tmp = px[k*n + id];
					w0 += tmp * c1[index];
					w1 += tmp * c2[index];
					w2 += tmp * c3[index];
					index++;
				}
				tmp = py[j*n + id];
				v0 += tmp * w0;
				v1 += tmp * w1;
				v2 += tmp * w2;
					
			}
			tmp = pz[i*n + id];
			u0 += tmp * v0;
			u1 += tmp * v1;
			u2 += tmp * v2;	
		}
		tmp_out[id] = u0;
		tmp_out[n + id] = u1;
		tmp_out[2 * n + id] = u2;	
		
	}
}

// just for reference
// hardcoded for d = 5
template <class Real_t>
__global__ void 
vec_eval_kernel_dof3_complete_unroll_hardcode_smart2 (int n, int d, Real_t* px_, Real_t* py_, Real_t* pz_, Real_t* coeff_, Real_t* tmp_out)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	int sharedid = threadIdx.x;
		
	__shared__ Real_t coeff[128];	
	
	
	if (sharedid < 105)
	{
		coeff[sharedid] = coeff_[sharedid];
		
	}
	__syncthreads();
	
	Real_t r0 = 0;
	Real_t u0 = 0;
	Real_t v0 = 0;
	
	Real_t r1 = 0;
	Real_t u1 = 0;
	Real_t v1 = 0;
	
	Real_t r2 = 0;
	Real_t u2 = 0;
	Real_t v2 = 0;
	int index = 0;
	
	if (id < n)
	{
	index = 0;
	
	Real_t* px = &px_[id];
	Real_t* py = &py_[id];
	Real_t* pz = &pz_[id];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	u0 += coeff[index] * px[2*n];
	u1 += coeff[35 + index] * px[2*n];
	u2 += coeff[70 + index] * px[2*n];
	index++;
	u0 += coeff[index] * px[3*n];
	u1 += coeff[35 + index] * px[3*n];
	u2 += coeff[70 + index] * px[3*n];
	index++;
	u0 += coeff[index] * px[4*n];
	u1 += coeff[35 + index] * px[4*n];
	u2 += coeff[70 + index] * px[4*n];
	index++;
	
	v0 = u0 * py[0];
	v1 = u1 * py[0];
	v2 = u2 * py[0];
	
	u0 = coeff[index] * px[0];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	u0 += coeff[index] * px[2*n];
	u1 += coeff[35 + index] * px[2*n];
	u2 += coeff[70 + index] * px[2*n];
	index++;
	u0 += coeff[index] * px[3*n];
	u1 += coeff[35 + index] * px[3*n];
	u2 += coeff[70 + index] * px[3*n];
	index++;
	
	v0 += u0 * py[n];
	v1 += u1 * py[n];
	v2 += u2 * py[n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	u0 += coeff[index] * px[2*n];
	u1 += coeff[35 + index] * px[2*n];
	u2 += coeff[70 + index] * px[2*n];
	index++;
	
	v0 += u0 * py[2*n];
	v1 += u1 * py[2*n];
	v2 += u2 * py[2*n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	
	v0 += u0 * py[3*n];
	v1 += u1 * py[3*n];
	v2 += u2 * py[3*n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	
	v0 += u0 * py[4*n];
	v1 += u1 * py[4*n];
	v2 += u2 * py[4*n];
	
	r0 = v0 * pz[0];
	r1 = v1 * pz[0];
	r2 = v2 * pz[0];
	
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	u0 += coeff[index] * px[2*n];
	u1 += coeff[35 + index] * px[2*n];
	u2 += coeff[70 + index] * px[2*n];
	index++;
	u0 += coeff[index] * px[3*n];
	u1 += coeff[35 + index] * px[3*n];
	u2 += coeff[70 + index] * px[3*n];
	index++;
	
	v0 = u0 * py[0];
	v1 = u1 * py[0];
	v2 = u2 * py[0];
	
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	u0 += coeff[index] * px[2*n];
	u1 += coeff[35 + index] * px[2*n];
	u2 += coeff[70 + index] * px[2*n];
	index++;
	
	v0 += u0 * py[n];
	v1 += u1 * py[n];
	v2 += u2 * py[n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	
	v0 += u0 * py[2*n];
	v1 += u1 * py[2*n];
	v2 += u2 * py[2*n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	
	v0 += u0 * py[3*n];	
	v1 += u1 * py[3*n];
	v2 += u2 * py[3*n];
	
	r0 += v0 * pz[n];
	r1 += v1 * pz[n];
	r2 += v2 * pz[n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	u0 += coeff[index] * px[2*n];
	u1 += coeff[35 + index] * px[2*n];
	u2 += coeff[70 + index] * px[2*n];
	index++;
	
	v0 = u0 * py[0];
	v1 = u1 * py[0];
	v2 = u2 * py[0];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	
	v0 += u0 * py[n];
	v1 += u1 * py[n];
	v2 += u2 * py[n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	
	v0 += u0 * py[2*n];
	v1 += u1 * py[2*n];
	v2 += u2 * py[2*n];
	
	r0 += v0 * pz[2*n];
	r1 += v1 * pz[2*n];
	r2 += v2 * pz[2*n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	u0 += coeff[index] * px[n];
	u1 += coeff[35 + index] * px[n];
	u2 += coeff[70 + index] * px[n];
	index++;
	
	v0 = u0 * py[0];
	v1 = u1 * py[0];
	v2 = u2 * py[0];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	
	v0 += u0 * py[n];	
	v1 += u1 * py[n];
	v2 += u2 * py[n];
	
	r0 += v0 * pz[3*n];
	r1 += v1 * pz[3*n];
	r2 += v2 * pz[3*n];
	
	u0 = coeff[index] * px[0];
	u1 = coeff[35 + index] * px[0];
	u2 = coeff[70 + index] * px[0];
	index++;
	
	v0 = u0 * py[0];
	v1 = u1 * py[0];
	v2 = u2 * py[0];
	
	r0 += v0 * pz[4*n];
	r1 += v1 * pz[4*n];
	r2 += v2 * pz[4*n];
	
	/////////////////////
	tmp_out[id] = r0;
	tmp_out[n + id] = r1;
	tmp_out[2 * n + id] = r2;	
	
	
	}
}


// unrolled function for dof 3
// Preprocessor defines in preprocessordefines.h and setChebyshevDegree.h
template <class Real_t>
__global__ void 
//__launch_bounds__(512, 1)
vec_eval_kernel_dof3_complete_unroll_preproctest_complete_shared (int n, Real_t* px_, Real_t* py_, Real_t* pz_, Real_t* coeff_, Real_t* tmp_out, int* pointnode)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int startid = blockIdx.x * blockDim.x;
	int idshared = threadIdx.x;
	int startnode = pointnode[startid];
	int coeffsize = NUMCOEFF*3;
	__shared__ Real_t coeff_s[NUMCOEFF*3];	
	
	Real_t* ctemp = &coeff_[startnode * coeffsize];
	
	for (int shloop = idshared; shloop < coeffsize; shloop += blockDim.x)
	{
		coeff_s[shloop] = ctemp[shloop];
		
	}
	__syncthreads();
	
	if (id < n) {
	
		Real_t r0 = 0;
		Real_t v0 = 0;
		Real_t u0 = 0;
		Real_t r1 = 0;
		Real_t v1 = 0;
		Real_t u1 = 0;
		Real_t r2 = 0;
		Real_t v2 = 0;
		Real_t u2 = 0;
		Real_t* px = &px_[id];
		Real_t* py = &py_[id];
		Real_t* pz = &pz_[id];
		Real_t* coeff = &coeff_s[0];
		if (startnode != pointnode[id])
			coeff = &coeff_[pointnode[id] * coeffsize];
		
		
		int N = n;

		COMP();
		
		tmp_out[id] = r0;
		tmp_out[n + id] = r1;
		tmp_out[2 * n + id] = r2;
	}
}


// unrolled function for dof 1
// Preprocessor defines in preprocessordefines.h and setChebyshevDegree.h
template <class Real_t>
__global__ void 
//__launch_bounds__(512, 1)
vec_eval_kernel_dof1_complete_unroll_preproctest_complete_shared (int n, Real_t* px_, Real_t* py_, Real_t* pz_, Real_t* coeff_, Real_t* tmp_out, int* pointnode)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int startid = blockIdx.x * blockDim.x;
	int idshared = threadIdx.x;
	int startnode = pointnode[startid];
	int coeffsize = NUMCOEFF;
	__shared__ Real_t coeff_s[NUMCOEFF];	
	
	Real_t* ctemp = &coeff_[startnode * coeffsize];
	
	for (int shloop = idshared; shloop < coeffsize; shloop += blockDim.x)
	{
		coeff_s[shloop] = ctemp[shloop];
		
	}
	__syncthreads();
	
	if (id < n) {
	
		Real_t r0 = 0;
		Real_t v0 = 0;
		Real_t u0 = 0;
		Real_t* px = &px_[id];
		Real_t* py = &py_[id];
		Real_t* pz = &pz_[id];
		Real_t* coeff = &coeff_s[0];;
		if(startnode != pointnode[id])
			coeff = &coeff_[pointnode[id] * coeffsize];
		
		int N = n;

		COMP_1();
		
		tmp_out[id] = r0;
	}
}

/**
 * reorders the results
 * shared memory acceses are unaligned
 */
template <class Real_t>
__global__ void reorderResult_shared (Real_t* in, Real_t* out, int n, int dof)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int shared_id = threadIdx.x;
	
	extern __shared__ Real_t sh_out[];
	
	if (idx < n)
	{	
			sh_out[shared_id*3] = in[idx];
		    sh_out[shared_id*3 + 1] = in[n + idx];
		    sh_out[shared_id*3 + 2] = in[2 * n + idx];	
			
	}		
	__syncthreads();
	Real_t* out_pointer = &out[blockIdx.x * blockDim.x*3];
			
	out_pointer[shared_id] = sh_out[shared_id];
	out_pointer[blockDim.x + shared_id] = sh_out[blockDim.x + shared_id];
	out_pointer[2*blockDim.x + shared_id] = sh_out[2*blockDim.x + shared_id];
				
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void reorderResult_helper(double* in, double* out, int n, int dof, int numBlocks, int threadsPerBlock)
{
	if (dof == 3)
		reorderResult_shared<<<numBlocks, threadsPerBlock, threadsPerBlock * 3 * sizeof(double)>>> (in, out, n, dof);
		
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
}

void chebPoly_helper(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock)
{
	
	chebPoly_unrolled<<<numBlocks, threadsPerBlock>>> (in, out, d, n);
	
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
}

// Chebyshev polynomial helper function for execution with streams
void chebPoly_helper_stream(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock, cudaStream_t stream)
{
	
	chebPoly_unrolled<<<numBlocks, threadsPerBlock, 0, stream>>> (in, out, d, n);
	
	getLastCudaError("Kernel execution failed");	

}


// helper function for old/obsolete kernels
// just for reference, not used anymore
void vec_eval_kernel_helper_dof3(int n, int d0, int d, int dof, double* px, double* py, double* pz, double* coeff, double* tmp_out, int numBlocks, int threadsPerBlock)
{
	// fastest triple for -loop version
	//vec_eval_kernel_dof3_shared_new<<<numBlocks, threadsPerBlock>>> (n, d0, d, px, py, pz, coeff, tmp_out);
		
	// unrolled by hand version for d = 5
	//vec_eval_kernel_dof3_complete_unroll_hardcode_smart2<<<numBlocks, threadsPerBlock>>> (n, d, px, py, pz, coeff, tmp_out);
		
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
}

void vec_eval_kernel_helper_dof1_new_complete(int n, double* px, double* py, double* pz, double* coeff, double* tmp_out, int* pointnode, int numBlocks, int threadsPerBlock)
{
	
	vec_eval_kernel_dof1_complete_unroll_preproctest_complete_shared<<<numBlocks, threadsPerBlock>>> (n, px, py, pz, coeff, tmp_out, pointnode);
	
	getLastCudaError("Kernel execution failed");	
}


void vec_eval_kernel_helper_dof3_new_complete(int n, double* px, double* py, double* pz, double* coeff, double* tmp_out, int* pointnode, int numBlocks, int threadsPerBlock)
{
	
	vec_eval_kernel_dof3_complete_unroll_preproctest_complete_shared<<<numBlocks, threadsPerBlock>>> (n, px, py, pz, coeff, tmp_out, pointnode);
	
	getLastCudaError("Kernel execution failed");	
}



