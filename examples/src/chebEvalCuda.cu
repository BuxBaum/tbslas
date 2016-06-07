#include "chebEval.h"


/**
 * \brief Returns the values of all chebyshev polynomials up to degree d,
 * evaluated at points in the input vector. Output format:
 * { T0[in[0]], ..., T0[in[n-1]], T1[in[0]], ..., T(d)[in[n-1]] }
 */
 // naive implementation
template <class Real_t>
__global__ void chebPoly(Real_t* in, Real_t* out, int d, int n)
{
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i<n)
	{
	
		if(d==0){
			out[i] = (abs(in[i]) <= 1.0f?1.0f:0); // if absolute value of in[i] <= 1 then 1.0 else 0 (polynomials in range [-1;1]
		}
		else if (d==1) {
			out[i]= (abs(in[i]) <= 1.0f?1.0f:0); // T0 same as above
			out[i+n]=(abs(in[i])<=1.0f?in[i]:0); // if abs of in[i] <= 1 then T1(x) = x else 0
		}
		else {
			Real_t x = (abs(in[i])<=1.0f?in[i]:0);
			Real_t y0 = (abs(in[i]) <= 1.0f?1.0f:0);		
			
			out[i] = y0;
			out[i+n] = x;
			
			Real_t y1 = x;
			Real_t* y2 = &out[2*n+i];
			for (int j=2; j<=d; j++) {
				*y2 = 2*x*y1-y0;
				y0=y1;
				y1=*y2;
				y2=&y2[n];
			}
						
		}
	}
	
}

// test with shared memory (apparently not faster)
template <class Real_t>
__global__ void chebPoly_sharedMem(Real_t* in, Real_t* out, int d, int n)
{
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int i_shared = threadIdx.x;
	
	__shared__ Real_t s_in[1024];
	s_in[i_shared] = in[i];
	__syncthreads();
	
	if (i<n)
	{
	
		if(d==0){
			out[i] = (abs(s_in[i_shared]) <= 1.0f?1.0f:0); // if absolute value of in[i] <= 1 then 1.0 else 0 (polynomials in range [-1;1]
		}
		else if (d==1) {
			out[i]= (abs(s_in[i_shared]) <= 1.0f?1.0f:0); // T0 same as above
			out[i+n]=(abs(s_in[i_shared])<=1.0f?s_in[i_shared]:0); // if abs of in[i] <= 1 then T1(x) = x else 0
		}
		else {
			Real_t x = (abs(s_in[i_shared])<=1.0f?s_in[i_shared]:0);
			Real_t y0 = (abs(s_in[i_shared]) <= 1.0f?1.0f:0);		
			
			out[i] = y0;
			out[i+n] = x;
			
			Real_t y1 = x;
			Real_t* y2 = &out[2*n+i];
			for (int j=2; j<=d; j++) {
				*y2 = 2*x*y1-y0;
				y0=y1;
				y1=*y2;
				y2=&y2[n];
			}
						
		}
	}
	
}

void chebPoly_helper(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock)
{
	chebPoly<<<numBlocks, threadsPerBlock>>> (in, out, d, n);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
}

void chebPoly_helper_shared(double* in, double* out, int d, int n, int numBlocks, int threadsPerBlock)
{
	chebPoly_sharedMem<<<numBlocks, threadsPerBlock>>> (in, out, d, n);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
}
