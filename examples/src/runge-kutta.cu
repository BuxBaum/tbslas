#include "runge-kutta.h"

// kernel 1 for double
template<typename real_t>
__global__ void kernel1_unoptimized(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{
	
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// check if id is in bounds
	if (i < size)
	{		
	  pos_out[i] = pos_in[i] + 0.5 * dt * vel_in[i];
	}
}

// kernel 2 for double
template<typename real_t>
__global__ void kernel2_unoptimized(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{
	
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// check if id is in bounds
	if (i < size)
	{		
	  pos_out[i] = pos_in[i] + dt * vel_in[i];
	}
}

// float, unoptimized
template<typename real_t>
__global__ void kernel1_unoptimizedf(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{
	
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// check if id is in bounds
	if (i < size)
	{		
	  pos_out[i] = pos_in[i] + 0.5f * dt * vel_in[i];
	}
}

// fload, shared memory
template<typename real_t>
__global__ void kernel1_sharedf(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{
	
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int i_shared = threadIdx.x; 
	__shared__ real_t s_vel[1024];
	__shared__ real_t s_pos_in[1024];
	__shared__ float t;
	s_vel[i_shared] = vel_in[i];
	s_pos_in[i_shared] = pos_in[i];
	t = 0.5f*dt;
	__syncthreads();

	// check if id is in bounds
	if (i < size)
	{		
	  float tmp = s_vel[i_shared] * t;
	  s_pos_in[i_shared] += tmp;
	  __syncthreads();
	
	  pos_out[i] = s_pos_in[i_shared];
	}

	
}

// float, shared memory
template<typename real_t>
__global__ void kernel2_shared(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{

	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int i_shared = threadIdx.x; 

	__shared__ real_t s_vel[1024];
	__shared__ real_t s_pos_in[1024];

	s_vel[i_shared] = vel_in[i];
	s_pos_in[i_shared] = pos_in[i];
	__syncthreads();

	// check if id is in bounds
	if (i < size)
	{		
	  pos_out[i] = s_pos_in[i_shared] + dt * s_vel[i_shared];
	}
}

// float, computation divided into multiple steps (useless)
template<typename real_t>
__global__ void kernel1_multiplestepsf(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{

	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// check if id is in bounds
	if (i < size)
	{		
	  real_t temp = 0.5f * dt * vel_in[i];
	  real_t temp2 = pos_in[i] + temp;
	  pos_out[i] = temp2;
	}
}

// float, computation divided into multiple steps (useless)
template<typename real_t>
__global__ void kernel2_multiplesteps(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{
	
	// compute thread id
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// check if id is in bounds
	if (i < size)
	{		
	  real_t temp = dt * vel_in[i];
	  real_t temp2 = pos_in[i] + temp;
	  pos_out[i] = temp2;
	}
}

// float, multiple points in one thread
template<typename real_t>
__global__ void kernel1_reducethreadsf(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{
	
	// compute thread id
	int i = 2* threadIdx.x + blockIdx.x * blockDim.x;
	int i_shared = threadIdx.x; 
	__shared__ real_t s_vel[2048];
	__shared__ real_t s_pos_in[2048];
	__shared__ real_t s_pos_out[2048];

	s_vel[i_shared] = vel_in[i];
	s_vel[i_shared+1024] = vel_in[i+1024];
	s_pos_in[i_shared] = pos_in[i];
	s_pos_in[i_shared+1024] = pos_in[i+1024];
	__syncthreads();

	// check if id is in bounds
	if (i < size)
	{		
	  s_pos_out[i_shared] = s_pos_in[i_shared] + 0.5f * dt * s_vel[i_shared];
	  s_pos_out[i_shared+1024] = s_pos_in[i_shared+1024] + 0.5f * dt * s_vel[i_shared+1024];
	  __syncthreads();

	  pos_out[i] = s_pos_out[i_shared];
	  pos_out[i+1024] = s_pos_out[i_shared+1024];

	}
}

// float, multiple points per thread
template<typename real_t>
__global__ void kernel2_reducethreadsf(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size)
{

	// compute thread id
	int i = 2* threadIdx.x + blockIdx.x * blockDim.x;
	int i_shared = threadIdx.x; 
	__shared__ real_t s_vel[2048];
	__shared__ real_t s_pos_in[2048];
	__shared__ real_t s_pos_out[2048];

	s_vel[i_shared] = vel_in[i];
	s_vel[i_shared+1024] = vel_in[i+1024];
	s_pos_in[i_shared] = pos_in[i];
	s_pos_in[i_shared+1024] = pos_in[i+1024];
	__syncthreads();

	// check if id is in bounds
	if (i < size)
	{		
	  s_pos_out[i_shared] = s_pos_in[i_shared] + dt * s_vel[i_shared];
	  s_pos_out[i_shared+1024] = s_pos_in[i_shared+1024] + 0.5f * dt * s_vel[i_shared+1024];
	  __syncthreads();

	  pos_out[i] = s_pos_out[i_shared];
	  pos_out[i+1024] = s_pos_out[i_shared+1024];

	}
}

// helper functions for kernel calls

void runge_kutta_helper1(double* vel_in, double* pos_in, double* pos_out, double dt, int size, int numBlocks, int threadsPerBlock) {

	kernel1_unoptimized<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper2(double* vel_in, double* pos_in, double* pos_out, double dt, int size, int numBlocks, int threadsPerBlock) {

	kernel2_unoptimized<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper1float(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel1_unoptimizedf<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper2float(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel2_unoptimized<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper1sharedfloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel1_sharedf<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper2shared(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel2_shared<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper1multifloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel1_multiplestepsf<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper2multifloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel2_multiplesteps<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper1reducethreatsfloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel1_reducethreadsf<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

void runge_kutta_helper2reducethreatsfloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock) {

	kernel2_reducethreadsf<<<numBlocks, threadsPerBlock>>> (vel_in, pos_in, pos_out, dt, size);
	getLastCudaError("Kernel execution failed");	

	checkCudaErrors(cudaDeviceSynchronize());
	
}

