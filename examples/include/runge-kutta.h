#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

template<typename real_t>
__global__ void kernel1_unoptimized(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size);

template<typename real_t>
__global__ void kernel2_unoptimized(real_t* vel_in, real_t* pos_in, real_t* pos_out, real_t dt, int size);

void runge_kutta_helper1(double* vel_in, double* pos_in, double* pos_out, double dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper2(double* vel_in, double* pos_in, double* pos_out, double dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper1float(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper2float(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper1sharedfloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper2shared(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper1multifloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper2multifloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper1reducethreatsfloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);

void runge_kutta_helper2reducethreatsfloat(float* vel_in, float* pos_in, float* pos_out, float dt, int size, int numBlocks, int threadsPerBlock);
