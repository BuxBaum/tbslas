// *************************************************************************
// Example with Gaussian Function
//
// *************************************************************************

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <math.h>

#include <pvfmm_common.hpp>
#include <mpi_tree.hpp>
#include <cheb_node.hpp>
#include <utils.hpp>
#include <vector.hpp>
#include <cheb_utils.hpp>
#include <profile.hpp>

#include <utils/common.h>
#include <utils/metadata.h>
#include <utils/reporter.h>
#include <utils/fields.h>

#include <tree/tree_semilag.h>
#include <tree/tree_utils.h>

#include <tree.hpp>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>    

#include <stdlib.h>

#include "runge-kutta.h"

typedef pvfmm::Cheb_Node<double> Node_t;
typedef pvfmm::Cheb_Node<float> Node_t_f;
typedef pvfmm::MPI_Tree<Node_t> Tree_t;
typedef pvfmm::MPI_Tree<Node_t_f> Tree_t_f;

typedef tbslas::MetaData<std::string,
                         std::string,
                         std::string> MetaData_t;

typedef typename Tree_t::Node_t NodeType;
typedef typename Tree_t_f::Node_t NodeType_f;		

double sigma_in, xc_in, yc_in, zc_in = 0.7;
double c_in = 0.0559;
double amp_in = 1;
char* fname = (char*) "gaussian-test";
		
void (*fn_val)(const double* , int , double*)=NULL;
void (*fn_vel)(const double* , int , double*)=NULL;
void (*fn_vel_f)(const float* , int , float*)=NULL;

// gaussian function using xc, yc, zc and c
template <class Real_t>
void gaussian_function(const Real_t* coord,
					   int n,
					   Real_t* out,	
					   const double xc = 0.5,
                       const double yc = 0.5,
                       const double zc = 0.5,
                       const double c  = 0.0559,
                       const double amp = 1.0) {

	for (int i=0;i<n;i++) {
		const Real_t* co =&coord[i*COORD_DIM];
		Real_t expo = - ((co[0] - xc) * (co[0] - xc) + (co[1] - yc) * (co[1] - yc) + (co[2] - zc) * (co[2] - zc))/(2*c*c);
		Real_t val = amp * exp(expo);
		out[i] = val;
		}
}

// gaussian kernel using sigma (similar to gaussian function)
template <class Real_t>
void my_gaussian_kernel(const Real_t* coord,
				    	int n,			
					    Real_t* out,	
					    Real_t sigma = 0.7
					    ) {
	const Real_t* c;
	Real_t expo;
	Real_t val;

	for (int i=0;i<n;i++) {
		c =&coord[i*COORD_DIM];
		expo = - (c[0]*c[0]+c[1]*c[1]+c[2]*c[2])/2*sigma*sigma;
		val = 1/(2*M_PI*sqrt(2*M_PI)*sigma*sigma*sigma) * exp(expo);
		out[i] = val;
	}

}

// for testing purposes
// copy of tbslas::gaussian_kernel
template <class Real_t>
void gaussian_kernel(const Real_t* coord,
                     int n,
                     Real_t* out,
                     const Real_t xc = 0.5,
                     const Real_t yc = 0.5,
                     const Real_t zc = 0.5,
                     const int a  = -160,
                     const Real_t amp = 1.0) { //Output potential
  int dof=1;
  for(int i=0;i<n;i++) {
    const Real_t* c=&coord[i*COORD_DIM];
    {
      Real_t r_2=(c[0]-xc)*(c[0]-xc)+(c[1]-yc)*(c[1]-yc)+(c[2]-zc)*(c[2]-zc);
      out[i*dof+0]=amp*exp(a*r_2);
    }
  }
}



// wraper for gaussian_kernel
template <class Real_t>
void get_gaussian_kernel_wraper(const Real_t* coord,
                     int n,
                     Real_t* out) {
  const Real_t xc  = 0.7;
  const Real_t yc  = 0.7;
  const Real_t zc  = 0.7;
  //tbslas::gaussian_kernel(coord, n, out, xc, yc, zc);
  gaussian_kernel(coord, n, out, xc, yc, zc);
}

// wraper for my_gaussian_kernel
template <class Real_t>
void get_my_gaussian_kernel_wraper(const Real_t* coord,
                     int n,
                     Real_t* out) {
  Real_t sigma = sigma_in;

  my_gaussian_kernel(coord, n, out, sigma);
}

// wraper for gaussian_function
template <class Real_t>
void get_gaussian_function_wraper(const Real_t* coord,
                     int n,
                     Real_t* out) {
  double xc = xc_in;
  double yc = yc_in;
  double zc = zc_in;
  double c = c_in;
  double amp = amp_in;
	
  gaussian_function(coord, n, out, xc, yc, zc, c, amp);

}

/////////////////////////////////////////////////////////////////
// CUDA FUNCTIONS
////////////////////////////////////////////////////////////////

	// for testing
	// copy array tp GPU and back
	template <class Real_t, class Real_t2>
	void copyToGPUArrays(int argc, char **argv, Real_t* evPoints, Real_t2* chebCoeff, Real_t* evPointsOut, Real_t2* chebCoeffOut, int evPointsSize, int chebCoeffSize, int chebCoeffDim)
	{
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
        devID, props.name, props.major, props.minor);

		unsigned int mem_size_points = sizeof(Real_t) * evPointsSize;
		unsigned int mem_size_cheb = sizeof(Real_t2) * chebCoeffSize * chebCoeffDim;
		std::cout << "mem_size_points: " << mem_size_points << " Real_t: " << sizeof(Real_t) << " evPoints: " << evPointsSize << std::endl;
		std::cout << "mem_size_cheb: " << mem_size_cheb << " Real_t2: " << sizeof(Real_t2) << " chebCoeff: " << chebCoeffSize << " chebCoeff values: " << chebCoeffDim << std::endl;
		
		// device pointer
		Real_t* dPoints = NULL; 
		Real_t2* dCheb = NULL;	
	
		// allocate GPU memory
		checkCudaErrors(cudaMalloc((void**) &dPoints, mem_size_points)); 
		checkCudaErrors(cudaMalloc((void**) &dCheb, mem_size_cheb)); 

		// copy to GPU
		checkCudaErrors(cudaMemcpy(dPoints, evPoints, mem_size_points,
                               cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(dCheb, chebCoeff, mem_size_cheb,
                               cudaMemcpyHostToDevice));
		// CUDA KERNEL CALL HERE


		//output		
		
		checkCudaErrors(cudaMemcpy(evPointsOut, dPoints, mem_size_points,
                               cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(chebCoeffOut, dCheb, mem_size_cheb,
                               cudaMemcpyDeviceToHost));
				
		checkCudaErrors(cudaFree(dPoints));
		checkCudaErrors(cudaFree(dCheb));
		cudaDeviceReset();

	}

	// for testing
	// copy data from vectors to GPU and back
	template <class Real_t, class Y>
	void copyToGPUVectors(int argc, char **argv, std::vector<Real_t>& evPoints, std::vector<Y>& chebCoeff, std::vector<Real_t>& evPointsOut, std::vector<Y>& chebCoeffOut)
	{

		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
        devID, props.name, props.major, props.minor);
    		
		//unsigned int num_threads = 32;
    	unsigned int mem_size_points = sizeof(Real_t) * evPoints.size();
		unsigned int mem_size_cheb = sizeof(double) * chebCoeff.size() * chebCoeff[0].Dim();
		std::cout << "mem_size_points: " << mem_size_points << " Real_t: " << sizeof(Real_t) << " evPoints: " << evPoints.size() << std::endl;
		std::cout << "mem_size_cheb: " << mem_size_cheb << " double: " << sizeof(Real_t) << " chebCoeff: " << chebCoeff.size() << " chebCoeff values: " << chebCoeff[0].Dim() << std::endl;
		
		// vector to array	
		Real_t* hPoints = &evPoints[0];
		
		double* hCheb = new double[chebCoeff.size() * chebCoeff[0].Dim()];
		for (int i = 0; i < chebCoeff.size(); i++)
		{
			double* temp = &chebCoeff[i][0];
			for (int j = 0; j < chebCoeff[0].Dim(); j++) {
				hCheb[i*chebCoeff[0].Dim() + j] = temp[j];	
			}		
		}
		// NOW: all cheb coeffs in hCheb

		// device pointer
		Real_t* dPoints = NULL; 
		double* dCheb = NULL;	
	
		// allocate GPU memory
		checkCudaErrors(cudaMalloc((void**) &dPoints, mem_size_points)); 
		checkCudaErrors(cudaMalloc((void**) &dCheb, mem_size_cheb)); 

		// copy to GPU
		checkCudaErrors(cudaMemcpy(dPoints, hPoints, mem_size_points,
                               cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(dCheb, hCheb, mem_size_cheb,
                               cudaMemcpyHostToDevice));
		// CUDA KERNEL CALL HERE



		//output
		Real_t* hPointsOut = new Real_t[evPoints.size()];
		double* hChebOut = new double[chebCoeff.size() * chebCoeff[0].Dim()];
		checkCudaErrors(cudaMemcpy(hPointsOut, dPoints, mem_size_points,
                               cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(hChebOut, dCheb, mem_size_cheb,
                               cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < evPoints.size(); i++)
		{
			evPointsOut[i] = hPointsOut[i];
		
		}

		for (int i = 0; i < chebCoeff.size(); i++)
		{
			
			for (int j = 0; j < chebCoeff[0].Dim(); j++)
			{
				chebCoeffOut[i][j] = hChebOut[i * chebCoeff[0].Dim() + j];
			}
		}
		
		checkCudaErrors(cudaFree(dPoints));
		checkCudaErrors(cudaFree(dCheb));
		delete[] hPointsOut;
		delete[] hCheb;
		delete[] hChebOut;
		cudaDeviceReset();

	}
	////////////////////////////////////////////////////////////////////
	// RUNGE_KUTTA
	////////////////////////////////////////////////////////////////////
	
	
	// run runge kutta with doubles, evaluation on CPU
	void runge_kutta_run (Tree_t* velT, std::vector<double> points, int num_rk_step, double dt, int argc, char **argv, MPI_Comm* comm) {
		
		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "START RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;
		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

		int pointsSize = points.size();
		int num_points = pointsSize/3;

		// same timestep as in traj
		double tinit = 0;
                double tfinal = tinit + num_rk_step*dt;
		double tau = (tfinal - tinit)/num_rk_step;
		
		pvfmm::Profile::Tic("Runge-Kutta on GPU", comm, true, 5);
		pvfmm::Profile::Tic("initialization", comm, true, 5);

		// Allocate GPU Memory
		double* d_vel = NULL;
		double* d_posIn = NULL;
		double* d_posOut = NULL;		

		unsigned int mem_size = sizeof(double) * pointsSize;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		double* h_vel = new double[pointsSize];
		double* h_posIn = new double[pointsSize];
		double* h_posTmp = new double[pointsSize];
			
		for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = points[i];
			}

		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = points.size()/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
		
		tbslas::NodeFieldFunctor<double,Tree_t> field_fn = tbslas::NodeFieldFunctor<double,Tree_t>(velT);

		pvfmm::Profile::Toc();		

		// run runge kutta on GPU
		for (int i = 0; i < num_rk_step; i++) {

			pvfmm::Profile::Tic("actual computations", comm, true, 5);

			// first evaluation	    		
	    	field_fn(h_posIn, num_points, h_vel);

			pvfmm::Profile::Tic("first copy to GPU", comm, true, 5);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_vel, mem_size,
                               cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posIn, mem_size,
                               cudaMemcpyHostToDevice));

			pvfmm::Profile::Toc();
			
			pvfmm::Profile::Tic("compute k1 on GPU", comm, true, 5);

			// compute k1 and temp position on GPU
			runge_kutta_helper1(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
			//getLastCudaError("Kernel execution failed");
			
			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("first copy from GPU", comm, true, 5);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
                               cudaMemcpyDeviceToHost));

			pvfmm::Profile::Toc();
			
			// second evaluation
			field_fn(h_posTmp, num_points, h_vel);

			pvfmm::Profile::Tic("second copy to GPU", comm, true, 5);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_vel, mem_size,
                               cudaMemcpyHostToDevice));

			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("compute k2 on GPU", comm, true, 5);

			// compute k2 and final position on GPU
			runge_kutta_helper2(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);

			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("second copy from GPU", comm, true, 5);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
                               cudaMemcpyDeviceToHost));

			pvfmm::Profile::Toc();
			
			// set values for next step
			for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = h_posTmp[i];
			}

			pvfmm::Profile::Toc();

		}

		pvfmm::Profile::Toc();

		std::cout << "start CPU comparison" << std::endl;
		// CPU function for comparison
		std::vector<double> cpuPosIn(pointsSize);
		std::vector<double> cpuPosOut(pointsSize);

		for (int i = 0; i < pointsSize; i++)
			{
				cpuPosIn[i] = points[i];
			}

		tbslas::ComputeTrajRK2(field_fn, 
                           cpuPosIn,
                           tinit,
                           tfinal,
                           num_rk_step,
                           cpuPosOut);

		std::cout << "num_rk_steps: " << num_rk_step << " dt: " << tau << " random point count: " << num_points << std::endl;

		std::cout << "final GPU points: " << h_posIn[0] << " " << h_posIn[1] << " " << h_posIn[2] << std::endl;
		std::cout << "final CPU points: " << cpuPosOut[0] << " " << cpuPosOut[1] << " " << cpuPosOut[2] << std::endl;

		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "END RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;


		// clean up
		delete[] h_vel;
		delete[] h_posIn;
		delete[] h_posTmp;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}
	
	// run runge kutta with floats, evaluation on CPU
	void runge_kutta_run_float (Tree_t_f* velT, std::vector<double> points, int num_rk_step, double dt, int argc, char **argv, MPI_Comm* comm) {
		
		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "START RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;
		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);

		int pointsSize = points.size();
		int num_points = pointsSize/3;

		// same timestep as in traj
		double tinit = 0;
                double tfinal = tinit + num_rk_step*dt;
		double tau = (tfinal - tinit)/num_rk_step;
		
		pvfmm::Profile::Tic("Runge-Kutta on GPU - float", comm, true, 5);
		//cudaDeviceSynchronize();
		pvfmm::Profile::Tic("initialization", comm, true, 5);

		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		float* h_vel = new float[pointsSize];
		float* h_posIn = new float[pointsSize];
		float* h_posTmp = new float[pointsSize];
			
		//h_posIn = points.data();
		for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = (float) points[i];
			}

		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = points.size()/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
		
		tbslas::NodeFieldFunctor<float,Tree_t_f> field_fn = tbslas::NodeFieldFunctor<float,Tree_t_f>(velT);

		pvfmm::Profile::Toc();		

		// run runge kutta on GPU
		for (int i = 0; i < num_rk_step; i++) {
			
			pvfmm::Profile::Tic("actual computations", comm, true, 5);

			// first evaluation	    		
	    	field_fn(h_posIn, num_points, h_vel);

			pvfmm::Profile::Tic("first copy to GPU", comm, true, 5);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_vel, mem_size,
                               cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posIn, mem_size,
                               cudaMemcpyHostToDevice));
			
			pvfmm::Profile::Toc();
			
			pvfmm::Profile::Tic("compute k1 on GPU", comm, true, 5);

			// compute k1 and temp position on GPU
			runge_kutta_helper1float(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
			
			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("first copy from GPU", comm, true, 5);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
                               cudaMemcpyDeviceToHost));
			
			pvfmm::Profile::Toc();
			
			// second evaluation
			field_fn(h_posTmp, num_points, h_vel);

			pvfmm::Profile::Tic("second copy to GPU", comm, true, 5);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_vel, mem_size,
                               cudaMemcpyHostToDevice));
			
			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("compute k2 on GPU", comm, true, 5);

			// compute k2 and final position on GPU
			runge_kutta_helper2float(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
			
			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("second copy from GPU", comm, true, 5);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
                               cudaMemcpyDeviceToHost));
			
			pvfmm::Profile::Toc();
			
			// set values for next step
			for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = h_posTmp[i];
			}

			pvfmm::Profile::Toc();
			
		}

		pvfmm::Profile::Toc();

		std::cout << "start CPU comparison" << std::endl;
		// CPU function for comparison
		std::vector<float> cpuPosIn(pointsSize);
		std::vector<float> cpuPosOut(pointsSize);

		for (int i = 0; i < pointsSize; i++)
			{
				cpuPosIn[i] = (float) points[i];
			}

		tbslas::ComputeTrajRK2(field_fn, 
                           cpuPosIn,
                           (float) tinit,
                           (float) tfinal,
                           num_rk_step,
                           cpuPosOut);

		std::cout << "num_rk_steps: " << num_rk_step << " dt: " << tau << " random point count: " << num_points << std::endl;

		std::cout << "final GPU points: " << h_posIn[0] << " " << h_posIn[1] << " " << h_posIn[2] << std::endl;
		std::cout << "final CPU points: " << cpuPosOut[0] << " " << cpuPosOut[1] << " " << cpuPosOut[2] << std::endl;

		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "END RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;


		// clean up
		delete[] h_vel;
		delete[] h_posIn;
		delete[] h_posTmp;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}
	
	// run runge kutta with float and zerocopy, evaluation on CPU
	void runge_kutta_run_zerocopyfloat (Tree_t_f* velT, std::vector<double> points, int num_rk_step, double dt, int argc, char **argv, MPI_Comm* comm) {
		
		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "START RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;
		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

   		//Get GPU information
   		checkCudaErrors(cudaGetDevice(&devID));
   		checkCudaErrors(cudaGetDeviceProperties(&props, devID));
   		printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);

		checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

		int pointsSize = points.size();
		int num_points = pointsSize/3;

		// same timestep as in traj
		double tinit = 0;
                double tfinal = tinit + num_rk_step*dt;
		double tau = (tfinal - tinit)/num_rk_step;
		
		pvfmm::Profile::Tic("Runge-Kutta on GPU - zc", comm, true, 5);
		
		pvfmm::Profile::Tic("initialization", comm, true, 5);

		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		// host arrays
		float* h_vel = NULL;
		float* h_posIn = NULL;
		float* h_posTmp = NULL;
			
		checkCudaErrors(cudaHostAlloc((void **)&h_vel, mem_size, cudaHostAllocMapped));
		checkCudaErrors(cudaHostAlloc((void **)&h_posIn, mem_size, cudaHostAllocMapped));
		checkCudaErrors(cudaHostAlloc((void **)&h_posTmp, mem_size, cudaHostAllocMapped));
			

		for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = (float) points[i];
			}
			
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_vel, (void *) h_vel, 0));
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_posIn, (void *) h_posIn, 0));
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_posOut, (void *) h_posTmp, 0));

		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = points.size()/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
		
		tbslas::NodeFieldFunctor<float,Tree_t_f> field_fn = tbslas::NodeFieldFunctor<float,Tree_t_f>(velT);

		
		pvfmm::Profile::Toc();		

		// run runge kutta on GPU
		for (int i = 0; i < num_rk_step; i++) {
			
			pvfmm::Profile::Tic("actual computations", comm, true, 5);

			// first evaluation	    		
	    	field_fn(h_posIn, num_points, h_vel);

			pvfmm::Profile::Tic("compute k1 on GPU", comm, true, 5);

			// compute k1 and temp position on GPU
			runge_kutta_helper1float(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);		
			pvfmm::Profile::Toc();
		
			// second evaluation
			field_fn(h_posTmp, num_points, h_vel);

			pvfmm::Profile::Tic("compute k2 on GPU", comm, true, 5);

			// compute k2 and final position on GPU
			runge_kutta_helper2float(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
			
			pvfmm::Profile::Toc();
			
			// set values for next step
			for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = h_posTmp[i];
			}

			pvfmm::Profile::Toc();
			
		}

		pvfmm::Profile::Toc();

		std::cout << "start CPU comparison" << std::endl;
		// CPU function for comparison
		std::vector<float> cpuPosIn(pointsSize);
		std::vector<float> cpuPosOut(pointsSize);

		for (int i = 0; i < pointsSize; i++)
			{
				cpuPosIn[i] = (float) points[i];
			}

		tbslas::ComputeTrajRK2(field_fn, 
                           cpuPosIn,
                           (float) tinit,
                           (float) tfinal,
                           num_rk_step,
                           cpuPosOut);

		std::cout << "num_rk_steps: " << num_rk_step << " dt: " << tau << " random point count: " << num_points << std::endl;

		std::cout << "final GPU points: " << h_posIn[0] << " " << h_posIn[1] << " " << h_posIn[2] << std::endl;
		std::cout << "final CPU points: " << cpuPosOut[0] << " " << cpuPosOut[1] << " " << cpuPosOut[2] << std::endl;

		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "END RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;


		// clean up
		checkCudaErrors(cudaFreeHost(h_vel));
		checkCudaErrors(cudaFreeHost(h_posIn));
		checkCudaErrors(cudaFreeHost(h_posTmp));
		cudaDeviceReset();

	}
	
	// run runge kutta with float and unified memory, evaluation on CPU
	void runge_kutta_run_unifiedmemoryfloat (Tree_t_f* velT, std::vector<double> points, int num_rk_step, double dt, int argc, char **argv, MPI_Comm* comm) {
		
		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "START RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;

		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);


		int pointsSize = points.size();
		int num_points = pointsSize/3;

		// same timestep as in traj
		double tinit = 0;
                double tfinal = tinit + num_rk_step*dt;
		double tau = (tfinal - tinit)/num_rk_step;
		
		pvfmm::Profile::Tic("Runge-Kutta on GPU - um", comm, true, 5);
		
		pvfmm::Profile::Tic("initialization", comm, true, 5);

		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		checkCudaErrors(cudaMallocManaged(&d_vel, mem_size)); 
		checkCudaErrors(cudaMallocManaged(&d_posIn, mem_size)); 
		checkCudaErrors(cudaMallocManaged(&d_posOut, mem_size)); 		
			
		for (int i = 0; i < pointsSize; i++)
			{
				d_posIn[i] = (float) points[i];
				d_vel[i] = 0.0f;
				d_posOut[i] = 0.0f;
			}
			
		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = points.size()/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
		
		tbslas::NodeFieldFunctor<float,Tree_t_f> field_fn = tbslas::NodeFieldFunctor<float,Tree_t_f>(velT);

		pvfmm::Profile::Toc();		

		// run runge kutta on GPU
		for (int i = 0; i < num_rk_step; i++) {
			
			pvfmm::Profile::Tic("actual computations", comm, true, 5);

			cudaDeviceSynchronize();
			// first evaluation	    		
	    	field_fn(d_posIn, num_points, d_vel);

			pvfmm::Profile::Tic("compute k1 on GPU", comm, true, 5);

			// compute k1 and temp position on GPU
			runge_kutta_helper1float(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
		
			pvfmm::Profile::Toc();
			
			// second evaluation
			field_fn(d_posOut, num_points, d_vel);

			pvfmm::Profile::Tic("compute k2 on GPU", comm, true, 5);

			// compute k2 and final position on GPU
			runge_kutta_helper2float(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
			
			pvfmm::Profile::Toc();
			
			// set values for next step
			for (int i = 0; i < pointsSize; i++)
			{
				d_posIn[i] = d_posOut[i];
			}

			pvfmm::Profile::Toc();
			cudaDeviceSynchronize();
		}

		pvfmm::Profile::Toc();

		std::cout << "start CPU comparison" << std::endl;
		// CPU function for comparison
		std::vector<float> cpuPosIn(pointsSize);
		std::vector<float> cpuPosOut(pointsSize);

		for (int i = 0; i < pointsSize; i++)
			{
				cpuPosIn[i] = (float) points[i];
			}



		tbslas::ComputeTrajRK2(field_fn, 
                           cpuPosIn,
                           (float) tinit,
                           (float) tfinal,
                           num_rk_step,
                           cpuPosOut);

		std::cout << "num_rk_steps: " << num_rk_step << " dt: " << tau << " random point count: " << num_points << std::endl;

		std::cout << "final GPU points: " << d_posIn[0] << " " << d_posIn[1] << " " << d_posIn[2] << std::endl;
		std::cout << "final CPU points: " << cpuPosOut[0] << " " << cpuPosOut[1] << " " << cpuPosOut[2] << std::endl;

		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "END RUNGE KUTTA" << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;


		// clean up
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}

	// ---------------------------------------------------
	// CUDA COMPARISONS
	// ---------------------------------------------------
	// compare different GPU implementations without evaluation
	
	// double, unoptimized
	void original_unoptimized (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		
		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);
	
		// Allocate GPU Memory
		double* d_vel = NULL;
		double* d_posIn = NULL;
		double* d_posOut = NULL;		

		unsigned int mem_size = sizeof(double) * pointsSize;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		double* h_posTmp = new double[pointsSize];
		
		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = pointsSize/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
				
		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_vel, mem_size,
		                      cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posIn, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k1 and temp position on GPU
			runge_kutta_helper1(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);
			
			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                      cudaMemcpyDeviceToHost));

			// skip evaluation

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_vel, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k2 and final position on GPU
			runge_kutta_helper2(d_vel, d_posIn, d_posOut, tau, pointsSize, blockCount, threadsPerBlock);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                       cudaMemcpyDeviceToHost));

		}

		// clean up
		delete[] h_posTmp;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}

	// float, unoptimized
	void original_float (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		
			
		// Get CUDA device
		int devID;
    	cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
        devID, props.name, props.major, props.minor);
	
		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		float* h_posTmp = new float[pointsSize];
		float* h_velf = new float[pointsSize];
		float* h_posInf = new float[pointsSize];

		for (int i = 0; i < pointsSize; i++) 
		{	
			h_velf[i] = (float) h_vel[i];
			h_posInf[i] = (float) h_posIn[i];
		}

		float tauf = (float) tau;
		
		// needed block count
		int threadsPerBlock = 1024; //1024;
		int blockCount = pointsSize/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
				
		std::cout << "total number of threads: " << blockCount*threadsPerBlock << std::endl;

		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{
			
			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posInf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k1 and temp position on GPU
			runge_kutta_helper1float(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);			

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                      cudaMemcpyDeviceToHost));

			// skip evaluation

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k2 and final position on GPU
			runge_kutta_helper2float(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                       cudaMemcpyDeviceToHost));

		}

		// clean up
		delete[] h_posTmp;
		delete[] h_velf;
		delete[] h_posInf;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}

	// float, shared memory
	void shared_float (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		
		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);
	
		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		float* h_posTmp = new float[pointsSize];
		float* h_velf = new float[pointsSize];
		float* h_posInf = new float[pointsSize];

		for (int i = 0; i < pointsSize; i++) 
		{	
			h_velf[i] = (float) h_vel[i];
			h_posInf[i] = (float) h_posIn[i];
		}

		float tauf = (float) tau;
		
		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = pointsSize/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
				
		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{
			
			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posInf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k1 and temp position on GPU
			runge_kutta_helper1sharedfloat(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);
		
			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                      cudaMemcpyDeviceToHost));
			
			// skip second evaluation
			//field_fn(h_posTmp, num_points, h_vel);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k2 and final position on GPU
			runge_kutta_helper2shared(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                       cudaMemcpyDeviceToHost));

		}

		// clean up
		delete[] h_posTmp;
		delete[] h_velf;
		delete[] h_posInf;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}

	// float, computations divided into a few steps (useless)
	void multi_float (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		
		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);
	
		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		float* h_posTmp = new float[pointsSize];
		float* h_velf = new float[pointsSize];
		float* h_posInf = new float[pointsSize];

		for (int i = 0; i < pointsSize; i++) 
		{	
			h_velf[i] = (float) h_vel[i];
			h_posInf[i] = (float) h_posIn[i];
		}

		float tauf = (float) tau;
		
		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = pointsSize/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
				
		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{
			
			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posInf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k1 and temp position on GPU
			runge_kutta_helper1multifloat(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);
			
			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                      cudaMemcpyDeviceToHost));

			// skip second evaluation
			//field_fn(h_posTmp, num_points, h_vel);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k2 and final position on GPU
			runge_kutta_helper2multifloat(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                       cudaMemcpyDeviceToHost));

		}

		// clean up
		delete[] h_posTmp;
		delete[] h_velf;
		delete[] h_posInf;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}

	// float, multiple points per thread
	void reducethreads_float (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		

		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);
	
		// pointsize needs to be multiple of 2048
		int r = pointsSize % 2048;
		int pointsSizeNew = pointsSize + 2048-r;

		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSizeNew;

		checkCudaErrors(cudaMalloc((void**) &d_vel, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posIn, mem_size)); 
		checkCudaErrors(cudaMalloc((void**) &d_posOut, mem_size)); 		

		// host arrays
		float* h_posTmp = new float[pointsSizeNew];
		float* h_velf = new float[pointsSizeNew];
		float* h_posInf = new float[pointsSizeNew];

		for (int i = 0; i < pointsSize; i++) 
		{	
			h_velf[i] = (float) h_vel[i];
			h_posInf[i] = (float) h_posIn[i];
		}

		for (int i = pointsSize; i < pointsSizeNew; i++)
		{
			h_velf[i] = 0.0f;
			h_posInf[i] = 0.0f;
		}

		float tauf = (float) tau;
		
		// needed block count
		int threadsPerBlock = 1024;
		int blockCount = pointsSizeNew/(2*threadsPerBlock) + ((pointsSizeNew/2)%threadsPerBlock == 0 ? 0:1);
				
		std::cout << "total number of threads: " << blockCount*threadsPerBlock << std::endl;

		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{
			
			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_posIn, h_posInf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k1 and temp position on GPU
			runge_kutta_helper1reducethreatsfloat(d_vel, d_posIn, d_posOut, tauf, pointsSizeNew/2, blockCount, threadsPerBlock);
			
			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                      cudaMemcpyDeviceToHost));

			// skip second evaluation
			//field_fn(h_posTmp, num_points, h_vel);

			// copy data to GPU
			checkCudaErrors(cudaMemcpy(d_vel, h_velf, mem_size,
		                      cudaMemcpyHostToDevice));

			// compute k2 and final position on GPU
			runge_kutta_helper2reducethreatsfloat(d_vel, d_posIn, d_posOut, tauf, pointsSizeNew/2, blockCount, threadsPerBlock);

			// copy data to CPU
			checkCudaErrors(cudaMemcpy(h_posTmp, d_posOut, mem_size,
		                       cudaMemcpyDeviceToHost));
			
		}

		// clean up
		delete[] h_posTmp;
		delete[] h_velf;
		delete[] h_posInf;
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}
	
	// float, using zerocopy
	void zerocopy_float (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		

		
		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);
	
		checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	
		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		// host arrays
		float* h_posTmp = NULL;
		float* h_velf = NULL;
		float* h_posInf = NULL;

		// Allocate host memory
		checkCudaErrors(cudaHostAlloc((void **) &h_posTmp, mem_size, cudaHostAllocMapped));
		checkCudaErrors(cudaHostAlloc((void **) &h_velf, mem_size, cudaHostAllocMapped));
		checkCudaErrors(cudaHostAlloc((void **) &h_posInf, mem_size, cudaHostAllocMapped));

		for (int i = 0; i < pointsSize; i++) 
		{	
			h_velf[i] = (float) h_vel[i];
			h_posInf[i] = (float) h_posIn[i];
		}
		
		// Set device Pointer
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_vel, (void *) h_velf, 0));
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_posIn, (void *) h_posInf, 0));
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_posOut, (void *) h_posTmp, 0));

		float tauf = (float) tau;
		
		// needed block count
		int threadsPerBlock = 1024; //1024;
		int blockCount = pointsSize/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
				
		std::cout << "total number of threads: " << blockCount*threadsPerBlock << std::endl;

		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{
				
			// compute k1 and temp position on GPU
			runge_kutta_helper1float(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);
			// skip second evaluation
			//field_fn(h_posTmp, num_points, h_vel);

			// compute k2 and final position on GPU
			runge_kutta_helper2float(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);
		
		}

		// clean up
		checkCudaErrors(cudaFreeHost(h_posTmp));
		checkCudaErrors(cudaFreeHost(h_velf));
		checkCudaErrors(cudaFreeHost(h_posInf));
		cudaDeviceReset();

	}
	
	// float, using unified memory
	void unifiedmemory_float (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		

		// Get CUDA device
		int devID;
    		cudaDeviceProp props;

    	// This will pick the best possible CUDA capable device
    	devID = findCudaDevice(argc, (const char **)argv);

    	//Get GPU information
    	checkCudaErrors(cudaGetDevice(&devID));
    	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           	devID, props.name, props.major, props.minor);
	
	
		// Allocate GPU Memory
		float* d_vel = NULL;
		float* d_posIn = NULL;
		float* d_posOut = NULL;		

		unsigned int mem_size = sizeof(float) * pointsSize;

		checkCudaErrors(cudaMallocManaged(&d_vel, mem_size));
		checkCudaErrors(cudaMallocManaged(&d_posIn, mem_size));
		checkCudaErrors(cudaMallocManaged(&d_posOut, mem_size));

		for (int i = 0; i < pointsSize; i++) 
		{	
			d_vel[i] = (float) h_vel[i];
			d_posIn[i] = (float) h_posIn[i];
		}
		

		float tauf = (float) tau;
		
		// needed block count
		int threadsPerBlock = 1024; //1024;
		int blockCount = pointsSize/threadsPerBlock + (pointsSize%threadsPerBlock == 0 ? 0:1);
				
		std::cout << "total number of threads: " << blockCount*threadsPerBlock << std::endl;

		// run on GPU
		for (int i = 0; i < test_runs; i++)
		{
		
			// compute k1 and temp position on GPU
			runge_kutta_helper1float(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);
			//getLastCudaError("Kernel execution failed");
			
			// skip second evaluation
			//field_fn(h_posTmp, num_points, h_vel);

			// compute k2 and final position on GPU
			runge_kutta_helper2float(d_vel, d_posIn, d_posOut, tauf, pointsSize, blockCount, threadsPerBlock);

		}

		
		// clean up
		checkCudaErrors(cudaFree(d_vel));
		checkCudaErrors(cudaFree(d_posIn));
		checkCudaErrors(cudaFree(d_posOut));
		cudaDeviceReset();

	}

	// cpu version, double
	void cpu_version (double* h_vel, double* h_posIn, int pointsSize, double tau, int test_runs, int argc, char **argv) {		

		

		double* h_posTmp = new double[pointsSize];
		
		
		for (int i = 0; i < test_runs; i++)
		{

			for (int i = 0; i < pointsSize; i++)
	    			h_posTmp[i] = h_posIn[i] + 0.5*tau*h_vel[i];

			// skip second evaluation

			for (int i = 0; i < pointsSize; i++)
	    			h_posTmp[i] = h_posIn[i] + tau*h_vel[i];

		}
		
		// clean up
		delete[] h_posTmp;
	}

	// cuda optimization speed tests
	void cuda_comparisons (Tree_t* velT, std::vector<double> points, int num_rk_step, double dt, int argc, char **argv) {
		std::cout << "\n-------------------------------------------------------" << std::endl;
		std::cout << "START SPEED TESTS" << std::endl;
		std::cout << "-------------------------------------------------------\n" << std::endl;

		int pointsSize = points.size();
		int num_points = pointsSize/3;		
		int test_runs = 100;

		// same timestep as in traj
		double tinit = 0;
                double tfinal = tinit + num_rk_step*dt;
		double tau = (tfinal - tinit)/num_rk_step;		

		tbslas::NodeFieldFunctor<double,Tree_t> field_fn = tbslas::NodeFieldFunctor<double,Tree_t>(velT);
		
		double* h_vel = new double[pointsSize];
		double* h_posIn = new double[pointsSize];

		for (int i = 0; i < pointsSize; i++)
			{
				h_posIn[i] = points[i];
				//skip evaluation
				h_vel[i] = 0;
			}

		// evaluate speed
		//field_fn(h_posIn, num_points, h_vel);
		
		std::cout << "\ntest runs: " << test_runs << std::endl;

		std::cout << "original_unoptimized\n" << std::endl;
		original_unoptimized (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);
		std::cout << "\noriginal_float\n" << std::endl;
		original_float (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);
		std::cout << "\nshared_float\n" << std::endl;
		shared_float (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);
		std::cout << "\nmulti_float\n" << std::endl;
		multi_float (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);	
		std::cout << "\nreducethreats_float\n" << std::endl;
		reducethreads_float (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);	
		std::cout << "\nzerocopy_float\n" << std::endl;
		zerocopy_float (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);
		std::cout << "\nunifiedmemory_float\n" << std::endl;
		unifiedmemory_float (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);	
		std::cout << "\ncpu_version\n" << std::endl;
		cpu_version (h_vel, h_posIn, pointsSize, tau, test_runs, argc, argv);

		std::cout << "\n-------------------------------------------------------" << std::endl;
		std::cout << "END SPEED TESTS" << std::endl;
		std::cout << "-------------------------------------------------------\n" << std::endl;

	}

/////////////////////////////////////////////////////////////////
// END OF CUDA FUNCTIONS
////////////////////////////////////////////////////////////////

int main (int argc, char **argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm comm=MPI_COMM_WORLD;
	int np;
	MPI_Comm_size(comm, &np);
	int myrank;
	MPI_Comm_rank(comm, &myrank);
	
	parse_command_line_options(argc, argv);
	
	int   test = strtoul(commandline_option(argc, argv, "-test",     "1", false,
                                          "-test <int> = (1)    : 1) Gaussian function 2) My Gaussian kernel 3) existing Gaussian kernel"),NULL,10);
    	sigma_in = strtod(commandline_option(argc, argv, "-sigma",     "0.7", false,
                                          "-sigma <double> = (0.7)    : Sigma for my_gaussian_kernel"),NULL);
	xc_in = strtod(commandline_option(argc, argv, "-xc",     "0.7", false,
                                          "-xc <double> = (0.7)    : xc for gaussian_function"),NULL);
	yc_in = strtod(commandline_option(argc, argv, "-yc",     "0.7", false,
                                          "-yc <double> = (0.7)    : yc for gaussian_function"),NULL);
	zc_in = strtod(commandline_option(argc, argv, "-zc",     "0.7", false,
                                          "-zc <double> = (0.7)    : zc for gaussian_function"),NULL);
	c_in = strtod(commandline_option(argc, argv, "-c",     "0.0559", false,
                                          "-c <double> = (0.0559)    : c for gaussian_function"),NULL);
	amp_in = strtod(commandline_option(argc, argv, "-amp",     "1", false,
                                          "-amp <double> = (1)    : amp for gaussian_function"),NULL);
	int   numEvPoints = strtoul(commandline_option(argc, argv, "-numEvP",     "1", false,
                                          "-numEvP <int> = (1)    : how many points for evaluation"),NULL,10);

	// =========================================================================
    // SIMULATION PARAMETERS
    // =========================================================================
    tbslas::SimConfig* sim_config       = tbslas::SimConfigSingleton::Instance();
    sim_config->vtk_filename_variable   = "conc";
    pvfmm::Profile::Enable(sim_config->profile);
    // =========================================================================
    // PRINT METADATA
    // =========================================================================
    if (!myrank) {
      MetaData_t::Print();
    }
	// =========================================================================
    // TEST CASE
    // =========================================================================
	switch(test) {
		case(1): 
			fname = (char*) "gaussian-function";
			fn_val = get_gaussian_function_wraper<double>;
			fn_vel = tbslas::get_vorticity_field<double,3>;
			fn_vel_f = tbslas::get_vorticity_field<float,3>;
			break;
		case(2):
			fname = (char*) "my-gaussian-kernel";
			fn_val = get_my_gaussian_kernel_wraper<double>;
			fn_vel = tbslas::get_vorticity_field<double,3>;
			fn_vel_f = tbslas::get_vorticity_field<float,3>;
			break;
		case(3):
			fname = (char*) "gaussian-kernel";
			fn_val = get_gaussian_kernel_wraper<double>;
			fn_vel = tbslas::get_vorticity_field<double,3>;
			fn_vel_f = tbslas::get_vorticity_field<float,3>;
		default:			
			break;
	}
	// =========================================================================
    // CONSTRUCT TREE
    // =========================================================================
    Tree_t tree(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_val,
                                  1,
                                  tree);

    // Create velocity tree for Runge-Kutte
    int max_depth_vel=0;
    Tree_t tvel(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  max_depth_vel?max_depth_vel:sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel,
                                  3,
                                  tvel);
	
    // Create float velocity tree for Runge-Kutte
    //int max_depth_vel=0;
    Tree_t_f tvel_f(comm);
    tbslas::ConstructTree<Tree_t_f>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  max_depth_vel?max_depth_vel:sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel_f,
                                  3,
                                  tvel_f);



						
	// create random points and evaluate the tree values with NodeFieldFunctor
	int num_points = numEvPoints;
    	std::vector<double> xtmp(num_points*3);
	for (int i = 0; i < xtmp.size(); i++) 
	{
	  xtmp[i] = ((double) (rand()%10000)/10000.0); // random points between 0 and 1
	  //std::cout << "xtmp: " << xtmp[i] << std::endl; 
	}   	
	
    std::vector<double> vtmp(num_points);
    tbslas::NodeFieldFunctor<double,Tree_t> cfun = tbslas::NodeFieldFunctor<double,Tree_t>(&tree);
	// evaluation not needed for speed comparison, reactivate later
    	//cfun(xtmp.data(), num_points, vtmp.data());

	// Start Runge Kutta
	runge_kutta_run (&tvel, xtmp, sim_config->total_num_timestep, sim_config->dt, argc, argv, &sim_config->comm);
	//runge_kutta_run_float (&tvel_f, xtmp, sim_config->total_num_timestep, sim_config->dt, argc, argv, &sim_config->comm);
	//runge_kutta_run_zerocopyfloat (&tvel_f, xtmp, sim_config->total_num_timestep, sim_config->dt, argc, argv, &sim_config->comm);
	//runge_kutta_run_unifiedmemoryfloat (&tvel_f, xtmp, sim_config->total_num_timestep, sim_config->dt, argc, argv, &sim_config->comm);
	
	//cuda_comparisons (&tvel, xtmp, sim_config->total_num_timestep, sim_config->dt, argc, argv);

	if (sim_config->vtk_save_rate) {
	
      //tree.Write2File(tbslas::GetVTKFileName(0, sim_config->vtk_filename_variable).c_str(),
       //               sim_config->vtk_order);

	
	tree.Write2File(fname,
                      sim_config->vtk_order);
    }
	// =========================================================================
    // COMPUTE ERROR
    // =========================================================================
	double rli, rl2;
    double ali, al2;
    CheckChebOutput<Tree_t>(&tree,
                            fn_val,
                            1,
                            al2,rl2,ali,rli,
                            std::string("Output"));


    // check velocity tree
    double in_al2,in_rl2,in_ali,in_rli;
    CheckChebOutput<Tree_t>(&tvel,
                            fn_vel,
                            3,
                            in_al2,in_rl2,in_ali,in_rli,
                            std::string("Input-Vel"));


							
								  
	int num_leaves = tbslas::CountNumLeafNodes(tree);


    // =========================================================================
    // chebyshev coeff and evaluation points in buffers
    // =========================================================================
	// commented out because only used for testing if copying to GPU works
	/*
  	NodeType* n_next = tree.PostorderFirst();
  	//int num_leaf_nodes = 0;
	std::vector<pvfmm::Vector<double>> chebdata;


	chebdata.reserve(num_leaves);
  	while (n_next != NULL) {
  	  if(!n_next->IsGhost() && n_next->IsLeaf())
	    {
	    chebdata.push_back(n_next->ChebData());

	    //temp = n_next->ChebData();
	    //std::cout << "temp dim: " << temp.Dim() <<std::endl;
	    //std::cout << "temp Cap: " << temp.Capacity() <<std::endl;
	    //std::cout << "temp values: " << temp[0] << " " << temp[1] << " " << temp[2] <<std::endl;
  	    //num_leaf_nodes++;
            }
  	  n_next = tree.PostorderNxt(n_next);
  	}
	*/

	/*
	std::cout << "chebdata size: " << chebdata.size() <<std::endl;
	// to check if everything has the same size
	//std::cout << "chebdata value sizes: " << chebdata[0].Dim() << " " << chebdata[1].Dim()<< " " << chebdata[50].Dim() << std::endl;

	std::vector<double> xtmpOut;
	std::vector<pvfmm::Vector<double>> chebdataOut;
	xtmpOut.reserve(num_points*3);
	chebdataOut.reserve(num_leaves);
	*/

	/*pvfmm::Vector<double> temp;
	double* tempArray = new double[chebdata[0].Dim()];
	temp.ReInit(chebdata[0].Dim(), tempArray, false);
	temp.SetZero();
	std::cout << "Dim: " << temp.Dim() << std::endl;
	chebdataOut.push_back(temp);
	std::cout << "Dim: " << temp.Dim() << std::endl;*/
	/*
	for (int i = 0; i < chebdata.size(); i++) {
		pvfmm::Vector<double> temp;
		double* tempArray = new double[chebdata[0].Dim()];
		temp.ReInit(chebdata[0].Dim(), tempArray, false);
		temp.SetZero();		
		chebdataOut.push_back(temp);	
	}	
	
	// Arrays for copyToGPUArrays
	double* evPoints = &xtmp[0];
	double* evPointsOut = new double[num_points*3];
	double* chebCoeff = new double[chebdata.size() * chebdata[0].Dim()];
		for (int i = 0; i < chebdata.size(); i++)
		{
			double* temp = &chebdata[i][0];
			for (int j = 0; j < chebdata[0].Dim(); j++) {
				chebCoeff[i*chebdata[0].Dim() + j] = temp[j];	
			}		
		}
	double* chebCoeffOut = new double[chebdata.size() * chebdata[0].Dim()];

	double** chebCoeff2D = new double*[chebdata.size()];
	double** chebCoeff2DOut = new double*[chebdata.size()];
	for (int i = 0; i < chebdata.size(); i++)
	{
		chebCoeff2D[i] = new double[chebdata[0].Dim()];
		chebCoeff2DOut[i] = new double[chebdata[0].Dim()];

		for (int j = 0; j < chebdata[0].Dim(); j++) {
				chebCoeff2D[i][j] = chebdata[i][j];	
				//chebCoeff2DOut[i][j] = chebdata[i][j];	
			}
	}
	*/

	// GPU FUNCTION CALL
	//copyToGPUVectors(argc, argv, xtmp, chebdata, xtmpOut, chebdataOut);
	//copyToGPUArrays(argc, argv, evPoints, chebCoeff, evPointsOut, chebCoeffOut, num_points * 3, chebdata.size(), chebdata[0].Dim());    
       
    ////////////////////////////////////////////////////////////////////   
       
       
	// =========================================================================
    // REPORT RESULTS
    // =========================================================================
	int tree_max_depth = 0;
	tbslas::GetTreeMaxDepth(tree, tree_max_depth);
	
	typedef tbslas::Reporter<double> Rep;
	if(!myrank) {
		Rep::AddData("NP", np, tbslas::REP_INT);
		Rep::AddData("OMP", sim_config->num_omp_threads, tbslas::REP_INT);	
			
		Rep::AddData("TOL", sim_config->tree_tolerance);
		Rep::AddData("Q", sim_config->tree_chebyshev_order, tbslas::REP_INT);
		Rep::AddData("NOCT", num_leaves, tbslas::REP_INT);
		
		Rep::AddData("MaxD", sim_config->tree_max_depth, tbslas::REP_INT);
		Rep::AddData("TMaxD", tree_max_depth, tbslas::REP_INT);

		Rep::AddData("CUBIC", sim_config->use_cubic?1:0, tbslas::REP_INT);
		Rep::AddData("CUF", sim_config->cubic_upsampling_factor, tbslas::REP_INT);
		
		Rep::AddData("TEST", test, tbslas::REP_INT);

		Rep::AddData("OutAL2", al2);
		Rep::AddData("OutALINF", ali);

		Rep::Report();
  }
  
  //Output Profiling results.
   pvfmm::Profile::print(&comm);
  
  // clean up 	
  /*
  for(int i = 0; i < chebdata.size(); ++i) {
    	//delete [] chebCoeff2D[i];
	delete [] chebCoeff2DOut[i];
  }	
  delete [] chebCoeff2D;
  delete [] chebCoeff2DOut;
  delete[] evPointsOut;
  delete[] chebCoeff;
  delete[] chebCoeffOut;
  */

  // Shut down MPI
  MPI_Finalize();
  return 0;
}
