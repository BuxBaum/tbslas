/*
 * Contains test function and examples
 * 
 * Important tests are 6,7,8
 * 
 */

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
#include <tree/node_field_functor_cuda.h>
#include <tree/tree_functor_cuda_vec.h>

#include <tree.hpp>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>    

#include <stdlib.h>

#include "chebEval.h"
#include "runge-kutta.h"
#include "cusparse.h"

#include <field_wrappers.h>

#include <cstdio>

typedef pvfmm::Cheb_Node<double> Node_t;
typedef pvfmm::Cheb_Node<float> Node_t_f;
typedef pvfmm::MPI_Tree<Node_t> Tree_t;
typedef pvfmm::MPI_Tree<Node_t_f> Tree_t_f;

typedef tbslas::MetaData<std::string,
                         std::string,
                         std::string> MetaData_t;

typedef typename Tree_t::Node_t NodeType;
typedef typename Tree_t_f::Node_t NodeType_f;	

char* fname = (char*) "velocityTree";

void (*fn_vel)(const double* , int , double*)=NULL;
void (*fn_vel2)(const double* , int , double*)=NULL;
void (*fn_vel_f)(const float* , int , float*)=NULL;	
void (*fn_val)(const double* , int , double*)=NULL;

// compute Chebyshev polynomials on GPU
// compare to CPU version
// d : cheb degree, in: input coords, n : coord count
template <class Real_t>
void chebPoly(int d, Real_t* in, int n) 
{
	d++;
	
	tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
	MPI_Comm* comm = &sim_config->comm;
		
	// Get CUDA device
	int devID;
   	cudaDeviceProp props;

   	// This will pick the best possible CUDA capable device
   	devID = findCudaDevice(0, (const char **)"");

   	//Get GPU information
   	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		devID, props.name, props.major, props.minor);
		
	// split in to x,y,z arrays
	Real_t* x_in = new Real_t[n];
	Real_t* y_in = new Real_t[n];
	Real_t* z_in = new Real_t[n];
	Real_t* x_out = new Real_t[n*(d+1)];
	Real_t* y_out = new Real_t[n*(d+1)];
	Real_t* z_out = new Real_t[n*(d+1)];
	
	for (int i = 0; i < n; i++)
	{
		x_in[i] = in[i*COORD_DIM+0];
		y_in[i] = in[i*COORD_DIM+1];
		z_in[i] = in[i*COORD_DIM+2];
	}
	
	pvfmm::Profile::Tic("Chebyshev Polynomials on GPU", comm, true, 5);
	pvfmm::Profile::Tic("Initialization", comm, true, 5);
	
	// allocate GPU memory
	Real_t* d_x_in = NULL;
	Real_t* d_y_in = NULL;
	Real_t* d_z_in = NULL;
	Real_t* d_x_out = NULL;
	Real_t* d_y_out = NULL;
	Real_t* d_z_out = NULL;
	
	unsigned int mem_size_in = sizeof(Real_t) * n;
	unsigned int mem_size_out = sizeof(Real_t) * n * (d+1);
	
	checkCudaErrors(cudaMalloc((void**) &d_x_in, mem_size_in)); 
	checkCudaErrors(cudaMalloc((void**) &d_y_in, mem_size_in)); 
	checkCudaErrors(cudaMalloc((void**) &d_z_in, mem_size_in)); 
	checkCudaErrors(cudaMalloc((void**) &d_x_out, mem_size_out)); 
	checkCudaErrors(cudaMalloc((void**) &d_y_out, mem_size_out));
	checkCudaErrors(cudaMalloc((void**) &d_z_out, mem_size_out));
	
	// needed block count
	// n threads in total per direction, for now just blocks with max thread number
	int threadsPerBlock = 1024;
	int blockCount = n/threadsPerBlock + (n%threadsPerBlock == 0 ? 0:1);
	
	pvfmm::Profile::Toc();
	pvfmm::Profile::Tic("Copy to GPU", comm, true, 5);
	
	// Copy data to GPU
	checkCudaErrors(cudaMemcpy(d_x_in, x_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y_in, y_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_z_in, z_in, mem_size_in,
                               cudaMemcpyHostToDevice));
                               
    pvfmm::Profile::Toc();
    pvfmm::Profile::Tic("Calculate polynomials", comm, true, 5);
                               
    // Calculate chebPoly for x,y,z
    pvfmm::Profile::Tic("X", comm, true, 5);
    chebPoly_helper(d_x_in, d_x_out, d, n, blockCount, threadsPerBlock);
    pvfmm::Profile::Toc();
    pvfmm::Profile::Tic("Y", comm, true, 5);
    chebPoly_helper(d_y_in, d_y_out, d, n, blockCount, threadsPerBlock);
    pvfmm::Profile::Toc();
    pvfmm::Profile::Tic("Z", comm, true, 5);
    chebPoly_helper(d_z_in, d_z_out, d, n, blockCount, threadsPerBlock);
    pvfmm::Profile::Toc();
    pvfmm::Profile::Toc();
    
    pvfmm::Profile::Tic("Copy to CPU", comm, true, 5);
    // Copy data back to CPU
    checkCudaErrors(cudaMemcpy(x_out, d_x_out, mem_size_out,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y_out, d_y_out, mem_size_out,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(z_out, d_z_out, mem_size_out,
                               cudaMemcpyDeviceToHost));
    pvfmm::Profile::Toc();
    pvfmm::Profile::Toc();
    
                               
                               
     // Run on CPU for comparison
    Real_t* x_out_cpu = new Real_t[n*(d+1)];
	Real_t* y_out_cpu = new Real_t[n*(d+1)];
	Real_t* z_out_cpu = new Real_t[n*(d+1)];
	
	pvfmm::Profile::Tic("Chebyshev Polynomials on CPU", comm, true, 5);
	
	pvfmm::cheb_poly(d,&(x_in[0]),n,&(x_out_cpu[0]));
	pvfmm::cheb_poly(d,&(y_in[0]),n,&(y_out_cpu[0]));
	pvfmm::cheb_poly(d,&(z_in[0]),n,&(z_out_cpu[0]));
	
	pvfmm::Profile::Toc();
	
	// compare Values
	bool passed = true;
	Real_t tolerance = 1e-6f;
    for (int i = 0; i < n*(d+1); i++)
    {
		if (std::abs(x_out[i] - x_out_cpu[i]) > tolerance ||
		    std::abs(y_out[i] - y_out_cpu[i]) > tolerance ||
			std::abs(z_out[i] - z_out_cpu[i]) > tolerance) {	
			std::cout << "x_out: " << x_out[i] << "x_out_cpu: " << x_out_cpu[i] << std::endl;	
			std::cout << "y_out: " << y_out[i] << "y_out_cpu: " << y_out_cpu[i] << std::endl;	
			std::cout << "z_out: " << z_out[i] << "z_out_cpu: " << z_out_cpu[i] << std::endl;	
			passed = false;
			std::cout << "Fail at i: " << i << std::endl;
			break;
		}
	}     
	if (passed == true)
	std::cout << "chebPoly passed tolerance" << std::endl;                      
                               
    // clean up
    delete [] x_in;
    delete [] y_in;
    delete [] z_in;
    delete [] x_out;
    delete [] y_out;
    delete [] z_out;
    delete [] x_out_cpu;
    delete [] y_out_cpu;
    delete [] z_out_cpu;
    checkCudaErrors(cudaFree(d_x_in));
    checkCudaErrors(cudaFree(d_y_in));
    checkCudaErrors(cudaFree(d_z_in));
	checkCudaErrors(cudaFree(d_x_out));
    checkCudaErrors(cudaFree(d_y_out));
    checkCudaErrors(cudaFree(d_z_out));
	cudaDeviceReset();
}

// tests for matrix multiplication
// used to find out how to use cuBLAS in the best way
// not relevant for vector evaluation
/**
 * Results:
 * CPU GEMM, Cublas GEMM Switched, pvfmm Cublas Gemm: correct results but stored like a b c a b c ...
 * CuBLAS Gemm Transposed: correct results and stored like a a a b b b c c c ...
 * Cublas Gemm, pvfmm Cublas Gemm switched, pvfmm cublas gemm changed: WRONG
 * 
 * USE CUBLAS GEMM TRANSPOSED TO AVOID EXTRA TRANSPOSING
 * 
 */ 
 // test differen matrix multiplications
void matrixMulTests()
{
	// expected results: 33 36 39 75 82 89 117 128 139 159 174 189
	// needed results to avoid transposing: 33 75 117 159 36 82 128 174 39 89 139 189
		
	std::vector<double> d1(4*2);
	std::vector<double> d2(2*3);
	std::vector<double> d0(4*3);
	
	// set test values
	for (int i = 0; i < 8; i++) {
		d1[i] = (double) i+1.0;
	}
	for (int i = 0; i < 6; i++) {
		d2[i] = (double) i+9;
	}
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	
	pvfmm::Matrix<double> M1  ( 4, 2,&d1[0],false);
	pvfmm::Matrix<double> M2  ( 2, 3,&d2[0],false);
	pvfmm::Matrix<double> Mo  ( 4,3,&d0[0],false);
    
	// CPU mul
	pvfmm::Matrix<double>::GEMM(Mo, M1, M2);
	
	std::cout << "CPU Mul: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
	
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	
// init GPU
	// Get CUDA device
	int devID;
   	cudaDeviceProp props;

   	// This will pick the best possible CUDA capable device
   	devID = findCudaDevice(0, (const char **)"");

   	//Get GPU information
   	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    //printf("Device %d: \"%s\" with Compute %d.%d capability\n",
	//	devID, props.name, props.major, props.minor);
	
	double* d_d1 = NULL;
	double* d_d2 = NULL;
	double* d_d0 = NULL;
	
	checkCudaErrors(cudaMalloc((void**) &d_d1, sizeof(double) * 8)); 
	checkCudaErrors(cudaMalloc((void**) &d_d2, sizeof(double) * 6)); 
	checkCudaErrors(cudaMalloc((void**) &d_d0, sizeof(double) * 12)); 
	
	checkCudaErrors(cudaMemcpy(d_d1, &d1[0], sizeof(double) * 8,
                               cudaMemcpyHostToDevice));   
    checkCudaErrors(cudaMemcpy(d_d2, &d2[0], sizeof(double) * 6,
                               cudaMemcpyHostToDevice));                          
    checkCudaErrors(cudaMemcpy(d_d0, &d0[0], sizeof(double) * 12,
                               cudaMemcpyHostToDevice));    
                               
    cublasHandle_t handle;
    cublasCreate(&handle); 
                        
	// CuBLAS 
	int M = 4;  // number of rows in A and C
	int N = 3;  // number of colums in B and C
	int K = 2;  // number of colums in A and rows in B
	double alpha = 1.0; // scalar used for multiplication (keep at 1)
	double beta = 0.0; // scalar used for multiplication (keep at 0)
	int lda = M; // leading dimension of matrix A
	int ldb = K; // leading dimension of matrix B
	int ldc = M; // leading dimension of Matrix C
	// non transposed, non switched cublas gemm
	cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_d1, lda, d_d2, ldb, &beta, d_d0, ldc);
	
	checkCudaErrors(cudaMemcpy(&d0[0], d_d0, sizeof(double) * 12,
                               cudaMemcpyDeviceToHost));    
	
	std::cout << "CuBLAS Mul: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
	
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	
	// CuBLAS switched
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], sizeof(double) * 12,
                               cudaMemcpyHostToDevice));
    
    M = 3;
    N = 4; 
    K = 2;
    lda = M;                           
    ldb = K;
    ldc = M;
    // non transposed, switched cublas gemm                           
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_d2, lda, d_d1, ldb, &beta, d_d0, ldc);
	
	checkCudaErrors(cudaMemcpy(&d0[0], d_d0, sizeof(double) * 12,
                               cudaMemcpyDeviceToHost));                          
	
	std::cout << "CuBLAS Mul Switched: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
	
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	
	// CuBLAS transposed not switched
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], sizeof(double) * 12,
                               cudaMemcpyHostToDevice));
	
	M = 4;
    N = 3; 
    K = 2;
    lda = 2;                           
    ldb = 3;
    ldc = M;
	// transposed, non switched cublas gemm
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_d1, lda, d_d2, ldb, &beta, d_d0, ldc);
	
	checkCudaErrors(cudaMemcpy(&d0[0], d_d0, sizeof(double) * 12,
                               cudaMemcpyDeviceToHost));                          
	
	std::cout << "CuBLAS Mul Transposed: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
	
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	
	// pvfmm cublasgemm
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], sizeof(double) * 12,
                               cudaMemcpyHostToDevice));
                   
    pvfmm::Matrix<double> M1_c  ( 4, 2,&d_d1[0],false);
	pvfmm::Matrix<double> M2_c  ( 2, 3,&d_d2[0],false);
	pvfmm::Matrix<double> Mo_c  ( 4,3,&d_d0[0],false);           
    // cublas gemm using pvfmm wraper                           
    pvfmm::Matrix<double>::CUBLASGEMM(Mo_c, M1_c, M2_c); 
    
    checkCudaErrors(cudaMemcpy(&d0[0], d_d0, sizeof(double) * 12,
                               cudaMemcpyDeviceToHost));        
    
    std::cout << "pvfmm CuBLAS Mul: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
	
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	
	// pvfmm cublasgemm switched
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], sizeof(double) * 12,
                               cudaMemcpyHostToDevice));
    // pvfmm cublas gemm wraper with matrices switched                     
	pvfmm::Matrix<double>::CUBLASGEMM(Mo_c, M2_c, M1_c);
	
	
	checkCudaErrors(cudaMemcpy(&d0[0], d_d0, sizeof(double) * 12,
                               cudaMemcpyDeviceToHost));        
    
    std::cout << "pvfmm CuBLAS Mul switched: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
		
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}  
	
	   
	// pvfmm cublasgemm matrixes changed
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], sizeof(double) * 12,
                               cudaMemcpyHostToDevice));  
                               
	pvfmm::Matrix<double> M1_c2  ( 2, 4,&d_d1[0],false);
	pvfmm::Matrix<double> M2_c2  ( 3, 2,&d_d2[0],false);
	pvfmm::Matrix<double> Mo_c2  ( 3,4,&d_d0[0],false);	  
	// pvfmm cublas gemm wraper with different matrix sizes                                
	pvfmm::Matrix<double>::CUBLASGEMM(Mo_c2, M2_c2, M1_c2); 
	
	checkCudaErrors(cudaMemcpy(&d0[0], d_d0, sizeof(double) * 12,
                               cudaMemcpyDeviceToHost));        
    
    std::cout << "pvfmm CuBLAS Mul changed: " << std::endl;
	for (int i = 0; i < 12; i++) {
		std::cout << d0[i] << " ";
	}
	std::cout << std::endl;
		
	for (int i = 0; i < 12; i++) {
		d0[i] = (double) 0.0;
	}
	                    
	//clean up
	cublasDestroy(handle);
	checkCudaErrors(cudaFree(d_d1));
    checkCudaErrors(cudaFree(d_d2));
    checkCudaErrors(cudaFree(d_d0));
	cudaDeviceReset();
}

// simulate first multiplikation
// comparing CPU, cuBLAS, cuSPARSE
void matrixMul_Manypoints_mul1(int numPoints, int d)
{
	tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
	MPI_Comm* comm = &sim_config->comm;
	
	d=d+1;
	
	int dof = 3;
	// vectors for multipikation
	std::vector<double> coeff (d*d*d*dof);
	std::vector<double> poly (numPoints * d);
	std::vector<double> resCPU (numPoints*d*d*dof);
	std::vector<double> resGPU (numPoints*d*d*dof);
	int sizeCoeff = d*d*d*dof;
	int sizePoly = numPoints*d;
	int sizeRes = numPoints*d*d*dof;
	for (int i = 0; i < sizeCoeff; i++)
		{
			//coeff[i] = (double) i;
			//if (i%2==0 || i%3 == 0 || i > 15)
			//	coeff[i] = 0.0;
			
			coeff[i] = 0.0;
			if (i%10==0)
				coeff[i] = (double) i;
		}
		
	for (int i = 0; i < sizePoly; i++)
		{
			poly[i] = (double) i;
		}
	
	pvfmm::Profile::Tic("Multiplication on GPU", comm, true, 5);
		
	// Get CUDA device
	int devID;
   	cudaDeviceProp props;

   	// This will pick the best possible CUDA capable device
   	devID = findCudaDevice(0, (const char **)"");

   	//Get GPU information
   	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		devID, props.name, props.major, props.minor);
		
	pvfmm::Profile::Tic("Allocate buffers", comm, true, 5);
	// Allocate Buffers
	int mem_size_poly = sizeof(double) * sizePoly;
	int mem_size_coeff = sizeof(double) * sizeCoeff;
	int mem_size_res = sizeof(double) * sizeRes;
	
	double* d_poly = NULL;
	double* d_coeff = NULL;
	double* d_res = NULL;
	
	checkCudaErrors(cudaMalloc((void**) &d_poly, mem_size_poly));
	checkCudaErrors(cudaMalloc((void**) &d_coeff, mem_size_coeff));
	checkCudaErrors(cudaMalloc((void**) &d_res, mem_size_res));
	pvfmm::Profile::Toc();
	pvfmm::Profile::Tic("Copy data to GPU", comm, true, 5);
	// Copy data to GPU
	checkCudaErrors(cudaMemcpy(d_poly, &poly[0], mem_size_poly,
                               cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_coeff, &coeff[0], mem_size_coeff,
                               cudaMemcpyHostToDevice));    
	pvfmm::Profile::Toc();
	
	// Init Cublas
	cublasHandle_t handle;
    cublasCreate(&handle); 
    cublasStatus_t status;
                         
	int M = d*d*dof;
	int N = numPoints;
	int K = d;
	double alpha = 1.0;
	double beta = 0.0;
	int lda = K;
	int ldb = N;
	int ldc = M;
	
	pvfmm::Profile::Tic("Multiplication cuBLAS", comm, true, 5);
	// Multipication
	checkCudaErrors(cudaDeviceSynchronize());
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_coeff, lda, d_poly, ldb, &beta, d_res, ldc);
	checkCudaErrors(cudaDeviceSynchronize());
	pvfmm::Profile::Toc();
	
	pvfmm::Profile::Tic("Copy data from GPU", comm, true, 5);
	// Copy data back to CPU
	checkCudaErrors(cudaMemcpy(&resGPU[0], d_res, mem_size_res,
                              cudaMemcpyDeviceToHost)); 
    pvfmm::Profile::Toc();                           
    
 
    pvfmm::Profile::Toc();
    
    // cuSPARSE
    cusparseStatus_t sparsestatus;
	cusparseHandle_t sparsehandle;
	sparsestatus = cusparseCreate(&sparsehandle);
	if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
		std::cout << "cusparse init failed" << std::endl;
			
	cusparseMatDescr_t descrA = 0;
	sparsestatus = cusparseCreateMatDescr(&descrA);
	if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
		std::cout << "cusparse MatDescr failed" << std::endl;
	cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    
	pvfmm::Profile::Tic("cuSPARSE", comm, false, 5);	
    
	pvfmm::Profile::Tic("nnz", comm, false, 5);	
    
			// apparently uses column major
    
			M = d;          // rows
			N = d*d*dof; // columns
			lda = M;
				
			int* nnzPerC = new int[N];
			int nnzTotal;
			
			int mem_size_nnz = sizeof(int) * N;
			
			int* d_nnz = NULL;
			
			checkCudaErrors(cudaMalloc((void**) &d_nnz, mem_size_nnz));
				
			// for csr	
			
			sparsestatus = cusparseDnnz(sparsehandle, CUSPARSE_DIRECTION_COLUMN, M, 
             N, descrA, 
             d_coeff, 
             lda, d_nnz, &nnzTotal);
             
			if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
				std::cout << "cusparse nnz failed" << std::endl;

			checkCudaErrors(cudaDeviceSynchronize());
			pvfmm::Profile::Toc();
			
			pvfmm::Profile::Tic("dense to csr", comm, false, 5);	
				
			// convert dense to csr matrix
			int mem_size_val = sizeof(double) * nnzTotal;
			int mem_size_colPtr = sizeof(int) * N+1;//M+1;	
			int mem_size_rowInd = sizeof(int) * nnzTotal;				
			double* d_valA = NULL;	
			int* d_csrColPtrA = NULL;
			int* d_csrRowIndA = NULL;
			
			checkCudaErrors(cudaMalloc((void**) &d_valA, mem_size_val));					
			checkCudaErrors(cudaMalloc((void**) &d_csrRowIndA, mem_size_rowInd));
			checkCudaErrors(cudaMalloc((void**) &d_csrColPtrA, mem_size_colPtr));					
					
           sparsestatus = cusparseDdense2csc(sparsehandle, M, N, 
                descrA, 
                d_coeff, 
                lda, d_nnz, 
                d_valA, 
                d_csrRowIndA, d_csrColPtrA);
                
            if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
				std::cout << "cusparse dense2csc failed" << std::endl;
            
            checkCudaErrors(cudaDeviceSynchronize());
            pvfmm::Profile::Toc();			
										
			pvfmm::Profile::Tic("poly transpose", comm, true, 5);
			// transpose poly
			double* d_polytrans = NULL;
			checkCudaErrors(cudaMalloc((void**) &d_polytrans, mem_size_poly));
			M = d; // number of rows
			N = numPoints; // number of columns
			lda = N;
			ldb = M;
			status = cublasDgeam(handle,
								  CUBLAS_OP_T, CUBLAS_OP_N,
								  M, N,
								  &alpha,
								  d_poly, lda,
								  &beta,
								  d_polytrans, ldb,
								  d_polytrans, ldb);
								  
			checkCudaErrors(cudaDeviceSynchronize());					  
			pvfmm::Profile::Toc();			
												
			pvfmm::Profile::Tic("mul", comm, false, 5);	
			
			M = d*d*dof;   // rows A
			N = numPoints; // columns B
			K = d;
			lda = M;
			ldb = K;
			ldc = M;
			
			sparsestatus = cusparseDcsrmm(sparsehandle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE, 
			M, 
			N, 
			K, 
			nnzTotal, 
			&alpha, 
			descrA, 
			d_valA, 
			d_csrColPtrA, 
			d_csrRowIndA,
			d_polytrans, 
			ldb,
			&beta,
			d_res, 
			ldc);

			if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
				std::cout << "cusparse dcsrmm failed" << std::endl;

			checkCudaErrors(cudaDeviceSynchronize());
			pvfmm::Profile::Toc();
			pvfmm::Profile::Toc();
							
			double* outtest = new double[ldc*numPoints];
			int mem_size_outtest = sizeof(double) * ldc * numPoints;
			checkCudaErrors(cudaMemcpy(outtest, d_res, mem_size_outtest,
                               cudaMemcpyDeviceToHost));		
					
			pvfmm::Profile::Tic("cuSPARSE transpose included", comm, false, 5);	
			sparsestatus = cusparseDcsrmm2(sparsehandle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_TRANSPOSE, 
			M, 
			N, 
			K, 
			nnzTotal, 
			&alpha, 
			descrA, 
			d_valA, 
			d_csrColPtrA, 
			d_csrRowIndA,
			d_poly, 
			N,
			&beta,
			d_res, 
			ldc);

			if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
				std::cout << "cusparse dcsrmm failed" << std::endl;
			checkCudaErrors(cudaDeviceSynchronize());
			pvfmm::Profile::Toc();		
								  
    pvfmm::Profile::Tic("Multiplication on CPU", comm, true, 5);
    // CPU Multiplication
    {   
			std::vector<double> tmpRes (sizeRes);
			
			pvfmm::Matrix<double> Mi  (d*d*dof, d,&coeff[0],false);
			pvfmm::Matrix<double> Mp  (d, numPoints,&poly[0],false);
			pvfmm::Matrix<double> Mo  (d*d*dof,numPoints,&tmpRes[0],false);
			pvfmm::Profile::Tic("Multiplication", comm, true, 5);
			pvfmm::Matrix<double>::GEMM(Mo, Mi, Mp);         
			pvfmm::Profile::Toc();
			pvfmm::Profile::Tic("Transpose", comm, true, 5);
			pvfmm::Matrix<double> Mo_t(numPoints,d*d*dof,&resCPU[0],false);
			for(size_t i=0;i<Mo.Dim(0);i++)
			for(size_t j=0;j<Mo.Dim(1);j++){
			  Mo_t[j][i]=Mo[i][j];
			}
			pvfmm::Profile::Toc();
		  }
    pvfmm::Profile::Toc();
    
    // check for Correctness
    bool passed = true;
	double tolerance = 1e-6f;
	
    for (int i = 0; i < sizeRes; i++)
    {
		if (std::abs(resCPU[i] - resGPU[i]) > tolerance) {	
			std::cout << "resCPU: " << resCPU[i] << "resGPU: " << resGPU[i] << std::endl;	
			passed = false;
			std::cout << "Fail at i: " << i << std::endl;
			break;
		}
	}     
	if (passed == true)
	std::cout << "Multiplication passed tolerance" << std::endl;
	
	passed = true;
	for (int i = 0; i < sizeRes; i++)
    {
		if (std::abs(resCPU[i] - outtest[i]) > tolerance) {	
			std::cout << "resCPU: " << resCPU[i] << "cuSPARSE: " << outtest[i] << std::endl;	
			passed = false;
			std::cout << "Fail at i: " << i << std::endl;
			break;
		}
	}     
	if (passed == true)
	std::cout << "cuSPARSE passed tolerance" << std::endl;
	
	// cleanup
	cublasDestroy(handle);
	checkCudaErrors(cudaFree(d_poly));
	checkCudaErrors(cudaFree(d_polytrans));
    checkCudaErrors(cudaFree(d_coeff));
    checkCudaErrors(cudaFree(d_res));
    checkCudaErrors(cudaFree(d_valA));
    checkCudaErrors(cudaFree(d_csrColPtrA));
    checkCudaErrors(cudaFree(d_csrRowIndA));
    checkCudaErrors(cudaFree(d_nnz));
    cudaDeviceReset();
}


// complete Evaluation on GPU using cuBLAS
// used for comparing to CPU matrix mul, but not important
template <class Real_t>
void chebEval(Tree_t* tree, Real_t* in, pvfmm::Vector<Real_t>& out, int numPoints)
{
	tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
	MPI_Comm* comm = &sim_config->comm;
	
	// get tree nodes and cheb coeffs
	NodeType* n_next = tree->PostorderFirst();
  	int num_leaves = tbslas::CountNumLeafNodes(*tree);
  	int cheb_deg;
  	size_t dof;
	std::vector<pvfmm::Vector<double> > chebdata;

	chebdata.reserve(num_leaves);
  	while (n_next != NULL) {
  	  if(!n_next->IsGhost() && n_next->IsLeaf())
	    {
	    chebdata.push_back(n_next->ChebData());
	    dof = n_next->DataDOF();
	    cheb_deg = n_next->ChebDeg();
        }
  	  n_next = tree->PostorderNxt(n_next);
  	} // NOW: vector with Chebdata Vectors
	
	
	// Get CUDA device
	int devID;
   	cudaDeviceProp props;

   	// This will pick the best possible CUDA capable device
   	devID = findCudaDevice(0, (const char **)"");

   	//Get GPU information
   	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		devID, props.name, props.major, props.minor);
	
	
	// set additional values
	size_t d = (size_t) cheb_deg+1;
	size_t n_coeff=(d*(d+1)*(d+2))/6;
	
	// buffer for coeffs	
	Real_t* coeff_buff = new Real_t[d * d * d * dof];
	// split in to x,y,z arrays
	Real_t* x_in = new Real_t[numPoints];
	Real_t* y_in = new Real_t[numPoints];
	Real_t* z_in = new Real_t[numPoints];
	Real_t* x_out = new Real_t[numPoints*(cheb_deg+1)];
	Real_t* y_out = new Real_t[numPoints*(cheb_deg+1)];
	Real_t* z_out = new Real_t[numPoints*(cheb_deg+1)];
	
	for (int i = 0; i < numPoints; i++)
	{
		x_in[i] = in[i*COORD_DIM+0];
		y_in[i] = in[i*COORD_DIM+1];
		z_in[i] = in[i*COORD_DIM+2];
	}
	
	// out vector for CPU - nodes * numPoints * numPoints * numPoints * dof;
	out.Resize(num_leaves * numPoints * numPoints * numPoints * dof);
	int pos = numPoints * numPoints * numPoints * dof;
	
	pvfmm::Profile::Tic("Chebyshev Evaluation on GPU", comm, true, 5);
	pvfmm::Profile::Tic("Chebyshev Polynomials", comm, true, 5);
	// Calculate polynomials and copy to GPU
	pvfmm::Profile::Tic("Allocate buffers", comm, true, 5);
    // allocate GPU memory
	Real_t* d_x_in = NULL;
	Real_t* d_y_in = NULL;
	Real_t* d_z_in = NULL;
	Real_t* d_x_poly = NULL;
	Real_t* d_y_poly = NULL;
	Real_t* d_z_poly = NULL;
	
	unsigned int mem_size_in = sizeof(Real_t) * numPoints;
	unsigned int mem_size_out = sizeof(Real_t) * numPoints * d;
	
	checkCudaErrors(cudaMalloc((void**) &d_x_in, mem_size_in)); 
	checkCudaErrors(cudaMalloc((void**) &d_y_in, mem_size_in)); 
	checkCudaErrors(cudaMalloc((void**) &d_z_in, mem_size_in)); 
	checkCudaErrors(cudaMalloc((void**) &d_x_poly, mem_size_out)); 
	checkCudaErrors(cudaMalloc((void**) &d_y_poly, mem_size_out));
	checkCudaErrors(cudaMalloc((void**) &d_z_poly, mem_size_out));	
	pvfmm::Profile::Toc();
	// needed block count
	// n threads in total per direction, for now just blocks with max thread number
	int threadsPerBlock = 1024;
	int blockCount = numPoints/threadsPerBlock + (numPoints%threadsPerBlock == 0 ? 0:1);
	pvfmm::Profile::Tic("Copy to GPU", comm, true, 5);
	// Copy data to GPU
	checkCudaErrors(cudaMemcpy(d_x_in, x_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y_in, y_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_z_in, z_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    pvfmm::Profile::Toc();                           
    pvfmm::Profile::Tic("Calculate polynomials", comm, true, 5);                                                          
    // Calculate chebPoly for x,y,z
    chebPoly_helper(d_x_in, d_x_poly, cheb_deg, numPoints, blockCount, threadsPerBlock);
    chebPoly_helper(d_y_in, d_y_poly, cheb_deg, numPoints, blockCount, threadsPerBlock);
    chebPoly_helper(d_z_in, d_z_poly, cheb_deg, numPoints, blockCount, threadsPerBlock);
    pvfmm::Profile::Toc();
    // free in_coords (not needed anymore)
    checkCudaErrors(cudaFree(d_x_in));
    checkCudaErrors(cudaFree(d_y_in));
    checkCudaErrors(cudaFree(d_z_in));
	pvfmm::Profile::Toc();
	pvfmm::Profile::Tic("Allocate Buffers", comm, true, 5);
	// allocate memory for computations
	//Real_t* d_coeffs = NULL;
	Real_t* d_buffer1 = NULL;
	Real_t* d_buffer2 = NULL;
	
	unsigned int mem_size_coeffs = sizeof(Real_t) * d * d * d * dof;
	unsigned int mem_size_buffers = sizeof(Real_t) * std::max(d,(size_t) numPoints)*std::max(d, (size_t) numPoints)*std::max(d,(size_t) numPoints)*dof;
	
	
	checkCudaErrors(cudaMalloc((void**) &d_buffer1, mem_size_buffers)); 
	checkCudaErrors(cudaMalloc((void**) &d_buffer2, mem_size_buffers));
	pvfmm::Profile::Toc();
	
	// Init cuBLAS
	cublasHandle_t handle;
    cublasCreate(&handle); 
                         
	int M;
	int N;
	int K;
	double alpha = 1.0;
	double beta = 0.0;
	int lda;
	int ldb;
	int ldc;
	
	cublasStatus_t status;
	
	
	// do evaluation for every leaf (not correct, but good for timing comparisons - for correct evaluation use NodeFieldFunctor)
	for (int countLeaves = 0; countLeaves < num_leaves; countLeaves++)
	{
		pvfmm::Profile::Tic("Evaluate on GPU", comm, true, 5);
		
		pvfmm::Profile::Tic("Coefficients", comm, true, 5);
		pvfmm::Profile::Tic("Rearrange coefficients", comm, true, 5);
		pvfmm::Vector<Real_t>& coeff_ = chebdata[countLeaves];
	    // FOR NOW ON CPU
		// Rearrange coefficients into a tensor.
		pvfmm::Vector<Real_t> coeff(d*d*d*dof,&coeff_buff[0],false);
		coeff.SetZero();
		size_t indx=0;
		for(size_t l=0;l<dof;l++){
		  for(size_t i=0;i<d;i++){
			for(size_t j=0;i+j<d;j++){
			  Real_t* coeff_ptr=&coeff[(j+(i+l*d)*d)*d];
			  for(size_t k=0;i+j+k<d;k++){
				coeff_ptr[k]=coeff_[indx];
				indx++;
			   }
			 }
		   }
		 }
		 pvfmm::Profile::Toc();
		 pvfmm::Profile::Tic("Copy coefficients to GPU", comm, true, 5);
		 // copy coefficients to GPU
		checkCudaErrors(cudaMemcpy(d_buffer1, &coeff[0], mem_size_coeffs,
                               cudaMemcpyHostToDevice));
		pvfmm::Profile::Toc();
		pvfmm::Profile::Toc();
		checkCudaErrors(cudaDeviceSynchronize());
		pvfmm::Profile::Tic("First multipication", comm, true, 5);
		// first Matrix mul - coeffs * x_poly
		M = d*d*dof;
		N = numPoints; 
		K = d;
		lda = K;                           
		ldb = N;
		ldc = M;
	
		status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_buffer1, lda, d_x_poly, ldb, &beta, d_buffer2, ldc);
		checkCudaErrors(cudaDeviceSynchronize());
		pvfmm::Profile::Toc();
		
		pvfmm::Profile::Tic("Second multipication", comm, true, 5);
		// second Matrix mul - mul1 * y_poly
		M = numPoints*d*dof;
		N = numPoints; 
		K = d;
		lda = K;                           
		ldb = N;
		ldc = M;
		
		status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_buffer2, lda, d_y_poly, ldb, &beta, d_buffer1, ldc);
		checkCudaErrors(cudaDeviceSynchronize());
		pvfmm::Profile::Toc();
		
		pvfmm::Profile::Tic("Third multipication", comm, true, 5);
		// third Matrix mul - mul2 * z_poly
		M = numPoints*numPoints*dof;
		N = numPoints; 
		K = d;
		lda = K;                           
		ldb = N;
		ldc = M;
		
		status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_buffer1, lda, d_z_poly, ldb, &beta, d_buffer2, ldc);
		checkCudaErrors(cudaDeviceSynchronize());
		pvfmm::Profile::Toc();
		
		{
		pvfmm::Profile::Tic("Copy to out", comm, true, 5);
		
		pvfmm::Profile::Tic("Copy from GPU", comm, true, 5);
		std::vector<Real_t> tmp (M*N);
	    int mem_size_out1 = sizeof(Real_t) * M * N;
		checkCudaErrors(cudaMemcpy(&tmp[0], d_buffer2, mem_size_out1,
                               cudaMemcpyDeviceToHost));
        pvfmm::Profile::Toc(); 
       
		
		pvfmm::Profile::Tic("Final transpose and copy", comm, true, 5);
		{ // Copy to out
			pvfmm::Matrix<Real_t> Mo  ( numPoints*numPoints*numPoints,dof,&tmp[0],false);
			pvfmm::Matrix<Real_t> Mo_t(dof,numPoints*numPoints*numPoints,&out[countLeaves*pos],false);
			for(size_t i=0;i<Mo.Dim(0);i++)
			for(size_t j=0;j<Mo.Dim(1);j++){
			Mo_t[j][i]=Mo[i][j];  
			}
		 } 
		 pvfmm::Profile::Toc();
		 pvfmm::Profile::Toc();
		
		}
		pvfmm::Profile::Toc();
	}
	pvfmm::Profile::Toc();
	
	// clean up
	delete [] x_in;
	delete [] y_in;
	delete [] z_in;
	delete [] x_out;
	delete [] y_out;
	delete [] z_out;
	checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));
    cublasDestroy(handle);
    cudaDeviceReset();
	
}

// Eval on CPU using Matrix multiplications
// used for comparison to GPU, but not important
template <class Real_t>
void chebEvalCPU(int numPoints, Tree_t* tree, Real_t* in, pvfmm::Vector<Real_t>& out) {
	tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
	MPI_Comm* comm = &sim_config->comm;
	
	// split in to x,y,z arrays
	Real_t* x_in = new Real_t[numPoints];
	Real_t* y_in = new Real_t[numPoints];
	Real_t* z_in = new Real_t[numPoints];
	
	for (int i = 0; i < numPoints; i++)
	{
		x_in[i] = in[i*COORD_DIM+0];
		y_in[i] = in[i*COORD_DIM+1];
		z_in[i] = in[i*COORD_DIM+2];
	}
	
	// get tree nodes and cheb coeffs
	NodeType* n_next = tree->PostorderFirst();
  	int num_leaves = tbslas::CountNumLeafNodes(*tree);
  	size_t dof;
  	int cheb_deg;
	std::vector<pvfmm::Vector<double> > chebdata;

	chebdata.reserve(num_leaves);
  	while (n_next != NULL) {
  	  if(!n_next->IsGhost() && n_next->IsLeaf())
	    {
	    chebdata.push_back(n_next->ChebData());
	    dof = n_next->DataDOF();
	    cheb_deg = n_next->ChebDeg();
        }
  	  n_next = tree->PostorderNxt(n_next);
  	} // NOW: vector with Chebdata Vectors
  	
  	size_t d=(size_t)cheb_deg+1;
  	
  	//////////////////////////////////////////////////////
  	// quik print values
  	{
  	  std::cout << "num_leaves: " << num_leaves << std::endl;
  	  std::cout << "dof: " << dof << std::endl;
  	  std::cout << "cheb_deg: " << cheb_deg << std::endl;
  	  std::cout << "chebdata size: " << chebdata[0].Dim() << std::endl;
  	
  	
  	
	}
  	////////////////////////////////////////////////////////
  	pvfmm::Profile::Tic("Chebyshev Evaluation on CPU", comm, true, 5);
  	
  	std::vector<Real_t> coord;
    pvfmm::Vector<Real_t> tmp_out;    // buffer used in chebyshev evaluation
	
    std::vector<Real_t> p1(numPoints*d);
    std::vector<Real_t> p2(numPoints*d);
    std::vector<Real_t> p3(numPoints*d);
    pvfmm::Profile::Tic("Chebyshev Polynomials", comm, true, 5);
    pvfmm::cheb_poly(cheb_deg,&x_in[0],numPoints,&p1[0]); // cheb polynomials at x,y,z
    pvfmm::cheb_poly(cheb_deg,&y_in[0],numPoints,&p2[0]);
    pvfmm::cheb_poly(cheb_deg,&z_in[0],numPoints,&p3[0]);
    pvfmm::Profile::Toc();
    pvfmm::Matrix<Real_t> Mp1(d,numPoints,&p1[0],false);
    pvfmm::Matrix<Real_t> Mp2(d,numPoints,&p2[0],false);
    pvfmm::Matrix<Real_t> Mp3(d,numPoints,&p3[0],false);
	
	pvfmm::Vector<Real_t> v1, v2;	
	out.Resize(num_leaves * numPoints * numPoints * numPoints * dof);
	size_t pos = numPoints * numPoints * numPoints * dof;
	
	{ 
	// Create work buffers
	size_t buff_size=std::max(cheb_deg+1,numPoints)*std::max(cheb_deg+1,numPoints)*std::max(cheb_deg+1,numPoints)*dof;
	v1.Resize(buff_size);
	v2.Resize(buff_size);

	}
	pvfmm::Profile::Tic("Computations", comm, true, 5);	
	// do evaluation for every leaf (not correct, but good for timing comparisons - for correct evaluation use NodeFieldFunctor)		
	for (int countLeaves = 0; countLeaves < num_leaves; countLeaves++)
	{
		{
		pvfmm::Vector<Real_t>& coeff_ = chebdata[countLeaves];
		
		size_t n_coeff=(d*(d+1)*(d+2))/6;
		size_t dof=coeff_.Dim()/n_coeff;
		assert(coeff_.Dim()==dof*n_coeff);
        pvfmm::Profile::Tic("Rearrange coefficients", comm, true, 5);
		{ // Rearrange coefficients into a tensor.
              pvfmm::Vector<Real_t> coeff(d*d*d*dof,&v1[0],false);
              coeff.SetZero();
              size_t indx=0;
              for(size_t l=0;l<dof;l++){
                for(size_t i=0;i<d;i++){
                  for(size_t j=0;j<d-i;j++){
                    Real_t* coeff_ptr=&coeff[(j+(i+l*d)*d)*d];
                    for(size_t k=0;k<d-i-j;k++){
                      coeff_ptr[k]=coeff_[indx];
                      indx++;
                    }
                  }
                }
              }
            }
          pvfmm::Profile::Toc();
		}
		
          pvfmm::Profile::Tic("Apply Mp1", comm, true, 5);             
		  { // Apply Mp1
		   pvfmm::Matrix<Real_t> Mi  ( d* d*dof, d,&v1[0],false);
		   pvfmm::Matrix<Real_t> Mo  ( d* d*dof,numPoints,&v2[0],false);
		   pvfmm::Matrix<Real_t>::GEMM(Mo, Mi, Mp1);

		   pvfmm::Matrix<Real_t> Mo_t(numPoints, d* d*dof,&v1[0],false);
		   for(size_t i=0;i<Mo.Dim(0);i++)
			  for(size_t j=0;j<Mo.Dim(1);j++){
				 Mo_t[j][i]=Mo[i][j];
			  }
		  }
		  pvfmm::Profile::Toc();
				  		 
		  pvfmm::Profile::Tic("Apply Mp2", comm, true, 5);
		  { // Apply Mp2   
			pvfmm::Matrix<Real_t> Mi  (numPoints* d*dof, d,&v1[0],false);
			pvfmm::Matrix<Real_t> Mo  (numPoints* d*dof,numPoints,&v2[0],false);
			pvfmm::Matrix<Real_t>::GEMM(Mo, Mi, Mp2);

			pvfmm::Matrix<Real_t> Mo_t(numPoints,numPoints* d*dof,&v1[0],false);
			for(size_t i=0;i<Mo.Dim(0);i++)
			for(size_t j=0;j<Mo.Dim(1);j++){
			  Mo_t[j][i]=Mo[i][j];
			}
		  }
		  pvfmm::Profile::Toc();
		
		  pvfmm::Profile::Tic("Apply Mp3", comm, true, 5);
		  { // Apply Mp3   
			pvfmm::Matrix<Real_t> Mi  (numPoints*numPoints*dof, d,&v1[0],false);
			pvfmm::Matrix<Real_t> Mo  (numPoints*numPoints*dof,numPoints,&v2[0],false);
			pvfmm::Matrix<Real_t>::GEMM(Mo, Mi, Mp3);         

			pvfmm::Matrix<Real_t> Mo_t(numPoints,numPoints*numPoints*dof,&v1[0],false);
			for(size_t i=0;i<Mo.Dim(0);i++)
			for(size_t j=0;j<Mo.Dim(1);j++){
			  Mo_t[j][i]=Mo[i][j];
			}
		  }
		pvfmm::Profile::Toc();  
				 
		 pvfmm::Profile::Tic("Copy to out", comm, true, 5);
		 { // Copy to out
			pvfmm::Matrix<Real_t> Mo  ( numPoints*numPoints*numPoints,dof,&v1[0],false);
			pvfmm::Matrix<Real_t> Mo_t(dof,numPoints*numPoints*numPoints,&out[countLeaves*pos],false);
			for(size_t i=0;i<Mo.Dim(0);i++)
			for(size_t j=0;j<Mo.Dim(1);j++){
			Mo_t[j][i]=Mo[i][j];  
			}
		 }            
		 pvfmm::Profile::Toc();
		
	}
	pvfmm::Profile::Toc();
	pvfmm::Profile::Toc();
}

// compare Eval on GPU and CPU, matrix multiplication
void compareEvaluations(Tree_t* tvel, double* coords, int num_points)
{
	pvfmm::Vector<double> cpu_out;
	pvfmm::Vector<double> gpu_out;
	
    chebEvalCPU(num_points, tvel, coords, cpu_out);
    std::cout << "cpu_out size: " << cpu_out.Dim() << std::endl;
    
    chebEval(tvel, coords, gpu_out, num_points);
	std::cout << "gpu_out size: " << cpu_out.Dim() << std::endl;
	
	// check for correctness
	bool passed = true;
	double tolerance = 1e-6f;
	size_t size_out = cpu_out.Dim();
	
	if (cpu_out.Dim() != cpu_out.Dim())
	{
		passed = false;
		std::cout << "Out Vectors have different dimensions" << std::endl;
	}
	
    for (size_t i = 0; i < size_out; i++)
    {
		if (std::abs(cpu_out[i] - gpu_out[i]) > tolerance) {	
			std::cout << "cpu_out: " << cpu_out[i] << "gpu_out: " << gpu_out[i] << std::endl;	
			passed = false;
			std::cout << "Fail at i: " << i << std::endl;
			break;
		}
	}     
	if (passed == true)
	std::cout << "Evaluation passed tolerance" << std::endl;
	
}

// compare CPU NodeFieldFunctor and GPU NodeFieldFunctor (using matrix multiplication)
void compareNodeFieldFunctors(Tree_t* tvel, double* point_pos, int num_points, int dof)
{
	double* cpu_out = new double[num_points*dof];
	double* gpu_out = new double[num_points*dof];
	
	// Evaluation on CPU
	{
	tbslas::NodeFieldFunctor<double,Tree_t> field_fn = tbslas::NodeFieldFunctor<double,Tree_t>(tvel);
	
	field_fn(point_pos, num_points, cpu_out);
	}
	
	//Evaluation on GPU
	{
	tbslas::NodeFieldFunctor_cuda<double,Tree_t> field_fn_cuda = tbslas::NodeFieldFunctor_cuda<double,Tree_t>(tvel);
	
	field_fn_cuda(point_pos, num_points, gpu_out);	
	}
			
	// check for correctness
	bool passed = true;
	double tolerance = 1e-6f;
	
    for (int i = 0; i < num_points*dof; i++)
    {
		if (std::abs(cpu_out[i] - gpu_out[i]) > tolerance) {	
			std::cout << "cpu_out: " << cpu_out[i] << "gpu_out: " << gpu_out[i] << std::endl;	
			passed = false;
			std::cout << "Fail at i: " << i << std::endl;
			break;
		}
	}     
	if (passed == true)
	std::cout << "Evaluation passed tolerance" << std::endl;
	
	
}

// compare CPU NodeFieldFunctor and GPU NodeFieldFunctor (using vec_eval)
template <typename Tree, class Real_t>
void compareNodeFieldFunctors_vec_eval(Tree* tvel, Real_t* point_pos, int num_points, int dof)
{
	Real_t* cpu_out = new Real_t[num_points*dof];
	
	
	// Evaluation on CPU
	{
	tbslas::NodeFieldFunctor<Real_t,Tree> field_fn = tbslas::NodeFieldFunctor<Real_t,Tree>(tvel);
	
	field_fn(point_pos, num_points, cpu_out);
	
	}
	
	Real_t* gpu_out = new Real_t[num_points*dof];
	//Evaluation on GPU
	{
	tbslas::NodeFieldFunctor_cuda_vec<Real_t,Tree> field_fn_cuda = tbslas::NodeFieldFunctor_cuda_vec<Real_t,Tree>(tvel);
	
	field_fn_cuda(point_pos, num_points, gpu_out);	
	
	}
			
	// check for correctness
	bool passed = true;
	double tolerance = 1e-6f;
	
    for (int i = 0; i < num_points*dof; i++)
    {
		//std::cout << "cpu_out: " << cpu_out[i] << "gpu_out: " << gpu_out[i] << std::endl;
		if (std::abs(cpu_out[i] - gpu_out[i]) > tolerance) {	
			std::cout << "cpu_out: " << cpu_out[i] << "gpu_out: " << gpu_out[i] << std::endl;	
			passed = false;
			std::cout << "Fail at i: " << i << std::endl;
			break;
		}
	}     
	if (passed == true)
	std::cout << "Evaluation passed tolerance" << std::endl;
}

// wraper for gaussian kernel
template <class Real_t>
void get_gaussian_kernel_wraper(const Real_t* coord,
                     int n,
                     Real_t* out) {
  const Real_t xc  = 0.7;
  const Real_t yc  = 0.7;
  const Real_t zc  = 0.7;
  tbslas::gaussian_kernel(coord, n, out, xc, yc, zc);

}

int main (int argc, char **argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm comm=MPI_COMM_WORLD;
	int np;
	MPI_Comm_size(comm, &np);
	int myrank;
	MPI_Comm_rank(comm, &myrank);
	
	parse_command_line_options(argc, argv);
	
	
	
	int   test = strtoul(commandline_option(argc, argv, "-test",     "1", false,
                                          "-test <int> = (1)      : 1 - NFF vel (matrix mul), 2 - NFF gauss (matrix mul), 3 - compare matrix mul evaluation, \
                                           4 - matrix mul comparisons (CPU, cuBLAS, cuSPARSE), 5 - chebPoly comparison, 6 - NFF vorticity field (vector evaluation), \
                                           7 - NFF Gaussian function (vector evaluation), 8 - NFF Hopf field (vector evaluation)"),NULL,10);
	int   numEvPoints = strtoul(commandline_option(argc, argv, "-numEvP",     "1", false,
                                          "-numEvP <int> = (1)    : how many points for evaluation"),NULL,10);
    int   degree = strtoul(commandline_option(argc, argv, "-degree",     "14", false,
                                          "-degree <int> = (14)    : chebyshev degree for chebPoly tests"),NULL,10);

	//printf("xc: %f, yc: %f, zc: %f, c: %f, amp: %f \n",xc_in,yc_in,zc_in,c_in,amp_in);

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
    
    fname = (char*) "velocityTree";
	fn_vel = tbslas::get_vorticity_field<double,3>;
	fn_vel_f = tbslas::get_vorticity_field<float,3>;
	fn_val = get_gaussian_kernel_wraper<double>;
    
	fn_vel2 = get_hopf_field_wrapper<double>;
	// =========================================================================
    // CONSTRUCT TREE
    // =========================================================================

    // Create velocity tree
    int max_depth_vel=0;
    Tree_t tvel(comm);
    if (test == 6)
    {
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
	}
	else
	{
	tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  max_depth_vel?max_depth_vel:sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel2,
                                  3,
                                  tvel);	
		
	}
	
	
	// gaussian 	
	Tree_t tgauss(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_val,
                                  1,
                                  tgauss);
	
						
	// create random points and evaluate the tree values with NodeFieldFunctor
	int num_points = numEvPoints;
    	std::vector<double> xtmp(num_points*3);
	for (int i = 0; i < xtmp.size(); i++) 
	{
	  xtmp[i] = ((double) (rand()%10000)/10000.0); // random points between 0 and 1
	  
	}   	
	
	
	if (sim_config->vtk_save_rate) {
	
      //tree.Write2File(tbslas::GetVTKFileName(0, sim_config->vtk_filename_variable).c_str(),
       //               sim_config->vtk_order);

	
	tvel.Write2File(fname,
                      sim_config->vtk_order);
    }
    
    
    FILE* file;
    
    switch(test) {
		case(1): // compare NFF with velocity tree			
			compareNodeFieldFunctors(&tvel, &xtmp[0], num_points, 3);	
			file = freopen("output_test1.txt","w",stdout);		
			break;
		case(2): // compare NFF with gaussian tree	
			compareNodeFieldFunctors(&tgauss, &xtmp[0], num_points, 1);	
			file = freopen("output_test2.txt","w",stdout);		
			break;
		case(3): // compare evaluations with matrix multiplication
			compareEvaluations(&tvel, &xtmp[0], num_points);
			file = freopen("output_test3.txt","w",stdout);
			break;
		case(4): // compare single matrix multiplication
			matrixMul_Manypoints_mul1(num_points, degree);
			file = freopen("output_test4.txt","w",stdout);
			break;
		case(5): // compare chebyshev polynomial computation
			chebPoly(degree, &xtmp[0], num_points);
			file = freopen("output_test5.txt","w",stdout);
			break;
		case(6): // compare NFF with velocity tree (vec_eval)
			compareNodeFieldFunctors_vec_eval(&tvel, &xtmp[0], num_points, 3);	
			file = freopen("output_test6.txt","w",stdout);		
			break;
		case(7): // compare NFF with gaussian tree (vec_eval)
			compareNodeFieldFunctors_vec_eval(&tgauss, &xtmp[0], num_points, 1);	
			file = freopen("output_test7.txt","w",stdout);		
			break;
		case(8): // compare NFF with hopf field tree (vec_eval)
			compareNodeFieldFunctors_vec_eval(&tvel, &xtmp[0], num_points, 3);	
			file = freopen("output_test8.txt","w",stdout);		
			break;	
		default:			
			break;
	}
    
 
    
	// =========================================================================
    // COMPUTE ERROR
    // =========================================================================

    // check velocity tree
    double al2,rl2,ali,rli;
    CheckChebOutput<Tree_t>(&tvel,
                            fn_vel,
                            3,
                            al2,rl2,ali,rli,
                            std::string("Input-Vel"));
                            								  
	int num_leaves = tbslas::CountNumLeafNodes(tvel);
       
	// =========================================================================
    // REPORT RESULTS
    // =========================================================================
    
	int tree_max_depth = 0;
	tbslas::GetTreeMaxDepth(tvel, tree_max_depth);
	
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
		Rep::AddData("NUMPOINTS", numEvPoints, tbslas::REP_INT);
		Rep::AddData("DEGREE", degree, tbslas::REP_INT);

		Rep::AddData("OutAL2", al2);
		Rep::AddData("OutALINF", ali);

		Rep::Report();
  }
  
  
  
  //Output Profiling results.
   pvfmm::Profile::print(&comm);
   fclose (stdout);	
  

  // Shut down MPI
  MPI_Finalize();
  return 0;
}
