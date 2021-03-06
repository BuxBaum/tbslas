// *************************************************************************
// Copyright (C) 2014 by Arash Bakhtiari
// You may not use this file except in compliance with the License.
// You obtain a copy of the License in the LICENSE file.

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// *************************************************************************


// edited 2016 by Benedikt Kucis for CUDA support

#ifndef SRC_TREE_NODE_FIELD_FUNCTOR_H_CUDA_
#define SRC_TREE_NODE_FIELD_FUNCTOR_H_CUDA_

#include <vector>
#include <cmath>
#include <pvfmm_common.hpp>
#include <cheb_node.hpp>
#include <profile.hpp>

#include <ompUtils.h>

#include "utils/common.h"
#include "chebEval.h"

namespace pvfmm {
template <class Real, class Vec = Real>
void vec_eval_cuda(int n, int d0, int d, int dof, Real *px, Real *py, Real *pz,
              Real *coeff, Real *tmp_out) {
  const int VecLen = sizeof(Vec) / sizeof(Real);

  for (int l0 = 0; l0 < dof; l0++) {
    for (int l1 = 0; l1 < n; l1 += 4*VecLen) {
      Vec u0 = zero_intrin<Vec>();
      Vec u1 = zero_intrin<Vec>();
      Vec u2 = zero_intrin<Vec>();
      Vec u3 = zero_intrin<Vec>();
      for (int i = 0; i < d; i++) {
        Vec v0 = zero_intrin<Vec>();
        Vec v1 = zero_intrin<Vec>();
        Vec v2 = zero_intrin<Vec>();
        Vec v3 = zero_intrin<Vec>();
        Vec pz0_ = load_intrin<Vec>(&pz[i * n + l1+0*VecLen]);
        Vec pz1_ = load_intrin<Vec>(&pz[i * n + l1+1*VecLen]);
        Vec pz2_ = load_intrin<Vec>(&pz[i * n + l1+2*VecLen]);
        Vec pz3_ = load_intrin<Vec>(&pz[i * n + l1+3*VecLen]);
        for (int j = 0; i + j < d; j++) {
          Vec w0 = zero_intrin<Vec>();
          Vec w1 = zero_intrin<Vec>();
          Vec w2 = zero_intrin<Vec>();
          Vec w3 = zero_intrin<Vec>();
          Vec py0_ = load_intrin<Vec>(&py[j * n + l1+0*VecLen]);
          Vec py1_ = load_intrin<Vec>(&py[j * n + l1+1*VecLen]);
          Vec py2_ = load_intrin<Vec>(&py[j * n + l1+2*VecLen]);
          Vec py3_ = load_intrin<Vec>(&py[j * n + l1+3*VecLen]);
          for (int k = 0; i + j + k < d; k++) {
            Vec px0_ = load_intrin<Vec>(&px[k * n + l1+0*VecLen]);
            Vec px1_ = load_intrin<Vec>(&px[k * n + l1+1*VecLen]);
            Vec px2_ = load_intrin<Vec>(&px[k * n + l1+2*VecLen]);
            Vec px3_ = load_intrin<Vec>(&px[k * n + l1+3*VecLen]);
            Vec c = set_intrin<Vec, Real>(coeff[k + d0 * (j + d0 * (i + d0 * l0))]);
            w0 = add_intrin(w0, mul_intrin(px0_, c));
            w1 = add_intrin(w1, mul_intrin(px1_, c));
            w2 = add_intrin(w2, mul_intrin(px2_, c));
            w3 = add_intrin(w3, mul_intrin(px3_, c));
          }
          v0 = add_intrin(v0, mul_intrin(py0_, w0));
          v1 = add_intrin(v1, mul_intrin(py1_, w1));
          v2 = add_intrin(v2, mul_intrin(py2_, w2));
          v3 = add_intrin(v3, mul_intrin(py3_, w3));
        }
        u0 = add_intrin(u0, mul_intrin(pz0_, v0));
        u1 = add_intrin(u1, mul_intrin(pz1_, v1));
        u2 = add_intrin(u2, mul_intrin(pz2_, v2));
        u3 = add_intrin(u3, mul_intrin(pz3_, v3));
      }
      store_intrin(&tmp_out[l0 * n + l1+0*VecLen], u0);
      store_intrin(&tmp_out[l0 * n + l1+1*VecLen], u1);
      store_intrin(&tmp_out[l0 * n + l1+2*VecLen], u2);
      store_intrin(&tmp_out[l0 * n + l1+3*VecLen], u3);
    }
  }
}
}

namespace tbslas {

// evaluation on GPU using matrix multiplication	
template <class Real_t>
void chebEval_cuda(Real_t* coeff, Real_t* x_in, Real_t* y_in, Real_t* z_in, pvfmm::Vector<Real_t>& out, int numPoints, int cheb_deg, int dof, 
			Real_t* d_x_in, Real_t* d_y_in, Real_t* d_z_in, Real_t* d_poly, Real_t* d_buffer1, Real_t* d_buffer2, cublasHandle_t handle)
{
	tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
	MPI_Comm* comm = &sim_config->comm;
	  	
   	// set additional values
	size_t d = (size_t) cheb_deg+1;
	
	out.Resize(numPoints * numPoints * numPoints * dof);
	                       
	int M;
	int N;
	int K;
	double alpha = 1.0;
	double beta = 0.0;
	int lda;
	int ldb;
	int ldc;
	
	cublasStatus_t status;
	
	int threadsPerBlock = 1024;
	int blockCount = numPoints/threadsPerBlock + (numPoints%threadsPerBlock == 0 ? 0:1);
	
	unsigned int mem_size_in = sizeof(Real_t) * numPoints;
	unsigned int mem_size_poly = sizeof(Real_t) * numPoints * d;
	unsigned int mem_size_buffers = sizeof(Real_t) * std::max(d,(size_t) numPoints)*std::max(d, (size_t) numPoints)*std::max(d,(size_t) numPoints)*dof;
	unsigned int mem_size_coeffs = sizeof(Real_t) * d * d * d * dof;
	
	pvfmm::Profile::Tic("Copy data to GPU", comm, false, 5);
	// copy data to GPU
	checkCudaErrors(cudaMemcpy(d_x_in, x_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y_in, y_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_z_in, z_in, mem_size_in,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_buffer1, coeff, mem_size_coeffs,
                               cudaMemcpyHostToDevice));
    pvfmm::Profile::Toc();
    pvfmm::Profile::Tic("Compute x poly", comm, false, 5);	
	// x polynomials
	chebPoly_helper(d_x_in, d_poly, cheb_deg, numPoints, blockCount, threadsPerBlock);   
	pvfmm::Profile::Toc();
	// first multiplication
	M = d*d*dof;
	N = numPoints; 
	K = d;
	lda = K;                           
	ldb = N;
	ldc = M;
	
	pvfmm::Profile::Tic("Mul 1", comm, false, 5);	
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_buffer1, lda, d_poly, ldb, &beta, d_buffer2, ldc);
	checkCudaErrors(cudaDeviceSynchronize());	  
	pvfmm::Profile::Toc();
	pvfmm::Profile::Tic("Compute y poly", comm, false, 5);	
	// y polynomials
	chebPoly_helper(d_y_in, d_poly, cheb_deg, numPoints, blockCount, threadsPerBlock);  
	pvfmm::Profile::Toc();
	// second multiplication
	M = numPoints*d*dof;
	N = numPoints; 
	K = d;
	lda = K;                           
	ldb = N;
	ldc = M;
	pvfmm::Profile::Tic("Mul 2", comm, false, 5);	
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_buffer2, lda, d_poly, ldb, &beta, d_buffer1, ldc);
	checkCudaErrors(cudaDeviceSynchronize());    
	pvfmm::Profile::Toc();
	pvfmm::Profile::Tic("Compute z poly", comm, false, 5);	
	// z polynomials
	chebPoly_helper(d_z_in, d_poly, cheb_deg, numPoints, blockCount, threadsPerBlock);  
	pvfmm::Profile::Toc();
	// third multiplication 
	M = numPoints*numPoints*dof;
	N = numPoints; 
	K = d;
	lda = K;                           
	ldb = N;
	ldc = M;
	
	pvfmm::Profile::Tic("Mul 3", comm, false, 5);	
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_buffer1, lda, d_poly, ldb, &beta, d_buffer2, ldc);
	checkCudaErrors(cudaDeviceSynchronize()); 
	pvfmm::Profile::Toc();
	pvfmm::Profile::Tic("Copy from GPU", comm, false, 5);
	// copy back from GPU
	{
	std::vector<Real_t> tmp (M*N);
	int mem_size_out = sizeof(Real_t) * M * N;
	checkCudaErrors(cudaMemcpy(&tmp[0], d_buffer2, mem_size_out,
                               cudaMemcpyDeviceToHost));      
    { // Copy to out
		pvfmm::Matrix<Real_t> Mo  ( numPoints*numPoints*numPoints,dof,&tmp[0],false);
		pvfmm::Matrix<Real_t> Mo_t(dof,numPoints*numPoints*numPoints,&out[0],false);
		for(size_t i=0;i<Mo.Dim(0);i++)
		for(size_t j=0;j<Mo.Dim(1);j++){
			Mo_t[j][i]=Mo[i][j];  
		}
    }             
   }    
   pvfmm::Profile::Toc();
   
                              
}	
	
	
	

template <class Real_t>
void fast_interp_cuda(const std::vector<Real_t>& reg_grid_vals, int data_dof,
                 int N_reg, const std::vector<Real_t>& query_points,
                 std::vector<Real_t>& query_values){

  Real_t lagr_denom[4];
  for(int i=0;i<4;i++){
    lagr_denom[i]=1;
    for(int j=0;j<4;j++){
      if(i!=j) lagr_denom[i]/=(Real_t)(i-j);
    }
  }

  int N_reg3=N_reg*N_reg*N_reg;
  int N_pts=query_points.size()/COORD_DIM;
  query_values.resize(N_pts*data_dof);

  for(int i=0;i<N_pts;i++){
    if( query_points[COORD_DIM*i+0] < 0 || query_points[COORD_DIM*i+0] > 1.0 ||
        query_points[COORD_DIM*i+1] < 0 || query_points[COORD_DIM*i+1] > 1.0 ||
        query_points[COORD_DIM*i+2] < 0 || query_points[COORD_DIM*i+2] > 1.0 ){
      for(int k=0;k<data_dof;k++) query_values[i*data_dof+k]=0;
      continue;
    }

    Real_t point[COORD_DIM];
    int grid_indx[COORD_DIM];
    for(int j=0;j<COORD_DIM;j++){
      point[j]=query_points[COORD_DIM*i+j]*(N_reg-1);
      grid_indx[j]=((int)point[j])-1;
      if(grid_indx[j]<      0) grid_indx[j]=      0;
      if(grid_indx[j]>N_reg-4) grid_indx[j]=N_reg-4;
      point[j]-=grid_indx[j];
    }

    Real_t M[3][4];
    for(int j=0;j<COORD_DIM;j++){
      Real_t x=point[j];
      for(int k=0;k<4;k++){
        M[j][k]=lagr_denom[k];
        for(int l=0;l<4;l++){
          if(k!=l) M[j][k]*=(x-l);
        }
      }
    }

    for(int k=0;k<data_dof;k++){
      Real_t val=0;
      for(int j2=0;j2<4;j2++){
        for(int j1=0;j1<4;j1++){
          Real_t M1M2=M[1][j1]*M[2][j2];
          long indx_j1j2 = N_reg*( (grid_indx[1]+j1) + N_reg*(grid_indx[2]+j2) );
          for(int j0=0;j0<4;j0++){
            long indx = (grid_indx[0]+j0) + indx_j1j2;
            val += M[0][j0]*M1M2 * reg_grid_vals[indx+k*N_reg3];
          }
        }
      }
      query_values[i*data_dof+k]=val;
    }
  }
}

template <class Real_t, class Tree_t>
void EvalNodesLocal_cuda(std::vector<typename Tree_t::Node_t*>& nodes,
                    pvfmm::Vector<Real_t>& trg_coord,
                    pvfmm::Vector<Real_t>& trg_value) { // Read nodes data
  size_t omp_p=omp_get_max_threads();
  tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
  size_t data_dof=nodes[0]->DataDOF();
  static pvfmm::Vector<pvfmm::MortonId> trg_mid;
  trg_mid.Resize(trg_coord.Dim()/COORD_DIM);
#pragma omp parallel for
  for(size_t i=0;i<trg_mid.Dim();i++){
    trg_mid[i] = pvfmm::MortonId(&trg_coord[i*COORD_DIM]);
  }

  std::vector<size_t> part_indx(nodes.size()+1);
  part_indx[nodes.size()] = trg_mid.Dim();
#pragma omp parallel for
  for (size_t j=0;j<nodes.size();j++) {
    part_indx[j]=std::lower_bound(&trg_mid[0],
                                  &trg_mid[0]+trg_mid.Dim(),
                                  nodes[j]->GetMortonId()) - &trg_mid[0];
  }

  bool use_cubic=0;//sim_config->use_cubic;
#pragma omp parallel for
  for (size_t pid=0;pid<omp_p;pid++) {
    size_t a=((pid+0)*nodes.size())/omp_p;
    size_t b=((pid+1)*nodes.size())/omp_p;

    std::vector<Real_t> coord;
    pvfmm::Vector<Real_t> tmp_out;    // buffer used in chebyshev evaluation
    std::vector<Real_t> query_values; // buffer used in cubic interpolation
    std::vector<Real_t> query_points;
    Real_t* output = NULL;

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    // ************************************************************
    // CONSTRUCT REGULAR GRID
    // ************************************************************
    int reg_grid_resolution =
        nodes[0]->ChebDeg()*sim_config->cubic_upsampling_factor;
    Real_t spacing = 1.0/(reg_grid_resolution-1);
    std::vector<Real_t> reg_grid_coord_1d(reg_grid_resolution);
    tbslas::get_reg_grid_points<Real_t, 1>(reg_grid_resolution,
                                           reg_grid_coord_1d.data());

    // ************************************************************
    // EVALUATE AT THE REGULAR GRID
    // ************************************************************
    int reg_grid_num_points = std::pow(static_cast<double>(reg_grid_resolution), COORD_DIM);
    std::vector<Real_t> reg_grid_vals(reg_grid_num_points*data_dof);

    // scale to [-1,1] -> used in cheb_eval
    std::vector<Real_t> x(reg_grid_resolution);
    for(size_t i=0;i<reg_grid_resolution;i++) {
      x[i] = -1.0+2.0*reg_grid_coord_1d[i];
    }

    pvfmm::Matrix<Real_t> Mp1;
    pvfmm::Vector<Real_t> v1, v2;
    { // Precomputation
      int cheb_deg=nodes[0]->ChebDeg();
      std::vector<Real_t> p1(reg_grid_resolution*(cheb_deg+1));
      pvfmm::cheb_poly(cheb_deg,&x[0],reg_grid_resolution,&p1[0]);
      Mp1.ReInit(cheb_deg+1,reg_grid_resolution,&p1[0]);

      // Create work buffers
      size_t buff_size=std::max(cheb_deg+1,reg_grid_resolution)*std::max(cheb_deg+1,reg_grid_resolution)*std::max(cheb_deg+1,reg_grid_resolution)*data_dof;
      v1.Resize(buff_size);
      v2.Resize(buff_size);
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    std::vector<Real_t> cx;
    std::vector<Real_t> cy;
    std::vector<Real_t> cz;
    pvfmm::Matrix<Real_t> px;
    pvfmm::Matrix<Real_t> py;
    pvfmm::Matrix<Real_t> pz;

    pvfmm::Vector<Real_t> coeff_;
    { // Set coeff_
      int d0=nodes[0]->ChebDeg()+1;//+1+8;
      coeff_.ReInit(d0*d0*d0*data_dof);
      coeff_.SetZero();
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////
        
    cublasHandle_t handle;         
    
    // initialize cuda if there are points to evaluate
    // do initialization here, to make sure only one initialization is needed
    if (trg_coord.Dim() > 0) { 
             
    //Init GPU and cublas
    pvfmm::Profile::Tic("Initialize GPU/CUBLAS", &sim_config->comm, false, 5);
	
	// Get CUDA device
	int devID;

   	// This will pick the best possible CUDA capable device
   	devID = findCudaDevice(0, (const char **)"");
   	 		
	// Init Cublas
	//cublasHandle_t handle;
    cublasCreate(&handle);                         
	
	pvfmm::Profile::Toc();
	
	}
    

    for (size_t j=a;j<b;j++) {
      const size_t n_pts=part_indx[j+1]-part_indx[j];     
      if(!n_pts) continue;

	  std::cout << "n_pts: " << n_pts << std::endl;
	  
	  
	  pvfmm::Profile::Tic("Allocate GPU Memory", &sim_config->comm, false, 5);
	  // allocate GPU memory
	  Real_t* d_x_in = NULL;
	  Real_t* d_y_in = NULL;
	  Real_t* d_z_in = NULL;
	  Real_t* d_poly = NULL;
	  Real_t* d_buffer1 = NULL;
	  Real_t* d_buffer2 = NULL;
	
	  size_t d0=nodes[0]->ChebDeg()+1;
	
	  unsigned int mem_size_in = sizeof(Real_t) * n_pts;
	  unsigned int mem_size_poly = sizeof(Real_t) * n_pts * d0;
	  unsigned int mem_size_buffers = sizeof(Real_t) * std::max(d0,(size_t) n_pts)*std::max(d0, (size_t) n_pts)*std::max(d0,(size_t) n_pts)*data_dof;
	  unsigned int mem_size_coeffs = sizeof(Real_t) * d0 * d0 * d0 * data_dof;
	
	  checkCudaErrors(cudaMalloc((void**) &d_x_in, mem_size_in)); 
	  checkCudaErrors(cudaMalloc((void**) &d_y_in, mem_size_in)); 
	  checkCudaErrors(cudaMalloc((void**) &d_z_in, mem_size_in)); 
	  checkCudaErrors(cudaMalloc((void**) &d_poly, mem_size_poly));  
	  checkCudaErrors(cudaMalloc((void**) &d_buffer1, mem_size_buffers)); 
	  checkCudaErrors(cudaMalloc((void**) &d_buffer2, mem_size_buffers));
	  pvfmm::Profile::Toc();
	   
	  

      Real_t* c = nodes[j]->Coord();
      size_t  d = nodes[j]->Depth();
      Real_t  s = (Real_t)(1ULL<<d);

      Real_t* coord_ptr = &trg_coord[0]+part_indx[j]*COORD_DIM;
      if (!use_cubic) { 
        //////////////////////////////////////////////////////////////
        // CHEBYSHEV INTERPOLATION
        //////////////////////////////////////////////////////////////
        coord.resize((n_pts+16)*COORD_DIM);
        for (size_t i=0;i<n_pts;i++) {
          // scale to [-1,1] -> used in cheb_eval
          coord[i*COORD_DIM+0]=(coord_ptr[i*COORD_DIM+0]-c[0])*2.0*s-1.0;
          coord[i*COORD_DIM+1]=(coord_ptr[i*COORD_DIM+1]-c[1])*2.0*s-1.0;
          coord[i*COORD_DIM+2]=(coord_ptr[i*COORD_DIM+2]-c[2])*2.0*s-1.0;
        }
        for (size_t i=n_pts;i<n_pts+16;i++) {
          // scale to [-1,1] -> used in cheb_eval
          coord[i*COORD_DIM+0]=0.0;
          coord[i*COORD_DIM+1]=0.0;
          coord[i*COORD_DIM+2]=0.0;
        }

        if(coord.size()){
          pvfmm::Vector<Real_t>& coeff=nodes[j]->ChebData();
          int cheb_deg=nodes[j]->ChebDeg();
          int d=cheb_deg+1;
          int d0 = d;
          //for(d0=d;d0%4;d0++);
          int n=n_pts;
          for(;n%16;n++);
          assert(coeff.Dim()==(size_t)(d*(d+1)*(d+2)*data_dof)/6);
          { // set coeff_
            long indx=0;
            for(int l0=0;l0<data_dof;l0++){
              for(int i=0;i<d;i++){
                for(int j=0;i+j<d;j++){
                  for(int k=0;i+j+k<d;k++){
                    coeff_[k+d0*(j+d0*(i+d0*l0))]=coeff[indx];
                    indx++;
                  }
                }
              }
            }
          }
          
          cx.resize(n);
          cy.resize(n);
          cz.resize(n);
          for(long i=0;i<n;i++){
            cx[i]=coord[i*COORD_DIM+0];
            cy[i]=coord[i*COORD_DIM+1];
            cz[i]=coord[i*COORD_DIM+2];
          }
		
		  pvfmm::Vector<Real_t> gpu_out;
		  // do chebyshev evaluation on GPU
		  pvfmm::Profile::Tic("chebEval", &sim_config->comm, false, 5);	
		  chebEval_cuda(&coeff_[0], &cx[0], &cy[0], &cz[0], gpu_out, n_pts, cheb_deg, data_dof,
					d_x_in, d_y_in, d_z_in, d_poly, d_buffer1, d_buffer2, handle);
		  pvfmm::Profile::Toc();
		  		  		  
		  // remove the useless values
		  int size = n_pts*n_pts*n_pts;
		  int step_length = 0;
		  if (n_pts > 1)
			step_length = (size-1)/(n_pts-1);
		  
		  if (tmp_out.Dim()<n*data_dof) {
            tmp_out.Resize(n*data_dof);
          }
          		  
		  int index = 0;
          for (int i = 0; i < n_pts * data_dof; i += data_dof)
          {
			  for (int dof = 0; dof < data_dof; dof++) 
			  {
				tmp_out[dof+i] = gpu_out[dof*size + index*step_length];
				
			}
			index++;
		  }
		  
		   { // Copy to trg_value
            Real_t *trg_value_ = &trg_value[0] + part_indx[j] * data_dof;
            for (long i = 0; i < n_pts*data_dof; i++) {
                trg_value_[i] = tmp_out[i];               
              }
            }          

        }

      } else {  // still unchanged
        query_values.resize(n_pts*data_dof);
        query_points.resize(n_pts*COORD_DIM);

        // ************************************************************
        // EVALUATE AT THE REGULAR GRID
        // ************************************************************
        // if(!sim_config->cubic_use_analytical) {
          pvfmm::Vector<Real_t>& coeff_=nodes[j]->ChebData();
          pvfmm::Vector<Real_t> reg_grid_vals_tmp(reg_grid_num_points*data_dof, &reg_grid_vals[0], false);
          { // cheb_eval
            int cheb_deg=nodes[0]->ChebDeg();
            size_t d=(size_t)cheb_deg+1;
            size_t n_coeff=(d*(d+1)*(d+2))/6;
            size_t dof=coeff_.Dim()/n_coeff;
            assert(coeff_.Dim()==dof*n_coeff);

            size_t n1=x.size();
            assert(reg_grid_vals_tmp.Dim()==n1*n1*n1*dof);

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
			pvfmm::Profile::Tic("Mul 1", &sim_config->comm, false, 5);
            { // Apply Mp1
              pvfmm::Matrix<Real_t> Mi  ( d* d*dof, d,&v1[0],false);
              pvfmm::Matrix<Real_t> Mo  ( d* d*dof,n1,&v2[0],false);
              pvfmm::Matrix<Real_t>::GEMM(Mo, Mi, Mp1);

              pvfmm::Matrix<Real_t> Mo_t(n1, d* d*dof,&v1[0],false);
              for(size_t i=0;i<Mo.Dim(0);i++)
                for(size_t j=0;j<Mo.Dim(1);j++){
                  Mo_t[j][i]=Mo[i][j];
                }
            }
            pvfmm::Profile::Toc();
            pvfmm::Profile::Tic("Mul 2", &sim_config->comm, false, 5);
            { // Apply Mp1
              pvfmm::Matrix<Real_t> Mi  (n1* d*dof, d,&v1[0],false);
              pvfmm::Matrix<Real_t> Mo  (n1* d*dof,n1,&v2[0],false);
              pvfmm::Matrix<Real_t>::GEMM(Mo, Mi, Mp1);

              pvfmm::Matrix<Real_t> Mo_t(n1,n1* d*dof,&v1[0],false);
              for(size_t i=0;i<Mo.Dim(0);i++)
                for(size_t j=0;j<Mo.Dim(1);j++){
                  Mo_t[j][i]=Mo[i][j];
                }
            }
            pvfmm::Profile::Toc();
            pvfmm::Profile::Tic("Mul 3", &sim_config->comm, false, 5);
            { // Apply Mp1
              pvfmm::Matrix<Real_t> Mi  (n1*n1*dof, d,&v1[0],false);
              pvfmm::Matrix<Real_t> Mo  (n1*n1*dof,n1,&v2[0],false);
              pvfmm::Matrix<Real_t>::GEMM(Mo, Mi, Mp1);

              pvfmm::Matrix<Real_t> Mo_t(n1,n1*n1*dof,&v1[0],false);
              for(size_t i=0;i<Mo.Dim(0);i++)
                for(size_t j=0;j<Mo.Dim(1);j++){
                  Mo_t[j][i]=Mo[i][j];
                }
            }
			pvfmm::Profile::Toc();
			
            { // Copy to reg_grid_vals_tmp
              pvfmm::Matrix<Real_t> Mo  ( n1*n1*n1,dof,&v1[0],false);
              pvfmm::Matrix<Real_t> Mo_t(dof,n1*n1*n1,&reg_grid_vals_tmp[0],false);
              for(size_t i=0;i<Mo.Dim(0);i++)
                for(size_t j=0;j<Mo.Dim(1);j++){
                  Mo_t[j][i]=Mo[i][j];
                }
            }
          }
        // } else {   // evaluate using analytical function
        //   std::vector<Real_t> reg_grid_anal_coord(3);
        //   std::vector<Real_t> reg_grid_anal_vals(1*data_dof);
        //   pvfmm::Vector<Real_t> reg_grid_vals_tmp(reg_grid_num_points*data_dof, &reg_grid_vals[0], false);
        //   int nx = reg_grid_resolution;
        //   for (int xi = 0; xi < nx; xi++) {
        //     for (int yi = 0; yi < nx; yi++) {
        //       for (int zi = 0; zi < nx; zi++) {
        //         reg_grid_anal_coord[0] = c[0] + reg_grid_coord_1d[xi]/s;
        //         reg_grid_anal_coord[1] = c[1] + reg_grid_coord_1d[yi]/s;
        //         reg_grid_anal_coord[2] = c[2] + reg_grid_coord_1d[zi]/s;
        //         assert(!nodes[j]->input_fn.IsEmpty());
        //         nodes[j]->input_fn(reg_grid_anal_coord.data(),
        //                            1,
        //                            reg_grid_anal_vals.data());
        //         for(int l=0;l<data_dof;l++)
        //           reg_grid_vals_tmp[xi+(yi+(zi+l*nx)*nx)*nx] = reg_grid_anal_vals[l];
        //       }
        //     }
        //   }
        // }
        // ************************************************************
        // 3D CUBIC INTERPOLATION
        // ************************************************************
        // scale to [0,1] in local node
        for ( int pi = 0; pi < n_pts; pi++) {
          query_points[pi*COORD_DIM+0] = (coord_ptr[pi*COORD_DIM+0]-c[0])*s;
          query_points[pi*COORD_DIM+1] = (coord_ptr[pi*COORD_DIM+1]-c[1])*s;
          query_points[pi*COORD_DIM+2] = (coord_ptr[pi*COORD_DIM+2]-c[2])*s;
        }
        
        fast_interp_cuda(reg_grid_vals, data_dof, reg_grid_resolution, query_points, query_values);
        output = &query_values[0];

        memcpy(&trg_value[0]+part_indx[j]*data_dof, output, n_pts*data_dof*sizeof(Real_t));
      } // end of cubic interpolation
    
   // cuda cleanup
   checkCudaErrors(cudaFree(d_buffer1));
   checkCudaErrors(cudaFree(d_buffer2));
   checkCudaErrors(cudaFree(d_x_in));
   checkCudaErrors(cudaFree(d_y_in));
   checkCudaErrors(cudaFree(d_z_in));
   checkCudaErrors(cudaFree(d_poly));
    
    }
    
   // cuda cleanup
   if (trg_coord.Dim() > 0) { 
	cublasDestroy(handle);
	cudaDeviceReset();   
	}
  }
  if (use_cubic){
    pvfmm::Profile::Add_FLOP(trg_coord.Dim()/COORD_DIM * (COORD_DIM*16 + data_dof*192)); // cubic interpolation
  }else{
    long d=nodes[0]->ChebDeg()+1;
    pvfmm::Profile::Add_FLOP(trg_coord.Dim()/COORD_DIM * ( COORD_DIM*d*3 + ((d*(d+1)*(d+2))/6) * data_dof * 2 ) );
  }
}

template <class Tree_t>
void EvalTree_cuda(Tree_t* tree,
              typename Tree_t::Real_t* trg_coord_,
              size_t N,
              typename Tree_t::Real_t* value,
              pvfmm::BoundaryType bc_type) {
  size_t omp_p=omp_get_max_threads();
  typedef typename Tree_t::Node_t Node_t;
  typedef typename Tree_t::Real_t Real_t;
  tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();

  int myrank;
  int np;
  MPI_Comm_rank(sim_config->comm, &myrank);
  MPI_Comm_size(sim_config->comm, &np);

  //////////////////////////////////////////////////////////////
  // GET LEAF NODES AND MINIMUM MORTON ID OF THE CURRENT PROCESS
  //////////////////////////////////////////////////////////////
  // pvfmm::Profile::Tic("MinMortonId", &sim_config->comm, false, 5);
  size_t data_dof=0;
  pvfmm::MortonId min_mid;
  std::vector<Node_t*> nodes;
  { // Get list of leaf nodes.
    const std::vector<Node_t*>& all_nodes=tree->GetNodeList();
    for (size_t i=0; i< all_nodes.size(); i++) {
      if (all_nodes[i]->IsLeaf() && !all_nodes[i]->IsGhost()) {
        nodes.push_back(all_nodes[i]);
      }
    }
    assert(nodes.size());
    min_mid=nodes[0]->GetMortonId();
    data_dof=nodes[0]->DataDOF();
  }
  // pvfmm::Profile::Toc();

  //////////////////////////////////////////////////////////////
  // GATHER MINIMUM MORTON IDS OF ALL PARTITIONS
  //////////////////////////////////////////////////////////////
  std::vector<pvfmm::MortonId> glb_min_mid(np);
  MPI_Allgather(&min_mid, 1, pvfmm::par::Mpi_datatype<pvfmm::MortonId>::value(),
                &glb_min_mid[0], 1, pvfmm::par::Mpi_datatype<pvfmm::MortonId>::value(),
                sim_config->comm);

  //////////////////////////////////////////////////////////////
  // APPLY PERIODIC BOUNDARY CONDITION
  //////////////////////////////////////////////////////////////
  if (bc_type == pvfmm::Periodic) {
#pragma omp parallel for
    for (size_t i = 0; i < N*COORD_DIM; i++) {
      Real_t& c = trg_coord_[i];
      if(c <  0.0) c = c + 1.0;
      if(c >= 1.0) c = c - 1.0;
    }
  }

  //////////////////////////////////////////////////////////////
  // LOCAL SORT
  //////////////////////////////////////////////////////////////
  typedef pvfmm::par::SortPair<pvfmm::MortonId,size_t> Pair_t;
  pvfmm::Vector<Pair_t> iarray_trg_mid(N);
  pvfmm::Vector<Pair_t> iarray_trg_mid_sorted(N);
  size_t lcl_start, lcl_end, trg_cnt_inside, trg_cnt_outside;

  // pvfmm::Profile::Tic("LclSort", &sim_config->comm, false, 5);
  //////////////////////////////////////////////////
  // LOCAL SORT WITH TRACKING THE INDICES
  //////////////////////////////////////////////////
  pvfmm::Profile::Tic("LclHQSort", &sim_config->comm, false, 5);
#pragma omp parallel for
  for(size_t i = 0; i < N; i++) {
    iarray_trg_mid[i].key  = pvfmm::MortonId(&trg_coord_[i*COORD_DIM]);
    iarray_trg_mid[i].data = i;
  }

  pvfmm::par::HyperQuickSort(iarray_trg_mid, iarray_trg_mid_sorted, MPI_COMM_SELF);

  Pair_t p1;
  p1.key = glb_min_mid[myrank];
  lcl_start = std::lower_bound(&iarray_trg_mid_sorted[0],
                               &iarray_trg_mid_sorted[0]+iarray_trg_mid_sorted.Dim(),
                               p1,
                               std::less<Pair_t>()) - &iarray_trg_mid_sorted[0];

  if (myrank+1 < np) {
    Pair_t p2; p2.key = glb_min_mid[myrank+1];
    lcl_end = std::lower_bound(&iarray_trg_mid_sorted[0],
                               &iarray_trg_mid_sorted[0]+iarray_trg_mid_sorted.Dim(),
                               p2,
                               std::less<Pair_t>()) - &iarray_trg_mid_sorted[0];
  } else {
    lcl_end = iarray_trg_mid_sorted.Dim();
  }

  // [lcl_start, lcl_end[
  trg_cnt_inside  = lcl_end - lcl_start;
  trg_cnt_outside = N - trg_cnt_inside;
  pvfmm::Profile::Toc();  // Sort

  //////////////////////////////////////////////////
  // COMMINUCATE THE OUTSIDER POINTS
  //////////////////////////////////////////////////
  static pvfmm::Vector<size_t> out_scatter_index;
  static pvfmm::Vector<Real_t> trg_coord_outside;

  // pvfmm::Profile::Tic("OutCpyCoord", &sim_config->comm, true, 5);
  trg_coord_outside.Resize(trg_cnt_outside*COORD_DIM);
#pragma omp parallel for
  for (int i = 0 ; i < lcl_start; i++) {
    size_t src_indx = iarray_trg_mid_sorted[i].data;
    size_t dst_indx = i;
    for(size_t j = 0; j < COORD_DIM; j++) {
      trg_coord_outside[dst_indx*COORD_DIM+j] = trg_coord_[src_indx*COORD_DIM+j];
    }
  }

#pragma omp parallel for
  for (int i = 0 ; i < (N-lcl_end); i++) {
    size_t src_indx = iarray_trg_mid_sorted[lcl_end+i].data;
    size_t dst_indx = lcl_start + i;
    for(size_t j = 0; j < COORD_DIM; j++) {
      trg_coord_outside[dst_indx*COORD_DIM+j] = trg_coord_[src_indx*COORD_DIM+j];
    }
  }
  // pvfmm::Profile::Toc();  // OUT COPY COORDINATES

  // pvfmm::Profile::Tic("OutMortonID", &sim_config->comm, true, 5);
  static pvfmm::Vector<pvfmm::MortonId> trg_mid_outside;
  trg_mid_outside.Resize(trg_cnt_outside);
#pragma omp parallel for
  for (size_t i = 0; i < trg_cnt_outside; i++) {
    trg_mid_outside[i] = pvfmm::MortonId(&trg_coord_outside[i*COORD_DIM]);
  }
  // pvfmm::Profile::Toc();

  pvfmm::Profile::Tic("OutScatterIndex", &sim_config->comm, true, 5);
  pvfmm::par::SortScatterIndex(trg_mid_outside, out_scatter_index, *tree->Comm(), &min_mid);
  pvfmm::Profile::Toc();

  pvfmm::Profile::Tic("OutScatterForward", &sim_config->comm, true, 5);
  pvfmm::par::ScatterForward(trg_coord_outside, out_scatter_index, *tree->Comm());
  pvfmm::Profile::Toc();

  size_t trg_cnt_others = trg_coord_outside.Dim()/COORD_DIM;
  size_t trg_cnt_total  = trg_cnt_inside + trg_cnt_others;

  //////////////////////////////////////////////////
  // EVALUATE THE OUTSIDER POINTS
  //////////////////////////////////////////////////
  static pvfmm::Vector<Real_t> trg_value_outsider;
  pvfmm::Profile::Tic("OutEvaluation", &sim_config->comm, false, 5);
  trg_value_outsider.Resize(trg_cnt_others*data_dof);
  EvalNodesLocal_cuda<Real_t, Tree_t>(nodes, trg_coord_outside, trg_value_outsider);
  pvfmm::Profile::Toc();

  //////////////////////////////////////////////////
  // SCATTER REVERSE THE OUTSIDER POINT'S VALUES
  //////////////////////////////////////////////////
  pvfmm::Profile::Tic("OutScatterReverse", &sim_config->comm, true, 5);
  pvfmm::par::ScatterReverse(trg_value_outsider, out_scatter_index, *tree->Comm(), trg_cnt_outside);
  pvfmm::Profile::Toc();

  //////////////////////////////////////////////////
  // SET OUTSIDER POINTS EVALUATION VALUES
  //////////////////////////////////////////////////
  // pvfmm::Profile::Tic("OutSetVal", &sim_config->comm, true, 5);
#pragma omp parallel for
  for (int i = 0 ; i < lcl_start; i++) {
    size_t src_indx = i;
    size_t dst_indx = iarray_trg_mid_sorted[i].data;
    for(size_t j = 0; j < data_dof; j++) {
      value[dst_indx*data_dof+j] = trg_value_outsider[src_indx*data_dof+j];
    }
  }
#pragma omp parallel for
  for (int i = 0 ; i < (N-lcl_end); i++) {
    size_t src_indx = lcl_start + i;
    size_t dst_indx = iarray_trg_mid_sorted[lcl_end+i].data;
    for(size_t j = 0; j < data_dof; j++) {
      value[dst_indx*data_dof+j] = trg_value_outsider[src_indx*data_dof+j];
    }
  }
  // pvfmm::Profile::Toc();  // OUT SET VALUES

  //////////////////////////////////////////////////
  // COLLECT THE COORDINATE VALUES
  //////////////////////////////////////////////////
  static  pvfmm::Vector<Real_t> trg_coord_inside;
  // pvfmm::Profile::Tic("InCpyCoord", &sim_config->comm, true, 5);
  trg_coord_inside.Resize(trg_cnt_inside*COORD_DIM);
#pragma omp parallel for
  for (size_t i = 0; i < trg_cnt_inside; i++) {
    size_t src_indx = iarray_trg_mid_sorted[lcl_start+i].data;
    size_t dst_indx = i;
    for(size_t j = 0; j < COORD_DIM;j++) {
      trg_coord_inside[dst_indx*COORD_DIM+j] = trg_coord_[src_indx*COORD_DIM+j];
    }
  }
  // pvfmm::Profile::Toc();

  //////////////////////////////////////////////////
  // EVALUATE THE LOCAL POINTS
  //////////////////////////////////////////////////
  static pvfmm::Vector<Real_t> trg_value_insider;
  pvfmm::Profile::Tic("InEvaluation", &sim_config->comm, false, 5);
  trg_value_insider.Resize(trg_cnt_inside*data_dof);
  EvalNodesLocal_cuda<Real_t, Tree_t>(nodes, trg_coord_inside, trg_value_insider);
  pvfmm::Profile::Toc();

  //////////////////////////////////////////////////
  // SET INSIDER POINTS EVALUATION VALUES
  //////////////////////////////////////////////////
  // pvfmm::Profile::Tic("InSetVal", &sim_config->comm, false, 5);
#pragma omp parallel for
  for (size_t i = 0; i < trg_cnt_inside; i++) {
    size_t src_indx = i;
    size_t dst_indx = iarray_trg_mid_sorted[lcl_start+i].data;
    for (int j = 0; j < data_dof; j++) {
      value[dst_indx*data_dof+j] = trg_value_insider[i*data_dof+j];
    }
  }
  // pvfmm::Profile::Toc();

  // pvfmm::Profile::Toc();  // LOCAL SORT

  //////////////////////////////////////////////////
  // PRINT TARGET COUNTS INFO
  //////////////////////////////////////////////////
  if (sim_config->profile) {
    size_t sbuff[4] = {trg_cnt_inside,
                       trg_cnt_outside,
                       trg_value_outsider.Dim()/data_dof,
                       trg_cnt_others};
    size_t* rbuff = (size_t *)malloc(np*4*sizeof(size_t));
    MPI_Gather(sbuff, 4, pvfmm::par::Mpi_datatype<size_t>::value(),
               rbuff, 4, pvfmm::par::Mpi_datatype<size_t>::value(),
               0, *tree->Comm());
    if (myrank == 0) {
      std::ostringstream os;
      os << "TRG_CNT_IN_TOT: ";
      for (int i = 0 ; i < np; i++) {
        size_t* data = &rbuff[i*4];
        std::cout
            << " PROC: " << i
            << " TRG_CNT_IN: "     << data[0]
            << " TRG_CNT_OUT: "    << data[1]
            << " TRG_CNT_RCV: "    << data[2]
            << " TRG_CNT_OTHRS: "  << data[3]
            << std::endl;
        os << data[0] + data[3] << " ";
      }
      std::cout << os.str() << std::endl;
    }
    delete rbuff;
  }
  //////////////////////////////////////////////////////////////
  // GLOBAL SORT
  //////////////////////////////////////////////////////////////
//   pvfmm::Profile::Tic("GlobalSort", &sim_config->comm, false, 5);
//   {
//     //////////////////////////////////////////////////////////////
//     // COMPUTE MORTON ID OF THE TARGET POINTS
//     //////////////////////////////////////////////////////////////
//     pvfmm::Profile::Tic("MortonId", &sim_config->comm, true, 5);
//     static pvfmm::Vector<pvfmm::MortonId> trg_mid; trg_mid.Resize(N);
// #pragma omp parallel for
//     for (size_t i = 0; i < N; i++) {
//       trg_mid[i] = pvfmm::MortonId(&trg_coord_[i*COORD_DIM]);
//     }
//     pvfmm::Profile::Toc();

//     //////////////////////////////////////////////////////////////
//     // SCATTER THE COORDINATES
//     //////////////////////////////////////////////////////////////
//     pvfmm::Profile::Tic("ScatterIndex", &sim_config->comm, true, 5);
//     static pvfmm::Vector<size_t> scatter_index;
//     pvfmm::par::SortScatterIndex(trg_mid, scatter_index, *tree->Comm(), &min_mid);
//     pvfmm::Profile::Toc();

//     static pvfmm::Vector<Real_t> trg_coord;
//     pvfmm::Profile::Tic("ScatterForward", &sim_config->comm, true, 5);
//     {
//       trg_coord.Resize(N*COORD_DIM);
// #pragma omp parallel for
//       for(size_t tid=0;tid<omp_p;tid++){
//         size_t a=N*COORD_DIM*(tid+0)/omp_p;
//         size_t b=N*COORD_DIM*(tid+1)/omp_p;
//         if(b-a) memcpy(&trg_coord[0]+a, &trg_coord_[0]+a, (b-a)*sizeof(Real_t));
//       }
//       pvfmm::par::ScatterForward(trg_coord, scatter_index, *tree->Comm());
//     }
//     pvfmm::Profile::Toc();

//     // std::cout << "P" << myrank << " TRG_CNT: " << trg_coord.Dim()/COORD_DIM << std::endl;

//     //////////////////////////////////////////////////////////////
//     // LOCAL POINTS EVALUATION
//     //////////////////////////////////////////////////////////////
//     size_t num_trg_points = trg_coord.Dim()/COORD_DIM;
//     static pvfmm::Vector<Real_t> trg_value;
//     pvfmm::Profile::Tic("Evaluation", &sim_config->comm, false, 5);
//     trg_value.Resize(num_trg_points*data_dof);
//     EvalNodesLocal<Real_t, Tree_t>(nodes, trg_coord, trg_value);
//     pvfmm::Profile::Toc();

//     //////////////////////////////////////////////////////////////
//     // GATHERING GLOBAL POINTS VALUES
//     //////////////////////////////////////////////////////////////
//     pvfmm::Profile::Tic("ScatterReverse", &sim_config->comm, true, 5);
//     pvfmm::par::ScatterReverse(trg_value, scatter_index, *tree->Comm(), N);
//     pvfmm::Profile::Toc();

//     //////////////////////////////////////////////////////////////
//     // SETTING EVALUATION VALUES
//     //////////////////////////////////////////////////////////////
//     // memcpy(value, &trg_value[0], trg_value.Dim()*sizeof(Real_t));
//   }  // GLOBAL SORT
  // pvfmm::Profile::Toc();
}

template<typename real_t,
         class NodeType>
class NodeFieldFunctor_cuda {

 public:
  explicit NodeFieldFunctor_cuda(NodeType* node):
      node_(node) {
  }

  virtual ~NodeFieldFunctor_cuda() {
  }

  void operator () (const real_t* points_pos,
                    int num_points,
                    real_t* out) {
    tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
    pvfmm::Profile::Tic("EvalTree - GPU", &sim_config->comm, true, 5);
    EvalTree_cuda(node_, const_cast<real_t*>(points_pos), num_points, out,sim_config->bc);
    pvfmm::Profile::Toc();
  }

 private:
  NodeType* node_;
};

}      // namespace tbslas
#endif  // SRC_TREE_NODE_FIELD_FUNCTOR_H_
