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
// SYSTEM
#include <mpi.h>
#include <pvfmm_common.hpp>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <utility>
// PVFMM
#include <profile.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <cheb_node.hpp>
// LOCAL
#include <utils.hpp>

// TBSLAS
#include <utils/common.h>
#include <utils/reporter.h>
#include <diffusion/kernel.h>
#include <tree/semilag_tree.h>
#include <tree/utils_tree.h>

int NUM_TIME_STEPS = 1;
double TBSLAS_DT;
double TBSLAS_DIFF_COEFF;
double TBSLAS_ALPHA;
// current simulation time
double tcurr = 0.25;

typedef tbslas::MetaData<std::string,
                         std::string,
                         std::string> MetaData_t;

template <class Real_t>
void fn_input_t1(const Real_t* coord,
                 int n,
                 Real_t* out) {
  const Real_t amp = 1e-2;
  const Real_t xc = 0.5;
  const Real_t yc = 0.5;
  const Real_t zc = 0.5;
  tbslas::diffusion_kernel(coord,
                           n,
                           out,
                           TBSLAS_DIFF_COEFF,
                           tcurr,
                           amp,
                           xc,
                           yc,
                           zc);
}

template <class Real_t>
void fn_input_t2(const Real_t* coord,
                 int n,
                 Real_t* out) {
  tbslas::gaussian_kernel_diffusion_input(coord,
                                          n,
                                          out,
                                          TBSLAS_ALPHA);
}

template <class Real_t>
void fn_poten_t2(const Real_t* coord,
                 int n,
                 Real_t* out) {
  tbslas::gaussian_kernel(coord,
                          n,
                          out);
}

template<class _FMM_Tree_Type,
         class _FMM_Mat_Type>
void
RunFMM(_FMM_Tree_Type* tree,
       _FMM_Mat_Type* fmm_mat,
       pvfmm::BoundaryType bndry) {
  tree->InitFMM_Tree(false,bndry);
  tree->SetupFMM(fmm_mat);
  tree->RunFMM();
  tree->Copy_FMMOutput(); //Copy FMM output to tree Data.
}

void
GetVTKFileName(char (*buffer)[300], int timestep) {
  tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
  snprintf(*buffer,
           sizeof(*buffer),
           sim_config->vtk_filename_format.c_str(),
           tbslas::get_result_dir().c_str(),
           sim_config->vtk_filename_prefix.c_str(),
           sim_config->vtk_filename_variable.c_str(),
           timestep);
}

template <class Real_t>
void RunAdvectDiff(int test_case, size_t N, size_t M, bool unif, int mult_order,
                   int cheb_deg, int depth, bool adap, Real_t tol, MPI_Comm comm) {
  typedef pvfmm::FMM_Node<pvfmm::Cheb_Node<Real_t> > FMMNode_t;
  typedef pvfmm::FMM_Cheb<FMMNode_t> FMM_Mat_t;
  typedef pvfmm::FMM_Tree<FMM_Mat_t> FMM_Tree_t;
  tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
  char out_name_buffer[300];
  // Find out number of OMP thereads.
  int omp_p=omp_get_max_threads();

  // **********************************************************************
  // SETUP FMM KERNEL
  // **********************************************************************
  const pvfmm::Kernel<Real_t>* mykernel=NULL;
  pvfmm::BoundaryType bndry;
  const pvfmm::Kernel<double> modified_laplace_kernel_d =
      pvfmm::BuildKernel<double, tbslas::modified_laplace_poten>
      (tbslas::GetModfiedLaplaceKernelName<double>(TBSLAS_ALPHA), 3, std::pair<int,int>(1,1));

  // **********************************************************************
  // SETUP TEST CASE
  // **********************************************************************
  void (*fn_input_)(const Real_t* , int , Real_t*)=NULL;
  void (*fn_poten_)(const Real_t* , int , Real_t*)=NULL;
  void (*fn_grad_ )(const Real_t* , int , Real_t*)=NULL;
  switch (test_case) {
    case 2:
      fn_input_ = fn_input_t2<Real_t>;
      fn_poten_ = fn_poten_t2<Real_t>;
      mykernel  = &modified_laplace_kernel_d;
      bndry = pvfmm::FreeSpace;
      break;
    case 1:
      fn_input_ = fn_input_t1<Real_t>;
      fn_poten_ = fn_input_t1<Real_t>;
      mykernel  = &modified_laplace_kernel_d;
      bndry = pvfmm::FreeSpace;
      break;
    default:
      fn_input_=NULL;
      fn_poten_=NULL;
      fn_grad_ =NULL;
      break;
  }
  // Find out my identity in the default communicator
  int myrank, p;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);

  // **********************************************************************
  // SETUP TREE
  // **********************************************************************
  typename FMMNode_t::NodeData tree_data;
  tree_data.dim       = COORD_DIM;
  tree_data.max_depth = depth;
  tree_data.cheb_deg  = cheb_deg;
  tree_data.input_fn  = fn_input_;
  tree_data.data_dof  = mykernel->ker_dim[0];
  tree_data.tol       = tol;
  //Set source coordinates.
  std::vector<Real_t> pt_coord;
  if(unif) pt_coord = tbslas::point_distrib<Real_t>(tbslas::UnifGrid,N,comm);
  else pt_coord = tbslas::point_distrib<Real_t>(tbslas::RandElps,N,comm); //RandElps, RandGaus
  tree_data.max_pts  = M; // Points per octant.
  tree_data.pt_coord = pt_coord;
  //Print various parameters.
  if (!myrank) {
    std::cout<<std::setprecision(2)<<std::scientific;
    std::cout<<"Number of MPI processes: "<<p<<'\n';
    std::cout<<"Number of OpenMP threads: "<<omp_p<<'\n';
    std::cout<<"Order of multipole expansions: "<<mult_order<<'\n';
    std::cout<<"Order of Chebyshev polynomials: "<<tree_data.cheb_deg<<'\n';
    std::cout<<"FMM Kernel name: "<<mykernel->ker_name<<'\n';
    std::cout<<"Number of point samples: "<<N<<'\n';
    std::cout<<"Uniform distribution: "<<(unif?"true":"false")<<'\n';
    std::cout<<"Maximum points per octant: "<<tree_data.max_pts<<'\n';
    std::cout<<"Chebyshev Tolerance: "<<tree_data.tol<<'\n';
    std::cout<<"Maximum Tree Depth: "<<depth<<'\n';
    std::cout<<"BoundaryType: "<<(bndry==pvfmm::Periodic?"Periodic":"FreeSpace")<<'\n';
  }
  //Create Tree and initialize with input data.
  FMM_Tree_t* tree = new FMM_Tree_t(comm);
  tree->Initialize(&tree_data);

  // **********************************************************************
  // SETUP FMM
  // **********************************************************************
  //Initialize FMM_Mat.
  FMM_Mat_t* fmm_mat = NULL;
  {
    fmm_mat = new FMM_Mat_t;
    fmm_mat->Initialize(mult_order,
                        cheb_deg,
                        comm,
                        mykernel);
  }

  // **********************************************************************
  // VELOCITY FIELD
  // **********************************************************************
  FMM_Tree_t tvel_curr(comm);
  tbslas::ConstructTree<FMM_Tree_t>(N,
                                    M,
                                    cheb_deg,
                                    depth,
                                    adap,
                                    tol,
                                    comm,
                                    tbslas::get_vorticity_field<double,3>,
                                    3,
                                    tvel_curr);
  snprintf(out_name_buffer,
           sizeof(out_name_buffer),
           sim_config->vtk_filename_format.c_str(),
           tbslas::get_result_dir().c_str(),
           sim_config->vtk_filename_prefix.c_str(),
           "velocity",
           0);
  tvel_curr.Write2File(out_name_buffer, cheb_deg);

  // =========================================================================
  // RUN
  // =========================================================================
  double in_al2,in_rl2,in_ali,in_rli;
  double di_al2,di_rl2,di_ali,di_rli;
  double ad_al2,ad_rl2,ad_ali,ad_rli;
  double al2,rl2,ali,rli;
  int timestep = 1;
  for (; timestep < 2*NUM_TIME_STEPS+1; timestep +=2) {
    // **********************************************************************
    // SOLVE DIFFUSION: FMM
    // **********************************************************************
    // RunFMM(tree, fmm_mat, bndry);
    tree->InitFMM_Tree(false,bndry);
    if (timestep==1)
      CheckChebOutput<FMM_Tree_t>(tree,
                                  fn_poten_,
                                  mykernel->ker_dim[1],
                                  in_al2,
                                  in_rl2,
                                  in_ali,
                                  in_rli,
                                  std::string("Input"));

    tree->SetupFMM(fmm_mat);
    tree->RunFMM();
    tree->Copy_FMMOutput(); //Copy FMM output to tree Data.
    // Write2File
    GetVTKFileName(&out_name_buffer, timestep);
    tree->Write2File(out_name_buffer, sim_config->vtk_order);
    tcurr += TBSLAS_DT;
    CheckChebOutput<FMM_Tree_t>(tree,
                                fn_poten_,
                                mykernel->ker_dim[1],
                                di_al2,
                                di_rl2,
                                di_ali,
                                di_rli,
                                std::string("Diffusion"));

    // **********************************************************************
    // SOLVE ADVECTION: SEMI-LAGRANGIAN
    // **********************************************************************
    tbslas::RunSemilagSimulationInSitu(&tvel_curr,
                                       tree,
                                       1,
                                       sim_config->dt,
                                       sim_config->num_rk_step,
                                       true,
                                       true);
    //Write2File
    GetVTKFileName(&out_name_buffer, timestep+1);
    tree->Write2File(out_name_buffer, sim_config->vtk_order);
    // tcurr += TBSLAS_DT;
    CheckChebOutput<FMM_Tree_t>(tree,
                                fn_poten_,
                                mykernel->ker_dim[1],
                                ad_al2,
                                ad_rl2,
                                ad_ali,
                                ad_rli,
                                std::string("Advection"));
  }

  // =========================================================================
  // REPORT RESULTS
  // =========================================================================
  int num_leaves = tbslas::CountNumLeafNodes(*tree);
  CheckChebOutput<FMM_Tree_t>(tree,
                              fn_poten_,
                              mykernel->ker_dim[1],
                              al2,rl2,ali,rli,
                              std::string("Output"));
  typedef tbslas::Reporter<Real_t> Rep;
  if(!myrank) {
    Rep::AddData("TOL", sim_config->tree_tolerance);
    Rep::AddData("MaxDEPTH", sim_config->tree_max_depth);
    Rep::AddData("NOCT", num_leaves);

    Rep::AddData("DT", sim_config->dt);
    Rep::AddData("TN", sim_config->total_num_timestep);

    Rep::AddData("DIFF", TBSLAS_DIFF_COEFF);
    Rep::AddData("ALPHA", TBSLAS_ALPHA);

    Rep::AddData("InAL2", in_al2);
    Rep::AddData("OutAL2", al2);

    Rep::AddData("InRL2", in_rl2);
    Rep::AddData("OutRL2", rl2);

    Rep::AddData("InALINF", in_ali);
    Rep::AddData("OutALINF", ali);

    Rep::AddData("InRLINF", in_rli);
    Rep::AddData("OutRLINF", rli);

    Rep::Report();
  }
  // **********************************************************************
  // CLEAN UP MEMORY
  // **********************************************************************
  //Delete matrices.
  if(fmm_mat)
    delete fmm_mat;

  //Delete the tree.
  delete tree;
}

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm=MPI_COMM_WORLD;
  int myrank;
  MPI_Comm_rank(comm, &myrank);

  parse_command_line_options(argc, argv);

  bool  unif=              (commandline_option(argc, argv, "-unif",    NULL, false, "-unif                : Uniform point distribution."        )!=NULL);
  int      m=       strtoul(commandline_option(argc, argv,    "-m",    "10", false, "-m    <int> = (10)   : Multipole order (+ve even integer)."),NULL,10);
  int   test=       strtoul(commandline_option(argc, argv, "-test",     "1", false,
                                               "-test <int> = (1)    : 1) Laplace, Smooth Gaussian, Periodic Boundary"),NULL,10);

  // =========================================================================
  // SIMULATION PARAMETERS
  // =========================================================================
  tbslas::SimConfig* sim_config     = tbslas::SimConfigSingleton::Instance();

  pvfmm::Profile::Enable(sim_config->profile);
  sim_config->vtk_filename_prefix   = "advdiff";
  sim_config->vtk_filename_variable = "conc";

  NUM_TIME_STEPS    = sim_config->total_num_timestep;
  TBSLAS_DT         = sim_config->dt*0.5; // dt/2 advection + dt/2 diffsuion
  TBSLAS_DIFF_COEFF = 0.01;
  TBSLAS_ALPHA      = (1.0)/TBSLAS_DT/TBSLAS_DIFF_COEFF;
  // =========================================================================
  // PRINT METADATA
  // =========================================================================
  if (!myrank) {
    MetaData_t::Print();
  }
  // =========================================================================
  // RUN
  // =========================================================================
  pvfmm::Profile::Tic("RunAdvectDiff",&comm,true);
  RunAdvectDiff<double>(test,
                        sim_config->tree_num_point_sources,
                        sim_config->tree_num_points_per_octanct,
                        unif,
                        m,
                        sim_config->tree_chebyshev_order,
                        sim_config->tree_max_depth,
                        sim_config->tree_adap,
                        sim_config->tree_tolerance,
                        comm);
  pvfmm::Profile::Toc();

  //Output Profiling results.
  pvfmm::Profile::print(&comm);

  // Shut down MPI
  MPI_Finalize();
  return 0;
}