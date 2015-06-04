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

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <string>

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
#include <tree/semilag_tree.h>
#include <tree/utils_tree.h>
#include <tree/field_set_functor.h>

typedef pvfmm::Cheb_Node<double> Node_t;
typedef pvfmm::MPI_Tree<Node_t> Tree_t;

typedef tbslas::MetaData<std::string,
                         std::string,
                         std::string> MetaData_t;
double tcurr = 0;

void (*fn_vel)(const double* , int , double*)=NULL;
void (*fn_con)(const double* , int , double*)=NULL;

template<typename real_t, int sdim>
void
get_velocity_field_atT(const real_t* points_pos,
                       int num_points,
                       real_t* points_values) {
  tbslas::get_vorticity_field_atT<real_t,sdim>(points_pos,
                                               num_points,
                                               tcurr,
                                               points_values);
}

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm=MPI_COMM_WORLD;
  int myrank;
  MPI_Comm_rank(comm, &myrank);

  parse_command_line_options(argc, argv);

  int   test = strtoul(commandline_option(argc, argv, "-test",     "1", false,
                                          "-test <int> = (1)    : 1) Gaussian profile 2) Zalesak disk"),NULL,10);

  {
    tbslas::SimConfig* sim_config       = tbslas::SimConfigSingleton::Instance();
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
    pvfmm::BoundaryType bc;
    switch(test) {
      case 1:
        fn_vel = get_velocity_field_atT<double,3>;
        fn_con = get_gaussian_field_cylinder_atT<double,3>;
        bc = pvfmm::FreeSpace;
        break;
      case 2:
        fn_vel = get_velocity_field_atT<double,3>;
        fn_con = get_slotted_cylinder_atT<double,3>;
        bc = pvfmm::FreeSpace;
        break;
      case 3:
        fn_vel = tbslas::get_vel_field_hom<double,3>;
        fn_con = get_gaussian_field_cylinder_atT<double,3>;
        bc = pvfmm::Periodic;
        break;
    }
    // =========================================================================
    // SIMULATION PARAMETERS
    // =========================================================================
    sim_config->vtk_filename_prefix     = "advection";
    sim_config->vtk_filename_variable   = "conc";
    sim_config->bc = bc;
    char out_name_buffer[300];
    // =========================================================================
    // INIT THE VELOCITY TREE
    // =========================================================================    
    // Tree_t tvel_curr(comm);
    // tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
    //                               sim_config->tree_num_points_per_octanct,
    //                               sim_config->tree_chebyshev_order,
    //                               sim_config->tree_max_depth,
    //                               sim_config->tree_adap,
    //                               sim_config->tree_tolerance,
    //                               comm,
    //                               fn_vel,
    //                               3,
    //                               tvel_curr);

    // if (sim_config->vtk_save) {
    //   snprintf(out_name_buffer,
    //            sizeof(out_name_buffer),
    //            sim_config->vtk_filename_format.c_str(),
    //            tbslas::get_result_dir().c_str(),
    //            sim_config->vtk_filename_prefix.c_str(),
    //            "vel",
    //            0);
    //   tvel_curr.Write2File(out_name_buffer, sim_config->vtk_order);
    // }

    // =========================================================================
    // INIT THE CONCENTRATION TREE
    // =========================================================================
    tcurr = 0;
    Tree_t tconc_curr(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_con,
                                  1,
                                  tconc_curr);
    double in_al2,in_rl2,in_ali,in_rli;
    CheckChebOutput<Tree_t>(&tconc_curr,
                            fn_con,
                            1,
                            in_al2,in_rl2,in_ali,in_rli,
                            std::string("Input"));
    typedef tbslas::Reporter<double> Rep;
    if(!myrank) {
      Rep::AddData("TOL", sim_config->tree_tolerance);
      Rep::AddData("ChbOrder", sim_config->tree_chebyshev_order);
      Rep::AddData("MaxDEPTH", sim_config->tree_max_depth);


      Rep::AddData("DT", sim_config->dt);
      Rep::AddData("TN", sim_config->total_num_timestep);

      Rep::AddData("InAL2", in_al2);
      Rep::AddData("InRL2", in_rl2);
      Rep::AddData("InALINF", in_ali);
      Rep::AddData("InRLINF", in_rli);
    }

    // =========================================================================
    // INIT THE VELOCITY TREES
    // =========================================================================
    std::vector<double>  vel_field_times;
    std::vector<Tree_t*> vel_field_elems;

    // =========================================================================
    // INIT THE TREE 1
    // =========================================================================
    tcurr = -sim_config->dt;
    Tree_t tree1(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel,
                                  3,
                                  tree1);

    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "vel",
               1);
      tree1.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    vel_field_times.push_back(tcurr);
    vel_field_elems.push_back(&tree1);

    // =========================================================================
    // INIT THE TREE 2
    // =========================================================================
    tcurr += sim_config->dt;
    Tree_t tree2(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel,
                                  3,
                                  tree2);
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "vel",
               2);
      tree2.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    vel_field_times.push_back(tcurr);
    vel_field_elems.push_back(&tree2);

    // =========================================================================
    // INIT THE TREE 3
    // =========================================================================
    tcurr += sim_config->dt;
    Tree_t tree3(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel,
                                  3,
                                  tree3);
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "vel",
               3);
      tree3.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    vel_field_times.push_back(tcurr);
    vel_field_elems.push_back(&tree3);

    // =========================================================================
    // INIT THE TREE 4
    // =========================================================================
    tcurr += sim_config->dt;
    Tree_t tree4(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_vel,
                                  3,
                                  tree4);
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "vel",
               4);
      tree4.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    vel_field_times.push_back(tcurr);
    vel_field_elems.push_back(&tree4);

    tbslas::FieldSetFunctor<double, Tree_t> vel_field_functor(vel_field_elems, vel_field_times);

    // =========================================================================
    // RUN
    // =========================================================================
    bool adaptive = true;
    // set the input_fn to NULL -> needed for adaptive refinement
    std::vector<Node_t*>  ncurr_list = tconc_curr.GetNodeList();
    for(int i = 0; i < ncurr_list.size(); i++) {
      ncurr_list[i]->input_fn = (void (*)(const double* , int , double*))NULL;
    }

    for (int tstep = 1; tstep < sim_config->total_num_timestep+1; tstep++) {
      if(!myrank) {
        printf("============================\n");
        printf("dt: %f tstep: %d \n", sim_config->dt, tstep);
        printf("============================\n");
      }
      // tbslas::NodeFieldFunctor<double,Tree_t> vel_field_functor(&tvel_curr);
      tbslas::SolveSemilagInSitu<Tree_t>(vel_field_functor,
                                         tconc_curr,
                                         tstep,
                                         sim_config->dt,
                                         sim_config->num_rk_step);
      if (adaptive) {
        // refine the tree according to the computed values
        pvfmm::Profile::Tic("RefineTree", &sim_config->comm, false, 5);
        tconc_curr.RefineTree();
        pvfmm::Profile::Toc();
      }
      tcurr += sim_config->dt;
      // save current time step data
      if (sim_config->vtk_save) {
        tconc_curr.Write2File(tbslas::GetVTKFileName(tstep, sim_config->vtk_filename_variable).c_str(), sim_config->vtk_order);
      }
    }
    // =========================================================================
    // COMPUTE ERROR
    // =========================================================================
    tcurr = sim_config->total_num_timestep*sim_config->dt;
    double al2,rl2,ali,rli;
    CheckChebOutput<Tree_t>(&tconc_curr,
                            fn_con,
                            1,
                            al2,rl2,ali,rli,
                            std::string("Output"));
    int num_leaves = tbslas::CountNumLeafNodes(tconc_curr);
    // =========================================================================
    // REPORT RESULTS
    // =========================================================================
    if(!myrank) {
      Rep::AddData("OutAL2", al2);
      Rep::AddData("OutRL2", rl2);
      Rep::AddData("OutALINF", ali);
      Rep::AddData("OutRLINF", rli);
      Rep::Report();
    }
    //Output Profiling results.
    pvfmm::Profile::print(&comm);
  }

  // Shut down MPI
  MPI_Finalize();
  return 0;
}
