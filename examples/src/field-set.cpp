// *************************************************************************
// Copyright (C) 2015 by Arash Bakhtiari
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

void (*fn_1)(const double* , int , double*)=NULL;
void (*fn_2)(const double* , int , double*)=NULL;

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
        fn_1 = tbslas::get_linear_field<double,3>;//tbslas::get_vorticity_field<double,3>;
        fn_2 = get_gaussian_field_cylinder_atT<double,3>;
        bc = pvfmm::FreeSpace;
        break;
    }
    // =========================================================================
    // SIMULATION PARAMETERS
    // =========================================================================
    sim_config->vtk_filename_prefix     = "merge";
    sim_config->vtk_filename_variable   = "conc";
    sim_config->bc = bc;
    double DT = 0.3925;
    std::vector<double>  tree_set_times;
    std::vector<Tree_t*> tree_set_elems;

    // =========================================================================
    // INIT THE TREE 1
    // =========================================================================
    tcurr = 0;
    Tree_t tree1(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_2,
                                  1,
                                  tree1);
    char out_name_buffer[300];
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "tree",
               1);
      tree1.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    tree_set_times.push_back(tcurr);
    tree_set_elems.push_back(&tree1);

    // =========================================================================
    // INIT THE TREE 2
    // =========================================================================
    tcurr += DT;
    Tree_t tree2(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_2,
                                  1,
                                  tree2);
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "tree",
               2);
      tree2.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    tree_set_times.push_back(tcurr);
    tree_set_elems.push_back(&tree2);

    // =========================================================================
    // INIT THE TREE 3
    // =========================================================================
    tcurr += DT;
    Tree_t tree3(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_2,
                                  1,
                                  tree3);
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "tree",
               3);
      tree3.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    tree_set_times.push_back(tcurr);
    tree_set_elems.push_back(&tree3);

    // =========================================================================
    // INIT THE TREE 4
    // =========================================================================
    tcurr += DT;
    Tree_t tree4(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_2,
                                  1,
                                  tree4);
    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "tree",
               4);
      tree4.Write2File(out_name_buffer, sim_config->vtk_order);
    }
    tree_set_times.push_back(tcurr);
    tree_set_elems.push_back(&tree4);

    tbslas::FieldSetFunctor<double, Tree_t> tree_set_functor(tree_set_elems, tree_set_times);

    // =========================================================================
    // MERGE
    // =========================================================================
    Tree_t merged_tree(comm);
    tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
                                  sim_config->tree_num_points_per_octanct,
                                  sim_config->tree_chebyshev_order,
                                  sim_config->tree_max_depth,
                                  sim_config->tree_adap,
                                  sim_config->tree_tolerance,
                                  comm,
                                  fn_1,
                                  1,
                                  merged_tree);
    tbslas::SyncTreeRefinement(tree4, merged_tree);
    // tbslas::SyncTreeRefinement(tree2, merged_tree);
    // // =========================================================================
    // //  LINEAR COMNINATION OF TREE VALUES
    // // =========================================================================
    // GET THE TREE PARAMETERS FROM CURRENT TREE
    Node_t* n_curr = merged_tree.PostorderFirst();
    while (n_curr != NULL) {
      if(!n_curr->IsGhost() && n_curr->IsLeaf())
        break;
      n_curr = merged_tree.PostorderNxt(n_curr);
    }
    int data_dof = n_curr->DataDOF();
    int cheb_deg = n_curr->ChebDeg();
    int sdim     = merged_tree.Dim();

    // COLLECT THE MERGED TREE POINTS
    std::vector<double> merged_tree_points_pos;
    tbslas::CollectChebTreeGridPoints(merged_tree, merged_tree_points_pos);

    // EVALUATE TREE 1 AND TREE 2 VALUES AT THE MERGED TREE POINTS
    int merged_tree_num_points = merged_tree_points_pos.size()/3;

    std::vector<double> merged_tree_points_val(merged_tree_num_points*data_dof);
    tree_set_functor(merged_tree_points_pos.data(),
                     merged_tree_num_points,
                     1.5*DT,
                     merged_tree_points_val.data());

    Node_t* n_next = merged_tree.PostorderFirst();
    while (n_next != NULL) {
      if(!n_next->IsGhost() && n_next->IsLeaf()) break;
      n_next = merged_tree.PostorderNxt(n_next);
    }

    int num_points_per_node = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1);
    int tree_next_node_counter = 0;
    while (n_next != NULL) {
      if (n_next->IsLeaf() && !n_next->IsGhost()) {
        pvfmm::cheb_approx<double, double>(
            &merged_tree_points_val[tree_next_node_counter*num_points_per_node*data_dof],
            cheb_deg,
            data_dof,
            &(n_next->ChebData()[0])
                                           );
        tree_next_node_counter++;
      }
      n_next = merged_tree.PostorderNxt(n_next);
    }

    if (sim_config->vtk_save) {
      snprintf(out_name_buffer,
               sizeof(out_name_buffer),
               sim_config->vtk_filename_format.c_str(),
               tbslas::get_result_dir().c_str(),
               sim_config->vtk_filename_prefix.c_str(),
               "merged_tree",
               0);
      merged_tree.Write2File(out_name_buffer, sim_config->vtk_order);
    }

    //Output Profiling results.
    pvfmm::Profile::print(&comm);
  }

  // Shut down MPI
  MPI_Finalize();
  return 0;
}
