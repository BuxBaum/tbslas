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

#ifndef SRC_TREE_FIELD_SET_FUNCTOR_H_
#define SRC_TREE_FIELD_SET_FUNCTOR_H_

#include <vector>

#include <pvfmm_common.hpp>
#include <cheb_node.hpp>
#include <profile.hpp>

#include "utils/common.h"
#include "semilag/cubic_interp_policy.h"

namespace tbslas {

template<typename Real_t,
         class Tree_t>
class FieldSetFunctor {

 public:
  explicit FieldSetFunctor(std::vector<Tree_t*> field_set_elems,
                           std::vector<Real_t>  field_set_times):
      field_set_elems_(field_set_elems),
      field_set_times_(field_set_times) {
    typedef typename Tree_t::Node_t Node_t;
    char out_name_buffer[300];

    tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();
    //////////////////////////////////////////////////////////////////////
    // GET THE TREES PARAMETERS
    //////////////////////////////////////////////////////////////////////
    Node_t* n_curr = field_set_elems[0]->PostorderFirst();
    while (n_curr != NULL) {
      if(!n_curr->IsGhost() && n_curr->IsLeaf())
        break;
      n_curr = field_set_elems[0]->PostorderNxt(n_curr);
    }
    data_dof_ = n_curr->DataDOF();
    cheb_deg_ = n_curr->ChebDeg();
    sdim_     = field_set_elems[0]->Dim();

    // //////////////////////////////////////////////////////////////////////
    // // CREATE THE MERGED TREES
    // //////////////////////////////////////////////////////////////////////
    // // TODO: clone tree instead of merging several times
    // for (int i = 0; i < field_set_elems.size(); i++) {
    //   Tree_t* merged_tree = new Tree_t(sim_config->comm);
    //   tbslas::ConstructTree<Tree_t>(sim_config->tree_num_point_sources,
    //                                 sim_config->tree_num_points_per_octanct,
    //                                 cheb_deg_,
    //                                 sim_config->tree_max_depth,
    //                                 sim_config->tree_adap,
    //                                 sim_config->tree_tolerance,
    //                                 sim_config->comm,
    //                                 tbslas::get_linear_field<Real_t,3>,
    //                                 data_dof_,
    //                                 *merged_tree);
    //   for (int i = 0; i < field_set_elems.size(); i++) {
    //     tbslas::SyncTreeRefinement(*(field_set_elems[i]), *merged_tree);
    //   }
    //   field_merged_.push_back(merged_tree);
    // }

    // //////////////////////////////////////////////////////////////////////
    // // INTERPOLATE THE MERGED TREE VALUES
    // //////////////////////////////////////////////////////////////////////
    // // COLLECT THE MERGED TREE POINTS
    // std::vector<Real_t> merged_tree_points_pos;
    // tbslas::CollectChebTreeGridPoints(*(field_merged_[0]), merged_tree_points_pos);

    // // EVALUATE TREE VALUES AT THE MERGED TREE POINTS
    // int merged_tree_num_points = merged_tree_points_pos.size()/3;
    // for (int i = 0 ; i < field_set_elems.size(); i++) {
    //   std::vector<Real_t> tree_points_val(merged_tree_num_points*data_dof_);
    //   tbslas::NodeFieldFunctor<Real_t,Tree_t> tree_func(field_set_elems[i]);
    //   tree_func(merged_tree_points_pos.data(),
    //              merged_tree_num_points,
    //             tree_points_val.data());

    //   Node_t* n_next = field_merged_[i]->PostorderFirst();
    //   while (n_next != NULL) {
    //     if(!n_next->IsGhost() && n_next->IsLeaf()) break;
    //     n_next = field_merged_[i]->PostorderNxt(n_next);
    //   }

    //   int num_points_per_node = (cheb_deg_+1)*(cheb_deg_+1)*(cheb_deg_+1);
    //   int tree_next_node_counter = 0;
    //   while (n_next != NULL) {
    //     if (n_next->IsLeaf() && !n_next->IsGhost()) {
    //       pvfmm::cheb_approx<Real_t, Real_t>(
    //           &tree_points_val[tree_next_node_counter*num_points_per_node*data_dof_],
    //           cheb_deg_,
    //           data_dof_,
    //           &(n_next->ChebData()[0])
    //                                          );
    //       tree_next_node_counter++;
    //     }
    //     n_next = field_merged_[i]->PostorderNxt(n_next);
    //   }

    //   if (sim_config->vtk_save) {
    //     snprintf(out_name_buffer,
    //              sizeof(out_name_buffer),
    //              sim_config->vtk_filename_format.c_str(),
    //              tbslas::get_result_dir().c_str(),
    //              sim_config->vtk_filename_prefix.c_str(),
    //              "merged_tree",
    //              i);
    //     field_merged_[i]->Write2File(out_name_buffer, sim_config->vtk_order);
    //   }
    // }
  }

  virtual ~FieldSetFunctor() {
  }

  void operator () (const Real_t* points_pos,
                    int num_points,
                    Real_t time,
                    Real_t* out) {
    tbslas::SimConfig* sim_config = tbslas::SimConfigSingleton::Instance();

    std::vector<std::vector<Real_t>*> field_set_points_val;
    for (int i = 0 ; i < field_set_elems_.size(); i++) {
      std::vector<Real_t>* pPoints_val = new std::vector<Real_t>(num_points*data_dof_);
      tbslas::NodeFieldFunctor<Real_t,Tree_t> tree_func(field_set_elems_[i]);
      tree_func(points_pos, num_points, (*pPoints_val).data());
      field_set_points_val.push_back(pPoints_val);
    }

    // INTERPOLATE IN TIME
    for (int i = 0; i < field_set_points_val[0]->size(); i++) {
      Real_t grid_vals[4] = {(*field_set_points_val[0])[i],
                             (*field_set_points_val[1])[i],
                             (*field_set_points_val[2])[i],
                             (*field_set_points_val[3])[i]};
      out[i] = tbslas::CubicInterpPolicy<Real_t>::InterpCubic1D
          (time, field_set_times_.data(), grid_vals);
    }
  }

 private:
  std::vector<Real_t>  field_set_times_;
  std::vector<Tree_t*> field_set_elems_;
  std::vector<Tree_t*> field_merged_;
  int data_dof_;
  int cheb_deg_;
  int sdim_;
};

}      // namespace tbslas
#endif  // SRC_TREE_FIELD_SET_FUNCTOR_H_
