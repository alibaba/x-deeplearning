/*
 * \file parallel_gemm_fusion_pattern_impl.cc 
 * \brief The parallel gemm fusion pattern.
 * Such as: op->GEMM/GEMM/GEMM/.../GEMM
 */
#include "blaze/graph/pattern/in_parallel/parallel_gemm_fusion_pattern_impl.h"

#include <sstream>

#include "blaze/common/blob.h"
#include "blaze/operator/common_helper.h"

namespace blaze {

void ParallelGemmFusionPatternImpl::Init() {
  gemm_node_info_.clear();
  fusion_candidate_.clear();
}

bool ParallelGemmFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                          const std::vector<Node*>& nodes,
                                          Graph* graph) {
  Node* node = nodes[0];
  int node_index = node->idx;
  int a_index = node->GetParentIdx(node->op.input(0));
  if (a_index < 0) return false;

  Node& a_node = graph->node(a_index);  // a_node must not ConstantFill node.
  for (const auto& children_iter : a_node.children) {
    int child = children_iter.first;
    Node& child_node = graph->node(child);
    if (child_node.op.type() != "Gemm") continue;

    // get argument node.
    int b_index = child_node.GetParentIdx(child_node.op.input(1));
    CHECK_TRUE(b_index >= 0, "b_index=", b_index, " iname=", child_node.op.input(1));
    Node& b_node = graph->node(b_index);

    ArgumentHelper child_helper(child_node.op);
    ArgumentHelper b_helper(b_node.op);
    
    GemmNodeInfo gemm_node_info;
    gemm_node_info.idx = child;
    gemm_node_info.transA = child_helper.GetSingleArgument<bool>("transA", false);
    gemm_node_info.transB = child_helper.GetSingleArgument<bool>("transB", false);
    gemm_node_info.alpha = child_helper.GetSingleArgument<float>("alpha", 1.0);
    gemm_node_info.beta = child_helper.GetSingleArgument<float>("beta", 1.0);

    gemm_node_info.shape = b_helper.GetRepeatedArgument<TIndex>("shape");
    gemm_node_info.weight_dtype = b_helper.GetSingleArgument<int>("dtype", kFloat);

    int c_index = -1;
    if (child_node.op.input_size() > 2) {
      c_index = child_node.GetParentIdx(child_node.op.input(2));
      Node& c_node = graph->node(c_index);

      ArgumentHelper c_helper(c_node.op);
      gemm_node_info.bias_shape = c_helper.GetRepeatedArgument<TIndex>("shape");
      gemm_node_info.bias_dtype = c_helper.GetSingleArgument<int>("dtype", kFloat);
    }

    gemm_node_info_.push_back(gemm_node_info);
  }
  GenerateFusionCandidate();
  if (fusion_candidate_.empty()) return false;
  return true;
}
  
void ParallelGemmFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                                 std::vector<Node*>& nodes,
                                                 Graph* graph) {
  for (const auto& candidate : fusion_candidate_) {
    int fusion_op_min_idx = candidate[0];
    std::set<int> fusion_idx_set;
    std::vector<int> fusion_idx_sequence;
    for (const auto& idx : candidate) {
      fusion_idx_set.insert(idx);
      fusion_idx_sequence.push_back(idx);
      if (fusion_op_min_idx > idx) {
        fusion_op_min_idx = idx;
      }
    }
    OperatorDef gemm_op, split_op;

    InitFusedGemmOp(gemm_op, fusion_idx_sequence, graph);
    split_op.add_input(gemm_op.output(0));
    InitFusedSplitOp(split_op, fusion_idx_sequence, graph);

    graph->DeactivateSubgraph(fusion_idx_sequence);

    graph->InsertNode(gemm_op);
    graph->InsertNode(split_op);
  }
}

void ParallelGemmFusionPatternImpl::GenerateFusionCandidate() {
  std::vector<bool> visited(gemm_node_info_.size(), false);
  for (size_t i = 0; i < gemm_node_info_.size(); ++i) {
    if (visited[i]) continue;
    visited[i] = true;
    std::vector<int> candidate;
    candidate.push_back(gemm_node_info_[i].idx);
    for (size_t j = i; j < gemm_node_info_.size(); ++j) {
      if (visited[j]) continue;
      if (CanFusion(i, j)) {
        visited[j] = true;
        candidate.push_back(gemm_node_info_[j].idx);
      }
    }
    if (candidate.size() > 1) {
      fusion_candidate_.push_back(candidate);
    }
  }
}

bool ParallelGemmFusionPatternImpl::CanFusion(int i, int j) {
  const GemmNodeInfo& m = gemm_node_info_[i];
  const GemmNodeInfo& n = gemm_node_info_[j];

  // fusion must be equal datatype
  if (m.weight_dtype != n.weight_dtype) return false;
  if (m.bias_dtype != n.bias_dtype) return false;
  if (m.transA != n.transA) return false;
  if (m.transB != n.transB) return false;
  if (m.alpha != n.alpha) return false;
  if (m.beta != n.beta) return false;
  if (m.shape[0] != n.shape[0] || m.shape[1] != n.shape[1]) return false;
  if (m.shape.size() != 2 || n.shape.size() != 2) return false;
  if (m.bias_shape.size() > 2 || n.bias_shape.size() > 2) return false;
  int diff_m = m.shape.size() - m.bias_shape.size();
  for (int i = 0; i < m.bias_shape.size(); ++i) {
    if (m.shape[i + diff_m] != m.bias_shape[i]) return false;
  }
  int diff_n = n.shape.size() - n.bias_shape.size();
  for (int i = 0; i < n.bias_shape.size(); ++i) {
    if (n.shape[i + diff_n] != n.bias_shape[i]) return false;
  }
  return true;
}

void ParallelGemmFusionPatternImpl::InitFusedGemmOp(OperatorDef& op,
                                                    const std::vector<int>& fusion_idx_sequence,
                                                    Graph* graph) {
  // set name of op
  static int transform_id = 1;
  std::stringstream name;
  name << "fused_parallel_gemm_" << transform_id++;
  op.set_name(name.str());
  // set type of op
  op.set_type("FusedParallelGemm");
  // set argument of op
  for (const auto& item : graph->node(fusion_idx_sequence[0]).op.arg()) {
    Argument* arg = op.add_arg();
    *arg = item;
  }
  Argument* arg = op.add_arg();
  arg->set_name(kOpArgNameParallelNum);
  arg->set_i(fusion_idx_sequence.size());
  // set input of op
  op.add_input(graph->node(fusion_idx_sequence[0]).op.input(0));
  op.add_input(graph->node(fusion_idx_sequence[0]).op.input(1));
  int fusion_bias_idx = -1;
  for (size_t i = 0; i < fusion_idx_sequence.size(); ++i) {
    int idx = fusion_idx_sequence[i];
    if (graph->node(idx).op.input_size() > 2) {  // has bias argument
      op.add_input(graph->node(idx).op.input(2));
      fusion_bias_idx = idx;
      break;
    }
  }
  // init the fused gemm weight.
  InitFusedGemmWeight(fusion_idx_sequence, fusion_idx_sequence[0], graph);
  InitFusedGemmBias(fusion_idx_sequence, fusion_bias_idx, graph);
  // set output of op
  op.add_output(name.str());
}

void ParallelGemmFusionPatternImpl::InitFusedGemmWeight(const std::vector<int>& fusion_idx_sequence,
                                                        int fusion_weight_idx,
                                                        Graph* graph) {
  DeviceOption cpu_device_option;
  Blob weight_blob(cpu_device_option);

  // Step1: Calc fusion weight shape
  const GemmNodeInfo* gni = GetGemmNodeInfo(fusion_idx_sequence[0]);
  weight_blob.set_data_type(DataType2PassDataType(gni->weight_dtype));
  std::vector<TIndex> weight_blob_shape = gni->shape;
  weight_blob_shape[0] *= fusion_idx_sequence.size();
  weight_blob.Reshape(weight_blob_shape);

  // Step2: Copy weight
  size_t offset = 0;
  for (const auto& fusion_idx : fusion_idx_sequence) {
    Node& node = graph->node(fusion_idx);
    int weight_idx = node.GetParentIdx(node.op.input(1));
    BLAZE_CONDITION_THROW(weight_idx >= 0, "weight_idx=", weight_idx);
    Node& weight_node = graph->node(weight_idx);
    // copy data.
    TYPE_SWITCH(weight_blob.data_type(), DType, {
      size_t count = weight_blob.size() / fusion_idx_sequence.size();
      ArgumentHelper temp(weight_node.op);
      std::vector<DType> temp_weight = temp.GetRepeatedArgument<DType>("value");
      memcpy(weight_blob.as<DType>() + offset, temp_weight.data(), count * sizeof(DType));
      offset += count;
    });
  }
  
  // Step3: update weight for ConstantFill Op.
  Node& node = graph->node(fusion_weight_idx);
  int weight_idx = node.GetParentIdx(node.op.input(1));
  Node& weight_node = graph->node(weight_idx);
  // update weight shape.
  ArgumentHelper::SetRepeatedArgument<TIndex>(weight_node.op, "shape", weight_blob_shape);
  // update weight data
  TYPE_SWITCH(weight_blob.data_type(), DType, {
    ArgumentHelper::SetRepeatedArgument<DType>(weight_node.op, "value",
                                               weight_blob.as<DType>(), weight_blob.size());
  });
}

void ParallelGemmFusionPatternImpl::InitFusedGemmBias(const std::vector<int>& fusion_idx_sequence,
                                                      int fusion_bias_idx,
                                                      Graph* graph) {
  if (fusion_bias_idx < 0) return;

  DeviceOption cpu_device_option;
  Blob bias_blob(cpu_device_option);

  // Step1: Calc fusion bias shape.
  std::vector<TIndex> fusion_bias_shape;
  for (const auto fusion_idx : fusion_idx_sequence) {
    const GemmNodeInfo* gni = GetGemmNodeInfo(fusion_idx);
    const std::vector<TIndex>& bias_shape = gni->bias_shape;
    if (bias_shape.size() > fusion_bias_shape.size()) {
      for (int z = 0; z < bias_shape.size() - fusion_bias_shape.size(); ++z) {
        fusion_bias_shape.insert(fusion_bias_shape.begin(), bias_shape[z]);
      }
    }
  }
  const GemmNodeInfo* gni = GetGemmNodeInfo(fusion_idx_sequence[0]);
  bias_blob.set_data_type(DataType2PassDataType(gni->bias_dtype));
  fusion_bias_shape[0] *= fusion_idx_sequence.size();
  bias_blob.Reshape(fusion_bias_shape);

  // Step2: Copy bias.
  size_t offset = 0;
  size_t copy_num = bias_blob.size() / fusion_idx_sequence.size();
  for (const auto fusion_idx : fusion_idx_sequence) {
    Node& node = graph->node(fusion_idx);
    int bias_idx = node.GetParentIdx(node.op.input(2));
    if (bias_idx < 0) {
      // make to be zero
      TYPE_SWITCH(bias_blob.data_type(), DType, {
        memset(bias_blob.as<DType>() + offset, 0, sizeof(DType) * copy_num);              
      });
      offset += copy_num;
    } else {
      Node& bias_node = graph->node(bias_idx);
      ArgumentHelper temp(bias_node.op);
      TYPE_SWITCH(bias_blob.data_type(), DType, {
        std::vector<DType> temp_weight = temp.GetRepeatedArgument<DType>("value");
        for (size_t k = 0; k < copy_num; k += temp_weight.size()) {
          memcpy(bias_blob.as<DType>() + offset, temp_weight.data(),
                 sizeof(DType) * temp_weight.size());
          offset += temp_weight.size();
        }
      });
    }
  }

  // Step3: update weight for ConstantFill op
  Node& node = graph->node(fusion_bias_idx);
  int bias_idx = node.GetParentIdx(node.op.input(2));
  Node& bias_node = graph->node(bias_idx);
  // update bias shape.
  ArgumentHelper::SetRepeatedArgument<TIndex>(bias_node.op, "shape", fusion_bias_shape);
  // update bias data
  TYPE_SWITCH(bias_blob.data_type(), DType, {
    ArgumentHelper::SetRepeatedArgument<DType>(bias_node.op, "value",
                                               bias_blob.as<DType>(), bias_blob.size());
  });
}

const ParallelGemmFusionPatternImpl::GemmNodeInfo* ParallelGemmFusionPatternImpl::GetGemmNodeInfo(int idx) {
  for (size_t k = 0; k < gemm_node_info_.size(); ++k) {
    if (gemm_node_info_[k].idx == idx) {
      return &(gemm_node_info_[k]);
    }
  }
  BLAZE_THROW("The idx: ", idx, " is not found");
  return nullptr;
}

void ParallelGemmFusionPatternImpl::InitFusedSplitOp(OperatorDef& op,
                                                     const std::vector<int>& fusion_idx_sequence,
                                                     Graph* graph) {
  // set name of op
  static int transform_id = 1;
  std::stringstream name;
  name << "fused_parallel_gemm_split_" << transform_id++;
  op.set_name(name.str());
  // set type of op
  op.set_type("Split");
  // set argument of op
  Argument* arg = op.add_arg();
  arg->set_name("axis");
  arg->set_i(0);
  // set output of op
  for (const auto& idx : fusion_idx_sequence) {
    op.add_output(graph->node(idx).op.output(0));
  }
}

}  // namespace blaze

