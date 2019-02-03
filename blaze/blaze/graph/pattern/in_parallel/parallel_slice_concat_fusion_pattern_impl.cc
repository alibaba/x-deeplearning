/*
 * \file parallel_slice_concat_fusion_pattern_impl.cc 
 * \brief The parallel slice concat fusion pattern implementation
 * Such as: slice, slice, ... -> Concat -> SliceConcat
 */
#include "blaze/graph/pattern/in_parallel/parallel_slice_concat_fusion_pattern_impl.h"

#include "blaze/operator/common_helper.h"

namespace blaze {

void ParallelSliceConcatFusionPatternImpl::Init() {
  slice_idx_.clear();
  starts_.clear();
  ends_.clear();
}

bool ParallelSliceConcatFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                                 const std::vector<Node*>& nodes,
                                                 Graph* graph) {
  Node *node0 = nodes[0], *node1 = nodes[1];
  BLAZE_CONDITION_THROW(node0->op.input_size() == 1, "node0->op.input_size()=", node0->op.input_size());
  int s0_parent_id = node0->GetParentIdx(node0->op.input(0));

  // node0 is Slice, node1 is Concat
  // if node1's parents are all Slice node has only one descendant,
  // and the axis of All Slice nodes are
  // equal, the Slice nodes and Concat node can be fused.
  auto axis0 = CommonHelper::GetSliceAxis(args[0]); 
  for (const auto& iname : node1->op.input()) {
    int id = node1->GetParentIdx(iname);
    Node& slice_node = graph->node(id);
    if (slice_node.children.size() != 1) return false;
    if (slice_node.op.type() != "Slice") return false;
    ArgumentHelper argument_helper(slice_node.op);
    auto axis = CommonHelper::GetSliceAxis(&argument_helper);
    if (axis != axis0) return false;
    BLAZE_CONDITION_THROW(slice_node.op.input_size() == 1, "slice_node.op.input_size()=", slice_node.op.input_size());
    int s_parent_id = slice_node.GetParentIdx(slice_node.op.input(0));
    if (s0_parent_id != s_parent_id) return false;

    slice_idx_.push_back(id);
    starts_.push_back(CommonHelper::GetSliceStart(&argument_helper));
    ends_.push_back(CommonHelper::GetSliceEnd(&argument_helper));
  }
  return true;
}

void ParallelSliceConcatFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                                        std::vector<Node*>& nodes,
                                                        Graph* graph) {
  Node *node0 = nodes[0], *node1 = nodes[1];
  // node1 is Concat.
  std::vector<int> to_be_deleted = slice_idx_;
  to_be_deleted.push_back(node1->idx);

  // prepare the inserted node
  OperatorDef new_op;
  new_op.set_type(this->pattern->fusion_op_name());
  new_op.set_name(node1->op.name());
  for (const auto& oname : node1->op.output()) {
    new_op.add_output(oname);
  }
  new_op.add_input(node0->op.input(0));

  // set attributes
  ArgumentHelper::SetSingleArgument<size_t>(new_op, "concat_axis", args[1]->GetSingleArgument<size_t>("axis", 0));
  ArgumentHelper::SetSingleArgument<size_t>(new_op, "slice_axis", CommonHelper::GetSliceAxis(args[0]));
  ArgumentHelper::SetRepeatedArgument<size_t>(new_op, "start", starts_);
  ArgumentHelper::SetRepeatedArgument<size_t>(new_op, "end", ends_);

  graph->DeactivateSubgraph(to_be_deleted);
  graph->InsertNode(new_op);
}

}  // namespace blaze
