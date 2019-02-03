/*
 * \file slice_slice_fusion_pattern_impl.cc
 * \brief The slice slice fusion pattern.
 */
#include "blaze/graph/pattern/in_order/slice_slice_fusion_pattern_impl.h"

#include "blaze/common/exception.h"

#include "blaze/operator/common_helper.h"

namespace blaze {

bool SliceSliceFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                        const std::vector<Node*>& nodes,
                                        Graph* graph) {
  size_t axis0 = CommonHelper::GetSliceAxis(args[0]);
  size_t axis1 = CommonHelper::GetSliceAxis(args[1]);
  return axis0 == axis1;
}

void SliceSliceFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                               std::vector<Node*>& nodes,
                                               Graph* graph) {
  // Process fused arguments
  size_t axis0 = CommonHelper::GetSliceAxis(args[0]);

  OperatorDef& op = nodes[nodes.size() - 1]->op;
  ArgumentHelper::ClearArgument(op);
  ArgumentHelper::SetSingleArgument<size_t>(op, "axis", axis0);

  size_t start1 = CommonHelper::GetSliceStart(args[0]);
  size_t end1 = CommonHelper::GetSliceEnd(args[0]);
  size_t start2 = CommonHelper::GetSliceStart(args[1]);
  size_t end2 = CommonHelper::GetSliceEnd(args[1]);
  BLAZE_CONDITION_THROW(start1 + end2 <= end1, "start1=", start1,
                        " end2=", end2, " end1=", end1);

  ArgumentHelper::SetSingleArgument<size_t>(op, "start", start1 + start2);
  ArgumentHelper::SetSingleArgument<size_t>(op, "end", start1 + end2);
}

}  // namespace blaze

