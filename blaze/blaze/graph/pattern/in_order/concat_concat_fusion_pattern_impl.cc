/*
 * \file concat_concat_fusion_pattern_impl.cc
 * \brief The concat concat fusion helper utility.
 */
#include "blaze/graph/pattern/in_order/concat_concat_fusion_pattern_impl.h"

#include "blaze/common/exception.h"

namespace blaze {

bool ConcatConcatFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                          const std::vector<Node*>& nodes,
                                          Graph* graph) {
  return args[0]->GetSingleArgument<size_t>("axis", 1) ==
      args[1]->GetSingleArgument<size_t>("axis", 1);
}

void ConcatConcatFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                                 std::vector<Node*>& nodes,
                                                 Graph* graph) { }

}  // namespace blaze

