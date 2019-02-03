/*
 * \file parallel_slice_concat_fusion_pattern_impl.h 
 * \brief The parallel slice concat fusion pattern
 * Such as: slice, slice, ... -> Concat -> SliceConcat
 */
#pragma once

#include "blaze/common/blob.h"
#include "blaze/graph/fusion_pattern.h"

namespace blaze {

class ParallelSliceConcatFusionPatternImpl : public FusionPatternImpl {
 public:
  virtual void Init() override;
  // The pattern is matched.
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) override;

  // Do graph rewrite
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) override;

 protected:
  std::vector<int> slice_idx_;
  std::vector<size_t> starts_, ends_;
};

}  // namespace blaze
