/*!
 * \file gemm_pass.h
 * \brief The gemm pass for Gemm op's param reinit
 */
#pragma once

#include "blaze/optimizer/pass.h"
#include "blaze/graph/graph.h"

namespace blaze {

class GemmPass : public Pass {
 public:
  GemmPass& Name(std::string name);
  GemmPass& Type(PassType pass_type);

  virtual NetDef RunPass(const NetDef& net_def);

 protected:
  bool RunGemmPass(Graph* graph, Node& node, void* arg);

  void ProcessGemmParam(Node& node, int min_size);
};

}  // namespace blaze
