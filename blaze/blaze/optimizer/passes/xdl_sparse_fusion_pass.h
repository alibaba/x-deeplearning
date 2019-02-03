/*!
 * \file xdl_sparse_fusion_pass.h
 * \brief The fusion pass for xdl sparse model.
 */
#pragma once

#include "blaze/optimizer/pass.h"
#include "blaze/graph/graph.h"

namespace blaze {

class XdlSparseFusionPass : public Pass {
 public:
  XdlSparseFusionPass& Name(std::string name);
  XdlSparseFusionPass& Type(PassType pass_type);

  virtual NetDef RunPass(const NetDef& net_def);

 protected:
  bool SparseNodeFusionPass(Graph* graph, Node& node, void* arg);
  // eliminate unique node, rename or remove descendant node input
  bool UniqueNodeEliminatePass(Graph* graph, Node& node, void* arg);
  // eliminate ps pull node, remove descendant node input
  bool PsPullNodeEliminatePass(Graph* graph, Node& node, void* arg);
  // fuse ps sparse pull node and ksum node to embedding node
  bool PsSparsePullNodeReplacePass(Graph* graph, Node& node, void* arg);
  // generate embedding args
  void GenEmbeddingArgs(const Node& node, const Node& child_node, OperatorDef* op_def);
};

}  // namespace blaze


