/*
 * \file fusion_pattern.h 
 * \brief The graph fusion pattern.
 * The subgraph match method for op fusion optimization.
 */
#pragma once

#include <unordered_map>

#include "blaze/common/proto_helper.h"
#include "blaze/graph/graph.h"
#include "blaze/proto/fusion_pattern.pb.h"

namespace blaze {

enum FusionPatternType {
  kInOrder = 0,        // In order ops fusion
  kInParallel,         // Batched operators in parallel
};

class FusionPattern;

// The fusion implementation abstract class,
// is mainly used in FMatch function for diffrent fusion pattern logic.
class FusionPatternImpl {
 public:
  // Init fusion pattern impl.
  virtual void Init() { }
  // If the fusion pattern is matched, Return true else false.
  virtual bool Match(const std::vector<ArgumentHelper*>& args,
                     const std::vector<Node*>& nodes,
                     Graph* graph) { return false; }
  // Do graph rewrite.
  virtual void GraphRewrite(const std::vector<ArgumentHelper*>& args,
                            std::vector<Node*>& nodes,
                            Graph* graph) { }

  FusionPattern* pattern;
};

// Fusion pattern node.
struct FPNode {
  FPNode() { }

  FusionPatternNodeDef node;
  int idx;

  // idx = index of child or parent
  std::map<int, std::vector<std::string>> parents;
  std::map<int, std::vector<std::string>> children;
};

// Fusion pattern graph.
struct FPGraph {
  explicit FPGraph(const FusionPatternDef& fusion_pattern_def);

  std::vector<FPNode> nodes;
  std::unordered_map<std::string, int> edge_parents;  // iname producer

  std::vector<int> input, output;
};

// Do graph matching
bool GraphMatch(const Graph& graph, int idx, const FPGraph& fp_graph,
                std::vector<int>* match_idx);

// The fusion pattern definition, which must be Fully Connected Graph.
//
// for example: we define Slice,Slice -> Slice
// 
// REGISTER_FUSION_PATTERN(SliceSlice)
//    .Type(kInOrder)
//    .AddOpNode("slice0", "Slice")
//    .AddOpNode("slice1", "Slice")
//    .AddConnect("slice0", "slice1")
//    .FusionOpName("Slice")
//    .SetFusionPatternImpl(new SliceSliceFusionPatternImpl())
//    .Init();
//
class FusionPattern {
 public:
  virtual ~FusionPattern();
  FusionPattern& Name(std::string name);
  inline const std::string& name() const { return name_; }

  FusionPattern& Type(FusionPatternType type);
  inline const FusionPatternType type() const { return type_; }

  FusionPattern& FusionOpName(std::string fusion_op_name);
  inline const std::string& fusion_op_name() const { return fusion_op_name_; }

  FusionPattern& SetFusionPatternImpl(FusionPatternImpl* fusion_pattern_impl);
  FusionPatternImpl* fusion_pattern_impl() { return fusion_pattern_impl_; }
  
  FusionPattern& Option(const std::string option_name, bool flag);
  bool option(const std::string& option_name) {
    auto ret = option_.find(option_name);
    if (ret == option_.end()) return false;
    return ret->second;
  }

  // The following two functions are used for PatternGraph generation.
  // The connection relationship is not used.
  FusionPattern& AddOpNode(const std::string& node_name, const std::string& op_name);
  FusionPattern& AddConnect(const std::string& edge_from, const std::string& edge_to);

  // Add op node of Fusion Pattern Graph
  // NOTE: Should add according to toplology order.
  FusionPattern& AddOpNode(const std::string& node_name, const std::string& op_name,
                           const std::vector<std::string>& iname,
                           const std::vector<std::string>& oname);

  // Initialize the pattern, such as: Fusion Pattern Graph Creation.
  FusionPattern& Init();

  // Do fusion pattern matching
  bool Match(Graph* graph);

  // Return input and output node names of pattern graph
  const std::vector<int>& input() const { return fp_graph_->input; }
  const std::vector<int>& output() const { return fp_graph_->output; }

  // Return the pattern def
  const FusionPatternDef& pattern_def() const { return pattern_def_; }

 protected:
  // Check valid
  void CheckValid();

  // Check the validity of inorder
  void CheckInOrderValid();
  // Check the validity of inparallel
  void CheckInParallelValid();

  // The fusion op name
  std::string name_;
  // The fusion type
  FusionPatternType type_;
  // The fusion op name
  std::string fusion_op_name_;

  FusionPatternImpl* fusion_pattern_impl_ = nullptr;
  // The option
  std::map<std::string, bool> option_;

  // The fusion pattern def
  FusionPatternDef pattern_def_;
  // The input and output
  std::shared_ptr<FPGraph> fp_graph_;
};

// Pattern register
struct FusionPatternRegisterer {
  static FusionPatternRegisterer* Get() {
    static std::shared_ptr<FusionPatternRegisterer> inst(new FusionPatternRegisterer());
    return inst.get();
  }
  FusionPattern& Register() {
    size_t idx = pattern.size();
    pattern.resize(idx + 1);
    pattern[idx].reset(new FusionPattern());
    return *(pattern[idx].get());
  }

  std::vector<std::shared_ptr<FusionPattern>> pattern;
};
#define REGISTER_FUSION_PATTERN(name)                             \
    static FusionPattern& ANONYMOUS_VARIABLE(name) =              \
      FusionPatternRegisterer::Get()->Register().Name(#name)

// Transform using patten
NetDef FusionPatternPass(const NetDef& net_def);

using FMatch=std::function<bool(FusionPattern*, Graph*)>;
extern std::unordered_map<int, FMatch> g_fusion_fmatch_map;

// Match function register
struct FMatchRegisterer {
  FMatchRegisterer(FusionPatternType type, FMatch match) {
    g_fusion_fmatch_map[type] = match;
  }
};

#define REGISTER_FMATCH(type, fmatch)                       \
    FMatchRegisterer ANONYMOUS_VARIABLE(type)(type, fmatch)

}  // namespace blaze

