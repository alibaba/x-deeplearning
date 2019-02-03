/*
 * \file graph.h 
 * \brief The graph definition
 */
#pragma once

#include <map>
#include <set>
#include <string>

#include "blaze/common/common_defines.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

// Graph node
struct Node {
  Node() { }

  // Get the node index, whose oname contains name.
  int GetParentIdx(const std::string& name) const;
  // Get the output name's index in op's output list
  int GetOutputIdx(const std::string& name) const;

  OperatorDef op;
  bool active = true;
  int idx;
 
  // idx = index of child or parent
  // tensor list = a list of string, containing the tensors that connects the
  // nodes.
  std::map<int, std::vector<std::string>> parents;
  std::map<int, std::vector<std::string>> children;
};

// The visitor function, The return function will be the visistor's return.
typedef std::function<bool(Node&, void* arg)> FVisitor;
typedef std::function<bool(std::vector<Node*>&, void* arg)> FVecVisitor;

// Graph
class Graph {
 public:
  const std::vector<std::pair<std::string, int>> GetSubgraphInput(const std::vector<int>& subgraph);
  const std::vector<std::pair<std::string, int>> GetSubgraphOutput(const std::vector<int>& subgraph);

  explicit Graph(const NetDef& net_def, bool inactive_solo_node = true);
  virtual ~Graph() {}

  // Remove inactive nodes.
  NetDef GetNetDef();

  // Deactivate subgraph for removing useless nodes.
  void DeactivateSubgraph(const std::vector<int>& subgraph);
  // Deactivate indepency node of oname
  void DeactivateIndependency(const std::string& oname);

  // get complementary nodes
  bool ComplementNodes(const std::vector<int>& src_nodes,
      std::vector<int>* complementary_nodes) const;

  // Add a ConcatFill node as input for node.
  // Return the added ConstantFill node idx.
  int AddConstantFillNode(Node* node);

  // Insert a new fusion node before idx-th node..
  int InsertNode(const OperatorDef& op);

  // Rename input of specified node     
  int RenameInput(int node_idx, const std::string& input_name,
      const std::string& new_input_name);

  // Remove input of specified node
  int RemoveInput(int node_idx, const std::string& input_name);

  // BFS Search, WARN: if you modify Graph in visitor function, you can not
  // refer the node function.
  bool BFS(const FVisitor& visitor, void* arg);

  // Max connected sub graph Search  
  bool MaxConnectedSearch(const FVisitor& visitor, void* arg);

  // Find sibling of node_idx, the sibling's op type is op_type.
  int FindSibling(int node_idx, const std::string& input_name, const std::string& op_type);
  int FindSibling(int node_idx, const std::string& input_name, const std::string& op_type, int output_size);
  // Check and deactivate node, if the node's children are empty
  void CheckAndDeactivateNode(int node_idx);

  inline const size_t size() const { return nodes_.size(); }
  inline void push_node(const Node& new_node) { nodes_.push_back(new_node); }
  inline void resize_nodes(size_t new_size) { nodes_.resize(new_size); }
  inline const Node& node(size_t idx) const { return nodes_.at(idx); }
  inline Node& node(size_t idx) { return nodes_.at(idx); }
  inline bool is_node_active(size_t idx) { return node(idx).active; }
  inline size_t active_size() const {
    size_t cnt = 0;
    for (auto& node : nodes_) {
      if (node.active) {
        cnt++;
      } 
    }
    return cnt;
  }

  inline const DeviceOption& device_option(const Node& node) const {
    return node.op.has_device_option() ?
      node.op.device_option() : net_def_.device_option(); 
  }

  std::string DeviceStr(const DeviceOption& device_option) {
    return std::to_string(device_option.device_type()) + "/" +
      std::to_string(device_option.device_id()) + "/" +
      std::to_string(device_option.is_pipe());
  }

  // Get debug string
  std::string DebugStr();

  inline const std::set<std::string>& external_input() const { return external_input_; }
  inline const std::set<std::string>& external_output() const { return external_output_; }

  // Return the not denpendency idx
  inline const std::vector<int>& not_dependency_idx() const { return not_dependency_idx_; }
  // Return the not dependent idx
  inline const std::vector<int>& not_be_dependent_idx() const { return not_be_dependent_idx_; }

 private:
  const std::vector<std::pair<std::string, int>> GetSubgraphParameterHelper(
      bool from_children, const std::vector<int>& match);

  NetDef net_def_;

  std::unordered_map<std::string, int> edge_parent_;  // iname producer
  std::unordered_map<std::string, std::vector<int>> edge_child_; // oname consumer

  std::set<std::string> external_input_;
  std::set<std::string> external_output_;

  std::vector<Node> nodes_;
  std::vector<int> not_dependency_idx_;
  std::vector<int> not_be_dependent_idx_;
};

}  // namespace blaze
