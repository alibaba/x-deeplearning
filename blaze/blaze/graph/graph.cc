/*
 * \file graph.cc 
 * \brief The graph definition
 */
#include "blaze/graph/graph.h"
#include "blaze/common/log.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace blaze {

int Node::GetParentIdx(const std::string& name) const {
  for (const auto& item : parents) {
    for (const auto& pname : item.second) {
      if (name == pname) return item.first;
    }
  }
  return -1;
}

int Node::GetOutputIdx(const std::string& name) const {
  for (int z = 0; z < op.output_size(); ++z) {
    if (op.output(z) == name) return z;
  }
  return -1;
}

Graph::Graph(const NetDef& net_def, bool inactive_solo_node) : net_def_(net_def) {
  nodes_.clear();
  nodes_.resize(net_def.op_size());

  // Copy over operators
  for (int x = 0; x < net_def.op_size(); ++x) {
    node(x).op = net_def.op(x);
    node(x).idx = x;
  }

  for (int i = 0; i < nodes_.size(); ++i) {
    bool dependency = false;
    for (const std::string& iname : node(i).op.input()) {
      auto it = edge_parent_.find(iname);
      if (it != edge_parent_.end()) {
        int j = it->second;
        node(i).parents[j].push_back(iname);
        node(j).children[i].push_back(iname);
        dependency = true;
      } else {
        external_input_.insert(iname);
      }
    }
    // The node has no dependency
    if (!dependency) not_dependency_idx_.push_back(i);

    for (const std::string& oname : node(i).op.output()) {
      edge_parent_[oname] = i;
    }
  }

  for (int i = nodes_.size() - 1; i >= 0; i--) {
    bool dependency = false;
    for (const std::string& oname : node(i).op.output()) {
      auto it = edge_child_.find(oname);
      if (it == edge_child_.end()) {
        external_output_.insert(oname);
      } else {
        dependency = true;
      }
    }
    // The node is not be dependent by othe node
    if (!dependency) not_be_dependent_idx_.push_back(i);

    for (const std::string& iname : node(i).op.input()) {
      edge_child_[iname].push_back(i);
    }
  }

  // set no input and output node to be inactive, which is solo node
  if (inactive_solo_node) {
    for (int i = 0; i < nodes_.size(); ++i) {
      Node& node = nodes_[i];
      if (node.parents.empty() && node.children.empty()) {
        node.active = false;
      }
    }
  }
}

// If hit the pattern return true.
bool Graph::BFS(const FVisitor& visitor, void* arg) {
  bool ret = false;
  std::queue<Node*> queue;
  for (size_t i = 0; i < this->size(); ++i) {
    Node& node = this->node(i);
    if (!node.parents.empty()) continue;
    queue.push(&node);
  }
  // Set the visited to be false.
  std::vector<bool> visited(this->size(), false);

  size_t count = queue.size(); 
  while (count) {
    while (count--) {
      Node* node = queue.front();
      queue.pop();
      if (visitor && !visited[node->idx]) {
        // If the visitor return true, it means has more pattern matching.
        ret = visitor(*node, arg);
        if (ret) return ret;
        visited[node->idx] = true;
      }
      for (const auto& item: node->children) {
        if (!visited[item.first]) {
          queue.push(&(this->node(item.first)));
        }
      }
    }
    count = queue.size();
  }
  return false;
}

bool Graph::MaxConnectedSearch(const FVisitor& visitor, void* arg) {
  bool ret = false;
  std::unordered_set<std::string> device_has_dependency;
  std::unordered_map<std::string, std::queue<Node*>> device_queue; 
  for (size_t i = 0; i < this->size(); ++i) {
    Node& node = this->node(i);
    auto& cur_device_option = device_option(node);  
    std::string cur_device_str = DeviceStr(cur_device_option);
    if (!node.active) continue;
    if (!node.parents.empty()) {
      for (auto& parent : node.parents) {
        auto& parent_device_option = device_option(this->node(parent.first));
        std::string parent_device_str = DeviceStr(parent_device_option);
        if (cur_device_str != parent_device_str) {
          device_has_dependency.insert(cur_device_str);
          break;
        }
      }
    } else {
      device_queue[cur_device_str].push(&node); 
    }
  }
  std::queue<Node*> queue;
  for (auto& dq : device_queue) {
    if (device_has_dependency.find(dq.first) == device_has_dependency.end()) {
      // no device dependency
      queue.swap(dq.second);
      break;
    }  
  }
  if (queue.empty()) {
    return false;
  } 
  // Set the visited to be false.
  std::vector<bool> visited(this->size(), false);

  size_t count = queue.size(); 
  while (count) {
    while (count--) {
      Node* node = queue.front();
      queue.pop();
      if (!node->active) {
        continue;
      }
      if (visitor && !visited[node->idx]) {
        ret = visitor(*node, arg);
        visited[node->idx] = true;
        if (!ret) {
          // if ret is false, don't continue to proceed
          continue;
        }
      }
      for (const auto& item: node->children) {
        if (!visited[item.first]) {
          queue.push(&(this->node(item.first)));
        }
      }
    }
    count = queue.size();
  }
  return false;
}

int Graph::FindSibling(int node_idx, const std::string& input_name, const std::string& op_type) {
  return FindSibling(node_idx, input_name, op_type, 0);
}

int Graph::FindSibling(int node_idx, const std::string& input_name, const std::string& op_type, int output_size) {
  Node& node = this->node(node_idx);
  int parent_idx = node.GetParentIdx(input_name);
  if (parent_idx < 0) return -1;
  
  Node& pnode = this->node(parent_idx);
  for (const auto& item : pnode.children) {
    Node& cnode = this->node(item.first);
    if (cnode.op.type() == op_type) {
      if (output_size > 0) {
        if (output_size == cnode.op.output_size()) return item.first;
      } else {
        return item.first;
      }
    }
  }
  return -1;
}

void Graph::CheckAndDeactivateNode(int node_idx) {
  Node& n = this->node(node_idx);
  if (n.children.empty()) {
    std::vector<int> subgraph;
    subgraph.push_back(node_idx);
    DeactivateSubgraph(subgraph);
  }
}

int Graph::AddConstantFillNode(Node* node) {
  Node new_node;
  static int gidx = 0;
  gidx++;
  
  std::stringstream ss_output_name;
  ss_output_name << node->op.name() << "_" << gidx << "_extention_output";
  const std::string& output_name = ss_output_name.str();

  std::stringstream ss_name;
  ss_name << node->op.name() << "_" << gidx << "_extention";
  const std::string& name = ss_name.str();

  new_node.op.set_type("ConstantFill");
  new_node.op.set_name(name);
  new_node.op.add_output(output_name);
  new_node.idx = size();
  node->op.add_input(output_name);
  
  // set new node's children
  std::vector<std::string> children = { output_name };
  new_node.children[node->idx] = children;

  // update parents of node
  std::vector<std::string> parents = { output_name };
  node->parents[new_node.idx] = parents;

  this->push_node(new_node);
  return new_node.idx;
}

int Graph::InsertNode(const OperatorDef& op) {
  Node new_node;
  new_node.idx = size();
  new_node.op = op;

  // process input related parents and children
  for (const std::string& iname : op.input()) {
    auto it = edge_parent_.find(iname);
    if (it != edge_parent_.end()) {
      int j = it->second;
      new_node.parents[j].push_back(iname);
      node(j).children[new_node.idx].push_back(iname);
    } else {
      external_input_.insert(iname);
    }
  }
  for (const std::string& oname : op.output()) {
    edge_parent_[oname] = new_node.idx;
  }

  // process output related parents and children
  for (const std::string& oname : op.output()) {
    auto it = edge_child_.find(oname);
    if (it != edge_child_.end()) {
      for (const auto& j : it->second) {
        new_node.children[j].push_back(oname);
        node(j).parents[new_node.idx].push_back(oname);
      }
    }
  }
  for (const std::string& iname : op.input()) {
    edge_child_[iname].push_back(new_node.idx);
  }
  
  this->push_node(new_node);
  return new_node.idx;
}

int Graph::RenameInput(int node_idx, const std::string& input_name,
    const std::string& new_input_name) {
  // first copy the operator def 
  OperatorDef new_op_def = node(node_idx).op; 
  // change the input, while other parts remain the same
  for (int i = 0; i < new_op_def.input_size(); ++i) {
    if (new_op_def.input(i) == input_name) {
      new_op_def.set_input(i, new_input_name); 
    }  
  } 
  // del the old op 
  DeactivateSubgraph({node_idx}); 
  // insert new op
  return InsertNode(new_op_def);
}

int Graph::RemoveInput(int node_idx, const std::string& input_name) {
  const OperatorDef old_op_def = node(node_idx).op;
  // first copy the operator def
  OperatorDef new_op_def = node(node_idx).op;
  // remove input name
  new_op_def.clear_input();
  for (int i = 0; i < old_op_def.input_size(); ++i) {
    if (old_op_def.input(i) == input_name) {
      continue;
    }
    new_op_def.add_input(old_op_def.input(i));
  }
  // del the old op
  DeactivateSubgraph({node_idx});
  // insert new op
  return InsertNode(new_op_def);
}

const std::vector<std::pair<std::string, int>> Graph::GetSubgraphInput(const std::vector<int>& match) {
  return GetSubgraphParameterHelper(true, match);
}

const std::vector<std::pair<std::string, int>> Graph::GetSubgraphOutput(const std::vector<int>& match) {
  return GetSubgraphParameterHelper(false, match);
}

const std::vector<std::pair<std::string, int>> Graph::GetSubgraphParameterHelper(
    bool from_children, const std::vector<int>& match) {
  std::vector<std::pair<std::string, int>> edge_list;
  std::unordered_set<int> match_set(match.begin(), match.end());
  for (int x = 0; x < nodes_.size(); ++x) {
    if (!is_node_active(x)) {
      continue;
    }
    if (!match_set.count(x)) {  // x is not in subgraph
      const auto& list = from_children ? node(x).children : node(x).parents;
      for (const auto& edge : list) {
        int parent = edge.first;
        const auto& names = edge.second;
        if (match_set.count(parent)) {  // but has a parent that is in subgraph
          for (const std::string& name : names) {
            edge_list.push_back({ name, x });
          }
        }
      }
    }
  }
  // return the list in sorted order, to allow binary searching
  std::sort(edge_list.begin(), edge_list.end());
  return edge_list;
}

NetDef Graph::GetNetDef() {
  std::vector<bool> visited(nodes_.size(), false);

  // Copy over all the properties of the NetDef we'are based on.
  NetDef net_def = net_def_;

  net_def.clear_op();

  std::vector<int> unchecked_parent_count;
  std::priority_queue<int, std::vector<int>, std::greater<int>> q;

  for (int i = 0; i < nodes_.size(); ++i) {
    unchecked_parent_count.push_back(node(i).parents.size());
    if (node(i).parents.size() == 0 && is_node_active(i)) {
      q.push(i);
      visited[i] = true;
    }
  }

  while (!q.empty()) {
    int idx = q.top();
    q.pop();
    if (!is_node_active(idx)) {
      continue;
    }
    auto& op = *(net_def.add_op());
    op = node(idx).op;
    for (const auto& edge : node(idx).children) {
      int child = edge.first;
      if (!visited[child] && is_node_active(child)) {
        unchecked_parent_count[child]--;
        if (unchecked_parent_count[child] == 0) {
          q.push(child);
          visited[child] = true;
        }
      }
    }
  }
  return net_def;
}

void Graph::DeactivateSubgraph(const std::vector<int>& subgraph) {
  for (int idx : subgraph) {
    // remove all edges connected to inactive node
    for (const auto& edge : node(idx).parents) {
      int parent = edge.first;
      node(parent).children.erase(idx);
    }
    for (const auto& edge : node(idx).children) {
      int child = edge.first;
      node(child).parents.erase(idx);
    }
    // remove edge_parent/edge_child
    for (const auto& oname : node(idx).op.output()) {
      edge_parent_.erase(oname);
    }
    for (const auto& iname : node(idx).op.input()) {
      std::unordered_map<std::string, std::vector<int>>::iterator iter = edge_child_.find(iname);
      for (std::vector<int>::iterator it = iter->second.begin(); it != iter->second.end(); ++it) {
        if (*it == idx) {
          iter->second.erase(it);
          break;
        }
      }
    }
    // mark flag
    node(idx).active = false;
  }
}

void Graph::DeactivateIndependency(const std::string& oname) {
  // Step1: find hit node.
  int hit_node = -1;
  for (int i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].GetOutputIdx(oname) != -1) {
      hit_node = i;
    }
  }
  if (hit_node < 0) return;

  // Step2: get active node
  std::vector<int> active_nodes;

  std::queue<int> queue;
  queue.push(hit_node);
  while (!queue.empty()) {
    int current = queue.front(); queue.pop();
    active_nodes.push_back(current);
    for (const auto& iter : nodes_[current].parents) {
      queue.push(iter.first);
    }
  }

  // Step3: get inactive node
  std::vector<int> inactive_nodes;
  ComplementNodes(active_nodes, &inactive_nodes);

  DeactivateSubgraph(inactive_nodes);
}

bool Graph::ComplementNodes(const std::vector<int>& src_nodes,
    std::vector<int>* complementary_nodes) const {
  if (complementary_nodes == nullptr) {
    return false;
  }
  // insert nodes into a HashSet 
  std::unordered_set<int> src_set; 
  for (auto node_id : src_nodes) {
    src_set.insert(node_id); 
  } 
  for (auto node : nodes_) {
    if (src_set.find(node.idx) == src_set.end()) {
      // cur node_id is not in src_set
      complementary_nodes->push_back(node.idx); 
    }   
  }
  return true;
}

std::string Graph::DebugStr() {
  std::stringstream ss;
  // Step1: input string
  const std::set<std::string>& external_input = this->external_input();
  ss << "Inputs:" << std::endl;
  for (const auto& name : external_input) {
    ss << "\t" << name << std::endl;
  }

  // Step2: output string
  ss << "Outputs:" << std::endl;
  const std::set<std::string>& external_output = this->external_output();
  for (const auto& name : external_output) {
    ss << "\t" << name << std::endl;
  }

  // Step3: not denpency idx
  ss << "Not dependency idx:" << std::endl;
  const std::vector<int>& not_dependency_idx = this->not_dependency_idx();
  for (const auto& idx : not_dependency_idx) {
    ss << "\t" << idx << std::endl;
  }

  // Step4: not be dependent idx
  ss << "Not be dependent idx:" << std::endl;
  const std::vector<int>& not_be_dependent_idx = this->not_be_dependent_idx();
  for (const auto& idx : not_be_dependent_idx) {
    ss << "\t" << idx << std::endl;
  }

  // Step5: graph structure
  ss << "Nodes:" << std::endl;
  for (size_t k = 0; k < this->size(); ++k) {
    const Node& node = this->node(k);
    
    ss << "\t" << "idx=" << k
        << " name=" << node.op.name()
        << " type=" << node.op.type() << std::endl;
    
    ss << "\t" << "parents:" << std::endl;
    for (const auto& item : node.parents) {
      ss << "\t\t" << item.first << " : ";
      for (const auto& name : item.second) {
        ss << name << " ";
      }
      ss << std::endl;
    }
    if (!node.parents.empty()) ss << std::endl;
    else ss << "\t\t" << "None" << std::endl;

    ss << "\t" << "children:" << std::endl;
    for (const auto& item : node.children) {
      ss << "\t\t" << item.first << " : ";
      for (const auto& name : item.second) {
        ss << name << " ";
      }
      ss << std::endl;
    }
    if (!node.children.empty()) ss << std::endl;
    else ss << "\t\t" << "None" << std::endl;
    
    ss << std::endl;
  }

  return ss.str();
}

}  // namespace blaze
