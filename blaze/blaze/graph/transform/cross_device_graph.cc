/*
 * \file cross_device_graph.cc 
 * \brief The definition of cross device graph 
 */

#include "blaze/graph/transform/cross_device_graph.h"
#include "blaze/common/log.h"

using std::string;
using std::vector;

namespace blaze {

CrossDeviceGraph::CrossDeviceGraph(const NetDef& net_def) :
    Graph(net_def) {
  std::vector<BoundaryNode> boundaries;
  FindBoundaries(&boundaries);
  LOG_DEBUG("Boundary nodes size:%u", boundaries.size());
  InsertBridgeNodes(boundaries); 
}

void CrossDeviceGraph::InsertBridgeNodes(const std::vector<BoundaryNode>& boundaries) {
  // init copy nodes
  int idx = 0;
  string base_name("bridge_"); 
  for (auto& boundary : boundaries) {
    OperatorDef op_def;
    DefineBridgeNode(base_name + std::to_string(idx),
        {boundary.input_name}, device_option(node(boundary.idx)), &op_def);
    InsertNode(op_def);
    // Rename the input of boundary op
    RenameInput(boundary.idx, boundary.input_name, op_def.output(0));
    idx++;
  }
}

void CrossDeviceGraph::DefineBridgeNode(const std::string& op_name,
    const std::set<std::string>& input_names,
    const DeviceOption& device_option,
    OperatorDef* op_def) {
  // set name of op
  op_def->set_name(op_name); 
  // set type of op
  op_def->set_type("Bridge");
  // set device type of op 
  op_def->mutable_device_option()->set_device_type(device_option.device_type());
  // set device id of op 
  op_def->mutable_device_option()->set_device_id(device_option.device_id());
  // enable is_pipe
  op_def->mutable_device_option()->set_is_pipe(true);
  // set arguments of op 
  for (auto& input_name : input_names) {
    // set input of op 
    op_def->add_input(input_name);   
    // set output of op 
    string output_name = input_name + "_bridge";
    op_def->add_output(output_name);
  }
}

bool CrossDeviceGraph::ShouldExclude(int idx) {
  if (node(idx).op.type() == "ConstantFill") {
    return true; 
  }
  return false;
}

void CrossDeviceGraph::FindBoundaries(std::vector<BoundaryNode>* boundaries) {
  // call graph.BFS to find boundary nodes
  BFS([this, boundaries](Node& node, void* arg) {
        // check the device type of parent node and current node 
        // are different or not
        auto& cur_device_option = this->device_option(node);
        if (node.parents.empty()) {
          // there is no parent nodes, meaning that current node is input nodes
          // but we don't need to handle this case  
        } else {
          // current node is non-input node
          for (auto& parent : node.parents) {
            auto& parent_device_option = this->device_option(
              this->node(parent.first));
            if (ShouldExclude(parent.first)) {
              continue;
            }
            if (cur_device_option.device_type() != parent_device_option.device_type() ||
                  cur_device_option.device_id() != parent_device_option.device_id()) {
              for (auto name : parent.second) {
                boundaries->emplace_back(node.idx, name);
                LOG_DEBUG("Boundary node id:%d; device_type:%d; input name:%s; parent device_type:%d",
                  node.idx, cur_device_option.device_type(),
                  name.c_str(), parent_device_option.device_type());
              }
            }
          } 
        }
        // NOTICE: The reason of returning false here is to find all the
        // boundary nodes rather than just the first one.
        return false;
      }, nullptr);
}

} // namespace blaze
