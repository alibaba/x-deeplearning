/*
 * \file cross_device_graph_manager.cc 
 * \brief The manager of cross device graph 
 */

#include <unordered_set>
#include <queue>
#include "blaze/graph/transform/cross_device_graph_manager.h"
#include "blaze/common/log.h"

using std::string;
using std::vector;
using std::unordered_set;

namespace blaze {

CrossDeviceGraphManager::CrossDeviceGraphManager(const NetDef& net_def) :
  net_def_(net_def) {
}

void CrossDeviceGraphManager::Transform() {
  std::vector<std::vector<int>> multi_parts;
  SplitToMultiParts(&net_def_, &multi_parts);
  LOG_DEBUG("Multi parts num: %u", multi_parts.size());
  SplitToMultiNetDefs(net_def_, multi_parts, &net_defs_);
  LOG_DEBUG("Net defs size:%u", net_defs_.size());
}

void CrossDeviceGraphManager::SplitToMultiParts(NetDef* net_def,
    std::vector<std::vector<int>>* multi_parts) {
  // recursively find max connected sub graph; when finding one sub graph,
  // deactivate it; then recursively handle the remaining active graph 
  CrossDeviceGraph cross_device_graph(*net_def);
  *net_def = cross_device_graph.GetNetDef();
  // NOTE: use new net_def to build graph 
  Graph graph(*net_def);
  while (graph.active_size() > 0) {
    vector<int> part;
    graph.MaxConnectedSearch(
        [&graph, &part](Node& node, void* arg) {
          if (part.empty()) {
            // just put the first node into part 
            part.push_back(node.idx);
            auto& first_device_option = graph.device_option(
              graph.node(part[0]));
            LOG_DEBUG("Splitted part, device type: %d, device id: %d, is_pipe:%d",
              first_device_option.device_type(), first_device_option.device_id(),
              first_device_option.is_pipe());
          } else {
            // always compare to the first node.idx to check equality 
            auto& cur_device_option = graph.device_option(node);
            auto& first_device_option = graph.device_option(
              graph.node(part[0]));
            if (cur_device_option.device_type() != first_device_option.device_type() ||
                  cur_device_option.device_id() != first_device_option.device_id() ||
                  cur_device_option.is_pipe() != first_device_option.is_pipe()) {
              // encounter different device or pipe, just stop to proceed  
              return false;
            }
            // the device type of current node is the same as the first node
            LOG_DEBUG("node id:%d name:%s", node.idx, node.op.name().c_str());
            part.push_back(node.idx);
          }

          // encounter the same device, just continue to proceed
          return true; 
        }, nullptr);  
    multi_parts->emplace_back(part);
    
    // deactivate current sub graph 
    graph.DeactivateSubgraph(part); 
  } 
}

void CrossDeviceGraphManager::SplitToMultiNetDefs(
    const NetDef& net_def,
    const std::vector<std::vector<int>>& multi_parts,
    std::vector<NetDef>* net_defs) {
  // get complementary node ids 
  for (auto& part : multi_parts) {
    Graph graph(net_def);
    std::vector<int> complementary_nodes;
    graph.ComplementNodes(part, &complementary_nodes);
    graph.DeactivateSubgraph(complementary_nodes);
    net_defs->emplace_back(graph.GetNetDef());
    auto& last_net_def = net_defs->at(net_defs->size() - 1);
    auto& node_device_option = graph.device_option(graph.node(part[0]));
    *(last_net_def.mutable_device_option()) = node_device_option;  
    LOG_DEBUG("CDGM, device type:%d device id:%d is pipe:%d",
        last_net_def.device_option().device_type(),
        last_net_def.device_option().device_id(),
        last_net_def.device_option().is_pipe());
  }
}

} // namespace blaze
