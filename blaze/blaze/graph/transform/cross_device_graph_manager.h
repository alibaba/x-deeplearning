/*
 * \file cross_device_graph_manager.h 
 * \brief The manager of cross device graph 
 */
#ifndef BLAZE_BLAZE_GRAPH_TRANSFORM_CROSS_DEVICE_GRAPH_MANAGER_
#define BLAZE_BLAZE_GRAPH_TRANSFORM_CROSS_DEVICE_GRAPH_MANAGER_

#include <memory>
#include "blaze/graph/transform/cross_device_graph.h"

namespace blaze {

class CrossDeviceGraphManager {
 public:
  CrossDeviceGraphManager(const NetDef& net_def);

  // transform to cross device graph
  void Transform();

  inline const std::vector<NetDef>& GetNetDefs() const {
    return net_defs_;
  }

  inline const NetDef& GetNetDef() const {
    return net_def_;
  } 

 private:
  // split node idxs into multi parts so that each part is
  // a max connected sub-graph having the same device option
  static void SplitToMultiParts(NetDef* net_def,
      std::vector<std::vector<int>>* multi_parts); 
  // split into multi net defs
  static void SplitToMultiNetDefs(const NetDef& net_def,
      const std::vector<std::vector<int>>& multi_parts,
      std::vector<NetDef>* net_defs);

  NetDef net_def_; 
  std::vector<NetDef> net_defs_;
};

} // namespace blaze

#endif  // BLAZE_BLAZE_GRAPH_TRANSFORM_CROSS_DEVICE_GRAPH_MANAGER_
