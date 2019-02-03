/*
 * \file cross_device_graph.h 
 * \brief The definition of cross device graph 
 */
#ifndef BLAZE_BLAZE_GRAPH_TRANSFORM_CROSS_DEVICE_GRAPH_
#define BLAZE_BLAZE_GRAPH_TRANSFORM_CROSS_DEVICE_GRAPH_

#include <vector>
#include <memory>
#include <string>
#include <set>

#include "blaze/proto/blaze.pb.h"
#include "blaze/graph/graph.h"

namespace blaze {

class CrossDeviceGraph : public Graph {
 public:
  explicit CrossDeviceGraph(const NetDef& net_def);
  virtual ~CrossDeviceGraph() {}

  // int Init(); 

 private:
  struct BoundaryNode {
    BoundaryNode(int idx, const std::string& name)
      : idx(idx), input_name(name) {}

    int idx;
    std::string input_name;
  };

  // insert bridge nodes before boundary nodes 
  void InsertBridgeNodes(const std::vector<BoundaryNode>& boundaries);
  // create one bridge node 
  void DefineBridgeNode(const std::string& op_name,
      const std::set<std::string>& input_names,
      const DeviceOption& device_option,
      OperatorDef* def); 

  // should exclude this node or not 
  bool ShouldExclude(int idx);

  // find the boundary nodes whose parent nodes have
  // different device types from these nodes;
  // if there are just non-cpu nodes, then the boundary nodes are 
  // the input nodes of this graph
  void FindBoundaries(std::vector<BoundaryNode>* boundaries);  
};

} // namespace blaze

#endif // BLAZE_BLAZE_GRAPH_TRANSFORM_CROSS_DEVICE_GRAPH_
