/*
 * \file flattened_batching.h 
 * \brief for batching flattened data
 */

#include <string>

#include "blaze/scheduler/flattened_batching.h"

using std::string;

namespace blaze {

FlattenedBatching::FlattenedBatching() {
}

bool FlattenedBatching::HandleIndicators(std::vector<Net*>& nets) {
  if (0 == nets.size()) {
    LOG_ERROR("Input nets's size should be larger than 0");
    return false; 
  }
  auto& blob_map = nets[0]->external_output_blob();
  // there is only one layer, so just use the first blob name    
  string one_name = blob_map.begin()->first; 

  level_lens_.resize(1u);
  // update lens of each net      
  for (auto& net : nets) {
    // NOTE: use input blob here instead of output blob,
    // since output blob of first net has been expanded
    auto input_blob = net->external_input_blob(one_name); 
    level_lens_[0].push_back(input_blob->dim(0)); 
  } 

  return true;
}

} // namespace blaze
