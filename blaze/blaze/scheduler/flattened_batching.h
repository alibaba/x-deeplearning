/*
 * \file flattened_batching.h 
 * \brief for batching flattened data
 */

#ifndef BLAZE_SCHEDULER_FLATTENED_BATCHING_H_
#define BLAZE_SCHEDULER_FLATTENED_BATCHING_H_

#include "blaze/scheduler/batching.h"

namespace blaze {

class FlattenedBatching : public Batching {
 public:
  FlattenedBatching();

 protected:
  bool HandleIndicators(std::vector<Net*>& nets);  
};

} // namespace blaze

#endif  // BLAZE_SCHEDULER_FLATTENED_BATCHING_H_
