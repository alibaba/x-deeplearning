/*
 * \file structured_batching.h 
 * \brief for batching structured data 
 */

#ifndef BLAZE_SCHEDULER_STRUCTURED_BATCHING_H_
#define BLAZE_SCHEDULER_STRUCTURED_BATCHING_H_

#include "blaze/scheduler/batching.h"

namespace blaze {

class StructuredBatching : public Batching {
 public:
  StructuredBatching();

 protected:
  virtual bool HandleIndicators(std::vector<Net*>& nets);  

 private:
  void CopyDeviceToHost(std::vector<Net*>& nets,
      const std::string& blob_name, std::vector<TIndex>* lens,
      Blob* host_indicators); 
  void AdjustVals(const std::vector<TIndex>& lens, Blob* host_indicators);
  void UpdateLevelLens(const std::vector<TIndex>& lens, int level_id); 
  TIndex TotalLen(const std::vector<TIndex>& lens);
  void ReshapeIndicator(const std::string& blob_name,
      TIndex total_len, Net* net);  
  void CopyHostToDevice(const std::string& blob_name, TIndex total_len,
      const Blob& host_indicators, Net* net);
};

} // namespace blaze

#endif  // BLAZE_SCHEDULER_STRUCTURED_BATCHING_H_
