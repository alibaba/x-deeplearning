/*
 * \file batching.h 
 * \brief high-level batching func to batch multi nets 
 */

#ifndef BLAZE_SCHEDULER_BATCHING_H_
#define BLAZE_SCHEDULER_BATCHING_H_

#include <vector>
#include <unordered_map>
#include "blaze/graph/net.h"

namespace blaze {

class CUDAContext;

class Batching {
 public:
  Batching();
  virtual ~Batching();
 
  // merge multi nets into one net 
  bool Merge(std::vector<Net*>& src_nets, Net** dst_net);

  // do split src net to multi nets
  // Requires: src_net == (*dst_nets)[0]
  // Requires: dst_nets != nullptr && dst_nets.size() > 0
  bool Split(Net* src_net, std::vector<Net*>* dst_nets);

 protected:
  // different batching implementations has differnt indicators
  virtual bool HandleIndicators(std::vector<Net*>& nets) = 0;  
  // backup the first dim of specified net
  void BackupFirstDim(const Net* net);
  // restore the first dim of specified net
  void RestoreFirstDim(Net* net);
  // get all blobs except indicators 
  void GetNonIndicatorBlobs(const Net* net,
      std::unordered_map<std::string, Blob*>* blobs_map);  
  // get the sum of first dim 
  TIndex CalcSumOfFirstDim(std::vector<Net*>& nets,
      const std::string& blob_name);
  // Backup the data of the first net's blob
  void BackupData(const Net* net, const std::string& blob_name,
      Blob* backup_blob);
  // Reshape merged net 
  void ReshapeMergedNet(std::vector<Net*>& nets, const std::string& blob_name, Net* merged_net); 
  // Restore the data from backup space
  void RestoreData(const Blob& backup_blob, Net* net,
      const std::string& blob_name, TIndex* offset);
  // Copy the data of some net to the specified location of merged net   
  void CopyDataToMergedNet(const Net* net, const std::string& blob_name,
      CUDAContext& context, Net* merged_net, TIndex* offset);

  // lens of each layer    
  std::vector<std::vector<TIndex>> level_lens_;
  // blob & net's first dim
  std::unordered_map<std::string, TIndex> blob_first_dim_; 
}; 

} // namespace blaze

#endif  // BLAZE_SCHEDULER_BATCHING_H_
