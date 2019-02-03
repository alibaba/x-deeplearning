/*
 * \file process.h 
 * \brief The blaze serving request process function.
 */
#pragma once

#include <string>

#include "blaze/api/cpp_api/predictor.h"
#include "thirdparty/protobuf/protobuf-3.6.0/src/google/protobuf/util/json_util.h"
#include "predict.pb.h"

namespace serving {

struct ProcessContext {
  // each feature name has many tensors.
  std::vector<Tensor> cache_tensor;
  // the indicator and empty feature's segments.
  Tensor indicator;
};

class PredictProcessor {
 public:
  bool Process(const std::string& input_str,
               ProcessContext* process_context,
               std::string* output_str);
  
 protected:
  bool GenerateProcessContext(const Request& request,
                              ProcessContext* process_context,
                              std::string* output_str);

  void FeedDenseFeature(size_t batch_size,
                        int index,
                        ProcessContext* process_context,
                        const std::string& input_name,
                        const blaze::FeedNameConfig& feed_name_config,
                        blaze::Predictor* predictor);

  void FeedSparseFeature(size_t batch_size,
                         int index,
                         ProcessContext* process_context,
                         const std::string& input_name,
                         const blaze::FeedNameConfig& feed_name_config,
                         blaze::Predictor* predictor);

  void FeedIndicator(size_t batch_size,
                     ProcessContext* process_context,
                     const std::string& input_name,
                     const blaze::FeedNameConfig& feed_name_config,
                     blaze::Predictor* predictor);

  void ClearTensor(Tensor& res);
  void FillTensor(size_t batch_size, const Tensor& tensor, Tensor& res);
};

} // namespace serving
