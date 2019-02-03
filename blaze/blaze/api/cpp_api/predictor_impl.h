/*!
 * \file predictor_impl.h
 * \brief The blaze predictor interface impl
 */
#pragma once

#include <memory>

#include "blaze/api/cpp_api/predictor.h"
#include "blaze/common/proto_helper.h"
#include "blaze/graph/workspace.h"
#include "blaze/common/func.h"

namespace blaze {

class PredictorImpl {
 public:
  PredictorImpl(std::shared_ptr<Net>& net); 

  FeedNameConfig GetFeedNameConfig(const std::string& feed_name);

  bool ReshapeInput(const char* name, const std::vector<size_t>& shape);
  bool ReshapeInput(size_t idx, const std::vector<size_t>& shape);

  bool Feed(const char* name, const void* data, size_t len);
  bool Feed(size_t idx, const void* data, size_t len);
  size_t InputSize() const;
  PredictDataType InputDataType(const char* name) const;
  PredictDataType InputDataType(size_t idx) const;
  const std::vector<std::string>& ListInputName() const; 

  bool Forward(const PredictorCallback&& cb); 

  bool Output(const char* name, void** data, size_t* len);
  bool Output(size_t idx, void** data, size_t* len);
  const std::vector<size_t>& OutputShape(size_t idx) const;
  const std::vector<size_t>& OutputShape(const char* name) const;
  PredictDataType OutputDataType(size_t idx) const;
  PredictDataType OutputDataType(const char* name) const;
  size_t OutputSize() const;
  const std::vector<std::string>& ListOutputName() const;

  std::vector<std::string> ListInternalName() const;
  bool InternalParam(const char* name, void** data, size_t* len);
  const std::vector<size_t>& InternalShape(const char* name) const;
  int InternalDataType(const char* name) const;

  void RegisterObservers(const std::vector<std::string>& oberver_names);
  void DumpObservers(std::unordered_map<std::string, std::string>* dump_map);

 protected:
  int InputName2Idx(const char* name) const;
  int OutputName2Idx(const char* name) const;

  std::vector<std::shared_ptr<Blob>> external_output_blob_cpu_;
  std::vector<Blob*> external_output_blob_;
  std::unordered_map<std::string, int> external_output_blob_index_;

  std::vector<Blob*> external_input_blob_;
  std::unordered_map<std::string, int> external_input_blob_index_;

  std::shared_ptr<Net> net_;
};

}  // namespace blaze
