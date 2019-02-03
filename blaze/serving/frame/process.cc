/* 
 * \file process.cc
 * \brief The blaze serving request process function.
 */
#include "blaze/common/queue.h"
#include "serving/frame/process.h"
#include "serving/model/model.h"
#include "serving/model/model_manager.h"

namespace serving {

void PredictProcessor::ClearTensor(Tensor& res) {
  res.clear_value();
  res.clear_key();
  res.clear_aux();
}

void PredictProcessor::FillTensor(size_t batch_size,
                                  const Tensor& tensor,
                                  Tensor& res) {
  if (tensor.key_size() == tensor.value_size()) {
    // sparse feature
    for (const auto& key : tensor.key()) {
      res.add_key(key);
    }
    for (const auto& value : tensor.value()) {
      res.add_value(value);
    }
    res.add_aux(tensor.key_size());
  } else {
    // dense feature
    for (const auto& value : tensor.value()) {
      res.add_value(value);
    }
  }
}

bool PredictProcessor::GenerateProcessContext(const Request& request,
                                              ProcessContext* process_context,
                                              std::string* output_str) {
  // Step1: generate indicator
  auto batch_size = request.ad_feature_size();
  process_context->indicator.mutable_aux()->Resize(batch_size, 0);

  // Step2: generate feature
  for (auto i = 0; i < request.feature_name_size(); ++i) {
    process_context->cache_tensor.resize(i + 1);
    ClearTensor(process_context->cache_tensor[i]);

    for (const auto& ad_feature : request.ad_feature()) {
      for (const auto& tensor : ad_feature.tensor()) {
        if (tensor.feature_name_index() >= request.feature_name_size()) {
          *output_str = "invalid feature_name_index";
          return false;
        }
        if (tensor.feature_name_index() == i) {
          FillTensor(batch_size, tensor, process_context->cache_tensor[i]);
        }
      }
    }
    for (const auto& tensor : request.user_feature().tensor()) {
      if (tensor.feature_name_index() >= request.feature_name_size()) {
        *output_str = "invalid feature_name_index";
        return false;
      }
      if (tensor.feature_name_index() == i) {
        FillTensor(batch_size, tensor, process_context->cache_tensor[i]);
      }
    }
  }  // for (auto i = 0; i < request.feature_name_size(); ++i) {
  return true;
}

void PredictProcessor::FeedDenseFeature(size_t batch_size,
                                        int index,
                                        ProcessContext* process_context,
                                        const std::string& input_name,
                                        const blaze::FeedNameConfig& feed_name_config,
                                        blaze::Predictor* predictor) {
  auto level = feed_name_config.level;
  const auto& tensor = process_context->cache_tensor[index];
  if (level == 0) {
    size_t dim = tensor.value_size();
    predictor->ReshapeInput(input_name.c_str(), { dim });
  } else {
    size_t dim = tensor.value_size();
    predictor->ReshapeInput(input_name.c_str(), { batch_size, dim / batch_size });
  }
  predictor->Feed(input_name.c_str(),
                  reinterpret_cast<const void*>(tensor.value().data()),
                  tensor.value_size() * sizeof(tensor.value(0)));
}

void PredictProcessor::FeedSparseFeature(size_t batch_size,
                                         int index,
                                         ProcessContext* process_context,
                                         const std::string& input_name,
                                         const blaze::FeedNameConfig& feed_name_config,
                                         blaze::Predictor* predictor) {
  switch (feed_name_config.sparse_feature_type) {
    case blaze::kSparseFeatureId:
      {
        if (index < 0) {
          predictor->ReshapeInput(input_name.c_str(), { 0 });
        } else {
          const auto& tensor = process_context->cache_tensor[index];
          predictor->ReshapeInput(input_name.c_str(), { static_cast<size_t>(tensor.key_size()) });
          predictor->Feed(input_name.c_str(),
                          reinterpret_cast<const void*>(tensor.key().data()),
                          tensor.key_size() * sizeof(tensor.key(0)));
        }
      }
      break;
    case blaze::kSparseFeatureValue:
      {
        if (index < 0) {
          predictor->ReshapeInput(input_name.c_str(), { 0 });
        } else {
          const auto& tensor = process_context->cache_tensor[index];
          predictor->ReshapeInput(input_name.c_str(), { static_cast<size_t>(tensor.value_size()) });
          predictor->Feed(input_name.c_str(),
                          reinterpret_cast<const void*>(tensor.value().data()),
                          tensor.value_size() * sizeof(tensor.value(0)));
        }
      }
      break;
    case blaze::kAuxSparseFeatureSegment:
      {
        if (index < 0) {
          size_t dim = 1;
          if (feed_name_config.level == 0) {
            dim = batch_size;
          }
          predictor->ReshapeInput(input_name.c_str(), { dim });
          const auto& tensor = process_context->indicator;
          predictor->Feed(input_name.c_str(),
                          reinterpret_cast<const void*>(tensor.aux().data()),
                          dim * sizeof(tensor.aux(0)));
          //std::cout << "-aux:" << input_name << " dim:" << dim << std::endl;
        } else {
          const auto& tensor = process_context->cache_tensor[index];
          predictor->ReshapeInput(input_name.c_str(), { static_cast<size_t>(tensor.aux_size()) });
          predictor->Feed(input_name.c_str(),
                          reinterpret_cast<const void*>(tensor.aux().data()),
                          tensor.aux_size() * sizeof(tensor.aux(0)));
          //std::cout << "aux: " << input_name << " dim:" << tensor.aux_size() << std::endl;
        }
      }
      break;
  }
}

void PredictProcessor::FeedIndicator(size_t batch_size,
                                     ProcessContext* process_context,
                                     const std::string& input_name,
                                     const blaze::FeedNameConfig& feed_name_config,
                                     blaze::Predictor* predictor) {
  // NOTE: Now only support 2 level.
  const auto& tensor = process_context->indicator;
  predictor->ReshapeInput(input_name.c_str(), { batch_size });
  predictor->Feed(input_name.c_str(),
                  reinterpret_cast<const void*>(tensor.aux().data()),
                  tensor.aux_size() * sizeof(tensor.aux(0)));
}

bool PredictProcessor::Process(const std::string &input_str,
                               ProcessContext* process_context,
                               std::string* output_str) {
  Request request;
  google::protobuf::util::Status status = google::protobuf::util::JsonStringToMessage(input_str, &request);
  if (!status.ok()) {
    *output_str = "Error: parse request failed!\n" + status.ToString();
    return false;
  }
  //std::cout << request.DebugString() << std::endl;

  blaze::Predictor* predictor = ModelManager::Instance()->CreatePredictor(request.model_version());
  if (predictor == nullptr) {
    *output_str = "[ERROR] create predictor for <" + request.model_version() + "> failed.\n";
    return false;
  }

  if (!GenerateProcessContext(request, process_context, output_str)) {
    return false;
  }

  auto batch_size = request.ad_feature_size();
  std::unordered_map<std::string, int> inverted_index;
  for (auto i = 0; i < request.feature_name_size(); ++i) {
    inverted_index[request.feature_name(i)] = i;
  }

  // feed input
  const auto& input_name_list = predictor->ListInputName();
  for (const auto& input_name : input_name_list) {
    auto feed_name_config = predictor->GetFeedNameConfig(input_name);
    int index = -1;
    auto iter = inverted_index.find(feed_name_config.feature_name);
    if (iter != inverted_index.end()) index = iter->second;

    switch (feed_name_config.feature_type) {
      case kDenseFeature:
        if (index < 0) {
          *output_str = " dense feature: " + feed_name_config.feature_name + " is missing";
          return false;
        } else {
          FeedDenseFeature(batch_size, index, process_context, input_name, feed_name_config, predictor);
        }
        break;
      case kSparseFeature:
        FeedSparseFeature(batch_size, index, process_context, input_name, feed_name_config, predictor);
        break;
      case kAuxIndicator:
        FeedIndicator(batch_size, process_context, input_name, feed_name_config, predictor);
        break;
    }
  }

  //calculate forward
  if (!predictor->Forward()) {
    *output_str = "[ERROR] forward <" + request.model_version() + "> failed.\n";
    delete(predictor);
    return false;
  }

  //get outputs
  size_t n = predictor->OutputSize();
  std::vector<float*> output_ptrs(n);
  std::vector<size_t> output_lens(n);
  for(size_t i = 0; i < n; ++i) {
    if (!predictor->Output(i, (void**)&(output_ptrs[i]), &(output_lens[i]))) {
      *output_str = "[ERROR] get output <" + (predictor->ListOutputName()[i]) + "> failed.\n";
      delete (predictor);
      return false;
    }
  }

  //construct response
  Response response;
  for (size_t i = 0; i < n; ++i){
    Tensor* tensor = response.add_output_list();
    //name
    const std::string& output_name = predictor->ListOutputName()[i];
    tensor->set_name(output_name);
    //shape
    const std::vector<size_t>& output_shape = predictor->OutputShape(i);
    for (const size_t dim: output_shape){
      tensor->add_shape(dim);
    }
    //value
    tensor->mutable_value()->Reserve(static_cast<int>(output_lens[i]));
    for (size_t j = 0; j < output_lens[i]/sizeof(float); ++j) {
      tensor->add_value(output_ptrs[i][j]);
    }
  }

  google::protobuf::util::MessageToJsonString(response, output_str);

  delete(predictor);
  return true;
};

} //namespace serving
