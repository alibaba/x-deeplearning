/*
 * \file cost_observer.cc
 * \beief The cost observer
 */
#include "blaze/graph/observer/cost_observer.h"

namespace blaze {

void CostOperatorObserver::Stop() {
  OperatorBase* op = subject();
  const std::string& type = op->type();
  OpSchema* schema = OpSchemaRegistry::Schema(type);
  const OperatorDef& def = op->operator_def();
  
  std::vector<TensorShape> input_tensor_shape;
  std::vector<DataType> input_data_type;
  for (size_t k = 0; k < op->InputSize(); ++k) {
    Blob* blob = op->Input(k);
    TensorShape ts;
    for (const auto& dim : blob->shape()) ts.add_dims(dim);
    input_tensor_shape.push_back(ts);
    input_data_type.push_back(static_cast<DataType>(blob->data_type()));
  }

  std::vector<TensorShape> output_tensor_shape;
  std::vector<DataType> output_data_type;
  for (size_t k = 0; k < op->OutputSize(); ++k) {
    Blob* blob = op->Output(k);
    TensorShape ts;
    for (const auto& dim : blob->shape()) ts.add_dims(dim);
    output_tensor_shape.push_back(ts);
    output_data_type.push_back(static_cast<DataType>(blob->data_type()));
  }

  cost_ = schema->InferCost(def, input_tensor_shape, input_data_type,
                            output_tensor_shape, output_data_type);
  cost_observer_->cost_ += cost_;
}

void CostObserver::Dump(std::string* out) {
  std::stringstream ss;
  ss << "GFlops per pv: " << cost_.flops / (1024.0 * 1024.0 * 1024.0) << "\n";
  ss << "GFlops: " << (cost_.flops / (1024.0 * 1024.0 * 1024.0)) / (end_time_ - start_time_) << "\n";
  ss << "BytesRead: " << (cost_.bytes_read / (1024.0 * 1024.0)) / (end_time_ - start_time_) << " MB/s\n";
  ss << "BytesWrite: " << (cost_.bytes_written / (1024.0 * 1024.0)) / (end_time_ - start_time_) << " MB/s\n";
  ss << "ParamsBytes: " << cost_.params_bytes / (1024.0 * 1024.0) << " MB";
  *out = ss.str();

#ifndef PROFILE_EXPORT
  LOG_INFO("\n%s", ss.str().c_str());
#endif
}

}  // namespace blaze

