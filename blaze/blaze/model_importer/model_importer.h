/*!
 * \file model_importer.h
 * \brief base model importer
 */
#pragma once

#include <functional>
#include <unordered_map>

#include "blaze/proto/blaze.pb.h"
#include "blaze/common/exception.h"

namespace blaze {

class ModelImporter {
 public:
  ModelImporter();

  // Load mxnet model from conf and data.
  virtual void LoadModel(const char* conf_file, const char* data_file) {
    BLAZE_THROW("Not implemented LoadModel");
  }

  // Dump blaze model.
  bool SaveToTextFile(const char* blaze_model_file);
  bool SaveToBinaryFile(const char* blaze_model_file);
  
  // The model calaculation input/output type, must set before LoadModel
  // interface.
  void set_data_type(DataType data_type) { data_type_ = data_type; }
  DataType data_type() const { return data_type_; }

  // The model weight data type(global), must set before LoadModel interface.
  void set_weight_type(DataType data_type) { weight_type_ = data_type; }
  DataType weight_type() { return weight_type_; }

  // The model op's weight data type(private), must set before LoadModel
  // interface.
  void set_op_weight_type(const std::string& op_type, DataType data_type) {
    op_weight_type_[op_type] = data_type;
  }
  DataType op_weight_type(const std::string& op_type) {
    const auto& iter = op_weight_type_.find(op_type);
    if (iter != op_weight_type_.end()) return iter->second;
    else return weight_type_;
  }

  const NetDef& net_def() const { return net_def_; }

 protected:
  bool ReadFileContent(const char* filename, std::string* content);
  bool SaveFileContent(const char* filename, const std::string& content);
  std::string GetParentPath(const std::string& path);

  std::unordered_map<std::string, DataType> op_weight_type_; 
  DataType weight_type_ = kFloat; // The model weight type
  DataType data_type_ = kFloat; // The model calculation input/output type
  
  NetDef net_def_;
};

}  // namepace blaze
