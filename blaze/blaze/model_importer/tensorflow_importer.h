/*
 * \file tensorflow_importer.h
 * \brief The tensorflow importer
 */
#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "blaze/common/exception.h"
#include "blaze/common/stream.h"
#include "blaze/common/types.h"
#include "blaze/model_importer/cnpy/cnpy.h"
#include "blaze/model_importer/model_importer.h"
#include "blaze/model_importer/tensorflow/core/framework/graph.pb.h"
#include "blaze/model_importer/tensorflow/core/protobuf/meta_graph.pb.h"

namespace blaze {

struct TFParam {
  struct NDArray {
    std::vector<size_t> shape;
    DataType data_type;
    union ValueType {
      float f;
      int64_t i;
    };
    std::vector<ValueType> data;
  };
  
  std::vector<NDArray> ndarray;
  std::vector<std::string> keys;

  void Load(const char* file_name) {
    auto npz = cnpy::npz_load(file_name);
    for (const auto& iter : npz) {
      keys.push_back(iter.first);
      const auto& value = iter.second;
      NDArray nd;
      nd.shape = value.shape;
      nd.data_type = kFloat; // We only support Float32
      NDArray::ValueType value_type;
      BLAZE_CONDITION_THROW(value.word_size == 4, "Only support float weights");
      for (auto i = 0; i < value.num_vals; ++i) {
        value_type.f = value.data<float>()[i];
        nd.data.push_back(value_type);
      }
      ndarray.push_back(nd);
    }
  }
};

// Convert TensorFlow model into blaze model
class TensorFlowImporter : public ModelImporter {
 public:
  TensorFlowImporter();

  // Load tensor model.
  virtual void LoadModel(const char* conf_file, const char* data_file);

  void LoadModelFromString(const std::string& conf_str, const char* data_file);

  // Process OPNode Function
  typedef std::function<void(const tensorflow::NodeDef& node_def)> ProcessOpNodeFunction;

 protected:
  void LoadCkpt(const char* data_file);
  void Tensorflow2Blaze();

  // Process node callback function
  void ProcessPlaceholderOp(const tensorflow::NodeDef& node);
  void ProcessConstOp(const tensorflow::NodeDef& node);
  void ProcessIdentityOp(const tensorflow::NodeDef& node);
  void ProcessMatMulOp(const tensorflow::NodeDef& node);
  void ProcessAddOp(const tensorflow::NodeDef& node);
  void ProcessMulOp(const tensorflow::NodeDef& node);
  void ProcessMaximumOp(const tensorflow::NodeDef& node);
  void ProcessSoftmaxOp(const tensorflow::NodeDef& node);
  // ...

  // Add operator definition.
  OperatorDef* AddOperatorDef(const tensorflow::NodeDef& node, const char* op_type, int onum = 1); 
  
  // Get input from input_str
  void GetInput(const std::string& input_str, std::string* node_name, int* index);
  DataType GetDataType(tensorflow::DataType data_type);
  void SetProcessNodeFunction(const std::string& name, ProcessOpNodeFunction function);

  // The tensorflow network graph def.
  tensorflow::GraphDef graph_def_;

  std::unordered_map<std::string, ProcessOpNodeFunction> op_process_func_map_;
  std::unordered_map<std::string, std::string> name_rewrite_;
  std::unordered_set<std::string> const_name_set_;

  TFParam tf_param_;

  friend class XdlImporter;
};

}  // namespace blaze

