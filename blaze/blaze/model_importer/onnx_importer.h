/*
 * \file onnx_importer.h
 * \brief The onnx importer
 */
#pragma once

#include <unordered_map>

#include "blaze/proto/blaze.pb.h"
#include "onnx/onnx_pb.h"
#include "blaze/model_importer/model_importer.h"

namespace blaze {

class OnnxAttributes {
 public:
  OnnxAttributes(const onnx::NodeProto& n);

  bool HasAttribute(const std::string& key) const {
    return onnx_attrs_.count(key);
  }
  google::protobuf::RepeatedPtrField<Argument> OnnxAttr2BlazeArg(const std::string& op_type);

 protected:
  bool ProcessAxesRelatedAttributes(Argument* arg, const onnx::AttributeProto& attr);
  void CopyOnnxAttrValueToBlazeArg(Argument* arg, const onnx::AttributeProto& attr,
                                   const std::string& op_type);

  std::unordered_map<std::string, const onnx::AttributeProto*> onnx_attrs_;
};

struct OnnxNode {
  OnnxNode(const onnx::NodeProto& n) : node(n), attributes(n) { }

  const onnx::NodeProto& node;
  OnnxAttributes attributes;
};

class ONNXImporter : public ModelImporter {
 public:
  ONNXImporter();
  virtual void LoadModel(const char* model_conf, const char* model_data = nullptr);

 protected:
  bool ONNX2Blaze();
  bool BuildConstantFillOp(const onnx::TensorProto& onnx_tensor);
  bool ONNXNode2BlazeNode(OnnxNode* onnx_node, int index);
  void RewriteOpDeviceOption();

  // The reshape constant name.
  std::unordered_set<std::string> reshape_constant_name_;
  // The onnx model
  onnx::ModelProto onnx_model_;
};

}  // namespace blaze
