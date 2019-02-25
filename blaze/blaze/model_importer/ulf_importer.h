/*!
 * \file ulf_importer.h
 * \brief Convert ulf model into blaze model.
 */
#pragma once

#include <functional>
#include <unordered_map>

#include "blaze/proto/blaze.pb.h"
#include "blaze/model_importer/ulf.pb.h"
#include "blaze/model_importer/model_importer.h"

namespace blaze {

// Convert ulf model into blaze model.
class ULFImporter : public ModelImporter {
 public:
  ULFImporter();

  virtual void LoadModel(const char* conf_file,const char* data_file);

  typedef std::function<bool(const ulf::LayerParameter& lp)> ProcessOpNodeFunction;

 protected:
  bool Ulf2Blaze();

  void InitLayerMap();
  bool CreateConstInputConstantFillNode(const ulf::LayerParameter& lp);
  bool CreateConstantFillNode();
  bool CreateOpNode();

  bool ProcessSliceLayer(const ulf::LayerParameter& lp);
  bool ProcessInnerProductLayer(const ulf::LayerParameter& lp);
  bool ProcessInnerProductLayerEx(const ulf::LayerParameter& lp);
  bool ProcessSoftmaxLayer(const ulf::LayerParameter& lp);
  bool ProcessFuseLayer(const ulf::LayerParameter& lp);
  bool ProcessGruLayer(const ulf::LayerParameter& lp);
  bool ProcessConcatLayer(const ulf::LayerParameter& lp);
  bool ProcessSumLayer(const ulf::LayerParameter& lp);
  bool ProcessMultiplyLayer(const ulf::LayerParameter& lp);
  bool ProcessSubLayer(const ulf::LayerParameter& lp);
  bool ProcessAddLayer(const ulf::LayerParameter& lp);
  bool ProcessBatchDotLayer(const ulf::LayerParameter& lp);
  bool ProcessPreluLayer(const ulf::LayerParameter& lp);
  bool ProcessReluLayer(const ulf::LayerParameter& lp);
  bool ProcessSigmoidLayer(const ulf::LayerParameter& lp);
  bool ProcessTanhLayer(const ulf::LayerParameter& lp);
  bool ProcessDiceLayer(const ulf::LayerParameter& lp);
  bool ProcessBnLayer(const ulf::LayerParameter& lp);
  bool ProcessEmbeddingLayer(const ulf::LayerParameter& lp);
  bool ProcessDivLayer(const ulf::LayerParameter& lp);
  bool ProcessConstantLayer(const ulf::LayerParameter& lp);
  bool ProcessReshapeLayer(const ulf::LayerParameter& lp);
  bool ProcessBroadcastToLayer(const ulf::LayerParameter& lp);
  bool ProcessWhereLayer(const ulf::LayerParameter& lp);

  void SetProcessNodeFunction(const std::string& name, ProcessOpNodeFunction function);
  bool SaveFileContent(const char* filename, const std::string& content);

  // The ulf net definition
  std::string conf_file_;
  ulf::NetParameter net_conf_;
  ulf::NetWeightsParameter net_param_;
  std::unordered_map<std::string, ulf::LayerWeightsParameter*> layer_param_map_; 
  std::unordered_map<std::string, ProcessOpNodeFunction> op_process_func_map_;
  std::unordered_map<std::string, std::string> layer_type_map_;
  std::unordered_map<std::string, int> sparse_level_;
};

}  // namespace blaze
