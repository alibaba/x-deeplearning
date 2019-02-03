/*
 * \file xdl_importer.h
 * \brief The xdl importer
 */
#pragma once

#include <unordered_map>
#include "blaze/proto/blaze.pb.h"
#include "blaze/proto/embedding.pb.h"
#include "blaze/model_importer/graph_def.pb.h"
#include "blaze/model_importer/model_importer.h"
#include "blaze/model_importer/mxnet_importer.h"
#include "blaze/model_importer/tensorflow_importer.h"

namespace blaze {

class XdlImporter : public ModelImporter {
 public:
  XdlImporter();

  virtual void LoadModel(const char* model_conf, const char* model_data);
  typedef std::function<void(const xdl::proto::NodeDef& nd)> ProcessOpNodeFunction;

 protected:
  void InitSparseInputAliasMap();
  void InitDenseInputAliasMap();
  void CreateOpNode();

  std::string GetSparseInputAliasName(const std::string& name) const;
  std::string GetDenseInputAliasName(const std::string& name) const;
  void GenOpInputOutputInfo(const xdl::proto::NodeDef &node,
                            size_t output_size,
                            OperatorDef *op);
  blaze::DataType DeduceInputDataType(const std::string& input_name);
  
  // convert xdl sparse graph to blaze sparse graph
  void XdlSparse2Blaze();
  // sparse optimization pass
  void SparseOptimizationPass();

  // Process node function.
  void ProcessGetBatchOp(const xdl::proto::NodeDef &node);
  void ProcessPsIsInitializedOp(const xdl::proto::NodeDef &node);
  void ProcessPsNormalInitializerOp(const xdl::proto::NodeDef &node);
  void ProcessPsRegisterVariableOp(const xdl::proto::NodeDef &node);
  void ProcessPsConstantInitializerOp(const xdl::proto::NodeDef &node);
  void ProcessPsTruncatedNormalInitializerOp(const xdl::proto::NodeDef &node);
  void ProcessPsIdentityInitializerOp(const xdl::proto::NodeDef &node);
  void ProcessUniqueOp(const xdl::proto::NodeDef &node);
  void ProcessTakeOp(const xdl::proto::NodeDef &node);
  void ProcessPsSparsePullOp(const xdl::proto::NodeDef &node);
  void ProcessPsPullOp(const xdl::proto::NodeDef &node);
  void ProcessConstantOp(const xdl::proto::NodeDef &node);
  void ProcessKSumOp(const xdl::proto::NodeDef &node);
  void ProcessTileOp(const xdl::proto::NodeDef &node);
  void ProcessMxnetBackendOp(const xdl::proto::NodeDef &node);
  void ProcessTFBackendOp(const xdl::proto::NodeDef& node);
  // ...

  void MergeDenseGraph();
  void SetProcessNodeFunction(const std::string& name, ProcessOpNodeFunction function);

  xdl::proto::GraphDef graph_def_;
  blaze::EmbeddingConfig embedding_config_;
  std::unordered_map<std::string, ProcessOpNodeFunction> op_process_func_map_;
  std::unordered_map<std::string, std::string> sparse_input_alias_map_;
  std::unordered_map<std::string, std::string> dense_input_alias_map_;
  std::unordered_map<std::string, int> var_dim_map_;

  MXNetImporter mxnet_importer_;
  TensorFlowImporter tensorflow_importer_;
  const char* data_file_;
  std::string backend_model_conf_;
  std::string backend_type_;
};

}  // namespace blaze

