/*
 * \file mxnet_importer.h
 * \brief The mxnet importer
 */
#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "blaze/common/exception.h"
#include "blaze/common/json.h"
#include "blaze/common/stream.h"
#include "blaze/common/types.h"
#include "blaze/proto/blaze.pb.h"
#include "blaze/model_importer/model_importer.h"

namespace blaze {

// MXNET nnvm::Node in JSON
struct JSONNode;
// MXNET nnvm::Symbol in JSON
struct JSONGraph;

struct JSONNode {
  // the node entry structure in serialized format
  struct Entry {
    uint32_t node_id;
    uint32_t index;
    uint32_t version;

    Entry() = default;
    Entry(uint32_t node_id, uint32_t index, uint32_t version) :
        node_id(node_id), index(index), version(version) { }
    
    void Load(JSONReader* reader) {
      reader->BeginArray();
      CHECK_TRUE(reader->NextArrayItem(), "invalid json format");
      reader->Read(&node_id);
      CHECK_TRUE(reader->NextArrayItem(), "invalid json format");
      reader->Read(&index);
      if (reader->NextArrayItem()) {
        reader->Read(&version);
        CHECK_TRUE(!reader->NextArrayItem(), "invalid json format");
      } else {
        version = 0;
      }
    }
  };

  struct Attrs {
    // name of node
    std::string name;
    // The dictionary representation of attributes.
    std::unordered_map<std::string, std::string> dict;
  };

  // op type.
  std::string op_type;
  // Attributes
  Attrs attrs;
  // inputs
  std::vector<Entry> inputs;
  // control flow dependencies
  std::vector<uint32_t> control_deps;
  // subgraphs
  std::vector<JSONGraph> subgraphs;

  // Load JSON Node.
  void Load(JSONReader* reader) {
    control_deps.clear();
    JSONObjectReadHelper helper;
    helper.DeclareField("op", &op_type);
    helper.DeclareField("name", &(attrs.name));
    helper.DeclareField("inputs", &inputs);
    helper.DeclareOptionalField("attrs", &(attrs.dict));
    helper.DeclareOptionalField("attr", &(attrs.dict));
    helper.DeclareOptionalField("control_deps", &control_deps);
    helper.DeclareOptionalField("subgraphs", &subgraphs);
    // backward compatible code with mxnet graph.
    int backward_source_id;
    std::unordered_map<std::string, std::string> param;
    helper.DeclareOptionalField("param", &param);
    helper.DeclareOptionalField("backward_source_id", &backward_source_id);
    helper.ReadAllFields(reader);
  }
};

struct JSONGraph {
  std::vector<JSONNode> nodes;
  std::vector<uint32_t> arg_nodes;
  std::vector<uint32_t> node_row_ptr;
  std::vector<JSONNode::Entry> heads;
  std::unordered_map<std::string, any > attrs;

  void Load(JSONReader* reader) {
    attrs.clear();
    JSONObjectReadHelper helper;
    helper.DeclareField("nodes", &nodes);
    helper.DeclareField("arg_nodes", &arg_nodes);
    helper.DeclareField("heads", &heads);
    helper.DeclareOptionalField("node_row_ptr", &node_row_ptr);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }
};

// The MXNet Model Param
struct MXParam {
  /* magic number for ndarray version 1, with int64_t TShape */
  static const uint32_t NDARRAY_V1_MAGIC = 0xF993fac8;
  /* magic number for ndarray version 2, with storage type */
  static const uint32_t NDARRAY_V2_MAGIC = 0xF993fac9;
  /* Magic code */
  const uint64_t kMXAPINDArrayListMagic = 0x112;

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

  void Load(Stream* stream) {
    uint64_t header, reserved;
    CHECK_TRUE(stream->Read(&header), "Invalid NDArray file format");
    CHECK_TRUE(stream->Read(&reserved), "Invalid NDArray file format");
    CHECK_EQ(header, kMXAPINDArrayListMagic, "Invalid NDArray file format");
    LoadNDArrayList(stream);
    LoadKeyList(stream);
    // Summary debug
    for (auto i = 0; i < keys.size(); ++i) {
      LOG_DEBUG("name=%s size=%u", keys[i].c_str(), ndarray[i].data.size());
    }
  }

 protected:
  // Load NDArray list
  void LoadNDArrayList(Stream* stream) {
    uint64_t sz;
    CHECK_EQ(stream->Read<uint64_t>(&sz), sizeof(sz));
    ndarray.resize(sz);
    for (auto i = 0; i < sz; ++i) {
      LoadNDArray(stream, &(ndarray[i]));
    }
  }

  // Load Key(Name) list
  void LoadKeyList(Stream* stream) {
    uint64_t sz;
    CHECK_EQ(stream->Read<uint64_t>(&sz), sizeof(sz));
    keys.resize(sz);
    CHECK_EQ(ndarray.size(), keys.size());
    for (auto i = 0; i < sz; ++i) {
      uint64_t len;
      CHECK_EQ(stream->Read<uint64_t>(&len), sizeof(len));
      keys[i].resize(len);
      CHECK_EQ(stream->Read(const_cast<char*>(keys[i].c_str()), len), len);
      if (strncmp(keys[i].c_str(), "aux:", 4) == 0 ||
          strncmp(keys[i].c_str(), "arg:", 4) == 0 ||
          strncmp(keys[i].c_str(), "var:", 4) == 0) {
        keys[i] = keys[i].substr(4);
      }
    }
  }

  // Load NDArray
  void LoadNDArray(Stream* stream, NDArray* ndarray) {
    uint32_t magic;
    CHECK_EQ(stream->Read(&magic, sizeof(magic)), sizeof(magic));
    if (magic != NDARRAY_V2_MAGIC) {
      return LoadLegacyNDArray(stream, magic, ndarray);
    }

    // NOTE: Not support sparse tensor type now.
    // load storage type
    int32_t stype;
    CHECK_EQ(stream->Read(&stype, sizeof(stype)), sizeof(stype));

    // load shape
    uint32_t ndim;
    std::vector<int64_t> shape;
    CHECK_EQ(stream->Read(&ndim, sizeof(ndim)), sizeof(ndim));
    shape.resize(ndim);
    ndarray->shape.resize(ndim);
    for (auto i = 0; i < ndim; ++i) {
      CHECK_EQ(stream->Read(&shape[i], sizeof(shape[0])), sizeof(shape[0]));
      ndarray->shape[i] = shape[i];
    }

    // load context
    int32_t dev_type, dev_id;
    CHECK_EQ(stream->Read(&dev_type, sizeof(dev_type)), sizeof(dev_type));
    CHECK_EQ(stream->Read(&dev_id, sizeof(dev_id)), sizeof(dev_id));

    // load type flag
    int32_t type_flag;
    CHECK_EQ(stream->Read(&type_flag, sizeof(type_flag)), sizeof(type_flag));
    ndarray->data_type = GetDataType(type_flag);

    // load data
    size_t size = 1;
    for (const auto dim : ndarray->shape) size *= dim;
    ndarray->data.resize(size);
    for (auto i = 0; i < size; ++i) {
      CONSTANT_FILL_TYPE_SWITCH(ndarray->data_type, DType, {
        DType value;
        CHECK_EQ(stream->Read(&value, sizeof(value)), sizeof(value));
        if (IsIntegerType(ndarray->data_type)) {
          ndarray->data[i].i = value;
        } else {
          ndarray->data[i].f = value;
        }
      });
    }
  }

  // Load Legacy NDArray
  void LoadLegacyNDArray(Stream* stream, uint32_t magic, NDArray* ndarray) {
    // load shape
    uint32_t ndim;
    std::vector<int64_t> shape;
    CHECK_EQ(stream->Read(&ndim, sizeof(ndim)), sizeof(ndim));
    shape.resize(ndim);
    ndarray->shape.resize(ndim);
    for (auto i = 0; i < ndim; ++i) {
      CHECK_EQ(stream->Read(&shape[i], sizeof(shape[0])), sizeof(shape[0]));
      ndarray->shape[i] = shape[i];
    }

    // load context
    int32_t dev_type, dev_id;
    CHECK_EQ(stream->Read(&dev_type, sizeof(dev_type)), sizeof(dev_type));
    CHECK_EQ(stream->Read(&dev_id, sizeof(dev_id)), sizeof(dev_id));

    // load type flag
    int32_t type_flag;
    CHECK_EQ(stream->Read(&type_flag, sizeof(type_flag)), sizeof(type_flag));
    ndarray->data_type = GetDataType(type_flag);

    // load data
    size_t size = 1;
    for (const auto dim : ndarray->shape) size *= dim;
    ndarray->data.resize(size);
    for (auto i = 0; i < size; ++i) {
      CONSTANT_FILL_TYPE_SWITCH(ndarray->data_type, DType, {
        DType value;
        CHECK_EQ(stream->Read(&value, sizeof(value)), sizeof(value));
        if (IsIntegerType(ndarray->data_type)) {
          ndarray->data[i].f = value;
        } else {
          ndarray->data[i].i = value;
        }
      });
    } 
  }

  // Convert type_flag to Blaze DataType
  DataType GetDataType(int32_t type_flag) {
    if (type_flag == 0) {
      return kFloat;
    } else if (type_flag == 1) {
      return kDouble;
    } else if (type_flag == 2) {
      return kFloat16;
    } else if (type_flag == 3) {
      return kUInt8;
    } else if (type_flag == 4) {
      return kInt32;
    } else if (type_flag == 5) {
      return kInt8;
    } else if (type_flag == 6) {
      return kInt64;
    } else {
      BLAZE_THROW("Unkown type_flag=", type_flag);
    }
  }
};

// Convert MXNet model into blaze model.
class MXNetImporter : public ModelImporter {
 public:
  MXNetImporter();

  // Load mxnet model from json and ndarray data.
  virtual void LoadModel(const char* conf_file, const char* data_file);

  // Process OPNode Function
  typedef std::function<void(const JSONNode& node)> ProcessOpNodeFunction;

 protected:
  void LoadModel(std::istream& is, const char* data_file);

  // MXNet2Blaze
  void MXNet2Blaze();
  // Create constant fill node based on data file
  void CreateConstantFillNode();
  // Create OP Nodes(which are not constatnt fill op)
  void CreateOpNode();

  // The Op Conversion Function
  void ProcessNullOp(const JSONNode& node);
  void ProcessConcatOp(const JSONNode& node);
  void ProcessFullyConnectedOp(const JSONNode& node);
  void ProcessBatchNormOp(const JSONNode& node);
  void ProcessActivationOp(const JSONNode& node);
  void ProcessRMimusScalarOp(const JSONNode& node);
  void ProcessElemwiseMulOp(const JSONNode& node);
  void ProcessElemwiseDivOp(const JSONNode& node);
  void ProcessBroadcastMulOp(const JSONNode& node);
  void ProcessElemwiseAddOp(const JSONNode& node);
  void ProcessSoftmaxOutputOp(const JSONNode& node);
  void ProcessTakeOp(const JSONNode& node);
  void ProcessSliceAxisOp(const JSONNode& node);
  void ProcessSumOp(const JSONNode& node);
  void ProcessReshapeOp(const JSONNode& node);
  void ProcessBatchdotOp(const JSONNode& node);
  void ProcessSoftmaxActivationOp(const JSONNode& node);
  void ProcessPlusScalarOp(const JSONNode& node);
  void ProcessSliceChannelOp(const JSONNode& node);
  void ProcessZerosOp(const JSONNode& node);
  void ProcessEqualScalarOp(const JSONNode& node);
  void ProcessNotEqualScalarOp(const JSONNode& node);
  void ProcessLeakyReLUOp(const JSONNode& node);
  void ProcessDivScalarOp(const JSONNode& node);
  void ProcessRDivScalarOp(const JSONNode& node);
  void ProcessMinusScalarOp(const JSONNode& node);
  void ProcessMulScalarOp(const JSONNode& node);
  void ProcessPowerScalarOp(const JSONNode& node);
  void ProcessBroadcastToOp(const JSONNode& node);
  void ProcessOnesOp(const JSONNode& node);
  void ProcessBroadcastAddOp(const JSONNode& node);
  void ProcessAddNOp(const JSONNode& node);
  // ...

  OperatorDef* AddOperatorDef(const JSONNode& node, const char* op_type, int onum = 1); 
  
  float GetFloatAttr(const JSONNode& node, const char* key, float default_val);
  int64_t GetIntegerAttr(const JSONNode& node, const char* key, int64_t default_val);
  const char* GetStrAttr(const JSONNode& node, const char* key, const char* default_val);

  void SetProcessNodeFunction(const std::string& name, ProcessOpNodeFunction function);
  DataType Str2DataType(const char* data_type_str);

  // The json graph and model params of MXNet
  JSONGraph jgraph_;
  MXParam mparam_;

  std::unordered_map<std::string, ProcessOpNodeFunction> op_process_func_map_;

  friend class XdlImporter;
};

}  // namespace blaze
