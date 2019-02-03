/*
 * \file c_api_blaze_converter.cc
 * Convert ONNX2 model file to Blaze model file.
 */
#include "blaze/api/c_api/c_api_blaze_converter.h"

#include "blaze/api/cpp_api/predictor.h"
#include "blaze/api/c_api/c_api_error.h"
#include "blaze/model_importer/onnx_importer.h"
#include "blaze/model_importer/mxnet_importer.h"
#include "blaze/model_importer/tensorflow_importer.h"
#include "blaze/model_importer/xdl_importer.h"
#include "blaze/model_importer/xdl_ulf_importer.h"
#include "blaze/model_importer/ulf_importer.h"
#include "blaze/common/exception.h"

using blaze::ONNXImporter;
using blaze::XdlImporter;
using blaze::MXNetImporter;
using blaze::ULFImporter;
using blaze::TensorFlowImporter;
using blaze::XdlULFImporter;

#ifndef SET_TYPE
#define SET_TYPE(handle)                                                                   \
    handle.set_weight_type(static_cast<blaze::DataType>(weight_type));                     \
    handle.set_data_type(static_cast<blaze::DataType>(io_type));                           \
    for (auto i = 0; i < x_type_num; ++i) {                                                \
      handle.set_op_weight_type(x_type_key[i], static_cast<blaze::DataType>(x_type_value[i]));    \
    }
#endif

// Convert mxnet model to blaze model
int Blaze_ConvertBlaze(const char* model_conf_file,
                       const char* model_data_file,
                       int model_type,
                       int weight_type,
                       int io_type,
                       int x_type_num,
                       const char** x_type_key,
                       const int* x_type_value,
                       const char* blaze_model_file,
                       int binary) {
  try {
    switch (model_type) {
      case blaze::kUlf:
        {
          ULFImporter ulf_importer;
          SET_TYPE(ulf_importer)
          ulf_importer.LoadModel(model_conf_file, model_data_file);
          if (binary) ulf_importer.SaveToBinaryFile(blaze_model_file);
          else ulf_importer.SaveToTextFile(blaze_model_file);
        }
        break;
      case blaze::kOnnx:
        {
          ONNXImporter onnx_importer;
          SET_TYPE(onnx_importer)
          onnx_importer.LoadModel(model_conf_file, model_data_file);
          if (binary) onnx_importer.SaveToBinaryFile(blaze_model_file);
          else onnx_importer.SaveToTextFile(blaze_model_file);
        }
        break;
      case blaze::kMxnet:
        {
          MXNetImporter mxnet_importer;
          SET_TYPE(mxnet_importer)
          mxnet_importer.LoadModel(model_conf_file, model_data_file);
          if (binary) mxnet_importer.SaveToBinaryFile(blaze_model_file);
          else mxnet_importer.SaveToTextFile(blaze_model_file);
        }
        break;
      case blaze::kTensorFlow:
        {
          TensorFlowImporter tf_importer;
          SET_TYPE(tf_importer)
          tf_importer.LoadModel(model_conf_file, model_data_file);
          if (binary) tf_importer.SaveToBinaryFile(blaze_model_file);
          else tf_importer.SaveToTextFile(blaze_model_file);
        }
        break;
      case blaze::kXDL:
        {
          XdlImporter xdl_importer;
          SET_TYPE(xdl_importer)
          xdl_importer.LoadModel(model_conf_file, model_data_file);
          if (binary) xdl_importer.SaveToBinaryFile(blaze_model_file);
          else xdl_importer.SaveToTextFile(blaze_model_file);
        }
        break;
      case blaze::kXDLUlf:
        {
           XdlULFImporter xdl_ulf_importer;
           SET_TYPE(xdl_ulf_importer);
           xdl_ulf_importer.LoadModel(model_conf_file, model_data_file);
           if (binary) xdl_ulf_importer.SaveToBinaryFile(blaze_model_file);
           else xdl_ulf_importer.SaveToTextFile(blaze_model_file);
        }
        break;
      default:
        {
          Blaze_SetLastErrorString("Unkown model_type:%d", model_type);
          return -1;
        }
        break;
    }
  } catch (blaze::Exception& e) {
    LOG_ERROR("Convert failed: %s", e.what());
    Blaze_SetLastErrorString("%s", e.what());
    return -1;
  }
  return 0;
}

#undef SET_TYPE
