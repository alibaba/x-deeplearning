/*
 * \file xdl_ulf_importer.cc
 * \brief The xdl ulf importer
 */
#include "blaze/model_importer/xdl_ulf_importer.h"

#include "blaze/common/exception.h"
#include "blaze/common/proto_configure.h"
#include "blaze/common/log.h"

namespace blaze {

XdlULFImporter::XdlULFImporter() : ULFImporter() {
  net_def_.set_run_mode("hybrid");
}

void XdlULFImporter::LoadModel(const char* conf_file, const char* data_file) {
  conf_file_ = conf_file;

  ProtoConfigure config;
  auto rc = config.Init("ulf.NetParameter", conf_file);
  if (rc != ProtoConfigure::kOK) {
    BLAZE_THROW("load model ulf.NetParameter from file:", conf_file, " failed");
  }
  net_conf_ = *(reinterpret_cast<const ulf::NetParameter*>(config.config()));

  if (net_conf_.backend() == "mxnet") {
    // if backend is mxnet
    LoadNetWeightsFromMxnet(data_file);
  } else if (net_conf_.backend() == "tensorflow") {
    // if backend is tensorflow
    LoadNetWeightsFromTF(data_file);
  } else {
    BLAZE_THROW("Unkown backend ", net_conf_.backend());
  }

  if (!Ulf2Blaze()) {
    BLAZE_THROW("ulf2blaze failed");
  }
}

void XdlULFImporter::LoadNetWeightsFromMxnet(const char* data_file) {
  FileStream stream(data_file, true);
  mx_param_.Load(&stream);

  std::unordered_map<std::string, MXParam::NDArray> param_map;
  for (auto i = 0; i < mx_param_.keys.size(); ++i) {
    const auto& name = mx_param_.keys[i];
    const auto& ndarray = mx_param_.ndarray[i];
    param_map[name] = ndarray;
  }

  for (const auto& layer_param : net_conf_.layer_params()) {
    if (layer_param.arg_names_size() <= 0) continue;

    auto layer_weight_parameter = net_param_.add_layer_weights_params();
    layer_weight_parameter->set_name(layer_param.name());

    for (const auto& name : layer_param.arg_names()) {
      auto blob_data = layer_weight_parameter->add_blob_datas();
      auto iter = param_map.find(name);
      CHECK(iter != param_map.end(), "name is not contained in mxnet data=", name.c_str());

      auto raw_blob = iter->second;
      for (const auto& dim : raw_blob.shape) {
        blob_data->add_shape(dim);
      }
      for (const auto& data : raw_blob.data) {
        blob_data->add_data(data.f);
      }
    }
  }
}

void XdlULFImporter::LoadNetWeightsFromTF(const char* data_file) {
  tf_param_.Load(data_file);

  std::unordered_map<std::string, TFParam::NDArray> param_map;
  for (auto i = 0; i < tf_param_.keys.size(); ++i) {
    const auto& name = tf_param_.keys[i];
    const auto& ndarray = tf_param_.ndarray[i];
    param_map[name] = ndarray;
    LOG_INFO("name=%s", name.c_str());
  }

  for (const auto& layer_param : net_conf_.layer_params()) {
    if (layer_param.arg_names_size() <= 0) continue;

    auto layer_weight_parameter = net_param_.add_layer_weights_params();
    layer_weight_parameter->set_name(layer_param.name());

    for (const auto& name : layer_param.arg_names()) {
      auto blob_data = layer_weight_parameter->add_blob_datas();
      auto iter = param_map.find(name);
      CHECK(iter != param_map.end(), "name is not contained in mxnet data=", name.c_str());

      auto raw_blob = iter->second;
      for (const auto& dim : raw_blob.shape) {
        blob_data->add_shape(dim);
      }
      for (const auto& data : raw_blob.data) {
        blob_data->add_data(data.f);
      }
    }
  }
}

}  // namespace blaze

