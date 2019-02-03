/*
 * \file xdl_ulf_importer.h
 * \brief The xdl ulf importer
 */
#pragma once

#include <unordered_map>
#include <functional>

#include "blaze/model_importer/ulf_importer.h"
#include "blaze/model_importer/mxnet_importer.h"
#include "blaze/model_importer/tensorflow_importer.h"

namespace blaze {

class XdlULFImporter : public ULFImporter {
 public:
  XdlULFImporter();

  virtual void LoadModel(const char* conf_file, const char* data_file);

 protected:
  void LoadNetWeightsFromMxnet(const char* data_file);
  void LoadNetWeightsFromTF(const char* data_file);

  MXParam mx_param_;
  TFParam tf_param_;
};

}  // namespace blaze

