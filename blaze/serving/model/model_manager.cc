#include <iostream>
#include <string>
#include <fcntl.h>

#include "./model_manager.h"
#include "predict.pb.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"


namespace serving {

ModelManager* ModelManager::Instance() {
  static ModelManager g_model_manager;
  return &g_model_manager;
}


bool ModelManager::Init(const std::string &config_file) {
  serving::Config config;

  int fd = open(config_file.c_str(), O_RDONLY);
  google::protobuf::io::FileInputStream file_input(fd);
  google::protobuf::TextFormat::Parse(&file_input, &config);

  for(const serving::ModelConfig& model_config: config.models()) {
    Model& model = model_version_map_[model_config.model_version()];

    if (model_config.has_sparse_model()) {
      if (!model.Init(model_config.sparse_model(), model_config.dense_model())) {
        std::cerr << "[Error] init <" << model_config.model_version() << "> failed." << std::endl;
        return false;
      };
    } else {
      if (!model.Init(model_config.dense_model())) {
        std::cerr << "[Error] init <" << model_config.model_version() << "> failed." << std::endl;
        return false;
      };
    }
  }

  return true;
}

Predictor* ModelManager::CreatePredictor(const std::string& model_version) {
  Model& model = model_version_map_[model_version];
  return model.CreatePredictor();
}

}
