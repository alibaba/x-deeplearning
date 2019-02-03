#include "./model.h"

namespace serving {

bool Model::Init(const std::string& dense_model,
                 PredictDataType data_type) {
  if (!pm_.LoadModel(dense_model.c_str(), true))
    return false;
  pm_.SetDataType(data_type);

  return true;
}

bool Model::Init(const std::string& sparse_model,
                 const std::string& dense_model,
                 PredictDataType data_type) {
  if (!pm_.LoadSparseModelWeight(sparse_model.c_str()))
    return false;

  if (!pm_.LoadModel(dense_model.c_str(), true))
    return false;

  pm_.SetDataType(data_type);

  return true;
}

Predictor* Model::CreatePredictor( PredictDeviceType device_type,
                                   int device_id) {
  return pm_.CreatePredictor(device_type, device_id);
};



}
