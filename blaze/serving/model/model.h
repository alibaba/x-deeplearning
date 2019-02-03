#ifndef __SERVING_MODEL_H__
#define __SERVING_MODEL_H__

#include <string>

#include "blaze/api/cpp_api/predictor.h"

namespace serving {

using namespace blaze;

class Model {
public:
  Model() {};
  ~Model() {};

  bool Init(const std::string& dense_model,
            PredictDataType data_type = kPDT_Float);

  bool Init(const std::string& sparse_model,
            const std::string& dense_model,
            PredictDataType data_type = kPDT_Float);

  Predictor * CreatePredictor(PredictDeviceType device_type = kPDT_Unkown,
                              int device_id = 0 );

private:
  PredictorManager pm_;
};

} // namespace serving

#endif
