/*!
 * \file predictor.cc
 * \brief The blaze predictor cpp interface implementation.
 */
#include "blaze/api/cpp_api/predictor.h"

#include <string>
#include <mutex>

#include "blaze/common/common_defines.h"
#include "blaze/optimizer/optimizer.h"
#include "blaze/store/sparse_puller.h"

namespace blaze {

class PredictorManagerImpl {
 public:
  PredictorManagerImpl() : data_type_(kFloat) { }

  // Set DataType
  void SetDataType(DataType data_type) { data_type_ = data_type; }
  // Set run mode
  void SetRunMode(const char* run_mode) { net_def_.set_run_mode(run_mode); }
  // load sparse model weight
  bool LoadSparseModelWeight(const char* uri, const char* ps_puller_type);
  // Load model
  bool LoadModel(const char* model_conf, const char* model_data, ModelType model_type, bool optimization_pass);
  // Create new predictor handle
  Predictor* CreatePredictor(PredictDeviceType predict_device, int device_id);

 protected:
  // Probe the available device.
  void ProbeDevice(DeviceOption* device_option);

  static const int kMaxNumDevice = 4;
  static const int kMaxNumDeviceId = 8;
  std::shared_ptr<Workspace> workspace_[kMaxNumDevice][kMaxNumDeviceId];

  DataType data_type_;
  NetDef net_def_;  // The model graph

  std::string model_conf_, model_data_;
  std::string sparse_db_uri_, ps_puller_type_;
  std::shared_ptr<SparsePuller> sparse_puller_;  // The sparse puller.
  std::mutex mutex_;
};

}  // namespace blaze
