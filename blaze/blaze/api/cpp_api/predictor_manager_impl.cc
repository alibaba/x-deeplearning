/*!
 * \file predictor_manager_impl.cc
 * \brief The blaze predictor manager cpp interface implementation.
 */
#include "blaze/api/cpp_api/predictor_manager_impl.h"

#include "blaze/api/cpp_api/predictor_impl.h"
#include "blaze/common/proto_helper.h"
#include "blaze/model_importer/ulf_importer.h"
#include "blaze/model_importer/onnx_importer.h"
#include "blaze/model_importer/mxnet_importer.h"
#include "blaze/model_importer/tensorflow_importer.h"
#include "blaze/model_importer/xdl_importer.h"
#include "blaze/model_importer/xdl_ulf_importer.h"

namespace blaze {

bool PredictorManagerImpl::LoadSparseModelWeight(const char* uri, const char* ps_puller_type) {
  sparse_db_uri_ = uri;
  ps_puller_type_ = ps_puller_type;

  sparse_puller_.reset(
      SparsePullerCreationRegisterer::Get()->CreateSparsePuller(ps_puller_type_));
  if (store::kOK != sparse_puller_->Load(sparse_db_uri_)) {
    LOG_ERROR("load sparse model %s failed", sparse_db_uri_.c_str());
    return false;
  }
  return true;
}

bool PredictorManagerImpl::LoadModel(
    const char* model_conf, const char* model_data, ModelType model_type, bool optimization_pass) {
  model_conf_ = model_conf;
  model_data_ = model_data;

  try {
    switch (model_type) {
      case kUlf:
        // load unified layer format model defined in blaze, for manaually
        // optimization.
        {
          ULFImporter ulf_importer;
          ulf_importer.set_data_type(data_type_);
          ulf_importer.LoadModel(model_conf_.c_str(), model_data_.c_str());
          net_def_ = ulf_importer.net_def();
        }
        break;
      case kBlaze:
        {
          bool success = NetDefHelper::LoadNetDefFromBinaryFile(model_conf, &net_def_);
          if (!success) {
            LOG_ERROR("Load model %s failed", model_conf);
            return false;
          }
        }
        break;
      case kOnnx:
        // load onnx model
        {
          ONNXImporter onnx_importer;
          onnx_importer.set_data_type(data_type_);
          onnx_importer.LoadModel(model_conf, model_data);
          net_def_ = onnx_importer.net_def();
        }
        break;
      case kMxnet:
        // load mxnet model
        {
          MXNetImporter mxnet_importer;
          mxnet_importer.set_data_type(data_type_);
          mxnet_importer.LoadModel(model_conf, model_data);
          net_def_ = mxnet_importer.net_def();
        }
        break;
      case kTensorFlow:
        // load tensorflow model
        {
          TensorFlowImporter tensorflow_importer;
          tensorflow_importer.set_data_type(data_type_);
          tensorflow_importer.LoadModel(model_conf, model_data);
          net_def_ = tensorflow_importer.net_def();
        }
        break;
      case kXDL:
        // load xdl model
        {
          XdlImporter xdl_importer;
          xdl_importer.set_data_type(data_type_);
          xdl_importer.LoadModel(model_conf, model_data);
          net_def_ = xdl_importer.net_def();
        }
        break;
      case kXDLUlf:
        // load xdl ulf model
        {
          XdlULFImporter xdl_ulf_importer;
          xdl_ulf_importer.set_data_type(data_type_);
          xdl_ulf_importer.LoadModel(model_conf, model_data);
          net_def_ = xdl_ulf_importer.net_def();
        }
        break;
      default:
        {
          LOG_FATAL("Unkown model_type: %d", model_type);
        }
        break;
    }
    if (optimization_pass) {
      // pass optimization.
      net_def_ = Optimizer::Get()->RunPass(net_def_); 
    }
  } catch (std::exception& e) {
    LOG_ERROR("load model conf=%s data=%s failed, msg=%s",
              model_conf_.c_str(), model_data_.c_str(), e.what());
    return false;
  }
  return true;
}

Predictor* PredictorManagerImpl::CreatePredictor(PredictDeviceType predict_device, int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    if (device_id < 0) device_id = 0;
    DeviceOption device_option;

    switch (predict_device) {
      case kPDT_CPU:
        device_option.set_device_type(kCPU);
        device_option.set_device_id(0);
        break;
      case kPDT_CUDA:
        device_option.set_device_type(kCUDA);
        device_option.set_device_id(device_id);
        break;
      default:
        ProbeDevice(&device_option);
        break;
    }
    int device_type = device_option.device_type();
    int device_id = device_option.device_id();

    if (workspace_[device_type][device_id].get() == nullptr) {
      workspace_[device_type][device_id].reset(new Workspace());
      workspace_[device_type][device_id]->Init(net_def_);

      auto mutable_device_option =
          workspace_[device_type][device_id]->net_def()->mutable_device_option();
      mutable_device_option->set_device_type(device_type);
      mutable_device_option->set_device_id(device_id);

      workspace_[device_type][device_id]->SetSparsePuller(sparse_puller_);
    }
    std::shared_ptr<Net> net = workspace_[device_type][device_id]->CreateNet();
    return new Predictor(new PredictorImpl(net));
  } catch (std::exception& e) {
    LOG_ERROR("Create Model Predictor failed, %s msg=%s",
              model_conf_.c_str(), e.what());
    return nullptr;
  }
}

void PredictorManagerImpl::ProbeDevice(DeviceOption* device_option) {
#ifdef USE_CUDA
  static int current_device_id = 0;
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count != 0) {
    device_option->set_device_type(kCUDA);
    device_option->set_device_id(current_device_id++ % count);
    // automaticlly allocate device id.
    return;
  }
#endif
  // at last use CPU prediction
  device_option->set_device_type(kCPU);
  device_option->set_device_id(0);
}

}  // namespace blaze
