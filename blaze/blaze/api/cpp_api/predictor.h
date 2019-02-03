/*!
 * \file predictor.h
 * \brief The blaze predictor cpp interface.
 */
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

namespace blaze {

class PredictorImpl;
class PredictorManagerImpl;

// The predict device
enum PredictDeviceType {
  kPDT_Unkown = -1,
  kPDT_CPU,
  kPDT_CUDA,
};

// The predict data type for input/output for each op.
enum PredictDataType {
  kPDT_Invalid = -1,
  kPDT_Float = 1,
  kPDT_Float16 = 12,
};

// The model type supported by blaze
enum ModelType {
  kBlaze = 0,
  kUlf,
  kOnnx,
  kMxnet,
  kTensorFlow,
  kXDL,
  kXDLUlf,
};

// The feature type of blaze input.
enum FeatureType {
  kDenseFeature = 0,
  kSparseFeature,
  kAuxIndicator
};

// The sparse feature type
enum SparseFeatureType {
  kSparseFeatureId = 0,
  kSparseFeatureValue,
  kAuxSparseFeatureSegment
};

// The feed name config 
struct FeedNameConfig {
  std::string feature_name;
  FeatureType feature_type;
  int level;
  // if feature_type equals kSparseFeature, sparse_feature_type will be available
  SparseFeatureType sparse_feature_type;
};

class FeedNameUtility {
 public:
  
  // Convert sparse feature name to real feed name.
  // @param sparse_feature_name: The name of sparse feature,
  // such as: cookie, category, etc.
  // @param sft: The sparse feature type.
  //
  // for example: if sparse_feature_name="category" sfp=kSparseFeatureId, the
  // Return "category.ids"
  static std::string SparseFeatureName2FeedName(
      const std::string& sparse_feature_name, SparseFeatureType sft);

  // Convert indicator level 2 feed name
  // @Param level: The level of indicator, 0 is ncommon, 1 is common
  // @Return the feed name of indicator level
  //
  // for example: if the level is 0, the feed name is "indicator.0"
  static std::string IndicatorLevel2FeedName(int level);
};

// The predictor forward async callback function.
using PredictorCallback = std::function<void()>;  

// The predictor interface of blaze, which is not thread-safe,
// The working thread of online-server should own one predictor handle.
class Predictor {
 public:
  virtual ~Predictor();
  
  // for example: feedname="indicator.0"
  // Return the FeedNameConfig={ indicator, kAuxIndicator, 0 }
  FeedNameConfig GetFeedNameConfig(const std::string& feed_name);

  // Reshape the input tensor(only support dense tensor) of model.
  // @param name: The tenor name
  // @param shape: The shape of tensor
  // @Return True: success False: failed
  bool ReshapeInput(const char* name, const std::vector<size_t>& shape);
  // Rehspae the idx-th input tensor, the idx is the index of ListInputName
  // function.
  // @param idx: The idx-th input tensor in ListInputName()
  // @param shape: The shape of tensor.
  // @Return True: success False: failed
  bool ReshapeInput(size_t idx, const std::vector<size_t>& shape);

  // Feed the input tensor with name, should Reshape the input tensor first.
  // @param name: The tensor name
  // @param data: The tensor's host memory address
  // @param len: The tensor's length in byte.
  // @Return True: success False: failed
  bool Feed(const char* name, const void* data, size_t len);
  // Feed the idx-th input tensor, should Reshape the input tensor first.
  // @param idx: The idx-th input tensor
  // @param data: The tensor's host memory address
  // @param len: The tensor's length in byte.
  // @Return True: success False: failed
  bool Feed(size_t idx, const void* data, size_t len);
 
  // Return the input tensor count
  size_t InputSize() const;

  // Return input tensor's data type with name
  // @param name: The input tensor's name
  // @Return data_type, if equals kPDT_Unkown means failed.
  PredictDataType InputDataType(const char* name) const;
  // Return idx-th input tensor's data type
  // @param name: The idx of input tensor in ListInputName()
  // @Return data_type, if equals kPDT_Unkown means failed.
  PredictDataType InputDataType(size_t idx) const;

  // Return the vector of input tensor name.
  const std::vector<std::string>& ListInputName() const;

  // Forward execution.
  // @Return True: success False: failed
  bool Forward(const PredictorCallback&& cb = nullptr);

  // Get the raw data of output tensor.
  // @param name: The output tensor name
  // @param data: The address
  // @param len: The raw data length in byte.
  // @Return True: success False: failed
  bool Output(const char* name, void** data, size_t* len);
  // Get the raw data of idx-th output tensor.
  // @param name: The index
  // @param data: The address
  // @param len: The raw data length in byte.
  // @Return True: success False: failed
  bool Output(size_t idx, void** data, size_t* len); 

  // Get the shape of idx-th output tensor
  // @param idx: The index
  // @Return The shape of idx-th output tensor
  const std::vector<size_t>& OutputShape(size_t idx) const;
  // Get the shape of output tensor
  // @param name: The name of output tensor
  // @Return The shape of output tensor of name
  const std::vector<size_t>& OutputShape(const char* name) const;

  // Get output tensor datatype
  // @param idx: the index
  // @Return the data type 
  PredictDataType OutputDataType(size_t idx) const;
  // Get output tensor datatype of name
  // @param name: The tensor name
  // @Return the data type
  PredictDataType OutputDataType(const char* name) const;

  // Return the output tensor count.
  size_t OutputSize() const;
  
  // Return the vector of output tensor name
  const std::vector<std::string>& ListOutputName() const;

  // NOTE:The following API is used for debugging.
  // Return vector of the internal tensor name
  std::vector<std::string> ListInternalName() const;

  // Get the raw-data of internel tensor
  // @param name: The name of internel tensor
  // @param data: the raw data address
  // @param len: the length of raw-data in byte
  // @Return True: success False: failed
  bool InternalParam(const char* name, void** data, size_t* len);

  // Get the shape of internal tensor
  // @param name: The name of internal tensor
  // @Return Shape of internal tensor with name
  const std::vector<size_t>& InternalShape(const char* name) const;

  // Return the datatype of internal tensor
  // @param name: The name of internal tensor
  // @Return the data type in blaze, defines in blaze.proto, serving system
  // neednot call the interface.
  int InternalDataType(const char* name) const;

  // Register obersers for internal observer,
  // Supported observer names: profile/cost 
  // @param observer_names: The obersver name list
  void RegisterObservers(const std::vector<std::string>& oberver_names);

  // Dump observer result.
  // @param dump_map: Save the observer result into dump_map
  void DumpObservers(std::unordered_map<std::string, std::string>* dump_map);

 protected:
  Predictor(PredictorImpl* impl);
  
  PredictorImpl* impl_;
  friend class PredictorManagerImpl;
};

// The predictor manager for managing one model's multiple predict handle.
// It doesn't support model hot switch, When model update, you should create
// a new PredictorManager instance.
class PredictorManager {
 public:
  PredictorManager();
  virtual ~PredictorManager();

  // Set data type for each op's input and output
  void SetDataType(PredictDataType data_type);

  // Set the run mode
  void SetRunMode(const char* run_mode);

  // set large-scale sparse model's weight.
  // @param uri: The sparse model uri
  // @param type: The sparse model storage backend type
  bool LoadSparseModelWeight(const char* uri, const char* type = "qed_sparse_puller");

  // load model of blaze format for online-serving.
  //
  // NOTE: if the model contains sparse model op, such as: Embedding,
  // you should load sparse model weight first. 
  //
  // @param filename: The blaze dense model filename
  // @param optimization_pass: if true, do optimization pass.
  bool LoadModel(const char* filename, bool optimization_pass = true);
  
  // load a model of ulf format
  // NOTE: Used in old model-serving system now, LoadDeepNetModel(ULF) function will
  // be deprecated, please use LoadModelEx.
  //
  // @param model_conf: the ulf conf file path
  // @param model_data: the ulf data file path
  // @param model_type: the model type
  // @param optimization_pass: if true, do optimization pass.
  // @Return True: success False: failed
  bool LoadDeepNetModel(const char* model_conf, const char* model_data, bool optimization_pass = true);
 
  // load a model of a user-specified model_type
  //
  // @param conf_file: The model conf file
  // @param data_file: The model data file
  // @param mode_type: model_type defined in ModelType enum.
  // @param optimization_pass: if do optimization_pass.
  bool LoadModel(const char* conf_file, const char* data_file,
                 ModelType model_type, bool optimization_pass = true);

  // Create a predictor.
  // if predict_device_type is kUnkown, blaze will first probe used device_type defined in
  // net_def, if device_option not defined in net_def, blaze will automaticlly
  // probe the available devices.
  // @param predict_device_type: The device type
  // @param device_id: the id of device
  // @Return return the predictor handle, the caller should manage the handle.
  Predictor* CreatePredictor(PredictDeviceType predict_device_type = kPDT_Unkown, int device_id = 0);

 protected:
  PredictorManagerImpl* impl_;
};

// Init Scheduler
bool InitScheduler(bool enable_batching,
                   int max_batch_size,
                   int batch_timeout_micros,
                   int num_threads_for_cpu,
                   int num_threads_for_cuda,
                   int num_threads_for_pipe);

}  // namespace blaze

