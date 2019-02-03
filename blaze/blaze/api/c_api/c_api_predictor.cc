/*
 * \file c_api_predictor.cc
 */
#include "blaze/api/c_api/c_api_predictor.h"

#include <string.h>

#include "blaze/api/c_api/c_api_error.h"
#include "blaze/api/cpp_api/predictor.h"
#include "blaze/common/log.h"
#include "blaze/common/thread_local.h"

using blaze::Predictor;
using blaze::PredictDeviceType;
using blaze::PredictorManager;

int Blaze_InitScheduler(int enable_batching,
                        int max_batch_size,
                        int batch_timeout_micros,
                        int num_threads_for_cpu,
                        int num_threads_for_cuda,
                        int num_threads_for_pipe) {
  if (!blaze::InitScheduler(enable_batching,
                            max_batch_size,
                            batch_timeout_micros,
                            num_threads_for_cpu,
                            num_threads_for_cuda,
                            num_threads_for_pipe)) {
    Blaze_SetLastErrorString("InitScheduler Failed");
    return -1;
  }
  return 0;
}

// Load model of PredictorManager
int Blaze_PredictorManagerCreate(PredictorManagerHandle* handle) {
  PredictorManager* pm = new PredictorManager();
  if (pm == nullptr) {
    Blaze_SetLastErrorString("Create PredictorManager Failed");
    return -1;
  }
  *handle = pm;
  return 0;
}

int Blaze_PredictorManagerDelete(PredictorManagerHandle handle) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(handle);
  delete pm;
  return 0;
}

int Blaze_PredictorManagerSetDataType(PredictorManagerHandle handle,
                                      int data_type) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(handle);
  if (pm == nullptr) {
    Blaze_SetLastErrorString("SetDataType: %d", data_type);
    return -1;
  }
  pm->SetDataType(static_cast<blaze::PredictDataType>(data_type));
  return 0;
}

int Blaze_PredictorManagerSetRunMode(PredictorManagerHandle handle,
                                      const char* run_mode) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(handle);
  if (pm == nullptr) {
    Blaze_SetLastErrorString("SetRunMode: %s", run_mode);
    return -1;
  }
  pm->SetRunMode(run_mode);
  return 0;
}

int Blaze_LoadSparseModelWeight(PredictorManagerHandle handle,
                                const char* sparse_model_weight_file) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(handle);
  if (pm == nullptr) {
    Blaze_SetLastErrorString("Load blaze model: %s failed", sparse_model_weight_file);
    return -1;
  }
  pm->LoadSparseModelWeight(sparse_model_weight_file);
  return 0;
}

int Blaze_PredcitorManagerLoadModel(PredictorManagerHandle handle,
                                    const char* blaze_model_file,
                                    int optimization_pass) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(handle);
  if (pm == nullptr) {
    Blaze_SetLastErrorString("Load blaze model: %s failed", blaze_model_file);
    return -1;
  }
  pm->LoadModel(blaze_model_file, optimization_pass);
  return 0;
}

int Blaze_PredictorManagerLoadDeepNetModel(PredictorManagerHandle handle,
                                           const char* deepnet_conf_file,
                                           const char* deepnet_param_file,
                                           int optimization_pass) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(handle);
  if (pm == nullptr) {
    Blaze_SetLastErrorString("The predict manager is nullptr");
    return -1;
  }
  bool ret = pm->LoadDeepNetModel(deepnet_conf_file, deepnet_param_file, optimization_pass);
  if (!ret) {
    Blaze_SetLastErrorString("Load deepnet model: %s %s failed", deepnet_conf_file, deepnet_param_file);
    return -1;
  }
  return 0;
}

// Predicator creation
int Blaze_PredictorCreate(PredictorManagerHandle predictor_mgr,
                          int device_type,
                          int device_id,
                          PredictorHandle* handle) {
  PredictorManager* pm = reinterpret_cast<PredictorManager*>(predictor_mgr);
  PredictDeviceType pdt = static_cast<PredictDeviceType>(device_type);
  Predictor* predictor = pm->CreatePredictor(pdt, device_id);
  if (predictor == nullptr) {
    Blaze_SetLastErrorString("Create Predictor Failed");
    return -1;
  }
  *handle = predictor;
  return 0;
}

int Blaze_PredictorDelete(PredictorHandle handle) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  delete predictor;
  return 0;
}

int Blaze_PredictorReshapeInput(PredictorHandle handle,
                                const char* name,
                                const int* data,
                                int num) {
  std::vector<size_t> shape;
  for (int i = 0; i < num; ++i) shape.push_back(data[i]);
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) {
    Blaze_SetLastErrorString("The predictor is nullptr");
    return -1;
  }
  bool ret = predictor->ReshapeInput(name, shape);
  if (!ret) {
    Blaze_SetLastErrorString("Reshape input: %s failed", name);
    return -1;
  }
  return 0;
}

int Blaze_PredictorFeed(PredictorHandle handle,
                        const char* name,
                        void* data,
                        int len) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) {
    Blaze_SetLastErrorString("The predictor is nullptr");
    return -1;
  }
  int data_type;
  if (!predictor->Feed(name, data, len)) {
    Blaze_SetLastErrorString("Feed %s failed", name);
    return -1;
  }
  return 0;
}

int Blaze_PredictorForward(PredictorHandle handle) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  if (!predictor->Forward()) {
    Blaze_SetLastErrorString("Forward failed");
    return -1;
  }
  return 0;
}

int Blaze_PredictorOutputShape(PredictorHandle handle,
                               const char* name,
                               size_t* ndim,
                               size_t** nshape) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  const std::vector<size_t>& shape = predictor->OutputShape(name);
  *ndim = shape.size();
  *nshape = const_cast<size_t*>(shape.data());
  return 0;
}

int Blaze_PredictorOutputDataType(PredictorHandle handle,
                                  const char* name,
                                  int* data_type) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  *data_type = predictor->OutputDataType(name);
  return 0;
}

int Blaze_PredictorOutput(PredictorHandle handle,
                          const char* name,
                          void* data,
                          size_t size) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  size_t slen = 0;
  void* dptr = nullptr;
  predictor->Output(name, &dptr, &slen);;
  memcpy(data, dptr, slen);
  return 0;
}

int Blaze_PredictorParamShape(PredictorHandle handle,
                              const char* name,
                              size_t* ndim,
                              size_t** nshape) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  const std::vector<size_t>& shape = predictor->InternalShape(name);
  *ndim = shape.size();
  *nshape = const_cast<size_t*>(shape.data());
  return 0;
}

int Blaze_PredictorParamDataType(PredictorHandle handle,
                                 const char* name,
                                 int* data_type) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  *data_type = predictor->InternalDataType(name);
  return 0;
}

int Blaze_PredictorParam(PredictorHandle handle,
                         const char* name,
                         void* data,
                         size_t len) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  size_t slen = 0;
  void* dptr = nullptr;
  predictor->InternalParam(name, &dptr, &slen);;
  memcpy(data, dptr, slen);
  return 0;
}

#define kMaxBufSize 1024 * 1024

struct StringVectorEntry {
  std::vector<char*> names;
  std::vector<char*> keys, values;
  char strings[kMaxBufSize];
};

typedef blaze::ThreadLocalStore<StringVectorEntry> ThreadLocalStringStore;

int Blaze_PredictorParamName(PredictorHandle handle,
                             size_t* ndim,
                             const char*** name) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  std::vector<std::string> names = predictor->ListInternalName();

  char* head = ThreadLocalStringStore::Get()->strings;
  std::vector<char*>& lnames = ThreadLocalStringStore::Get()->names;
  lnames.clear();

  for (const auto& oname : names) {
    lnames.push_back(head);
    memcpy(head, oname.c_str(), oname.length());
    head += oname.length();
    head[0] = '\0';
    head += 1;
  }
  *ndim = names.size();
  *name = (const char**)(lnames.data());
  return 0;
}

int Blaze_PredictorListInputNames(PredictorHandle handle,
                                  size_t* ndim,
                                  const char*** names) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  
  const std::vector<std::string>& input_names = predictor->ListInputName();
  char* head = ThreadLocalStringStore::Get()->strings;
  std::vector<char*>& lnames = ThreadLocalStringStore::Get()->names;
  lnames.clear();

  for (const auto& iname : input_names) {
    lnames.push_back(head);
    memcpy(head, iname.c_str(), iname.length());
    head += iname.length();
    head[0] = '\0';
    head += 1;
  }
  *ndim = input_names.size();
  *names = (const char**)(lnames.data());
  return 0;
}

int Blaze_PredictorListOutputNames(PredictorHandle handle,
                                   size_t* ndim,
                                   const char*** names) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  
  const std::vector<std::string>& output_names = predictor->ListOutputName();
  char* head = ThreadLocalStringStore::Get()->strings;
  std::vector<char*>& lnames = ThreadLocalStringStore::Get()->names;
  lnames.clear();

  for (const auto& oname : output_names) {
    lnames.push_back(head);
    memcpy(head, oname.c_str(), oname.length());
    head += oname.length();
    head[0] = '\0';
    head += 1;
  }
  *ndim = output_names.size();
  *names = (const char**)(lnames.data());
  return 0;
}

int Blaze_PredictorRegisterObservers(PredictorHandle handle,
                                     size_t ndim,
                                     const char** names) {
  std::vector<std::string> observer_names;
  for (size_t i = 0; i < ndim; ++i) {
    observer_names.push_back(names[i]);
  }
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;
  predictor->RegisterObservers(observer_names);
  return 0;
}

int Blaze_PredictorDumpObservers(PredictorHandle handle,
                                 size_t* ndim,
                                 const char*** key,
                                 const char*** value) {
  Predictor* predictor = reinterpret_cast<Predictor*>(handle);
  if (predictor == nullptr) return -1;

  std::unordered_map<std::string, std::string> dump_map;
  predictor->DumpObservers(&dump_map);
  
  char* head = ThreadLocalStringStore::Get()->strings;
  
  std::vector<char*>& lkeys = ThreadLocalStringStore::Get()->keys;
  lkeys.clear();
  std::vector<char*>& lvalues = ThreadLocalStringStore::Get()->values;
  lvalues.clear();

  for (const auto& iter : dump_map) {
    const std::string& key = iter.first;
    const std::string& value = iter.second;

    lkeys.push_back(head);
    memcpy(head, key.c_str(), key.length());
    head += key.length();
    head[0] = '\0';
    head += 1;

    lvalues.push_back(head);
    memcpy(head, value.c_str(), value.length());
    head += value.length();
    head[0] = '\0';
    head += 1;
  }
  *ndim = dump_map.size();
  *key = (const char**)(lkeys.data());
  *value = (const char**)(lvalues.data());
  return 0;
}

